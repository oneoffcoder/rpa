"""Two-stage LoRA for the pinned 12 Hz Qwen3-TTS Base on MLX."""

import hashlib
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.tuner.utils import linear_to_lora_layers

from . import mlx_backend as backend


ADAPTER_TYPE = "qwen3-tts-joint-lora-v1"
RESIDUAL_WEIGHT = 0.3
PROJECTIONS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
               "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
GROUPS = ("talker", "residual")


def stage(key):
    if not key.endswith((".lora_a", ".lora_b")):
        raise ValueError(f"Unexpected trainable parameter {key}")
    if key.startswith("model.layers."):
        return "talker"
    if key.startswith("code_predictor.model.layers."):
        return "residual"
    raise ValueError(f"Unexpected adapter location {key}")


def configure(wrapper, params):
    wrapper.full_model.freeze()
    talker = wrapper.full_model.talker
    options = {"rank": params["rank"], "scale": params["alpha"] / params["rank"],
               "dropout": params["dropout"], "keys": PROJECTIONS}
    for transformer in (talker.model, talker.code_predictor.model):
        linear_to_lora_layers(transformer, len(transformer.layers), options)
    weights = dict(tree_flatten(talker.trainable_parameters()))
    if {stage(k) for k in weights} != set(GROUPS):
        raise RuntimeError("Both talker and residual LoRA tensors must be trainable")
    if tree_flatten(wrapper.full_model.speech_tokenizer.trainable_parameters()):
        raise RuntimeError("Speech codec must remain frozen")
    return weights


def load(wrapper, directory: Path, params):
    expected = configure(wrapper, params)
    weights = mx.load(str(directory / "adapters.safetensors"))
    if set(weights) != set(expected) or any(
        v.shape != expected[k].shape or v.dtype != expected[k].dtype for k, v in weights.items()
    ):
        raise ValueError("Joint checkpoint does not match both sets of adapter tensors")
    wrapper.full_model.talker.load_weights(list(weights.items()), strict=False)
    wrapper.full_model.eval()
    mx.eval(wrapper.full_model.talker.trainable_parameters())


def tensor_hash(value):
    return hashlib.sha256(np.asarray(value.view(mx.uint8)).tobytes()).hexdigest()


def codec_hash(codec):
    digest = hashlib.sha256()
    for key, value in sorted(tree_flatten(codec.parameters())):
        digest.update(f"{key} {value.shape} {value.dtype} {tensor_hash(value)}\n".encode())
    return digest.hexdigest()


def training_example(wrapper, collator, sample):
    """Remove ignored prefix/EOS-input/padding positions using the original collator's labels."""
    batch = collator([sample])
    codes = wrapper.codec_adapter.encode_all_codebooks(sample["audio"]["array"],
                                                      sample["audio"]["sampling_rate"]).T
    labels = np.asarray(batch["labels"][0])
    active = np.flatnonzero(labels != -100)
    eos = wrapper.full_model.config.talker_config.codec_eos_token_id
    targets = np.append(np.asarray(codes[:, 0]), eos)
    if codes.shape[1] != 16 or active.size != len(targets) or active[0] < 1:
        raise ValueError("Unexpected codebook count or talker target layout")
    if not np.array_equal(labels[active], targets) or not np.all(np.diff(active) == 1):
        raise ValueError("Collator targets must contain contiguous code-0 frames and one EOS")
    # Last input is the last real frame and predicts EOS. Include the text prefix for attention.
    inputs = batch["inputs_embeds"][:, :int(active[-1])]
    return inputs, codes, mx.array(targets, dtype=mx.int32)


def logits(talker, inputs, codes):
    frames = codes.shape[0]
    first, hidden = talker(inputs)
    first = first[0, -(frames + 1):]
    # The hidden state that predicted code 0 conditions that SAME frame's depth model.
    frame_hidden = hidden[0, -(frames + 1):-1, None, :]
    predictor = talker.code_predictor
    depth_inputs = [frame_hidden, talker.get_input_embeddings()(codes[:, :1])]
    for group in range(1, codes.shape[1] - 1):
        depth_inputs.append(predictor.codec_embedding[group - 1](codes[:, group:group + 1]))
    depth = mx.concatenate(depth_inputs, axis=1)
    if predictor.small_to_mtp_projection is not None:
        depth = predictor.small_to_mtp_projection(depth)
    # Each real frame is an independent causal sequence [h, c0, ..., c14].
    depth = predictor.model(depth)
    residual = mx.stack([head(depth[:, j + 1]) for j, head in enumerate(predictor.lm_head)], axis=1)
    return first, residual


def prepare_examples(wrapper, rt, rows):
    collator = rt.TTSDataCollator(model=wrapper, tokenizer=wrapper.tokenizer, max_seq_length=2048,
                                  text_column="text", audio_column="audio")
    data = backend._AudioDataset(rows, np)[:]
    return {s["text"]: training_example(wrapper, collator, s) for s in data}


def fingerprints(talker, examples):
    return {text: [tensor_hash(x) for x in logits(talker, inputs, codes)]
            for text, (inputs, codes, _) in examples.items()}


def losses(talker, inputs, codes, targets):
    first, residual = logits(talker, inputs, codes)
    primary = nn.losses.cross_entropy(first.astype(mx.float32), targets, reduction="mean")
    secondary = nn.losses.cross_entropy(residual.astype(mx.float32), codes[:, 1:], reduction="mean")
    return primary + RESIDUAL_WEIGHT * secondary, (primary, secondary)


def score(talker, examples):
    sums = np.zeros((2, 3))
    for inputs, codes, targets in examples:
        first, residual = logits(talker, inputs, codes)
        for i, (scores, truth) in enumerate(((first, targets), (residual, codes[:, 1:]))):
            ce = nn.losses.cross_entropy(scores.astype(mx.float32), truth, reduction="sum")
            hits = (scores.argmax(-1) == truth).sum()
            sums[i] += [ce.item(), hits.item(), truth.size]
    return {name: {"ce": float(ce / count), "correct": int(hits), "targets": int(count)}
            for name, (ce, hits, count) in zip(GROUPS, sums)}


def norms_by_stage(weights):
    result = {}
    for group in GROUPS:
        arrays = [v for k, v in weights.items() if stage(k) == group]
        squared = mx.stack([mx.sum(mx.square(v.astype(mx.float32))) for v in arrays])
        values = np.asarray(squared)
        if not np.isfinite(values).all() or values.sum() <= 0:
            raise RuntimeError(f"{group} has no finite nonzero norm")
        result[group] = {"l2": float(np.sqrt(values.sum())), "nonzero_tensors": int((values > 0).sum()),
                         "tensors": len(arrays)}
    return result


def parity(wrapper, example, text):
    """Compare training logits with an independent replay of the cached inference calls."""
    inputs, codes, _ = example
    parallel = logits(wrapper.model, inputs, codes)
    full = wrapper.full_model
    talker = full.talker
    predictor = talker.code_predictor
    embed, trailing, pad = full._prepare_generation_inputs(text, language="auto")
    cache = talker.make_cache()
    first, depth = [], []
    for t in range(codes.shape[0] + 1):
        scores, h = talker(embed, cache=cache)
        first.append(scores[0, -1])
        if t == codes.shape[0]:  # EOS has no residual targets.
            break
        frame = codes[t:t + 1]
        audio = talker.get_input_embeddings()(frame[:, :1])
        entry = mx.concatenate([h[:, -1:], audio], axis=1)
        subcache = predictor.make_cache()
        heads = []
        for j in range(15):
            scores, subcache, _ = predictor(entry, cache=subcache, generation_step=j)
            heads.append(scores[0, -1])
            entry = predictor.codec_embedding[j](frame[:, j + 1:j + 2])
            audio = audio + entry
        depth.append(mx.stack(heads))
        embed = audio + (trailing[:, t:t + 1] if t < trailing.shape[1] else pad)
    result = {}
    for name, fast, incremental in zip(GROUPS, parallel, (mx.stack(first), mx.stack(depth))):
        a, b = (np.asarray(x.astype(mx.float32)) for x in (fast, incremental))
        relative = float(np.linalg.norm(a - b) / np.linalg.norm(b))
        if not np.isfinite(relative) or relative > 0.035:
            raise RuntimeError(f"{name} training and incremental logits disagree: {relative:.5f}")
        result[name] = relative
    return result
