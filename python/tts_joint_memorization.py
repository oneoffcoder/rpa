"""Train and evaluate two-stage LoRA on the six known training syllables."""
import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
import numpy as np

from rpa_tts import joint_mlx as joint, mlx_backend as backend
from tts_memorization import SEED, select_training_rows


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "docs/tts/memorization-six/measurements.json"
MILESTONES = (0, 25, 50, 75, 100, 125, 150, 200, 250, 300)


def checkpoint(output, step, wrapper, initial, meta, examples):
    folder = output / f"step-{step}"
    adapter = folder / "adapters"
    adapter.mkdir(parents=True)
    weights = dict(tree_flatten(wrapper.model.trainable_parameters()))
    delta = joint.norms_by_stage({k: v - initial[k] for k, v in weights.items()})
    mx.save_safetensors(str(adapter / "adapters.safetensors"), weights)
    backend._write_json(adapter / "adapter_config.json", {
        "adapter_type": joint.ADAPTER_TYPE, "lora_parameters": meta["lora_parameters"],
    })
    hashes = {p.name: backend._digest(p) for p in adapter.iterdir()}
    run = {**meta, "status": "trained", "optimizer_updates": step, "adapter_sha256": hashes,
           "parameter_change": delta, "logit_sha256": joint.fingerprints(wrapper.model, examples),
           "completed_at": backend._now()}
    backend._write_json(folder / "run.json", run)


def train(args):
    previous = backend._read_adapter_run(args.previous_adapter)
    baseline = json.loads(BASELINE.read_text())
    for key in ("adapter_sha256", "training_ids", "training_audio_sha256", "model_name", "model_revision"):
        if previous[key] != baseline[key]:
            raise ValueError(f"Previous run differs from the 588bd57 baseline in {key}")
    args.output.mkdir(parents=True, exist_ok=False)
    subset = args.output / "train-six.jsonl"
    selected = select_training_rows(args.train_manifest, subset)
    if {r["id"]: r["audio_sha256"] for r in selected} != previous["training_audio_sha256"]:
        raise ValueError("The six source recordings differ from the baseline")
    rt = backend._load_runtime()
    mx.random.seed(SEED)
    model, _, revision, _ = backend._model(rt, previous["model_name"], previous["model_revision"])
    params = {"rank": 16, "alpha": 16, "dropout": 0.0}
    initial = joint.configure(model, params)
    data = joint.prepare_examples(model, rt, selected)
    talker = model.model
    model.full_model.eval()
    alignment = {t: joint.parity(model, ex, t) for t, ex in data.items()}
    codec_digest = joint.codec_hash(model.full_model.speech_tokenizer)
    order = previous["training_ids"]
    meta = {"started_at": backend._now(), "model_name": previous["model_name"], "model_revision": revision,
            "training_ids": order, "training_audio_sha256": previous["training_audio_sha256"],
            "conditioning": "auto_without_reference", "lora_parameters": params,
            "codec_sha256": codec_digest, "base_incremental_relative_l2": alignment,
            "packages": backend._versions(), "source_revisions": previous["source_revisions"],
            "gradient_accumulation_steps": 4, "seed": SEED, "adam_learning_rate": 1e-4,
            "residual_loss_weight": joint.RESIDUAL_WEIGHT, "extension_total_updates": 300,
            "extension_rule": "residual CE at 150 <= 0.95 times residual CE at 125",
            "limitations": ["Six memorized training recordings. Pronunciation, tone and connected speech require listening."]}
    optimizer = optim.Adam(learning_rate=meta["adam_learning_rate"])
    grad_fn = nn.value_and_grad(talker, lambda x, c, y: joint.losses(talker, x, c, y))
    metrics = {"0": joint.score(talker, data.values())}
    meta["scores"] = metrics
    step = 0
    stop = 150
    start = time.monotonic()
    model.model.train()
    while step < stop:
        acc = None
        for micro in range(4):
            ex = data[order[(4 * step + micro) % len(order)]]
            (objective, _), grads = grad_fn(*ex)
            acc = grads if acc is None else tree_map(lambda a, b: a + b, acc, grads)
            mx.eval(objective, acc)
            if not np.isfinite(objective.item()):
                raise RuntimeError(f"Non-finite loss on update {step + 1}")
        acc = tree_map(lambda v: v / 4, acc)
        if step == 0:
            meta["first_gradient"] = joint.norms_by_stage(dict(tree_flatten(acc)))
        optimizer.update(talker, acc)
        mx.eval(talker.trainable_parameters(), optimizer.state)
        step += 1
        if step not in MILESTONES:
            continue
        scores = joint.score(talker, data.values())
        metrics[str(step)] = scores
        print(f"Update {step}: {scores}", flush=True)
        meta["elapsed_seconds"] = time.monotonic() - start
        if step in (150, 300):
            if joint.codec_hash(model.full_model.speech_tokenizer) != codec_digest:
                raise RuntimeError("Codec parameters changed during training")
            if step == 150 and scores["residual"]["ce"] <= 0.95 * metrics["125"]["residual"]["ce"]:
                stop = 300
            meta["final_update"] = stop
            checkpoint(args.output, step, model, initial, meta, data)
        backend._write_json(args.output / "training.json", meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "listen"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, default=ROOT / "build/tts/data/train.jsonl")
    parser.add_argument("--previous-adapter", type=Path,
                        default=ROOT / "build/tts/runs/memorization-six/model/adapters")
    args = parser.parse_args()
    if args.phase == "train":
        train(args)
    else:
        from tts_joint_listen import listen
        listen(args, BASELINE)
