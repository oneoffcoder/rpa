"""Paired text-only generation and local listening page for the two-stage experiment."""
import json
from html import escape
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from rpa_tts import joint_mlx as joint, mlx_backend as backend
from tts_memorization import MAX_TOKENS, SEED, SYLLABLES, reconstruct, record_audio


DECODING = dict(lang_code="auto", temperature=0.9, top_k=50, top_p=1.0, repetition_penalty=1.05,
                stream=False, max_tokens=MAX_TOKENS)
PHRASE = "Kuv yog Hmoob."


def generate(full, text):
    """Observe native sampling and decoding without passing in audio, codes or a duration."""
    sample_token, decode = full._sample_token, full.speech_tokenizer.decode
    decisions, output, first_scores = [], {}, {}

    def observe_sample(logits, *args, **kwargs):
        token = sample_token(logits, *args, **kwargs)
        primary = kwargs.get("eos_token_id") is not None
        group = "talker" if primary else "residual"
        if group not in first_scores:
            first_scores[group] = logits[0, -1].astype(mx.float32)
        if primary:
            decisions.append(token)
        return token

    def observe_decode(codes):
        wav, lengths = decode(codes)
        output.update(codes=codes[0], full_wave=wav[0])
        return wav, lengths

    mx.random.seed(SEED)
    with patch.object(full, "_sample_token", observe_sample), patch.object(full.speech_tokenizer, "decode", observe_decode):
        results = list(full.generate(text=text, **DECODING))
    eos = full.config.talker_config.codec_eos_token_id
    events = np.asarray(mx.concatenate(decisions, axis=1))[0].tolist()
    stopped = events[-1] == eos
    if not stopped and len(events) != MAX_TOKENS:
        raise RuntimeError("Generation exited without EOS or the frame limit")
    codes = np.asarray(output["codes"]) if output else np.empty((0, 16), dtype=np.int32)
    if codes[:, 0].tolist() != (events[:-1] if stopped else events):
        raise RuntimeError("Sampled first-stream decisions disagree with the decoded sequence")
    if len(codes):
        if len(results) != 1 or results[0].sample_rate != backend.SAMPLE_RATE:
            raise RuntimeError("Expected one 24 kHz synthesis result")
        wave = np.asarray(output["full_wave"], dtype=np.float32)
        if wave.size != len(codes) * full.speech_tokenizer.decode_upsample_rate:
            raise RuntimeError("Missing samples from the complete unpadded decoder output")
    else:
        wave = np.empty(0, dtype=np.float32)
    info = {"codec_frames": len(codes), "stop": "EOS" if stopped else "limit",
            "sampling_iterations": len(events), "hit_limit": not stopped,
            "duration_seconds": wave.size / backend.SAMPLE_RATE,
            "tokens_sha256": joint.tensor_hash(mx.array(codes)) if len(codes) else None}
    return wave, codes, info, {k: np.asarray(v) for k, v in first_scores.items()}


def aligned_matches(codes, reference):
    count = min(len(codes), len(reference))
    same = codes[:count] == reference[:count]
    return {"reference_frames": len(reference), "shared_frames": count,
            "first_correct_in_shared_frames": int(same[:, 0].sum()),
            "residual_correct_in_shared_frames": int(same[:, 1:].sum()),
            "exact_all_codes_and_length": np.array_equal(codes, reference)}


def render_variant(wrapper, variant, refs, folder, report):
    for text in [*SYLLABLES, PHRASE]:
        wave, codes, info, _ = generate(wrapper.full_model, text)
        path = folder / f"{'phrase' if text == PHRASE else text}-{variant}.wav"
        if wave.size:
            info["clipped_samples"] = backend._write_waveform(path, wave, np)
            info.update(record_audio(path))
        if text in refs:
            info.update(aligned_matches(codes, refs[text]))
        report[text][variant] = info


def ablate(wrapper):
    """Disable the learned update in one stage at a time during ordinary generation."""
    full = wrapper.full_model
    _, _, normal, before = generate(full, "kuv")
    weights = dict(tree_flatten(wrapper.model.trainable_parameters()))
    effects = {}
    for group in joint.GROUPS:
        branch = [(k, v) for k, v in weights.items() if joint.stage(k) == group and k.endswith(".lora_b")]
        try:
            wrapper.model.load_weights([(k, mx.zeros_like(v)) for k, v in branch], strict=False)
            _, _, probe, after = generate(full, "kuv")
        finally:
            wrapper.model.load_weights(branch, strict=False)
        delta = float(np.max(np.abs(after[group] - before[group])))
        if not np.isfinite(delta) or delta == 0:
            raise RuntimeError(f"Disabling {group} did not change its live inference logits")
        effects[group] = {"max_abs_change_first_logits": delta, "ablated_frames": probe["codec_frames"],
                          "ablated_stop": probe["stop"], "changed_generated_codes": probe["tokens_sha256"] != normal["tokens_sha256"]}
    return effects


def listening_page(report, steps, step):
    def player(item, label):
        if "file" not in item:
            return "<p>Immediate EOS, no audio emitted.</p>"
        path = escape(item["file"])
        return (f'<audio controls preload="metadata" aria-label="{escape(label)}" src="{path}"></audio>'
                f'<a href="{path}">WAV</a>')
    def group(text, columns):
        controls = []
        for key, name in columns:
            control = player(report[text][key], text + ", " + name)
            controls.append(f'<div><strong>{escape(name)}</strong>{control}</div>')
        return f'<section><h2 lang="hmn">{escape(text)}</h2><div class="players">' + "".join(controls) + "</div></section>"
    neural = [("previous", "Previous adapter"), (f"joint-{step}", f"Two-stage adapter · {step} updates")]
    cols = [("original", "Original recording"), ("reconstructed", "Codec reconstruction"), *neural]
    nav = ""
    if len(steps) > 1:
        nav = '<nav>' + ' · '.join(f'<a href="{page_name(s, steps)}">{s} updates</a>' for s in steps) + '</nav>'
    rows = "".join(group(s, cols) for s in SYLLABLES)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hmong listening comparison</title><style>
* {{box-sizing:border-box}} body {{max-width:78rem;margin:2rem auto;padding:0 1rem;color:#17262b;background:#f8f8f5;font:1rem/1.45 system-ui,sans-serif}}
h1 {{font: bold clamp(2rem,5vw,3rem)/1.1 Georgia,serif}} h2 {{margin:.5rem 0}} header p {{max-width:68ch}}
section {{padding:1rem 0;border-top:1px solid #acbcb5}} .players {{display:grid;gap:1rem;grid-template-columns:repeat(4,minmax(0,1fr))}}
audio {{display:block;width:100%;margin:.45rem 0}} a {{color:#135846}} @media(max-width:58rem){{.players{{grid-template-columns:1fr 1fr}}}}
@media(max-width:32rem){{.players{{grid-template-columns:1fr}}}}</style></head><body><main>
<header><h1>Hmong listening comparison</h1><p>First compare the original and reconstructed recordings for consonant, vowel and tone. Then compare the two models reading each syllable from text.</p>{nav}</header>
{rows}<h2>Exploratory phrase</h2>{group(PHRASE, neural)}</main></body></html>"""


def page_name(step, steps):
    return "listen.html" if step == steps[-1] else f"listen-{step}.html"


def evaluate(args, dest, baseline):
    rt = backend._load_runtime()
    report = {s: {} for s in [*SYLLABLES, PHRASE]}
    published = json.loads(baseline.read_text())
    previous = backend._read_adapter_run(args.previous_adapter)
    training = json.loads((args.output / "training.json").read_text())
    if (previous["adapter_sha256"] != published["adapter_sha256"]
            or previous["model_revision"] != training["model_revision"]):
        raise RuntimeError("Wrong baseline adapter or base model revision")
    rows = backend.read_training_manifest(args.output / "train-six.jsonl")
    if {r["id"]: r["audio_sha256"] for r in rows} != published["training_audio_sha256"]:
        raise RuntimeError("Audio does not match the six baseline recordings")
    folder = dest / "audio"
    folder.mkdir(parents=True)
    def fresh():
        return backend._model(rt, training["model_name"], training["model_revision"])[0]
    old = fresh()
    backend._apply_adapter(rt, old, args.previous_adapter)
    examples = joint.prepare_examples(old, rt, rows)
    refs = {n: np.asarray(ex[1]) for n, ex in examples.items()}
    for row, sample in zip(rows, backend._AudioDataset(rows, np)[:]):
        text = row["text"]
        entry = reconstruct(old, sample, row, folder)
        for key in ("original", "reconstructed"):
            report[text][key] = entry[key]
            if entry[key]["sha256"] != published["syllables"][text][key]["sha256"]:
                raise RuntimeError(f"{key} reference differs for {text}")
    diagnostics = {"previous": {"scores": joint.score(old.model, examples.values())}}
    render_variant(old, "previous", refs, folder, report)
    for text in SYLLABLES:
        if report[text]["previous"].get("sha256") != published["syllables"][text]["adapted"]["sha256"]:
            raise RuntimeError(f"Previous adapter render does not reproduce the baseline for {text}")
    del old
    steps = sorted({150, training["final_update"]})
    for step in steps:
        adapter = args.output / f"step-{step}" / "adapters"
        run = backend._read_adapter_run(adapter)
        new = fresh()
        backend._apply_adapter(rt, new, adapter)
        if joint.fingerprints(new.model, examples) != run["logit_sha256"]:
            raise RuntimeError("Reloaded logits differ from the in-memory checkpoint")
        if joint.codec_hash(new.full_model.speech_tokenizer) != training["codec_sha256"]:
            raise RuntimeError("Reloaded codec differs from the input codec")
        variant = f"joint-{step}"
        diagnostics[variant] = {"scores": joint.score(new.model, examples.values()),
                                "ablation_kuv": ablate(new), "reload_logits_exact": True,
                                "incremental_relative_l2": {t: joint.parity(new, ex, t) for t, ex in examples.items()}}
        render_variant(new, variant, refs, folder, report)
        for text in SYLLABLES:
            entry = report[text][variant]
            entry["matching_reconstructions"] = [
                name for name in SYLLABLES if report[name]["reconstructed"]["sha256"] == entry.get("sha256")
            ]
        del new
    for step in steps:
        (dest / page_name(step, steps)).write_text(listening_page(report, steps, step))
    backend._write_json(dest / "results.json", {"decoding": DECODING, "seed": SEED, "results": report,
                                                "diagnostics": diagnostics, "checkpoints": steps})


def listen(args, baseline):
    out = args.output / "listening"
    if out.exists():
        raise FileExistsError(out)
    with TemporaryDirectory(dir=args.output) as scratch:
        dest = Path(scratch) / "listening"
        evaluate(args, dest, baseline)
        dest.rename(out)
    print(out / "listen.html")
