"""Same-example held-out comparison across saved VITS checkpoints."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits

from rpa_tts.text.vits_tokens import TOKEN_TO_ID, encode

SR = 22050
HOP = 256
SEED = 3407
ROOT = Path("data/runs/vits-corpus-pass")
MANIFEST = Path("data/raw/mwwhdv-nt/manifest-quarantine.jsonl")
SPLIT = Path("data/raw/mwwhdv-nt/split-quarantine.jsonl")


def sanitize(text: str) -> str:
    allowed = set(TOKEN_TO_ID)
    t = text.lower().replace("'", " ").replace('"', " ").replace("“", " ").replace("”", " ")
    return "".join(c if c in allowed else " " for c in t)


def load_dev():
    rows = {r["id"]: r for r in (json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines())}
    splits = {r["id"]: r["split"] for r in (json.loads(line) for line in SPLIT.read_text(encoding="utf-8").splitlines())}
    out = []
    for rid, r in sorted(rows.items()):
        if splits.get(rid) != "dev":
            continue
        ids = encode(sanitize(r["text"]))
        out.append({"row": r, "ids": ids})
    return out


def make_pairs(dev, n=12):
    by_len: dict[int, list] = {}
    for p in dev:
        by_len.setdefault(len(p["ids"]), []).append(p)
    pairs = []
    used = set()
    for length in sorted(by_len):
        candidates = by_len[length]
        if len(candidates) < 2:
            continue
        a, b = candidates[0], candidates[1]
        if a["row"]["id"] in used or b["row"]["id"] in used:
            continue
        used.add(a["row"]["id"])
        used.add(b["row"]["id"])
        pairs.append((a, b))
        if len(pairs) == n:
            break
    return pairs


def synth(model, ids):
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        with torch.no_grad():
            out = model.inference(
                torch.tensor([ids], device="cuda", dtype=torch.long),
                aux_input={
                    "speaker_ids": torch.zeros(1, device="cuda", dtype=torch.long),
                    "d_vectors": None,
                    "language_ids": None,
                },
            )
    return out["model_outputs"].detach().float().cpu().numpy().reshape(-1)


def feats(gen, ref_path):
    y, sr = sf.read(ref_path, dtype="float32")
    y = y.mean(axis=1) if y.ndim > 1 else y
    mf = librosa.feature.mfcc(y=gen, sr=SR, n_mfcc=20).mean(axis=1)
    rmf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).mean(axis=1)
    f0, _, _ = librosa.pyin(gen, fmin=60, fmax=400, sr=SR, frame_length=1024)
    rf0, _, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr, frame_length=1024)
    v = f0[~np.isnan(f0)]
    rv = rf0[~np.isnan(rf0)]
    ratio = (len(gen) / SR) / (len(y) / sr)
    return {
        "duration_ratio": round(ratio, 4),
        "mfcc_distance": round(float(np.linalg.norm(mf - rmf)), 3),
        "f0_error_hz": round(abs(float(np.median(v)) - float(np.median(rv))), 2)
        if len(v) and len(rv)
        else None,
        "lag256": round(float(np.corrcoef(gen[:-HOP], gen[HOP:])[0, 1]), 4),
    }


def main() -> int:
    dev = load_dev()
    pairs = make_pairs(dev)
    steps = [50, 250, 500, 750, 1000, 1250, 1500]
    checkpoints = {}
    for s in steps:
        for name in (f"checkpoint-step{s:04d}.pt", f"checkpoint-u{s:03d}.pt"):
            p = ROOT / name
            if p.is_file():
                checkpoints[s] = p
                break
    outdir = ROOT / "same-example-eval"
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = VitsConfig()
    cfg.model_args.num_chars = len(TOKEN_TO_ID)
    cfg.model_args.use_speaker_embedding = False
    cfg.model_args.max_inference_len = 3000
    cfg.audio.sample_rate = SR
    report = {"pairs": len(pairs), "checkpoints": {}, "records": []}
    for s, ckpt in sorted(checkpoints.items()):
        ck = torch.load(ckpt, map_location="cuda", weights_only=False)
        model = Vits(cfg).cuda()
        model.load_state_dict(ck["model"])
        model.eval()
        records = []
        for correct_p, wrong_p in pairs:
            c = synth(model, correct_p["ids"])
            w = synth(model, wrong_p["ids"])
            cf = feats(c, correct_p["row"]["audio_path"])
            wf = feats(w, correct_p["row"]["audio_path"])
            cs = cf["mfcc_distance"] + 20 * abs(1 - cf["duration_ratio"])
            ws = wf["mfcc_distance"] + 20 * abs(1 - wf["duration_ratio"])
            records.append(
                {
                    "id": correct_p["row"]["id"],
                    "reference": correct_p["row"]["reference"],
                    "wrong": wrong_p["row"]["reference"],
                    "correct": cf,
                    "wrong": wf,
                    "correct_better_score": cs < ws,
                    "correct_better_mfcc": cf["mfcc_distance"] < wf["mfcc_distance"],
                    "correct_better_duration": abs(1 - cf["duration_ratio"])
                    < abs(1 - wf["duration_ratio"]),
                }
            )
            if s == max(checkpoints):
                d = outdir / f"step{s:04d}"
                d.mkdir(parents=True, exist_ok=True)
                sf.write(d / f"{correct_p['row']['id']}_correct.wav", np.clip(c, -1, 1), SR)
                sf.write(d / f"{correct_p['row']['id']}_wrong.wav", np.clip(w, -1, 1), SR)
        report["checkpoints"][str(s)] = {
            "checkpoint": str(ckpt),
            "sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
            "correct_better_score": sum(r["correct_better_score"] for r in records),
            "correct_better_mfcc": sum(r["correct_better_mfcc"] for r in records),
            "correct_better_duration": sum(r["correct_better_duration"] for r in records),
            "pairs": len(records),
        }
        report["records"].append({"step": s, "records": records})
        del model
        torch.cuda.empty_cache()
    report["max_step"] = max(checkpoints) if checkpoints else None
    (outdir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["checkpoints"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
