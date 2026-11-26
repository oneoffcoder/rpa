"""Corpus-scale paired Hmong VITS trainer (private diagnostic).

- Train rows: quarantined chapter-grouped train split, unique audio hashes.
- Every item carries its matching transcript (unsupported quote punctuation is
  mapped to spaces only; no transcript dropout).
- Waveform decoder initialized from the verified LJSpeech VITS decoder.
- Full model/optimizer/RNG/order state checkpoint; latest checkpoint is
  resumable.
- Fixed held-out dev prompts are synthesized each evaluation in two conditions:
  correct transcript and a different, length-matched dev transcript.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import librosa
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits

from rpa_tts.text.vits_tokens import TOKEN_TO_ID, encode

SR = 22050
HOP = 256
SEED = 3407
ROOT = Path("data/runs/vits-corpus")
MANIFEST = Path("data/raw/mwwhdv-nt/manifest-quarantine.jsonl")
SPLIT = Path("data/raw/mwwhdv-nt/split-quarantine.jsonl")
DECODER = Path("data/runs/vits-diagnostic/ljspeech/ljspeech_waveform_decoder.pt")


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def sanitize(text: str) -> str:
    allowed = set(TOKEN_TO_ID)
    t = text.lower().replace("'", " ").replace('"', " ").replace("“", " ").replace("”", " ")
    return "".join(c if c in allowed else " " for c in t)


def load_rows() -> tuple[list[dict], list[dict], dict]:
    rows = {
        r["id"]: r
        for r in (
            json.loads(line)
            for line in MANIFEST.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    splits = {
        r["id"]: r["split"]
        for r in (
            json.loads(line)
            for line in SPLIT.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    train = [r for rid, r in sorted(rows.items()) if splits.get(rid) == "train"]
    dev = [r for rid, r in sorted(rows.items()) if splits.get(rid) == "dev"]
    if len({r["id"] for r in train}) != len(train):
        raise SystemExit("duplicate train ids")
    if len({r["audio_sha256"] for r in train}) != len(train):
        raise SystemExit("duplicate train audio hashes")
    for r in train + dev:
        if not Path(r["audio_path"]).is_file():
            raise SystemExit(f"missing audio {r['audio_path']}")
    return train, dev, splits


def prepare(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        ids = encode(sanitize(r["text"]))
        out.append({"row": r, "ids": ids, "seconds": float(r["duration_seconds"])})
    return out


def make_batch(items: list[dict]) -> dict:
    b = len(items)
    max_t = max(len(p["ids"]) for p in items)
    max_w = max(p["wav"].shape[-1] for p in items)
    tokens = torch.zeros(b, max_t, dtype=torch.long, device="cuda")
    tlen = torch.tensor([len(p["ids"]) for p in items], device="cuda")
    wav = torch.zeros(b, 1, max_w, device="cuda")
    wlen = torch.tensor([p["wav"].shape[-1] for p in items], device="cuda")
    for i, p in enumerate(items):
        tokens[i, : len(p["ids"])] = torch.tensor(p["ids"], device="cuda")
        wav[i, :, : p["wav"].shape[-1]] = p["wav"].to("cuda")
    return {
        "tokens": tokens,
        "token_lens": tlen,
        "waveform": wav,
        "waveform_lens": wlen,
        "waveform_rel_lens": wlen / wlen.max(),
        "speaker_ids": torch.zeros(b, device="cuda", dtype=torch.long),
        "d_vectors": None,
        "language_ids": None,
    }


def read_wav(path: str) -> torch.Tensor:
    y, sr = sf.read(path, dtype="float32")
    if sr != SR:
        raise SystemExit(f"sample rate {sr} != {SR} for {path}")
    if y.ndim > 1:
        y = y.mean(axis=1)
    return torch.from_numpy(y).unsqueeze(0)


def features(path: Path, ref_path: Path) -> dict:
    y, sr = sf.read(path, dtype="float32")
    y = y.mean(axis=1) if y.ndim > 1 else y
    ry, rsr = sf.read(ref_path, dtype="float32")
    ry = ry.mean(axis=1) if ry.ndim > 1 else ry
    f0, _, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr, frame_length=1024)
    rf0, _, _ = librosa.pyin(ry, fmin=60, fmax=400, sr=rsr, frame_length=1024)
    v = f0[~np.isnan(f0)]
    rv = rf0[~np.isnan(rf0)]
    mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).mean(axis=1)
    rmf = librosa.feature.mfcc(y=ry, sr=rsr, n_mfcc=20).mean(axis=1)
    return {
        "seconds": round(len(y) / sr, 4),
        "reference_seconds": round(len(ry) / rsr, 4),
        "duration_ratio": round((len(y) / sr) / (len(ry) / rsr), 4),
        "median_f0": round(float(np.median(v)), 2) if len(v) else None,
        "reference_f0": round(float(np.median(rv)), 2) if len(rv) else None,
        "f0_error_hz": round(abs(float(np.median(v)) - float(np.median(rv))), 2)
        if len(v) and len(rv)
        else None,
        "mfcc_distance": round(float(np.linalg.norm(mf - rmf)), 3),
        "lag256": round(float(np.corrcoef(y[:-HOP], y[HOP:])[0, 1]), 4),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> int:
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--max-updates", type=int, default=1600)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=1200000)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    ROOT = args.root
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "pid.txt").write_text(str(os.getpid()), encoding="utf-8")
    train_rows, dev_rows, _ = load_rows()
    train = prepare(train_rows)
    dev = prepare(dev_rows)
    # fixed held-out dev prompts from distinct chapters
    held = []
    seen_chapters = set()
    for p in dev:
        chapter = p["row"]["reference"].split(":")[0]
        if chapter in seen_chapters:
            continue
        if 4.0 <= p["seconds"] <= 9.0:
            held.append(p)
            seen_chapters.add(chapter)
        if len(held) == 3:
            break
    if len(held) < 2:
        raise SystemExit("not enough held-out prompts")
    dataset = {
        "train_rows": len(train),
        "dev_rows": len(dev),
        "unique_train_ids": len({p["row"]["id"] for p in train}),
        "unique_train_hashes": len({p["row"]["audio_sha256"] for p in train}),
        "train_seconds": round(sum(p["seconds"] for p in train), 3),
        "dev_seconds": round(sum(p["seconds"] for p in dev), 3),
        "heldout": [
            {
                "id": p["row"]["id"],
                "reference": p["row"]["reference"],
                "seconds": p["seconds"],
                "tokens": len(p["ids"]),
            }
            for p in held
        ],
    }
    atomic_json(ROOT / "dataset.json", dataset)
    cfg = VitsConfig()
    cfg.model_args.num_chars = len(TOKEN_TO_ID)
    cfg.model_args.use_speaker_embedding = False
    cfg.model_args.max_inference_len = 3000
    cfg.audio.sample_rate = SR
    model = Vits(cfg).cuda()
    dec = torch.load(DECODER, map_location="cuda", weights_only=False)
    model.waveform_decoder.load_state_dict(dec["state_dict"])
    disc_crit, gen_crit = model.get_criterion()
    disc_opt = torch.optim.AdamW(
        model.disc.parameters(), lr=2e-4, betas=(0.8, 0.99), eps=1e-9, weight_decay=0.01
    )
    gen_params = [p for n, p in model.named_parameters() if not n.startswith("disc.")]
    gen_opt = torch.optim.AdamW(
        gen_params, lr=2e-4, betas=(0.8, 0.99), eps=1e-9, weight_decay=0.01
    )
    start_step = 0
    seen_unique: set[str] = set()
    epoch = 0
    position = 0
    order = list(range(len(train)))
    random.Random(SEED).shuffle(order)
    if args.resume and args.resume.is_file():
        ck = torch.load(args.resume, map_location="cuda", weights_only=False)
        model.load_state_dict(ck["model"])
        disc_opt.load_state_dict(ck["disc_opt"])
        gen_opt.load_state_dict(ck["gen_opt"])
        start_step = int(ck.get("step", 0))
        epoch = int(ck.get("epoch", 0))
        position = int(ck.get("position", 0))
        order = list(ck.get("order", order))
        seen_unique = set(ck.get("unique_seen", []))
        torch.set_rng_state(ck["torch_rng"].cpu().to(torch.uint8))
        if ck.get("cuda_rng"):
            torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in ck["cuda_rng"]])
    def save(step: int, eval_payload: dict | None = None) -> None:
        ckpt = ROOT / "checkpoint.pt"
        tmp = ROOT / "checkpoint.pt.tmp"
        torch.save(
            {
                "model": model.state_dict(),
                "gen_opt": gen_opt.state_dict(),
                "disc_opt": disc_opt.state_dict(),
                "step": step,
                "epoch": epoch,
                "position": position,
                "order": order,
                "unique_seen": sorted(seen_unique),
                "seed": SEED,
                "config": cfg.to_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
            },
            tmp,
        )
        os.replace(tmp, ckpt)
        status = {
            "pid": os.getpid(),
            "step": step,
            "epoch": epoch,
            "position": position,
            "unique_train_rows": dataset["train_rows"],
            "unique_rows_seen": len(seen_unique),
            "dev_rows": dataset["dev_rows"],
            "checkpoint": str(ckpt),
            "checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        if eval_payload:
            status["eval"] = eval_payload
        atomic_json(ROOT / "status.json", status)
    def synth(p: dict, ids: list[int]) -> tuple[np.ndarray, float]:
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
        return (
            out["model_outputs"].detach().float().cpu().numpy().reshape(-1),
            float(out["durations"].sum().detach().cpu()),
        )
    def wrong_ids(p: dict, used: set[str]) -> tuple[list[int], str]:
        target = len(p["ids"])
        candidates = [q for q in dev if q["row"]["id"] not in used and q["row"]["id"] != p["row"]["id"]]
        candidates.sort(key=lambda q: (abs(len(q["ids"]) - target), q["row"]["id"]))
        q = candidates[0]
        return q["ids"], q["row"]["reference"]
    def evaluate(step: int) -> dict:
        model.eval()
        outdir = ROOT / "eval" / f"step_{step:05d}"
        outdir.mkdir(parents=True, exist_ok=True)
        used = {p["row"]["id"] for p in held}
        records = []
        for p in held:
            wid, wref = wrong_ids(p, used)
            correct, cdur = synth(p, p["ids"])
            wrong, wdur = synth(p, wid)
            cp = outdir / f"{p['row']['id']}_correct.wav"
            wp = outdir / f"{p['row']['id']}_wrong.wav"
            sf.write(cp, np.clip(correct, -1, 1), SR)
            sf.write(wp, np.clip(wrong, -1, 1), SR)
            cf = features(cp, Path(p["row"]["audio_path"]))
            wf = features(wp, Path(p["row"]["audio_path"]))
            cf["duration_frames"] = cdur
            wf["duration_frames"] = wdur
            records.append(
                {
                    "id": p["row"]["id"],
                    "reference": p["row"]["reference"],
                    "wrong_reference": wref,
                    "correct": cf,
                    "wrong": wf,
                    "correct_better_mfcc": cf["mfcc_distance"] < wf["mfcc_distance"],
                }
            )
        payload = {
            "step": step,
            "prompts": len(held),
            "correct_better_mfcc": sum(r["correct_better_mfcc"] for r in records),
            "records": records,
        }
        atomic_json(outdir / "eval.json", payload)
        atomic_json(ROOT / "latest_eval.json", payload)
        return payload
    model.train()
    step = start_step
    rows_seen = 0
    losses = []
    started = time.time()
    with (ROOT / "train.jsonl").open("a", encoding="utf-8") as log:
        while step < args.max_updates:
            if position >= len(order):
                epoch += 1
                order = list(range(len(train)))
                random.Random(SEED + epoch).shuffle(order)
                position = 0
            items = []
            samples = 0
            while position < len(order) and len(items) < args.batch_size:
                p = train[order[position]]
                w = read_wav(p["row"]["audio_path"])
                if items and samples + w.shape[-1] > args.max_samples:
                    break
                p = dict(p)
                p["wav"] = w
                items.append(p)
                samples += w.shape[-1]
                position += 1
            if not items:
                continue
            batch = make_batch(items)
            batch = model.format_batch_on_device(batch)
            disc_opt.zero_grad(set_to_none=True)
            _, dl = model.train_step(batch, disc_crit, optimizer_idx=0)
            dl["loss"].backward()
            dg = float(torch.nn.utils.clip_grad_norm_(model.disc.parameters(), 1.0))
            disc_opt.step()
            gen_opt.zero_grad(set_to_none=True)
            _, gl = model.train_step(batch, gen_crit, optimizer_idx=1)
            gl["loss"].backward()
            gg = float(torch.nn.utils.clip_grad_norm_(gen_params, 1.0))
            gen_opt.step()
            step += 1
            rows_seen += len(items)
            seen_unique.update(p["row"]["id"] for p in items)
            rec = {
                "step": step,
                "epoch": epoch,
                "rows_seen": rows_seen,
                "batch": len(items),
                "unique_rows_seen": len(seen_unique),
                "disc_loss": float(dl["loss"].detach()),
                "gen_loss": float(gl["loss"].detach()),
                "disc_grad": dg,
                "gen_grad": gg,
                "elapsed_s": round(time.time() - started, 1),
            }
            losses.append(rec["gen_loss"])
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if step % 10 == 0:
                print(json.dumps(rec), flush=True)
            eval_payload = None
            if step % args.eval_every == 0 or step == 50:
                eval_payload = evaluate(step)
                model.train()
            if step % args.checkpoint_every == 0 or step == args.max_updates or step == 50:
                save(step, eval_payload)
            if not np.isfinite(rec["gen_loss"]) or not np.isfinite(rec["disc_loss"]):
                raise SystemExit(f"non-finite loss at step {step}")
    save(step)
    print(json.dumps({"done": True, "step": step, "rows_seen": rows_seen}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
