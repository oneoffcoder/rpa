"""Paired Hmong VITS training. Epoch counts completed passes over the train split."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from unittest.mock import patch

import librosa
import numpy as np
import soundfile as sf
import torch
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits
from TTS.tts.utils.text.tokenizer import TTSTokenizer

from rpa_tts.text.vits_tokens import TOKEN_TO_ID, encode

if __package__:
    from .vits_corpus_train import SEED, SR, load_rows, make_batch, prepare, read_wav
else:
    from vits_corpus_train import SEED, SR, load_rows, make_batch, prepare, read_wav

ROOT = Path("data/runs/vits-corpus-full")
START_CK = Path("data/runs/vits-compare/full-acoustic/checkpoint.pt")
LOSS_KEYS = ("loss_mel", "loss_duration", "loss_kl")


def replace(source, target):
    for attempt in range(8):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.05 * (attempt + 1))


def atomic_json(path, value):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    replace(tmp, path)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def epoch_order(size, completed):
    order = list(range(size))
    random.Random(SEED + completed).shuffle(order)
    return order


def resume_cursor(ck, size):
    epoch, position = ck["epoch"], ck["position"]
    if epoch < 0 or not 0 <= position <= size or ck["seed"] != SEED:
        raise ValueError("invalid checkpoint epoch, cursor or shuffle seed")
    expected = epoch_order(size, epoch)
    if ck["order"] != expected:
        # The initial loop stored the just-finished permutation at boundaries.
        if ("loop_version" not in ck and epoch > 0 and position in (0, size)
                and ck["order"] == epoch_order(size, epoch - 1)):
            # The old boundary had no supervised dev result. Evaluate these
            # exact weights once, then advance without another optimizer step.
            return epoch - 1, size, ck["order"]
        raise ValueError("checkpoint shuffle does not match the training split")
    return epoch, position, expected


def restore_training(ck, disc_opt, gen_opt):
    disc_opt.load_state_dict(ck["disc_opt"])
    gen_opt.load_state_dict(ck["gen_opt"])
    torch.set_rng_state(ck["torch_rng"].cpu())
    torch.cuda.set_rng_state_all([state.cpu() for state in ck["cuda_rng"]])


def next_batch(train, order, position, batch_size, sample_budget):
    items = []
    total = 0
    for index in order[position:position + batch_size]:
        item = train[index]
        wav = read_wav(item["row"]["audio_path"])
        if items and total + wav.shape[-1] > sample_budget:
            break
        items.append(dict(item, wav=wav))
        total += wav.shape[-1]
    return items


def update(model, batch, criterion, index, optimizer, parameters):
    optimizer.zero_grad(set_to_none=True)
    _, losses = model.train_step(batch, criterion, optimizer_idx=index)
    loss = losses["loss"]
    if not torch.isfinite(loss).item():
        raise FloatingPointError(f"non-finite VITS loss for optimizer {index}")
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    optimizer.step()
    return loss.detach().item(), grad.item()


@torch.no_grad()
def validation(model, criteria, rows):
    # Singleton batches avoid padding-dependent scores and variable batch weighting.
    totals = dict.fromkeys(LOSS_KEYS, 0.0)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(SEED)
        for item in rows:
            wav = read_wav(item["row"]["audio_path"])
            batch = model.format_batch_on_device(make_batch([dict(item, wav=wav)]))
            model.train_step(batch, criteria[0], optimizer_idx=0)
            _, losses = model.train_step(batch, criteria[1], optimizer_idx=1)
            for key in LOSS_KEYS:
                totals[key] += losses[key].item()
    means = {key: value / len(rows) for key, value in totals.items()}
    score = sum(means.values())
    if not math.isfinite(score):
        raise FloatingPointError("non-finite paired validation loss")
    return {"rows": len(rows), **means, "score": score}


def snapshot(source, target):
    tmp = target.with_suffix(".link.tmp")
    tmp.unlink(missing_ok=True)
    try:
        os.link(source, tmp)
    except OSError:
        shutil.copyfile(source, tmp)
    replace(tmp, target)


def save(root, ck, status, boundary):
    # Never truncate a published file, including one open by a Windows reader.
    path = root / "checkpoint.pt"
    tmp = root / "checkpoint.pt.tmp"
    torch.save(ck, tmp)
    try:
        replace(tmp, path)
    except PermissionError:
        path = root / f"checkpoint-step{ck['step']}.pt"
        replace(tmp, path)
        print(f"Checkpoint alias busy. Resume from {path}.", file=sys.stderr, flush=True)
    if boundary:
        snapshot(path, root / f"checkpoint-epoch{ck['epoch']}.pt")
    status.update(checkpoint=str(path), checkpoint_sha256=digest(path))
    atomic_json(root / "status.json", status)


def make_pairs(dev, n=12):
    by_length = {}
    for item in dev:
        by_length.setdefault(len(item["ids"]), []).append(item)
    return [(group[0], group[1]) for _, group in sorted(by_length.items())
            if len(group) >= 2][:n]


@torch.no_grad()
def synth(model, ids):
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(SEED)
        out = model.inference(
            torch.tensor([ids], device="cuda"),
            aux_input={"speaker_ids": torch.zeros(1, device="cuda", dtype=torch.long),
                       "d_vectors": None, "language_ids": None},
        )
    return [out[key].detach().float().cpu().numpy().reshape(-1)
            for key in ("model_outputs", "durations")]


def mfcc(audio):
    return librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=20).mean(axis=1)


def evaluate(model, criteria, dev_rows, pairs, held, root, completed):
    model.eval()
    try:
        val = validation(model, criteria, dev_rows)
        outdir = root / "boundaries" / f"epoch{completed}"
        outdir.mkdir(parents=True, exist_ok=True)
        prompts = [("spaced", "kuv yog hmoob", encode("kuv yog hmoob")),
                   ("unspaced", "kuvyoghmoob", encode("kuvyoghmoob"))]
        prompts.extend((p["row"]["id"], p["row"]["text"], p["ids"]) for p in held)
        samples = {}
        for label, text, ids in prompts:
            audio, _ = synth(model, ids)
            path = outdir / f"{label}.wav"
            if not np.isfinite(audio).all():
                raise FloatingPointError(f"non-finite synthesis for {label}")
            sf.write(path, np.clip(audio, -1, 1), SR, subtype="PCM_16")
            samples[label] = {"path": str(path), "text": text, "tokens": ids,
                              "seconds": len(audio) / SR, "sha256": digest(path)}
        records = []
        for right, wrong in pairs:
            a, da = synth(model, right["ids"])
            b, db = synth(model, wrong["ids"])
            ref = read_wav(right["row"]["audio_path"]).squeeze(0).numpy()
            ref_mfcc = mfcc(ref)
            n = min(len(a), len(b))
            corr = None
            if n > 1 and a[:n].std() > 0 and b[:n].std() > 0:
                corr = float(np.corrcoef(a[:n], b[:n])[0, 1])
            records.append({"id": right["row"]["id"], "wrong": wrong["row"]["reference"],
                            "correct_better_mfcc": bool(np.linalg.norm(mfcc(a) - ref_mfcc)
                                                        < np.linalg.norm(mfcc(b) - ref_mfcc)),
                            "corr": corr, "duration_equal": bool(np.array_equal(da, db)),
                            "correct_seconds": len(a) / SR, "wrong_seconds": len(b) / SR})
        corrs = [r["corr"] for r in records if r["corr"] is not None]
        result = {"epoch": completed, "validation": val, "samples": samples, "pairs": len(records),
                  "correct_better_mfcc": sum(r["correct_better_mfcc"] for r in records),
                  "mean_corr": float(np.mean(corrs)) if corrs else None,
                  "duration_equal_pairs": sum(r["duration_equal"] for r in records), "records": records}
        atomic_json(outdir / "eval.json", result)
        return result
    finally:
        model.train()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--start-checkpoint", type=Path, default=START_CK)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=1200000)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--validation-samples", type=int, default=32)
    parser.add_argument("--patience", type=int, default=0, help="epochs without improved validation; 0 disables")
    parser.add_argument("--max-updates", type=int, help="optional absolute optimizer-pair update limit")
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.max_samples, args.checkpoint_every, args.validation_samples) < 1:
        parser.error("epoch, batch, checkpoint and validation sizes must be positive")
    if args.patience < 0 or (args.max_updates is not None and args.max_updates < 1):
        parser.error("invalid patience or update limit")
    source = args.resume or args.start_checkpoint
    if not source.is_file():
        raise FileNotFoundError(source)
    root = args.root
    if not args.resume and (root / "status.json").exists():
        parser.error("existing run: choose --resume or a new --root")
    train_rows, dev_rows, _ = load_rows()
    train, dev = prepare(train_rows), prepare(dev_rows)
    if not train or not dev:
        raise ValueError("both paired training and development recordings are required")
    random_dev = random.Random(SEED).sample(dev, min(args.validation_samples, len(dev)))
    pairs = make_pairs(dev)
    held = []
    chapters = set()
    for item in dev:
        row = item["row"]
        chapter = row["reference"].split(":")[0]
        if chapter not in chapters and 4 <= row["duration_seconds"] <= 9:
            held.append(item)
            chapters.add(chapter)
            if len(held) == 4:
                break
    train_dev = [[(p["row"]["id"], p["row"]["audio_sha256"], p["ids"]) for p in split]
                 for split in (train, dev)]
    contract = {"corpus_sha256": hashlib.sha256(json.dumps(train_dev).encode()).hexdigest(),
                "batch_size": args.batch_size, "max_samples": args.max_samples,
                "validation_ids": [p["row"]["id"] for p in random_dev]}
    cfg = VitsConfig()
    cfg.model_args.num_chars = len(TOKEN_TO_ID)
    cfg.model_args.use_speaker_embedding = False
    cfg.model_args.max_inference_len = 3000
    cfg.audio.sample_rate = SR
    torch.manual_seed(SEED)
    # The inputs are already tokenized. Coqui's default tokenizer replaces num_chars.
    with patch.object(TTSTokenizer, "init_from_config", return_value=(None, cfg)):
        model = Vits(cfg).cuda()
    assert model.text_encoder.emb.num_embeddings == len(TOKEN_TO_ID)
    ck = torch.load(source, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"], strict=True)
    criteria = model.get_criterion()
    disc_params = list(model.disc.parameters())
    gen_params = [p for name, p in model.named_parameters() if not name.startswith("disc.")]
    opt = dict(lr=2e-4, betas=(0.8, 0.99), eps=1e-9, weight_decay=0.01)
    disc_opt = torch.optim.AdamW(disc_params, **opt)
    gen_opt = torch.optim.AdamW(gen_params, **opt)
    if args.resume:
        if "contract" in ck and contract != ck["contract"]:
            raise ValueError("corpus, batch settings or validation set changed since checkpoint")
        epoch, position, order = resume_cursor(ck, len(train))
        if "unique_seen" in ck:
            used = order[:position] if epoch == 0 else range(len(train))
            if set(ck["unique_seen"]) != {train[i]["row"]["id"] for i in used}:
                raise ValueError("legacy checkpoint IDs differ from the paired training split")
        step = ck["step"]
        origin = ck["start_checkpoint"]
        best, best_epoch, stale = (ck.get("best_score"), ck.get("best_epoch"), ck.get("stale_epochs", 0))
        restore_training(ck, disc_opt, gen_opt)
    else:
        epoch = position = step = stale = 0
        order = epoch_order(len(train), 0)
        best = best_epoch = None
        origin = str(source)
    del ck
    root.mkdir(parents=True, exist_ok=True)
    model.train()
    dataset = {"train_rows": len(train), "unique_train_ids": len({p["row"]["id"] for p in train}),
               "unique_train_hashes": len({p["row"]["audio_sha256"] for p in train}), "dev_rows": len(dev),
               "seed": SEED, "epochs": args.epochs, "start_checkpoint": origin, **contract,
               "heldout": [p["row"]["id"] for p in held],
               "pairs": [[a["row"]["id"], b["row"]["id"]] for a, b in pairs]}
    atomic_json(root / "dataset.json", dataset)
    (root / "pid.txt").write_text(str(os.getpid()), encoding="utf-8")
    print(json.dumps({"resumed": bool(args.resume), "source": str(source), "pid": os.getpid(),
                      "epoch": epoch, "position": position, "step": step,
                      "disc_optimizer_states": len(disc_opt.state),
                      "gen_optimizer_states": len(gen_opt.state)}), flush=True)
    started = time.monotonic()
    with (root / "train.jsonl").open("a", encoding="utf-8") as log:
        while epoch < args.epochs and (args.max_updates is None or step < args.max_updates):
            boundary = None
            if position < len(train):
                items = next_batch(train, order, position, args.batch_size, args.max_samples)
                batch = model.format_batch_on_device(make_batch(items))
                dl, dg = update(model, batch, criteria[0], 0, disc_opt, disc_params)
                gl, gg = update(model, batch, criteria[1], 1, gen_opt, gen_params)
                step += 1
                position += len(items)
                seen = epoch * len(train) + position
                rec = {"step": step, "epoch": epoch, "position": position, "rows_seen": seen,
                       "unique_seen": min(seen, len(train)), "batch": len(items),
                       "disc_loss": dl, "gen_loss": gl, "disc_grad": dg, "gen_grad": gg,
                       "elapsed_s": round(time.monotonic() - started, 1)}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                if step % 10 == 0:
                    print(json.dumps(rec), flush=True)
            if position == len(train):
                epoch += 1
                boundary = evaluate(model, criteria, random_dev, pairs, held, root, epoch)
                score = boundary["validation"]["score"]
                if best is None or score < best:
                    best, best_epoch, stale = score, epoch, 0
                else:
                    stale += 1
                order = epoch_order(len(train), epoch)
                position = 0
            stop = None
            if epoch >= args.epochs:
                stop = "epoch_limit"
            elif args.max_updates is not None and step >= args.max_updates:
                stop = "update_limit"
            elif boundary and args.patience and stale >= args.patience:
                stop = "validation_plateau"
            if boundary or stop or step % args.checkpoint_every == 0:
                state = {"loop_version": 2, "model": model.state_dict(), "disc_opt": disc_opt.state_dict(),
                         "gen_opt": gen_opt.state_dict(), "step": step, "epoch": epoch,
                         "position": position, "order": order, "contract": contract, "seed": SEED,
                         "config": cfg.to_dict(), "start_checkpoint": origin, "best_score": best,
                         "best_epoch": best_epoch, "stale_epochs": stale, "torch_rng": torch.get_rng_state(),
                         "cuda_rng": torch.cuda.get_rng_state_all()}
                seen = epoch * len(train) + position
                status = {"pid": os.getpid(), "step": step, "epoch": epoch, "position": position,
                          "rows_seen": seen, "unique_seen": min(seen, len(train)), "epoch_target": args.epochs,
                          "start_checkpoint": origin, "best_score": best, "best_epoch": best_epoch,
                          "best_checkpoint": f"checkpoint-epoch{best_epoch}.pt" if best_epoch else None,
                          "stale_epochs": stale, "stop_reason": stop,
                          "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
                if boundary:
                    status["boundary"] = boundary
                save(root, state, status, boundary)
                if boundary:
                    atomic_json(root / "latest_boundary.json", boundary)
            if stop:
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
