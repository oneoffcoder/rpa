"""Measure Coqui VITS training throughput from a common retained checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
from threading import Event, Thread, Timer

import torch
from trainer import Trainer, TrainerArgs
from vits_coqui_setup import MANIFEST, load_recipe, read_index


def observe_gpu(stop: Event, samples: list[tuple[float, float]]) -> None:
    while not stop.is_set():
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if result.returncode == 0:
            fields = result.stdout.strip().splitlines()[0].split(",")
            samples.append((float(fields[0]), float(fields[1])))
        stop.wait(0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--trial", required=True, type=Path)
    parser.add_argument("--batch", required=True, type=int)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--workers", default=0, type=int)
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--samples", default=512, type=int)
    parser.add_argument("--wall-limit", default=150, type=int)
    args = parser.parse_args()
    if args.batch < 1 or args.workers < 0 or args.samples < 32 or args.samples % args.batch:
        parser.error("samples must be at least 32 and divisible by a positive batch size")
    if args.wall_limit < 1:
        parser.error("wall limit must be positive")
    if not torch.cuda.is_available() or (
        args.precision == "bf16" and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("Requested CUDA/precision mode is unavailable")

    config, model, train, _ = load_recipe(args.config.resolve(strict=True))
    if args.samples > len(train):
        parser.error("samples exceed the training split")
    records = read_index(MANIFEST)
    by_length = sorted(
        train,
        key=lambda sample: (
            records[Path(sample["audio_file"]).stem]["duration_seconds"],
            Path(sample["audio_file"]).stem,
        ),
        reverse=True,
    )
    # Keep rare long utterances in the trial to exercise memory headroom.
    long_count = min(32, args.samples // 4)
    cohort = by_length[:long_count] + random.Random(20261126).sample(
        by_length[long_count:], args.samples - long_count
    )

    source = args.checkpoint.resolve(strict=True)
    state = torch.load(source, map_location="cpu", weights_only=False)
    step, epoch = state["step"], state["epoch"]
    opt_sizes = [len(optimizer["state"]) for optimizer in state["optimizer"]]
    if len(opt_sizes) != 2 or not all(opt_sizes) or len(state["scheduler"]) != 2:
        raise ValueError("Expected a VITS checkpoint with both optimizers and schedulers")
    del state
    trial = args.trial.resolve()
    trial.mkdir(parents=True)
    os.link(source, trial / f"checkpoint_{step}.pth")

    config.output_path = str(trial)
    config.batch_size = args.batch
    config.num_loader_workers = args.workers
    config.mixed_precision = args.precision != "fp32"
    config.allow_tf32 = args.tf32
    if config.mixed_precision:
        config.precision = args.precision
    config.epochs = epoch + 1
    config.run_eval = False
    config.test_delay_epochs = epoch + 2
    config.save_checkpoints = False
    config.save_on_interrupt = False
    config.save_best_after = step + args.samples + 1
    config.save_json(str(trial / "config.json"))

    stop = Event()
    gpu_samples: list[tuple[float, float]] = []
    monitor = Thread(target=observe_gpu, args=(stop, gpu_samples), daemon=True)
    start = 0.0
    start_step = 0
    metrics = {}

    def abort_trial() -> None:
        reason = {"status": "aborted", "seconds": args.wall_limit, "reason": "trial wall limit"}
        (trial / "aborted.json").write_text(json.dumps(reason), encoding="utf-8")
        print(json.dumps(reason), flush=True)
        os._exit(124)

    watchdog = Timer(args.wall_limit, abort_trial)
    watchdog.daemon = True

    def start_epoch(trainer: Trainer) -> None:
        nonlocal start, start_step
        start_step = trainer.total_steps_done
        if start_step != step + 1 or [len(o.state) for o in trainer.optimizer] != opt_sizes:
            raise RuntimeError("Source step or optimizer state did not restore")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        monitor.start()
        watchdog.start()

    def finish_epoch(trainer: Trainer) -> None:
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        completed = trainer.total_steps_done - start_step
        averages = trainer.keep_avg_train.avg_values
        losses = {key: float(value) for key, value in averages.items() if "loss" in key}
        if not losses or not all(math.isfinite(value) for value in losses.values()):
            raise RuntimeError(f"Invalid losses: {losses}")
        seen = completed * args.batch
        if seen != args.samples:
            raise RuntimeError(f"Trial processed {seen} utterances, expected {args.samples}")
        metrics.update(
            batch=args.batch,
            precision=args.precision,
            loader_workers=args.workers,
            tf32=args.tf32,
            checkpoint_step=step,
            samples=seen,
            updates=completed,
            wall_seconds=seconds,
            utterances_per_second=seen / seconds,
            peak_allocated_mib=round(torch.cuda.max_memory_allocated() / 2**20, 2),
            peak_reserved_mib=round(torch.cuda.max_memory_reserved() / 2**20, 2),
            losses=losses,
        )
        watchdog.cancel()
        stop.set()

    try:
        Trainer(
            TrainerArgs(continue_path=str(trial)),
            config,
            str(trial),
            model=model,
            train_samples=cohort,
            eval_samples=[],
            callbacks={"on_train_epoch_start": start_epoch, "on_train_epoch_end": finish_epoch},
            parse_command_line_args=False,
        ).fit()
    finally:
        watchdog.cancel()
        stop.set()
        if monitor.ident is not None:
            monitor.join()
    if not metrics:
        raise RuntimeError("The trial did not complete an epoch")
    if gpu_samples:
        metrics["gpu_samples"] = len(gpu_samples)
        metrics["mean_gpu_use_pct"] = sum(row[0] for row in gpu_samples) / len(gpu_samples)
        metrics["peak_board_mib"] = max(row[1] for row in gpu_samples)
    (trial / "result.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
