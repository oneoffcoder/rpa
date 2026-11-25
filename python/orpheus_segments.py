"""Revision 1.2 local development experiment. Run in the pinned Orpheus environment."""
import argparse
import ctypes
import hashlib
import json
import os
import platform
import threading
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/tts/orpheus-segmentation/protocol.json"
CACHE = ROOT / "build/tts/cache/huggingface/hub"
MODEL = "models--Pakorn2112--Orpheus-TTS-hmong-3b/snapshots/464d34449a778b1a6d9506bc10bd4e00f752c5c8"
CODEC = "models--hubertsiuzdak--snac_24khz/snapshots/d73ad176a12188fcf4f360ba3bf2c2fbbe8f58ec"
SETTINGS = dict(max_new_tokens=1200, temperature=0.6, top_p=0.95, top_k=50,
                repetition_penalty=1.1, eos_token_id=128258, pad_token_id=128263,
                do_sample=True, use_cache=True)
RATE = 24000


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def chunk_prompts(item):
    chunks = item["chunks"]
    if not chunks or any(not s.strip() for s in chunks) or " ".join(chunks).split() != item["text"].split():
        raise ValueError(f"Segmentation changes the text of {item['id']}")
    return [s if s.endswith((".", "?", "!")) else s + "." for s in chunks]


def assemble(waves):
    parts, joins = [waves[0]], []
    for left, right in zip(waves, waves[1:]):
        def edge(wave):
            active = np.flatnonzero(np.abs(wave.astype(float) / 32768) >= 10 ** (-50/20))
            if not len(active):
                raise ValueError("Silent segment")
            return int(active[0])
        trailing, leading = edge(left[::-1]), edge(right)
        added = max(0, round(.12 * RATE) - trailing - leading)
        joins.append(dict(left_quiet_samples=trailing, right_quiet_samples=leading, added_samples=added))
        parts.extend([np.zeros(added, dtype=np.int16), right])
    return np.concatenate(parts), joins


class Budget:
    """Sample physical footprint (Darwin rusage v0) and system VM pressure during inference."""
    def __init__(self, folder):
        self.folder = folder
        self.peak = self.call_peak = 0
        self.used = 0.
        self.start = None
        self.lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        self.check()
        threading.Thread(target=self.watch, daemon=True).start()

    def check(self):
        usage = ctypes.create_string_buffer(96)
        pressure = ctypes.c_int()
        size = ctypes.c_size_t(ctypes.sizeof(pressure))
        if self.lib.proc_pid_rusage(os.getpid(), 0, usage) or self.lib.sysctlbyname(
                b"kern.memorystatus_vm_pressure_level", ctypes.byref(pressure), ctypes.byref(size), None, 0):
            raise OSError("macOS resource telemetry unavailable")
        footprint = ctypes.c_uint64.from_buffer(usage, 16 + 7 * 8).value
        self.peak = max(self.peak, footprint)
        self.call_peak = max(self.call_peak, footprint)
        elapsed = self.used + (0 if self.start is None else time.monotonic() - self.start)
        if footprint >= 32 * 2**30 or pressure.value & 4 or elapsed >= 1800:
            save(self.folder / "resource-stop.json", dict(footprint=footprint, pressure=pressure.value,
                                                         generation_seconds=elapsed))
            os._exit(75)

    def watch(self):
        while True:
            try:
                self.check()
            except Exception as exc:
                save(self.folder / "telemetry-error.json", dict(error=str(exc)))
                os._exit(76)
            time.sleep(.25)


def load_models(cache):
    import torch
    from snac import SNAC
    from tokenizers import Tokenizer
    from transformers import LlamaConfig, LlamaForCausalLM
    if not torch.backends.mps.is_available():
        raise RuntimeError("This comparison requires the historical MPS runtime")
    torch.set_num_threads(4)
    repo = cache / MODEL
    raw = json.loads((repo / "config.json").read_text())
    rope = raw.pop("rope_parameters")
    raw["rope_theta"] = rope.pop("rope_theta")
    raw["rope_scaling"] = rope
    model = LlamaForCausalLM.from_pretrained(str(repo), config=LlamaConfig.from_dict(raw),
                local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("mps").eval()
    return model, Tokenizer.from_file(str(repo / "tokenizer.json")), SNAC.from_pretrained(str(cache / CODEC)).eval()


def complete_groups(speech):
    """Keep every intact SNAC frame; a malformed sequence is diagnostic-only."""
    ids = np.asarray(speech)
    slots = (ids - 128266) // 4096
    kept = np.zeros(len(ids), dtype=bool)
    frames = []
    for start in np.flatnonzero(slots == 0):
        if np.array_equal(slots[start:start+7], np.arange(7)):
            frames.append(ids[start:start+7] - 128266 - 4096*np.arange(7))
            kept[start:start+7] = True
    if not frames:
        raise ValueError("No complete SNAC frames")
    return np.stack(frames), np.flatnonzero(~kept).tolist()


def render(models, text, seed, path, budget):
    import soundfile as sf
    import torch
    model, vocab, codec = models
    text_ids = vocab.encode(text).ids
    assert text_ids[0] == 128000
    prompt = [128259, *text_ids, 128009, 128260]
    limit = model.config.max_position_embeddings
    if len(prompt) + SETTINGS["max_new_tokens"] > limit:
        raise ValueError("Request exceeds the configured model context")
    ids = torch.tensor([prompt], device="mps")
    record = dict(text=text, prompt_tokens=prompt, seed=seed, prompt_length=len(prompt),
                  model_context=limit, output_cap=SETTINGS["max_new_tokens"], status="started")
    sidecar = path.with_suffix(".json")
    previous = json.loads(sidecar.read_text()) if sidecar.exists() else None
    if previous:
        if any(previous[k] != record[k] for k in ("prompt_tokens","text","seed","output_cap")):
            raise ValueError("Saved render inputs changed")
        if previous["status"] in ("ok","diagnostic_only"):
            if digest(path) != previous["sha256"]:
                raise ValueError("Saved audio hash changed")
            return previous
        if previous["status"] != "error" or "output_ids" not in previous:
            raise ValueError("Cannot resume an unidentified previous generation")
        record["initial_error"] = previous["error"]
        record["initial_wall_seconds"] = previous["wall_seconds"]
        record["initial_peak_bytes"] = previous["sampled_peak_footprint_bytes"]
    if previous is None:
        save(sidecar, record)
    budget.call_peak = 0
    budget.start = time.monotonic()
    try:
        torch.manual_seed(seed)
        with torch.inference_mode():
            if previous:
                new = previous["output_ids"]
            else:
                sequence = model.generate(ids, attention_mask=torch.ones_like(ids), **SETTINGS)[0].tolist()
                if sequence[:len(prompt)] != prompt:
                    raise ValueError("Model returned a different input prefix")
                new = sequence[len(prompt):]
            stop = "speech_eos" if new and new[-1] == 128258 else "cap" if len(new) == 1200 else "unexpected_stop"
            record.update(output_ids=new, generated_tokens=len(new), stop=stop)
            if stop != "speech_eos" or 128257 not in new:
                raise ValueError(f"Unusable generation: {stop}")
            speech = new[new.index(128257)+1:-1]
            if not speech:
                record["stop"] = "empty_eos"
                raise ValueError("No audio codes before EOS")
            codes, rejected = complete_groups(speech)
            record["undecodable_audio_token_indices"] = rejected
            if codes.min() < 0 or codes.max() >= 4096:
                raise ValueError("Invalid SNAC codebook")
            layers = [torch.from_numpy(codes[:, slots].reshape(1, -1).copy()) for slots in ([0], [1,4], [2,3,5,6])]
            wave = codec.decode(layers).float().cpu().numpy().squeeze()
        if wave.shape != (len(codes)*2048,) or not np.isfinite(wave).all() or np.std(wave) < 1e-4:
            raise ValueError("Invalid decoded waveform")
        peak = float(np.abs(wave).max())
        sf.write(path, wave * (0.99 / peak if peak >= 1 else 1.), RATE, subtype="PCM_16")
        record.update(status="diagnostic_only" if rejected else "ok", frames=len(codes), samples=len(wave), seconds=len(wave)/RATE, sha256=digest(path))
    except Exception as exc:
        record.update(status="error", error=str(exc))
        raise
    finally:
        budget.check()
        record.update(wall_seconds=time.monotonic()-budget.start, sampled_peak_footprint_bytes=budget.call_peak)
        budget.used += record["wall_seconds"]
        budget.start = None
        save(sidecar, record)
    return record


def main():
    import soundfile as sf
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Fresh run directory")
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--repeat-seed", type=int, choices=[3408,3409])
    parser.add_argument("--resume", action="store_true", help="Reuse verified existing emissions of this run")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    for item in protocol["prompts"]:
        chunk_prompts(item)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    started = time.monotonic()
    budget = Budget(args.output)
    run = dict(protocol=protocol, protocol_sha256=digest(args.protocol), runner_sha256=digest(Path(__file__)),
               started=datetime.now(timezone.utc).isoformat(), settings=SETTINGS, model=MODEL, codec=CODEC,
               engine="MPS BF16 SDPA, CPU SNAC, text only, no speaker prefix", system=platform.platform(),
               versions={p:version(p) for p in ["torch","transformers","snac","tokenizers","numpy","soundfile","safetensors","accelerate"]},
               repeat_seed=args.repeat_seed, items=[])
    old = None
    if args.resume:
        old = json.loads((args.output/"run.json").read_text())
        for key in ("protocol_sha256","versions","settings","model","codec","repeat_seed"):
            if old[key] != run[key]:
                raise ValueError(f"Cannot resume after changing {key}")
        run["original_runner_sha256"] = old.get("original_runner_sha256",old["runner_sha256"])
        run["started"] = old["started"]
        prior = [json.loads(p.read_text()) for p in args.output.glob("*/*.json")]
        budget.used = sum(r.get("wall_seconds",0)+r.get("initial_wall_seconds",0) for r in prior)
        budget.peak = max([budget.peak]+[r.get("sampled_peak_footprint_bytes",0) for r in prior])
    run["model_files"] = {p.relative_to(args.cache).as_posix():digest(p) for ref in (MODEL,CODEC)
                          for p in sorted((args.cache/ref).iterdir()) if p.is_file()}
    if old and old["model_files"] != run["model_files"]:
        raise ValueError("Cannot resume with different weights")
    models = load_models(args.cache)
    run.update(setup_seconds=time.monotonic()-started+(old["setup_seconds"] if old else 0), configured_context=models[0].config.max_position_embeddings,
               input_truncation=False, runtime_output_limit=1200)
    save(args.output/"run.json", run)
    for item in protocol["prompts"]:
        folder = args.output/item["id"]
        folder.mkdir(exist_ok=args.resume)
        seed = args.repeat_seed if args.repeat_seed is not None else item["seed"]
        whole = render(models, item["text"], seed, folder/"whole.wav", budget)
        waves, chunks = [], []
        for i, text in enumerate(chunk_prompts(item)):
            path = folder/f"chunk-{i+1}.wav"
            chunks.append(render(models, text, seed+i, path, budget))
            waves.append(sf.read(path, dtype="int16")[0])
        joined, joins = assemble(waves)
        path = folder/"assembled.wav"
        sf.write(path, joined, RATE, subtype="PCM_16")
        run["items"].append(dict(id=item["id"], whole_status=whole["status"],
                                 assembled_status="ok" if all(c["status"]=="ok" for c in chunks) else "diagnostic_only",
                                 whole_sha256=whole["sha256"], chunks=[c["sha256"] for c in chunks],
                                 assembled_sha256=digest(path), joins=joins, samples=len(joined), seconds=len(joined)/RATE))
        run.update(generation_seconds=budget.used, sampled_peak_footprint_bytes=budget.peak)
        save(args.output/"run.json", run)
        print(item["id"],len(joined)/RATE,flush=True)


if __name__ == "__main__":
    main()
