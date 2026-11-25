"""Check SNAC packing on human references and replay the saved Orpheus emissions."""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from snac import SNAC

from orpheus_segments import CACHE, CODEC, RATE, complete_groups, digest


def decode(codec, layers, seed):
    # The pinned SNAC decoder has noise injection even in evaluation mode.
    torch.manual_seed(seed)
    return codec.decode(layers).flatten().cpu().numpy()


def publisher_layers(ids):
    """Index the three SNAC levels as in the Hmong publisher's inference notebook."""
    if len(ids) == 0 or len(ids) % 7:
        raise ValueError("Incomplete seven-token frame")
    levels = [[], [], []]
    for start in range(0, len(ids), 7):
        for slot, level in enumerate((0, 1, 2, 2, 1, 2, 2)):
            code = int(ids[start + slot]) - 128266 - 4096 * slot
            if not 0 <= code < 4096:
                raise ValueError("Invalid codebook position")
            levels[level].append(code)
    return [torch.tensor([codes]) for codes in levels]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("originals", type=Path)
    parser.add_argument("--cache", type=Path, default=CACHE)
    args = parser.parse_args()
    torch.set_num_threads(4)
    codec = SNAC.from_pretrained(str(args.cache / CODEC)).eval()
    results = {"natural_references": {}, "saved_emissions": {}}
    for path in sorted(args.originals.glob("*.m4a")):
        pcm = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1",
                                       "-ar", str(RATE), "-f", "f32le", "-"])
        samples = np.frombuffer(pcm, dtype="<f4").copy()
        original = codec.encode(torch.from_numpy(samples).view(1, 1, -1))
        first, second, third = [x.flatten().numpy() for x in original]
        columns = [first, second[::2], third[::4], third[1::4], second[1::2], third[2::4], third[3::4]]
        ids = (np.column_stack(columns) + np.arange(7) * 4096 + 128266).ravel()
        recovered = publisher_layers(ids)
        codes, rejected = complete_groups(ids)
        assert not rejected and np.array_equal(codes, np.column_stack(columns))
        assert all(torch.equal(a,b) for a,b in zip(original,recovered,strict=True))
        direct, unpacked = decode(codec, original, 3407), decode(codec, recovered, 3407)
        assert np.array_equal(direct, unpacked)
        results["natural_references"][path.stem] = dict(frames=len(first), original_samples=len(samples),
                decoded_samples=len(unpacked), equal_to_native_snac_decode=True)
    for path in sorted(args.run.glob("*/*.json")):
        record = json.loads(path.read_text())
        ids = record["output_ids"]
        text = record["text"]
        try:
            layers = publisher_layers(ids[ids.index(128257)+1:-1])
        except ValueError as exc:
            if record["status"] != "diagnostic_only":
                raise
            results["saved_emissions"][text] = str(exc)
            continue
        wave = decode(codec,layers,record["seed"])
        peak = float(np.abs(wave).max())
        with tempfile.TemporaryDirectory() as folder:
            replay = Path(folder)/"replay.wav"
            sf.write(replay,wave * (.99 / peak if peak >= 1 else 1.),RATE,subtype="PCM_16")
            byte_match = digest(replay) == digest(path.with_suffix(".wav"))
        results["saved_emissions"][str(path.relative_to(args.run))] = dict(text=text, byte_match=byte_match)
        assert byte_match, path
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
