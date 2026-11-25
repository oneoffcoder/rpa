# Hmong TTS: corpus preparation and a Mac training canary

## Current status

The first milestone is a reproducible experiment using the recordings already
in `source/_static/mp3`. The full corpus has been decoded and prepared; all 5,187
expected filenames are present and there are no exact source-byte or decoded-PCM
duplicates. The measured audio duration is **38.49 minutes**, not the earlier
rough estimate of one to two hours.

The [Apple Silicon canary](#measured-mac-canary) completed 100 optimizer updates
on 64 recordings and produced six WAVs from the saved adapter. Hmong
pronunciation, tone accuracy and connected-speech naturalness remain unverified.

The initial model is
[`mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16`](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16).
Hmong is outside [Qwen3-TTS's documented supported languages](https://github.com/QwenLM/Qwen3-TTS#introduction).
This is an adaptation experiment. It does not silently label Hmong as English
or invent a Hmong language token.

## 1. Install the data tools

Use a native arm64 terminal and Python on an Apple Silicon Mac. Python 3.11 or
newer is required; 3.11 is a conservative starting version for the optional
speech dependencies. The preparation commands also work on Linux with FFmpeg.

```bash
git clone --branch feat/hmong-tts-bootstrap https://github.com/oneoffcoder/rpa.git
cd rpa
brew install uv ffmpeg
uv sync --locked --python 3.11
uv run python -m unittest discover -s tests -v
```

If the repository is already checked out, fetch and switch to the feature branch
in a clean checkout instead of cloning into an existing directory. The existing
Sphinx book and its default Makefile targets remain independent of Python TTS
dependencies.

## 2. Audit the original recordings

```bash
uv run rpa-tts audit --workers 4
```

This writes `build/tts/audit/manifest.jsonl`, `report.json`, and `report.md`.
Every original recording is probed and decoded with FFmpeg. Damaged packets,
empty/zero-signal audio, missing labels, unknown labels and duplicate labels
make the audit command exit unsuccessfully. Amplitude and duration review flags
are advisory; they do not silently discard quiet or short syllables.

The manifest stores the relative source path, normalized RPA text, consonant,
vowel, tone, duration, original codec/rate/channels, source and decoded-PCM
SHA-256 hashes, and quality flags. The report records Git provenance, the audio
tree hash and the decoder version. Original MP3 files are never rewritten.

The inventory is `(56 + 1 empty onset) × 13 × 7 = 5,187`. The 13 bare vowels are
already represented by empty onset plus unmarked tone. `_` is an internal marker
for that tone and is never added to training text. The parser recognizes this
repository's inventory; it is not a validator for every Hmong dialect, loanword,
or spelling convention. For example, the book also discusses an occasional
final `d` that is outside the seven recorded tone markers.

See the [committed audit](corpus-audit-2026-11-24.md) for measured results and
source hashes.

## 3. Prepare train, validation and test data

```bash
uv run rpa-tts prepare --workers 4
```

This checks the audited source hashes, verifies text/CVT consistency, resamples
to **24 kHz mono PCM16 WAV**, and writes:

```text
build/tts/data/audio/<syllable>.wav
build/tts/data/train.jsonl
build/tts/data/valid.jsonl
build/tts/data/test.jsonl
build/tts/data/dataset.json
```

No VAD, denoising, loudness normalization, duration filter or concatenated fake
sentence examples are used. PCM16 conversion quantizes and saturates samples
outside its representable range; this is reported per recording and in the
summary, so it is not described as lossless. In the measured preparation,
60 samples across six files saturated after resampling. Review the original
high-level recordings before interpreting fine acoustic comparisons.

The default seed is 3407, with roughly 80/10/10 groups. The audited corpus gives
**4,151 train, 518 validation and 518 test recordings**. All observed consonants,
vowels and tones remain represented in training. Repeated text, identical source
files, and identical native-rate decoded waveforms stay together. Holdout is
assigned before conversion; generated audio is not fed back into training.

Manifest audio paths are relative to the manifest directory. Every row has an
explicit `split` and an output-audio hash. For a stronger compositional test,
hold out entire consonant/vowel families, keeping all seven tones together:

```bash
uv run rpa-tts prepare --split-unit cv --output build/tts/data-cv
```

Prepared output directories are immutable by convention: the command refuses to
overwrite an existing one. Use a new `--output` for a new seed, corpus or recipe.
Both split designs test unseen recorded combinations. Neither establishes
generalization to new speakers, sessions, dialects or natural connected speech.

## 4. Listen to a recording-based reference

```bash
uv run rpa-tts baseline --text "Kuv yog Hmoob." --output build/tts/baseline.wav
afplay build/tts/baseline.wav
```

This joins the original syllable recordings with explicit pauses. It is useful
as a recognizable reference and a pipeline smoke check, but it is not neural
TTS or a fair held-out-syllable model: it looks up the full recording inventory.
Unrecognized spellings and characters cause an error rather than disappearing
from the output. Numbers and abbreviations need manual expansion for now.

## 5. Check the Mac training environment

```bash
uv sync --locked --extra mlx
uv run --extra mlx rpa-tts doctor --manifest build/tts/data/train.jsonl
```

`doctor` checks native Apple Silicon Python, installed dependencies, Metal
availability and the training manifest. It does not download model weights.
Inspect its `ready` field. A Linux preparation machine should report
`apple_silicon: false`, `metal_available: false`, `ready: false`.

The optional dependencies are pinned to these inspected source revisions:

| Component | Revision | Why |
| --- | --- | --- |
| mlx-tune 0.6.0 | `9690fe1fed19c72f0810902b9c188d9a2625eb5b` | Qwen3-TTS LoRA trainer and collator |
| mlx-audio 0.4.2 | `91caeb9ed95b3839c7c08bfc58ee8e7d6216dea8` | Base-model encoder and matching generation API |

`uv.lock` resolves the full dependency graph. A newer mlx-audio 0.6.8 revision
was rejected during setup because its fsspec requirement conflicts with
mlx-tune's datasets constraint. The locked revisions installed on native arm64
Python 3.11.14 and ran on Metal with MLX 0.32.4.

If the configured Hugging Face cache points to an unavailable drive, set a
writable cache for this checkout before running training or synthesis. Use the
same cache for both commands.

```bash
export HF_HOME="$PWD/build/tts/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
```

## 6. Run a small training canary

```bash
uv run --extra mlx rpa-tts train \
  --manifest build/tts/data/train.jsonl \
  --output build/tts/runs/canary-001 \
  --steps 100 --max-samples 64 \
  --model-revision 1eccf1cb2519b5a4e8a95b5f0544f3303568164f
```

This downloads the pretrained Base BF16 model on first use, resolves its model
revision to an immutable Hugging Face snapshot, selects and shuffles 64 training
recordings with a fixed seed, and performs 100 optimizer updates. Defaults are
microbatch 1, gradient accumulation 4, LoRA rank/alpha 16, and learning rate
`1e-4`. The trainer repeats examples as needed. `--steps` counts optimizer
updates, not individual recordings.

The current upstream implementation trains LoRA parameters in the talker using
the first codec stream. Text embeddings, the speech codec and the remaining
codebook predictor stay frozen. It is not a custom CVT-embedding architecture
or full-model adaptation. The collator needs decoded float32 arrays, not paths;
the wrapper supplies real 24 kHz audio because merely passing a sample-rate label
to the upstream codec does not resample it.

The run produces:

```text
build/tts/runs/canary-001/run.json
build/tts/runs/canary-001/adapters/adapter_config.json
build/tts/runs/canary-001/adapters/adapters.safetensors
```

`run.json` records selected IDs and audio hashes, manifest hash, model revision,
package versions, training settings, measured timing, training loss, peak memory
when available, and saved-adapter hashes. It becomes `trained` only after finite
loss and real checkpoint artifacts exist; failures are recorded as `failed`.
Training rejects `valid` and `test` rows and nonempty existing run directories.
The pinned trainer has no validation loop, early stopping or optimizer-state
resume. The `valid` and `test` files are held aside for separate evaluation.

After a successful canary, a first larger pilot can use every training example:

```bash
uv run --extra mlx rpa-tts train \
  --manifest build/tts/data/train.jsonl \
  --output build/tts/runs/pilot-001 \
  --steps 2000 --max-samples 0
```

These are experiment settings, not a proven Hmong recipe. Use canary measurements
to estimate runtime before scaling. `--model-revision <commit>` can repeat an
exact model snapshot recorded by a previous run.

## 7. Generate and evaluate speech

```bash
uv run --extra mlx rpa-tts synthesize \
  --adapter build/tts/runs/canary-001/adapters \
  --text "Kuv yog Hmoob." --output build/tts/neural-example.wav
afplay build/tts/neural-example.wav
```

Inference restores the exact model revision and saved LoRA configuration, checks
adapter hashes, and calls the full Qwen model's generation method. It uses the
same automatic-language/no-reference conditioning as training. No test recording
is supplied as a voice reference. A metadata sidecar accompanies each output.

Start evaluation with isolated training and held-out syllables. Ask native
listeners to transcribe the consonant, vowel and tone without seeing the expected
label, then rate naturalness separately. Keep tone accuracy distinct from timbre
similarity and training loss. Review high-level flagged recordings before treating
them as acoustic ground truth. Compare minimal tone sets and all onset/vowel
groups, including empty onsets. Keep validation for experiment choices and reserve
test until the recipe is fixed. Sentence listening is an exploratory check until
real sentence recordings and a sentence test set exist.

## Measured Mac canary

The canary ran on November 25, 2026 on an Apple M4 Pro with 48 GiB of unified
memory and macOS 27.0.1. The local audit decoded all 5,187 MP3s with FFmpeg
9.0.2 and found zero errors. The prepared split contained 4,151 training, 518
validation and 518 test recordings.

| Measurement | Result |
| --- | ---: |
| Optimizer updates / recordings sampled | 100 / 64 |
| Trainer elapsed time | 28.22 s |
| Training command wall time with the model cached | 31.10 s |
| Peak MLX allocated memory | 2,992,435,892 bytes (2.79 GiB) |
| macOS peak process memory footprint (`time -l`) | 4,150,152,672 bytes (3.87 GiB) |
| Mean training loss | 0.803044 |
| Last update loss (trainer log) | 0.0830 |

Run metadata and adapter weights are in
`build/tts/runs/canary-001/run.json` and
`build/tts/runs/canary-001/adapters/adapters.safetensors`. Neural synthesis
reloaded that checkpoint and produced `po.wav`, `poj.wav`, `pos.wav`,
`ntxwj.wav`, `uab.wav` and `kuv-yog-hmoob.wav` under
`build/tts/runs/canary-001/audio/`. Each file has a JSON sidecar.
`ntxwj` was in the canary sample. The three `po` forms were in the training
split and outside the 64 sampled recordings. `uab` was in the validation split.
The recording-based phrase reference is `build/tts/baseline.wav`.

All six generated WAVs were non-silent 24 kHz mono PCM16 with matching
sidecar checksums and zero clipped samples. These checks establish that the
saved adapter reloads and generates audio. Native-listener transcription and
ratings are still needed to assess Hmong consonants, vowels, tones and prosody.

## Source references

- [Original repository and CC BY 4.0 notice](https://github.com/oneoffcoder/rpa/tree/cb0401e0c38652f998ca1b3bfe49d2084a279be7)
- [Qwen3-TTS official model and language documentation](https://github.com/QwenLM/Qwen3-TTS)
- [Pinned mlx-tune Qwen fine-tuning example](https://github.com/ARahim3/mlx-tune/blob/9690fe1fed19c72f0810902b9c188d9a2625eb5b/examples/20_qwen3_tts_finetuning.py)
- [Pinned trainer and collator implementation](https://github.com/ARahim3/mlx-tune/blob/9690fe1fed19c72f0810902b9c188d9a2625eb5b/mlx_tune/tts.py)
- [Pinned audio codec implementation](https://github.com/ARahim3/mlx-tune/blob/9690fe1fed19c72f0810902b9c188d9a2625eb5b/mlx_tune/audio_codecs.py)
- [Pinned MLX Qwen generation implementation](https://github.com/Neywa-Labs/mlx-audio/blob/91caeb9ed95b3839c7c08bfc58ee8e7d6216dea8/mlx_audio/tts/models/qwen3_tts/qwen3_tts.py)

The corpus retains its existing attribution/license notice. Generated datasets,
model downloads and checkpoints are local build artifacts and are not committed.
