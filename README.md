![RPA logo](https://rpa.oneoffcoder.com/_static/logo-small.png)

![CICD](https://codebuild.us-east-1.amazonaws.com/badges?uuid=eyJlbmNyeXB0ZWREYXRhIjoiblF0U29uRlVWYndLNUt3OFpLMHVhSHNkNDFpbC9BekJkZnpra2Z5aTRSdHB3MlhkVTE5NFhUVzR6QnJpT1JhTUovR3R4NmRtM3ZnNmc3YVg3U3RPUVk0PSIsIml2UGFyYW1ldGVyU3BlYyI6InllemU0L2ZHeVpjYkNKN2giLCJtYXRlcmlhbFNldFNlcmlhbCI6MX0%3D&branch=publish "CICD")

# Romanized Popular Alphabet

A book to read, write and speak Hmong. The published site is at [http://rpa.oneoffcoder.com](http://rpa.oneoffcoder.com).

## Hmong text-to-speech research

The repository contains **5,187 isolated-syllable MP3 recordings**, covering its
56 consonants plus an empty onset, 13 vowels, and seven tones. A complete decode
audit found **38.49 minutes of audio**. The recordings support a pronunciation
experiment; natural sentence speech still needs separate evaluation and data.

The new `rpa-tts` Python package audits and prepares the corpus, provides a
recording-concatenation baseline, and includes an **experimental Qwen3-TTS 0.6B
LoRA workflow for Apple Silicon**. The complete corpus has been prepared. A 64-recording canary
and a separate six-syllable memorization experiment ran on an Apple M4 Pro.
The [six-syllable listening set](docs/tts/memorization-six) includes original,
codec-reconstructed, base-model and adapted-model WAVs.

Start from the repository root with Python 3.11 or newer, [uv](https://docs.astral.sh/uv/),
and FFmpeg installed:

```bash
uv sync --locked
uv run rpa-tts audit
uv run rpa-tts prepare
uv run rpa-tts baseline --text "Kuv yog Hmoob." --output build/tts/baseline.wav
```

On an Apple Silicon Mac, the first small training experiment is:

```bash
uv sync --locked --extra mlx
uv run --extra mlx rpa-tts doctor --manifest build/tts/data/train.jsonl
uv run --extra mlx rpa-tts train
```

The default is a **100-update canary using 64 training recordings**, not a full
training run. It downloads the pretrained model on first use. See the
[Mac runbook and research limitations](docs/tts/README.md), the
[measured corpus audit](docs/tts/corpus-audit-2026-11-24.md), and the
[reproducible preparation summary](docs/tts/preparation-2026-11-24.json).

# Copyright Stuff

![One-Off Coder Logo](logo.png "One-Off Coder")

Copyright 2019 One-Off Coder

This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/) by [One-Off Coder](https://www.oneoffcoder.com).

![Creative Commons Attribution 4.0 International License](https://i.creativecommons.org/l/by/4.0/88x31.png "Creative Commons Attribution 4.0 International License")

# Contact

One-Off Coder

* [Website](https://www.oneoffcoder.com)
* [Twitter](https://twitter.com/oneoffcoder)
* [Facebook](https://www.facebook.com/oneoffcoder)
* [YouTube](https://www.youtube.com/channel/UCCCv8Glpb2dq2mhUj5mcHCQ)
