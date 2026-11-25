# Six-syllable Hmong listening experiment

Open the player from the repository root.

```sh
open docs/tts/memorization-six/listen.html
```

Every row contains the same prompt in four versions. The original is the training
WAV prepared from One-Off Coder’s MP3. The reconstruction decodes all 16 streams
of the frozen speech codec, including its complete final frame. The base and
adapted models each generate speech from the RPA text alone, using the same seed
and sampling settings. The WAVs retain their individual levels.

| Syllable | Original | Reconstructed | Base | Adapted |
| --- | --- | --- | --- | --- |
| kuv | [Original](audio/kuv-original.wav) | [Reconstructed](audio/kuv-reconstructed.wav) | [Base](audio/kuv-base.wav) | [Adapted](audio/kuv-adapted.wav) |
| yog | [Original](audio/yog-original.wav) | [Reconstructed](audio/yog-reconstructed.wav) | [Base](audio/yog-base.wav) | [Adapted](audio/yog-adapted.wav) |
| hmoob | [Original](audio/hmoob-original.wav) | [Reconstructed](audio/hmoob-reconstructed.wav) | [Base](audio/hmoob-base.wav) | [Adapted](audio/hmoob-adapted.wav) |
| po | [Original](audio/po-original.wav) | [Reconstructed](audio/po-reconstructed.wav) | [Base](audio/po-base.wav) | [Adapted](audio/po-adapted.wav) |
| poj | [Original](audio/poj-original.wav) | [Reconstructed](audio/poj-reconstructed.wav) | [Base](audio/poj-base.wav) | [Adapted](audio/poj-adapted.wav) |
| pos | [Original](audio/pos-original.wav) | [Reconstructed](audio/pos-reconstructed.wav) | [Base](audio/pos-base.wav) | [Adapted](audio/pos-adapted.wav) |

Listen for the intended consonant, vowel and tone. The last three prompts share
the same consonant and vowel. Their tones should distinguish them. Listener
judgments on this set are pending.

## What the model learned

The six train-split recordings total 2.305 s. The talker LoRA saw each recording
100 times across 150 optimizer updates, using the original Qwen backend and its
frozen codec and residual-code predictor. Training took 32.1 s on the M4 Pro.
Mean training loss was 0.05608. The experiment reads only the training manifest
and these six WAVs.

| Syllable | Original vs reconstructed waveform correlation | Reference codec frames | Base generated frames | Adapted generated frames | Adapted first divergent frame (1-based) |
| --- | ---: | ---: | ---: | ---: | ---: |
| kuv | 0.940 | 5 | 18 | 7 | 3 |
| yog | 0.911 | 6 | 11 | 16 | 3 |
| hmoob | 0.947 | 6 | 13 | 9 | 2 |
| po | 0.921 | 5 | 8 | 8 | 2 |
| poj | 0.895 | 5 | 8 | 8 | 3 |
| pos | 0.864 | 5 | 20 | 9 | 4 |

Waveform correlations use the original sample times, with no lag search or
time stretching. All six original/reconstruction pairs are available for judging
tone by ear. Before training, the previous 64-recording canary adapter changed the
highest-scoring first token on every prompt. Repeated base-only probes matched
exactly. Reloading the new six-item adapter also changed the top choice on every
prompt and selected the recorded first token for all six.

Given the correct preceding audio frames, the adapted model predicted **38/38**
first-stream targets, including the six end tokens. The base predicted **11/38**.
Generating from text, the adapted model diverged from the recording by frame
2–4 on every prompt. All six outputs ran longer than their recorded code sequences.

At the first divergence we replayed each prompt twice through the talker.
The previous first-stream tokens were identical. Changing only the other 15
streams from their recorded values to their generated values changed the next
highest-scoring token from the reference token to a different token in **6/6**.
That isolates a source of drift in this run. The test measures exact codec-token
agreement. Different token sequences can still produce intelligible speech.

The project now seeds MLX immediately before generation. The old synthesis path
seeded before adapter initialization, which consumed random numbers before the
first sample. The corrected CLI reproduced the checked-in adapted `kuv` WAV
byte for byte. [Per-syllable predictions, source hashes and WAV hashes](measurements.json)

## Reproduce

Use the prepared training split and the saved 64-recording canary adapter from
the [Mac runbook](../README.md#measured-mac-canary). Use a fresh output directory.
The runner first reconstructs the recordings, renders the base model and probes
the canary. It then trains the six-item adapter, reloads the checkpoint, renders
the adapted clips and replays the first divergence. The model uses raw RPA text
and its usual end token throughout.

```sh
uv run --locked --extra mlx python python/tts_memorization.py \
  --train-manifest build/tts/data/train.jsonl \
  --canary-adapter build/tts/runs/canary-001/adapters \
  --output build/tts/runs/memorization-six-repeat
```

Each run writes `train-six.jsonl`, `model/run.json`, `model/adapters/`, 24 WAVs
under `audio/` and `measurements.json`. Model revision
`1eccf1cb2519b5a4e8a95b5f0544f3303568164f` and synthesis seed `3407`
identify this listening set. Original recordings are by One-Off Coder.
[Corpus attribution and license](../../../README.md#copyright-stuff)
