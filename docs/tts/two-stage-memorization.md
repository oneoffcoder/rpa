# Two-stage Hmong memorization on the M4 Pro

At synthesis seed 3407 the two-stage adapter generated **five of six training
syllables exactly as the codec reconstructed them**. The previous adapter at
`588bd57` matched zero. The exception, **`poj`, produced the exact `po` WAV**.
The listener found most isolated syllables recognizable with the two-stage
adapter. The 150-update phrase **failed the listening test**. Only the first
two syllables were audible to the listener, who described the delivery as
unnaturally pinched and nasal. The codec reconstructions have yet to receive
a separate listening judgment.

```sh
open /Users/super/git/oneoffcoder/rpa/build/tts/runs/joint-six/listening/listen.html
```

## Text-only generation

| Prompt | Previous duration | Two-stage duration | Two-stage WAV matches |
| --- | ---: | ---: | --- |
| kuv | 0.56 s | 0.40 s | kuv reconstruction |
| yog | 1.28 s | 0.48 s | yog reconstruction |
| hmoob | 0.72 s | 0.48 s | hmoob reconstruction |
| po | 0.64 s | 0.40 s | po reconstruction |
| poj | 0.64 s | 0.40 s | **po reconstruction** |
| pos | 0.72 s | 0.40 s | pos reconstruction |
| Kuv yog Hmoob. | 2.08 s | 0.96 s | Listener heard only the first two syllables |

All 14 model renders emitted EOS within the 128-frame limit. Each WAV contains
the complete decoder output. Both adapters used automatic language, temperature
0.9, top-k 50, top-p 1.0 and repetition penalty 1.05. The six previous WAV hashes
reproduced the published baseline. Seven independent runs of the synthesis CLI
reproduced every new WAV with reference-audio and corpus paths blocked to Python
file opens.

Replaying the exact phrase reproduced its saved codec-token hash. All seven text
tokens, including the whole spelling of “Hmoob,” had reached the talker before
it emitted EOS after 12 audio frames. The first four frames matched the isolated
`kuv` recording in all 16 codebooks. With the same seed, lowercase
`kuv yog hmoob` stopped after five frames (0.40 s), with or without the period.
The training set contains six isolated utterances and no examples of continuing
from one Hmong syllable into the next. The cause of the nasal quality is unresolved.

The initial text embedding is identical for `po` (token 5368) and `poj` (tokens
5368, 73). The extra token first enters after the model has generated a whole
audio frame. Their recordings already differ in the first frame’s first residual
code (420 versus 1362). With matching randomness, this streaming prompt gives
the predictor identical inputs at that first decision. The entire generated
tone contrast collapsed in this run.

## Training and verification

The same six train-split recordings, sample order, base revision
`1eccf1cb2519b5a4e8a95b5f0544f3303568164f`, rank/alpha 16, Adam rate 0.0001
and four-recording gradient accumulation as the previous experiment were used.
The 150 updates took 32.1 s. The three split-manifest hashes still match their
preparation record. Training and evaluation used only the six training recordings.

| Teacher-forced measure | Previous adapter | Two-stage adapter |
| --- | ---: | ---: |
| First-codebook and EOS cross-entropy | 0.0000507 | 0.0000556 |
| First-codebook and EOS correct | 38/38 | 38/38 |
| Residual cross-entropy | 6.35363 | 0.003583 |
| Residual codes correct | 24/480 | 479/480 |

The local MLX trainer scores code 0 and EOS using the preceding talker state.
For each real frame a second causal pass uses that same state and codes 0–14 to
predict codes 1–15. Prompt and padding positions contribute no loss. EOS belongs
only to the first stream. The joint objective is first-stream mean cross-entropy
plus 0.3 times residual mean cross-entropy. The cached single-step inference
path was independently replayed against the training calculation. [Qwen reference trainer](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/finetuning/sft_12hz.py) · [Pinned MLX generation](https://github.com/Neywa-Labs/mlx-audio/blob/91caeb9ed95b3839c7c08bfc58ee8e7d6216dea8/mlx_audio/tts/models/qwen3_tts/qwen3_tts.py)

Both adapter groups received gradients on update one (L2 norms 18.06 and 0.679).
All 392 talker and 70 residual LoRA tensors changed. The codec weight hash stayed
identical. All six prompts reproduced both stages’ pre-save logits after reload.
During a separate `kuv` inference check, disabling the residual update changed
the generated sequence from 5 to 7 frames. Disabling the talker update hit the
128-frame limit. Residual loss fell from 0.003691 at update 125 to 0.003583 at
150, a 2.9% decrease. The prespecified threshold for a single extension to 300
was 5%, so the saved comparator is update 150.

## Reproduce

With the existing [Mac environment and prepared data](README.md), run from the
repository root using a fresh output directory.

```sh
uv run --locked --extra mlx python python/tts_joint_memorization.py train --output build/tts/runs/joint-repeat
uv run --locked --extra mlx python python/tts_joint_memorization.py listen --output build/tts/runs/joint-repeat
open build/tts/runs/joint-repeat/listening/listen.html
```

The local run is under `build/tts/runs/joint-six/`. It contains
`step-150/adapters/`, `step-150/run.json`, `training.json`, and
`listening/results.json` with the WAV hashes, stopping events and diagnostics.
Checkpoint SHA-256 is `2951133372a5ca332a8b86233a6bfb3b04d8a851647a7f1e8a65c1638900bb28`.
