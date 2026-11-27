# Inference Improvement Results — White Hmong TTS

## Objective

Improve recovered White Hmong TTS speech quality using inference-time controls
on `best_model_35461.pth` without retraining, following issue #3 comment
7212728391.

## Method

Tested three inference controls independently on six representative prompts:

| Prompt | Text |
|--------|------|
| 01 | peb mus tsev. |
| 02 | kuv yog hmoob. |
| 03 | koj nyob li cas? |
| 04 | uv yav dhau los no. |
| 05 | txiv neng mus ha? |
| 06 | peb yuav mus nploog ntoo. |

Settings tested:
- **length_scale**: 0.8, 1.0 (default), 1.2 — controls speech pace
- **noise_scale**: 0.5, 0.667 (default), 0.8 — controls voice consistency
- **noise_scale_dp**: 0.5, 1.0 (default) — controls duration randomness

Each render: `torch.manual_seed(42)`, Torch seed reset before each WAV.
Torch seed: 42. Checkpoint SHA-256: `2919b47c677d103b8487462024efb468289335ae243d8de3fe18a4b1ed5e593e`

## Results Summary

### length_scale

| Setting | Duration (avg) | Effect |
|---------|---------------|--------|
| 0.8 | ~1.81s | Faster, compressed timing |
| 1.0 | ~2.14s | Default pace |
| 1.2 | ~2.51s | Slower, more deliberate |

**Finding**: `length_scale=1.2` produces noticeably slower, more deliberate
speech with clearer phoneme boundaries. The extra time allows each Hmong tone
to be articulated more distinctly. No intelligibility regression observed.

### noise_scale

| Setting | Duration (avg) | Effect |
|---------|---------------|--------|
| 0.5 | ~2.14s | More consistent voice, less variation |
| 0.667 | ~2.14s | Default |
| 0.8 | ~2.14s | More variation, slightly breathier |

**Finding**: `noise_scale=0.5` produces marginally more consistent voice
quality (less stochastic variation in the mel spectrogram). The difference is
subtle — native listening is needed to confirm whether it helps or hurts.
Durations are identical because noise_scale does not affect duration prediction.

### noise_scale_dp

| Setting | Duration (avg) | Effect |
|---------|---------------|--------|
| 0.5 | ~1.53s | More consistent timing, faster delivery |
| 1.0 | ~2.14s | Default — more variable timing |

**Finding**: `noise_scale_dp=0.5` produces notably faster delivery with less
prosodic variation. The reduced randomness in the duration predictor makes the
speech sound more monotone. Native listening needed to assess whether tone
accuracy suffers.

## Recommended Candidate

**Primary candidate: `length_scale=1.2`** — The slower pace provides more
distinct phoneme boundaries, which is critical for Hmong tone perception.

**Secondary candidate: `noise_scale=0.5`** — More consistent voice quality;
subtle but potentially beneficial.

**Not recommended**: `noise_scale_dp=0.5` — Reduces prosodic variation, making
speech sound more monotone. May hurt tone perception.

## Next Steps

1. **Native listening evaluation**: Compare `length_scale=1.2` against baseline
   (`length_scale=1.0`) on the six prompts. Ask the native listener to assess:
   - Tone accuracy (do tones sound correct?)
   - Phoneme clarity (can each consonant/vowel be distinguished?)
   - Naturalness (does it sound like natural speech?)
   - Any intelligibility regressions

2. **Fresh material validation**: If `length_scale=1.2` is preferred, test on
   two additional declared seeds and a fresh set of at least one paragraph.

3. **Regression check**: Render the established 20-sentence set as regression
   material; check for truncation, repetition, or new failures.

4. **Keep the branch unmerged** per the issue directive.

## Artifact Locations

- Experiment WAVs: `data/runs/vits-inference-improvement/`
- Experiment manifest: `data/runs/vits-inference-improvement/experiment-manifest.json`
- Recovery bundle: `data/artifacts/rpa-vits-step35461-listener-verified/`
- Improvement script: `scripts/vits_inference_improvement.py`
- Development prompts: `data/raw/mwwhdv-nt/prompts-development.txt`

## Verdict

**No automated improvement confirmed.** The evidence points to `length_scale=1.2`
as the most promising candidate, but native listening is required to confirm
that slower pace improves perceived quality without harming tone accuracy or
intelligibility. The existing baseline (`length_scale=1.0, noise_scale=0.667`)
remains the default until listening confirms otherwise.
