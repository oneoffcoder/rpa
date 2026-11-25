# Orpheus whole-sentence and phrase comparison

**Revision 1.2 primary run, November 25, 2026.** Orpheus remains the active research model following the earlier listener's preference on short Hmong sentences. F5 and Qwen are deferred. The frozen policy produced eight listening clips. The family-sentence assembly contains a diagnostic recovery from an invalid model emission and fails the technical gate. Word coverage, tones, voice, pauses and naturalness await native assessment. All four whole-request WAVs exactly match the previous audit.

## Listening on the Mac

Open the live, local blind panel from Terminal.

```sh
open http://127.0.0.1:8766/
```

If the Mac or server has restarted, run the server from the repository root, then open the same address.

```sh
python3 python/orpheus_listen.py build/tts/orpheus-segments-primary
```

Use the same browser to resume. The panel shuffles eight opaque IDs once and saves the separate mapping in the local run directory. It locks the first transcription, confidence, listener alias, dialect, exposure and timestamp for each clip in SQLite. After all eight first responses it displays target text and collects word order, consonants, vowels, tones, phonation, omissions, additions, repetitions, voice consistency, pauses, naturalness and continuity. Only after all ratings does it reveal model conditions and individually playable chunks. The recordings are familiar development probes. The earlier named comparison was informal. The unchanged short controls intentionally occur twice.

## Primary results

| RPA text | Seed | Whole | Phrase assembly | Join |
|---|---:|---:|---:|---|
| Kuv yog Hmoob. | 3407 | 1.37 s | 1.37 s | Single request, exact same WAV |
| Kuv mus kawm ntawv. | 3409 | 2.22 s | 2.22 s | Single request, exact same WAV |
| Kuv hlub kuv niam kuv txiv thiab kuv tej me nyuam. | 3410 | 4.44 s | 4.69 s | After “txiv”, 308 ms observed quiet. Diagnostic assembly |
| Nyuj ua rau nyuj thauj nees ua rau nees ris. | 3412 | 5.03 s | 5.63 s | After “thauj”, 256 ms observed quiet |

Each whole request and chunk emitted speech EOS before 1,200 new tokens. There were ten model calls. None truncated its prompt. The longest used 416 generated tokens. EOS measures model termination only. The model had a configured context of 131,072 positions, and the local wrapper supplied the entire tokenized prompt without an additional text cutoff. No target recording, target duration or voice reference entered inference.

For the family sentence's first chunk, the model emitted 202 audio tokens where SNAC expects ordered groups of seven. The strict decoder rejected the output. Zero-based audio-token offsets 168 through 173 formed an incomplete, out-of-order group inside the sequence. The diagnostic decoder used the same saved emission, recovered all 28 complete frames including those following the error, and kept all 202 raw audio tokens in the sidecar. Six tokens had no decodable frame. No replacement seed or request was used. The primary assembly remains technically failed regardless of how the diagnostic example sounds. Its second chunk and both chunks of the other long sentence decoded normally.

The runner kept each complete decoded sample in its raw WAV. The join rule measured edge silence below −50 dBFS and would add only enough silence to bring a clause boundary to 120 ms. Both boundaries already exceeded that value, so each assembly consists of the unmodified PCM chunks in order. An audible gap or voice change can still occur. An independent byte comparison checked every assembled sample against the source chunks.

The measured model-call and decode time totaled **128.18 seconds**, including the failed call and recovery decoding. Setup across the two processes took **20.21 seconds**. The maximum sampled macOS physical footprint was **11.21 GiB** with 250 ms polling, within the 30-minute and 32-GiB bounds. New model downloads were unnecessary. PyTorch 2.8.0 on MPS used the original BF16 weights, SDPA, the historical tokenizer and speaker-free prompt, and the CPU SNAC decoder at 24 kHz.

**Decision at handoff.** The full primary policy cannot qualify for opt-in integration, because one long assembly required a decoder recovery. Whether the clean long pair shows a useful improvement awaits native responses. Repeats at seeds 3408/3409, fresh balanced development passages, product integration, F5 timing trials and training were not run. If a native listener finds a useful change, retain the technical failure when deciding whether to repeat. The six fresh passages require a passing policy. Neither a familiar sentence nor an identical short control establishes generalization.

## Reproduce and verify

The [predeclared text and boundary rules](protocol.json) retain spelling and case. A nonfinal phrase gets a terminal period for delivery. The first request uses the historical seed, and following chunks add their zero-based index. Two clear long coordination/parallel structures were selected before generation. The third historical long sentence had a less certain internal boundary. [Machine results with prompt tokens, hashes, stops, durations, gain-independent sample counts, resource measurements and listener counts](primary-results.json).

From this checkout on the Mac, use the original isolated runtime and the cached snapshots. Supply a fresh directory for a new measurement.

```sh
HF_HUB_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
  build/tts/hmong-model-audit/runtime/bin/python -u python/orpheus_segments.py \
  build/tts/orpheus-segments-repeat-primary
python3 python/orpheus_report.py build/tts/orpheus-segments-repeat-primary
```

The measured run used the same command with destination `build/tts/orpheus-segments-primary`. A second invocation with `--resume` decoded the saved malformed emission and continued the remaining calls. Model hashes, old emissions, inputs and settings are checked on resume. The source hashes before and after diagnostic recovery are in the machine results. The executed recovery runner is also stored in the local run directory. The repository runner subsequently moved the optional SoundFile import into the inference entry points so fixture tests can run with the base project dependencies.

The active inference dependencies and exact file hashes are in the machine results. For another Mac, install the listed versions in a Python 3.11 environment. Obtain the two pinned snapshots from [Orpheus Hmong 3B](https://huggingface.co/Pakorn2112/Orpheus-TTS-hmong-3b/tree/464d34449a778b1a6d9506bc10bd4e00f752c5c8) and [SNAC 24 kHz](https://huggingface.co/hubertsiuzdak/snac_24khz/tree/d73ad176a12188fcf4f360ba3bf2c2fbbe8f58ec). The runner accepts the Hugging Face hub cache root through `--cache`. The Hmong publisher lists Apache-2.0 metadata and separately describes research/education use with a commercial data-rights caveat. The Unsloth and Canopy Orpheus base cards list Apache-2.0 and name Meta Llama 3.2 among the upstream bases. The Meta base lists the Llama 3.2 Community License. This was a local research evaluation. [Hmong model card](https://huggingface.co/Pakorn2112/Orpheus-TTS-hmong-3b) · [Unsloth base](https://huggingface.co/unsloth/orpheus-3b-0.1-ft) · [Canopy base](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft) · [Llama base](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)

All named media, generated token arrays, reveal keys, SQLite responses and the server log stay on the Mac under `build/tts/orpheus-segments-primary/`. Each prompt directory contains `whole.wav`, numbered raw chunks, their JSON sidecars and `assembled.wav`. The server binds only to loopback. Read current listener counts and completed ratings with the command below.

```sh
python3 python/orpheus_report.py build/tts/orpheus-segments-primary
.venv/bin/python -m unittest discover -s tests -q
```

At handoff, 36 repository tests passed. A separate Chrome panel backed by a disposable database played all eight WAVs, saved and locked first answers before showing targets, completed the second scoring phase, revealed conditions and raw chunks, and resumed after reload. Those fixture responses are excluded from the study.

## Prior run index

[Historical manifests, hashes, references, settings and available timings](run-index.json). Paths starting with `build/` are local ignored artifacts. Fields absent from the original logs remain unknown. No historical model was rerun to fill missing fields.

| Experiment | Command or entry point | Evidence and listener result |
|---|---|---|
| Original-syllable assembly | `rpa-tts baseline --text "Kuv yog Hmoob." --output build/tts/baseline.wav` | 5,187 isolated clips, 38.49 minutes. Joined syllables sounded unnaturally separated. |
| Qwen 64-clip canary | [Pinned training and synthesis commands](../README.md#measured-mac-canary) | 100 updates. Phrase incorrect. The sampled clips omitted all three target syllables. |
| Qwen first-stream adapter | [Six-syllable runner and commands](../memorization-six/README.md) | Correct prior history gave 38/38 first-stream targets. Free generation diverged on all six. |
| Qwen two-stage adapter | [Joint training and replay commands](../two-stage-memorization.md#reproduce) | 150 updates, five exact codec reconstructions, “poj” became “po”. Only the first two syllables of the phrase were audible. |
| Natural phrase codec control | `build/tts/phrase-codec-check/comparison.json` | Original and 1.92 s reconstruction both sounded good to Jee. Historical rendering command unknown. |
| Qwen five connected phrases | Local `build/tts/connected-five/run_experiment.py` and `render.py` | Five training phrases, 18.212 seconds, 150 updates. Held-out phrase failed. Practiced “kuv noj mov” sounded close but unnatural. |
| Orpheus and F5 six-sentence comparison | Local `build/tts/hmong-model-audit/runtime/bin/python` with `render_orpheus.py` or `render_f5.py` in the audit directory | Audition page at `~/Downloads/hmong-tts-audit/index.html`. Orpheus did well on short sentences and degraded on longer ones. F5 failed this configuration. The index identifies each F5 cross-sentence reference and actual duration. |

At preflight, the checkout was clean on `feat/hmong-tts-bootstrap`, and both local and remote branch heads were `6b94dbb`. For `git merge-base --is-ancestor`, the checks `588bd57 6b94dbb` and `aa3caab7 6b94dbb` returned 0. The check `588bd57 5919bb0` returned 1. The remote main head was `5919bb0`. Historical Qwen pins were MLX 0.32.4, mlx-lm 0.31.1, mlx-tune 9690fe1, mlx-audio 91caeb9 and Qwen snapshot 1eccf1cb. The sealed 518-item syllable test split was unused in this comparison.
