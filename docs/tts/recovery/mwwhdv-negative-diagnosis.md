# Negative listening result and controlled pipeline diagnosis

Private investigation record. Authorized 2000 UBS corpus; audio/weights private.
No release or native acceptance claim.

## Listener result (negative)

The owner heard all **eight** clips in the corrected blinded A/B (four prompts,
base and trained slots). **None sounded like Hmong**, for either condition. The
owner also judged the **LUK 19:22 codec reconstruction** and the matched-token
trained Luke output unrecognizable as Hmong, after confirming the original pair.
A stock English generation from the untouched checkpoint was independently
transcribed correctly on the owner's MBP. This is not a preference result and
does not establish a ranking.

## Source pairs are correct

The owner confirmed both source pairs. Independent recuts reproduce them exactly:

| Pair | Interval | Duration | Chapter SHA-256 | Segment SHA-256 | Recut match |
| --- | --- | ---: | --- | --- | --- |
| MRK 1:1 | 3.75–9.01 | 5.26 s | `c76fd063…` | `38c24a74…` | yes |
| LUK 19:22 | 183.09–199.17 | 16.08 s | `1750b832…` | `af31b3ce…` | yes |

MRK 1:1: 76 chars / 15 words. LUK 19:22: 274 chars / 58 words.

## Tokenization trace

- `text_ce_tokenizer` is at aggregated offset **96**, 384 tokens; text embedding
  has 3,359 rows; EOS id 3358.
- Training tokens use `model.tokenizer.encode(text, tokenizer_name="text_ce_tokenizer")`
  (no EOS). Inference uses `chunk_text_for_inference`, which **appends EOS** and
  chunks long text.
- MRK 1:1: training **77** tokens vs inference **78** (one EOS); EOS-augmented
  training loss changes only ~1e-4.
- LUK 19:22: training **275** tokens vs inference **3 chunks** (first 76 tokens)
  plus EOS. This is a real long-text train/inference mismatch, but it does not
  explain the short A/B prompts.
- The text route is active: `use_text_conditioning_encoder=True`,
  `decoder_ce`/`text_ce_tokenizer`, mask sums equal lengths, encoder output
  norm ≈78–136, and two different short texts produced different audio
  (`e6a9dea8…` vs `d19f9687…`), same seed/speaker/settings.
- Audio codec round-trip: 16.1146 s decoded vs 16.08 s input.
- Checkpoint load changes the text embedding (`dc7c6110…` → `a98ed503…`) and
  lowers LUK 19:22 teacher-forced loss from **11.4771** (base) to **10.6136**
  (trained). Under teacher forcing, correct vs random text gives nearly identical
  loss (10.6136 vs 10.6132), so that loss alone is not evidence text is unused.

## Controlled overfit diagnostic

Starting from the base model, training **one known pair (MAT 1:1)** for 200 steps
at 1e-4 drops loss **10.4703 → 0.0198** (held-out same-pair eval 0.0139). The
pipeline can memorize a text/audio pair; token wiring, codec conditioning and
checkpoint updates are functional.

## Codec audit (upstream of TTS)

The original LUK 19:22 segment is 22,050 Hz float32, 16.08 s, RMS 0.1664,
max 0.8094, shape `(1,T)`. The Magpie codec and a separately restored NVIDIA
NanoCodec (`fc00890b…`) were compared directly:

- Codec sample rate **22,050**; no resampling path; `converter=None`; no codebook
  offset conversion; source level did not clip.
- Magpie codec state has **713** keys; standalone has 1,249. All **713** common
  keys are exactly equal; the extra 536 are `discriminator.*` training-only keys.
- Both encoders produced **exactly equal codes** `(1, 8, 347)`; no padding/length
  discrepancy.
- Both decoders produced **16.1146 s**, RMS 0.16155, max ≈0.8183. Helper vs direct
  waveform correlation **0.99999996**, max abs difference 0.0009766 (PCM16
  quantisation), SNR **8.3869 dB**; source-vs-codec correlation 0.9259.
- The codec round-trip is therefore **not** corrupted by helper layout, dtype,
  level, resampling, padding, conversion or export. The direct NVIDIA round-trip
  and the helper round-trip are the same signal.

The codec is a **1.89 kbps** NanoCodec. Its reconstruction of the Hmong drama is
not recognizable as Hmong even though English generation through the same stack
was transcribed correctly. TTS training and inference both consume this codec's
tokens/decoder, so no amount of TTS fine-tuning can recover detail the codec
discards. This is the root cause identified by the controls, not the transcript
pairing.

## Transcript/EOS contract

Pinned `text_to_speech_dataset.py:490` appends `model.eos_id`; the prior
`make_verse_batch` did not. `encode_transcript` now returns **tokens + EOS, no
BOS**, and both training batches and the controlled one-chunk inference use it.
Regression against the pinned native tokenizer:

- `text_ce_tokenizer` offset **96**, 384 tokens; text embedding 3,359 rows;
  EOS **3358**.
- MRK 1:1: 77 text tokens + EOS = **78**, exact native IDs.
- LUK 19:22: 275 text tokens + EOS = **276**, exact native IDs.
- No vocabulary was added.

The prior checkpoints were trained without EOS and are not aligned with this
contract; their results are diagnostic only. The aligned 200-step LUK probe
(loss 11.5577 → 0.0301) generated **9.9846 s** for LUK and **22.059 s** for
MRK 1:1 at the same seed/settings; outputs are distinct (MFCC distance 16.54),
but remain codec-limited and are not evidence of usable speech.

## Aligned control pack

`data/runs/pair-trace/pack.json` and `listen.html` retain:

- LUK source and helper/direct codec round-trips;
- LUK/MRK 3ede outputs (no-EOS training, diagnostic only);
- LUK/MRK aligned-probe outputs;
- English base native-path output (`Hello, how are you today?`, 1.904 s).

## Upstream guidance

NVIDIA's “Adding a New Language” guide selects language-specific train/dev
tokenizer configuration, uses a constant Adam-family LR of 1e-5 with the
scheduler removed, and defaults to batch size 16. Those are a **comparison arm**,
not a claimed cause of the failure; the codec audit above bounds what any of them
can achieve.

## Duplicate ambiguity

Manifest audit: 7,935 rows, **7,934 distinct WAV hashes**. `REV 22:20` and
`REV 22:21` share hash `76a070d8…` and interval **298.23–327.183625 s** with
different transcripts. Both are quarantined pending combined-verse alignment.
- The 4,000-step checkpoint's cumulative selection includes `REV 22:21` from its
  first 2,000 rows; the continuation block 2,000–3,999 contains neither
  ambiguous row. The clean-split 2,000-step candidate also includes `REV 22:21`.
  Those results are labelled accordingly; A/B Mark prompts are unaffected.
- Quarantine split: 7,933 rows, **5,759 train / 714 dev / 1,460 test**, no
  ambiguous rows.

## Diagnosis

The immediate bottleneck is the **1.89 kbps NanoCodec**, not the source pairing,
the transcript tokenizer, the codec helper, checkpoint loading or the TTS text
route. A higher-fidelity codec/model (or a different architecture) is required
before further TTS training can produce recognizable Hmong. Training stays
paused; no candidate is accepted or release-ready.
