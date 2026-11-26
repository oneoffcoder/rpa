# Coverage ledger

**Status:** No accepted corpus exists yet, so this ledger is empty. It is
generated from a frozen manifest plus the language contract:

```text
rpa-tts corpus coverage \
  --manifest data/manifests/<dataset-version>/accepted.jsonl \
  --contract docs/tts/recovery/language-contract.yaml \
  --output-md docs/tts/recovery/coverage-ledger.md \
  --output-json data/runs/coverage/<dataset-version>.json
```

## Screening rule

For every attested prioritized onset-by-tone and vowel-by-tone cell (and every
declared critical contrast), require at least five distinct training examples
and a separate held-out context. These are project screening counts, not
completeness guarantees, and not a claim that 57 × 13 × 7 or marginal tone
counts form a complete language inventory.

The report lists every missing, underfilled and explicitly unattested cell. It
counts distinct utterances (not repeated tokens) and reports observed
onset×tone and vowel×tone totals for context. A passing screen means the
declared coverage was present; it says nothing about acoustic balance, model
quality or release readiness.

## Current data

| Dataset | Manifest SHA-256 | Unique accepted hours | Declared cells | Screen |
| --- | --- | ---: | ---: | --- |
| — (none) | — | 0 | 0 (contract draft) | Not started |

## Notes

- The repository's existing `source/_static/mp3/` recordings are 5,187 isolated
  syllables. They are a pronunciation/lexicon reference with unverified speaker
  identity and dialect, not connected-speech training hours; they cannot satisfy
  this ledger.
- The language contract is a draft with no attested contrast cells. The native
  lead enumerates them (see `language-contract.yaml`); until then the screen
  cannot pass, and the coverage command exits non-zero.
