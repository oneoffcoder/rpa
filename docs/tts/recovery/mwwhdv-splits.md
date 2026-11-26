# MWWHDV private chapter-grouped splits

Private development artifact; authorized 2000 UBS corpus; audio/weights private.
Not an accepted or frozen dataset and not the F-07 locked qualification test.

## Method

- `assign_splits` from `rpa_tts.data.split`, seed 3407, dev 0.1 / test 0.2.
- Group = chapter (`MAT 1`, etc.), bucket = book, so no chapter spans splits and
  each book contributes train/test (and dev where it has multiple chapters).
- 260 chapter groups / 7,935 rows:
  - train: 184 groups, **5,761 rows / 972.61 min**
  - dev: 28 groups, **714 rows / 125.31 min**
  - test: 48 groups, **1,460 rows / 253.46 min**
- Split manifest SHA-256
  `5dd757eef85b2076a7f23b2077cfbe9bb389e6f7cc68ca3f95de29047c103f34`.
- Warnings: the single-chapter books 2JN, 3JN, JUD and PHM have no dev/test
  groups; they remain training-only. No group-level leakage.

## Boundaries

This is a reproducible private split for development, not a native-approved or
frozen corpus, and not the locked F-07 suite. Audio/text/weights remain private.
