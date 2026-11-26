# VITS five-epoch corpus run (private)

Measured training result only; no native acceptance or release claim.

## Terminal state

- PID `39012`; output
  `data/runs/vits-corpus-full-validated-20261126T1344Z`.
- Started from `data/runs/vits-compare/full-acoustic/checkpoint.pt`.
- Terminal status: **step 7,330, epoch 5, position 0**, 28,795 rows dispatched,
  **5,759 unique train rows**; **stop_reason `epoch_limit`**.
- Selected epoch: **5**, validation score **28.0979** (lower is better).
- Selected checkpoint: `checkpoint-epoch5.pt`, SHA-256
  `757926d0…`; identical bytes/hash to the terminal `checkpoint.pt`.
  It contains 111 discriminator and 838 generator AdamW state entries, the
  5,759-row order and cursor.

## Boundary validation (32 fixed dev verses)

| Epoch | Score | Duration | KL | Correct-vs-wrong (MFCC) | Mean corr | Duration-equal |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 28.7395 | 2.1640 | 2.4713 | 7/12 | −0.002646 | 0 |
| 2 | 28.9923 | 2.0988 | 2.1184 | 7/12 | 0.008237 | 0 |
| 3 | 28.8145 | 2.0771 | 2.2305 | 7/12 | 0.006570 | 0 |
| 4 | 28.7711 | 2.0669 | 2.3314 | 8/12 | 0.007522 | 0 |
| 5 | **28.0979** | **2.0497** | 2.1931 | 7/12 | 0.004177 | 0 |

The model is text-sensitive on the paired-audio diagnostic (mean correct/wrong
correlation near zero, no equal-duration pairs), but the correct transcript wins
only 7–8 of 12. This is **not** an intelligibility or native-listening result.
Earlier owner listening on step 1500 was buzzing and failed, both unspaced and
spaced; no new listening claim is made here.

## Boundaries retained

- Five epoch checkpoints `checkpoint-epoch1.pt` … `checkpoint-epoch5.pt`.
- Boundary samples and 12-pair records under `boundaries/epoch1…5/`.
- Exact spaced `kuv yog hmoob` and unspaced samples are retained per boundary.
- The user-requested step-1500 sample remains preserved separately under
  `data/runs/vits-corpus-pass/owner-samples/`.

Native intelligibility, F-07 qualification and release remain open.
