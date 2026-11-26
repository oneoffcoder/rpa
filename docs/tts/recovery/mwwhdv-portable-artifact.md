# MWWHDV private portable candidate artifact

Private, authorized 2000 UBS corpus; audio/weights private. Not release, not an
accepted package, and not native-approved.

## Artifact

- Source candidate: clean chapter-split checkpoint `a16f4cb3…` (5,761 train rows;
  dev mean 11.9123 → 11.0115). The candidate includes one ambiguous row
  (`REV 22:21`) from the first 2,000 training rows; quarantined going forward.
- Exported with the artifact-path workaround required by NeMo's Windows `nemo:`
  copy defect:
  `data/runs/f04-split-train/model.nemo`
- Size **1,459,773,440 bytes**; SHA-256 `87d16174…`.
- Reloaded through the **pinned NeMo commit** path; 372,919,838 parameters, all
  on `cuda:0`; exact parameter fingerprint equals the source model.
- 1.207 s private sample written (`kuv hlub koj`), unlistened.

## What this does and does not prove

Proves the trained candidate can be saved as a portable `.nemo` and restored
bit-exactly with the pinned stack. It is not a release package: no F-07 acceptance,
no second-worker package inspection, no distribution rights, no native listening.
