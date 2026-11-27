# Coqui VITS character map recovery

The runtime character order for `best_model_35461.pth` was inferred from paired White Hmong audio using frozen model weights. Two independent fits converged on the same 27 shuffled IDs. The original checkpoint, run configuration and TensorBoard records examined contain no historical ordered map. The project owner verified two original prompts in the controlled comparison, then accepted all 20 additional prompts in a separate native listening evaluation.

For “peb mus tsev.” and “kuv yog hmoob.”, the A/B/A comparison used one loaded model, the same checkpoint, input, CUDA device, output encoder and Torch seed 42. The only intervention was assigning the ordered vocabulary. For both phrases, the original fresh map and its reversion reproduced the previously emailed unintelligible WAV byte for byte. The owner found both recovered-map WAVs intelligible and both failed controls unintelligible. The trainer previews also sounded decent to the owner. This establishes the effect of the map on these two phrases. All 20 additional prompts passed the owner’s native listening evaluation. This evaluation is diagnostic evidence and does not confer F-07 qualification. [Investigation and evidence](https://docs.google.com/document/d/1IV09sUpHPPF0qBJnzJP8CCwSRutSAA757viA6oAswbY/edit)

The recovery bundle on Radiance is `C:/Users/jee/git/rpa/data/artifacts/rpa-vits-step35461-listener-verified`. It contains a verified copy of the original checkpoint (SHA-256 `2919b47c677d103b8487462024efb468289335ae243d8de3fe18a4b1ed5e593e`), the original and recovered configs, the A/B/A provenance, and a checksum manifest. The recovered config stores the ordered token list in `characters.vocab_dict`. Each list index is the token ID. The manifest binds this order to the checkpoint using canonical digests.

The repository’s recipe writes an explicit map for new runs. Resume verifies the full tokenizer contract and embedding row count of both checkpoints selected by Trainer against the launch config. Passing a checkpoint file to the render command uses its colocated config and verifies the same contract. A legacy checkpoint without an embedded map needs explicit recovery evidence for inference. The recovered checkpoint remains a legacy checkpoint with its original model bytes.

With the pinned Coqui environment and `src` on `PYTHONPATH`, reproduce the frozen inference path:

```text
python scripts/vits_character_map.py render <bundle> <one-sentence-per-line.txt> <fresh-output-directory> --cuda
```

The loader verifies the checkpoint, map, and configuration before constructing the synthesizer. The output manifest records each WAV's sentence, model text, duration and hash.

A second copy of the recovery bundle was transferred to the Mac under `~/Downloads/RPA-map-recovery/verified-checkpoint`. The checkpoint, both configs, and A/B/A receipt matched the Radiance manifest.
