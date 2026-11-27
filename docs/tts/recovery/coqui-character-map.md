# Coqui VITS character map compatibility

Character IDs index the text embedding. A tokenizer must persist the exact ordered
vocabulary with its configuration. In Coqui, `characters.vocab_dict` is an ordered
list, with the list index serving as the token ID.

The RPA recipe writes an explicit map for new runs. Resume checks both
Trainer-selected checkpoints against the current config, including token order,
reserved roles, blank and boundary flags, and the embedding's row count. Inference
applies the same compatibility checks before loading model weights.

An older checkpoint without an explicit vocabulary requires a verified recovery
bundle. Preserve the original model, configuration, and recovery receipt together.

With the pinned Coqui environment and `src` on `PYTHONPATH`, use the checked
inference path:

```text
python scripts/vits_character_map.py render <checkpoint-or-bundle> <sentences.txt> <output-dir>
```

A regular checkpoint must have its matching `config.json` alongside it. The
render manifest records the checkpoint and vocabulary digests and the hash of
each WAV. Formal listening qualification uses the staged pack and the
[qualification workflow](listening-instrument.md).
