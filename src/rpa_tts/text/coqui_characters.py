"""Ordered character IDs for the upstream Coqui VITS tokenizer."""

from __future__ import annotations

import hashlib
import json
import string
from collections.abc import Mapping
from typing import Any

from TTS.tts.configs.shared_configs import BaseTTSConfig
from TTS.tts.utils.text.characters import BaseCharacters

from .normalize import PUNCTUATION

PAD_TOKEN = "<pad>"
EOS_TOKEN = "<eos>"
BOS_TOKEN = "<bos>"
BLANK_TOKEN = "<blank>"
SPECIALS = (PAD_TOKEN, EOS_TOKEN, BOS_TOKEN, BLANK_TOKEN)
LETTERS = string.ascii_lowercase + " "
MARKS = "".join(sorted(PUNCTUATION))
DEFAULT_VOCABULARY = (*SPECIALS, *LETTERS, *MARKS)
CHARACTERS_CLASS = "rpa_tts.text.coqui_characters.RpaCharacters"


def ordered_vocabulary(vocab_dict: list[str] | None) -> tuple[str, ...]:
    """Validate Coqui's ordered vocabulary. Each list index is the token ID."""
    if not vocab_dict:
        raise ValueError("RPA config has no saved character map")
    if not isinstance(vocab_dict, list) or not all(isinstance(token, str) for token in vocab_dict):
        raise ValueError("RPA character map must be an ordered list of strings")
    tokens = tuple(vocab_dict)
    if len(tokens) != len(DEFAULT_VOCABULARY) or set(tokens) != set(DEFAULT_VOCABULARY):
        raise ValueError("RPA character map has duplicate, missing, or unexpected tokens")
    if tokens[: len(SPECIALS)] != SPECIALS:
        raise ValueError("RPA character map moved a reserved token")
    return tokens


def vocabulary_sha256(tokens: tuple[str, ...]) -> str:
    encoded = json.dumps(tokens, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tokenizer_contract(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Return the model-facing tokenizer settings that must survive checkpoint reload."""
    characters = settings.get("characters") or {}
    if characters.get("characters_class") != CHARACTERS_CLASS:
        raise ValueError("Checkpoint or config does not select the RPA character class")
    tokens = ordered_vocabulary(characters.get("vocab_dict"))
    if tuple(characters.get(key) for key in ("pad", "eos", "bos", "blank")) != SPECIALS:
        raise ValueError("RPA character config changed the reserved token roles")
    if settings.get("use_phonemes") or settings.get("text_cleaner") is not None:
        raise ValueError("RPA VITS requires the project grapheme frontend")
    if (settings.get("model_args") or {}).get("num_chars") != len(tokens):
        raise ValueError("RPA character map does not match the configured model dimension")
    flags = {key: settings.get(key) for key in ("add_blank", "enable_eos_bos_chars")}
    if any(type(flag) is not bool for flag in flags.values()):
        raise ValueError("RPA tokenizer blank and boundary flags must be explicit booleans")
    return {"tokens": tokens, **flags}


def verify_checkpoint_tokenizer(
    saved: Mapping[str, Any],
    config: BaseTTSConfig,
    *,
    verified_legacy_settings: Mapping[str, Any] | None = None,
) -> None:
    """Reject incompatible embeddings before installing checkpoint model or optimizer state."""
    launch = tokenizer_contract(config.to_dict())
    raw = saved.get("config") or {}
    if verified_legacy_settings is not None and not (raw.get("characters") or {}).get("vocab_dict"):
        # The caller has already authenticated the original config/checkpoint pair.
        raw = {**raw, "characters": verified_legacy_settings["characters"]}
    if tokenizer_contract(raw) != launch:
        raise ValueError("Checkpoint tokenizer differs from the requested tokenizer")
    embedding = (saved.get("model") or {}).get("text_encoder.emb.weight")
    if embedding is None or embedding.ndim != 2 or embedding.shape[0] != len(launch["tokens"]):
        raise ValueError("Checkpoint text embedding does not match the character map")


class RpaCharacters(BaseCharacters):
    """Keep the explicit character-to-embedding mapping used by the checkpoint."""

    def __init__(self, tokens: tuple[str, ...] = DEFAULT_VOCABULARY) -> None:
        super().__init__(
            characters=LETTERS,
            punctuations=MARKS,
            pad=PAD_TOKEN,
            eos=EOS_TOKEN,
            bos=BOS_TOKEN,
            blank=BLANK_TOKEN,
            is_unique=False,
            is_sorted=False,
        )
        self.vocab = list(ordered_vocabulary(list(tokens)))

    def to_config(self):
        config = super().to_config()
        config.characters_class = CHARACTERS_CLASS
        config.vocab_dict = self.vocab.copy()
        return config

    @staticmethod
    def init_from_config(config: BaseTTSConfig) -> tuple[RpaCharacters, BaseTTSConfig]:
        tokens = tokenizer_contract(config.to_dict())["tokens"]
        return RpaCharacters(tokens), config


__all__ = ["BOS_TOKEN", "EOS_TOKEN", "PAD_TOKEN", "RpaCharacters"]
