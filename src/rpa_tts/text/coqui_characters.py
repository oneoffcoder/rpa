"""Ordered character IDs for the upstream Coqui VITS tokenizer."""

from __future__ import annotations

import hashlib
import json
import string

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
    tokens = tuple(vocab_dict)
    if len(tokens) != len(DEFAULT_VOCABULARY) or set(tokens) != set(DEFAULT_VOCABULARY):
        raise ValueError("RPA character map has duplicate, missing, or unexpected tokens")
    if tokens[: len(SPECIALS)] != SPECIALS:
        raise ValueError("RPA character map moved a reserved token")
    return tokens


def vocabulary_sha256(tokens: tuple[str, ...]) -> str:
    encoded = json.dumps(tokens, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
        self.vocab = list(tokens)

    def to_config(self):
        config = super().to_config()
        config.characters_class = CHARACTERS_CLASS
        config.vocab_dict = self.vocab.copy()
        return config

    @staticmethod
    def init_from_config(config: BaseTTSConfig) -> tuple[RpaCharacters, BaseTTSConfig]:
        tokens = ordered_vocabulary(config.characters.vocab_dict)
        if config.model_args.num_chars != len(tokens):
            raise ValueError("RPA character map does not match the configured model dimension")
        return RpaCharacters(tokens), config


__all__ = ["BOS_TOKEN", "EOS_TOKEN", "PAD_TOKEN", "RpaCharacters"]
