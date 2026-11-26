"""RPA character class for the upstream Coqui VITS tokenizer."""

from __future__ import annotations

import string

from TTS.tts.configs.shared_configs import BaseTTSConfig
from TTS.tts.utils.text.characters import BaseCharacters

from .normalize import PUNCTUATION

PAD_TOKEN = "<pad>"
EOS_TOKEN = "<eos>"
BOS_TOKEN = "<bos>"


class RpaCharacters(BaseCharacters):
    """Coqui character vocabulary for ``rpa-chars-1`` (36 tokens, no UNK).

    ``rpa-norm-1`` rejects characters outside this set before tokenization, so
    an UNK token is unnecessary. The ordering follows Coqui's
    ``[PAD, EOS, BOS, characters, punctuations]`` contract rather than the
    project's internal 37-token ID order; symbols and normalization are shared.
    """

    def __init__(self) -> None:
        super().__init__(
            characters=string.ascii_lowercase + " ",
            punctuations="".join(sorted(PUNCTUATION)),
            pad=PAD_TOKEN,
            eos=EOS_TOKEN,
            bos=BOS_TOKEN,
            blank=None,
            is_unique=True,
            is_sorted=False,
        )

    @staticmethod
    def init_from_config(config: BaseTTSConfig) -> tuple[RpaCharacters, BaseTTSConfig]:
        return RpaCharacters(), config


__all__ = ["BOS_TOKEN", "EOS_TOKEN", "PAD_TOKEN", "RpaCharacters"]
