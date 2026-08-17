"""Type stubs for fuzzgpu.distance rapidfuzz-compatible submodule."""
from __future__ import annotations

from ._common import (
    Editop,
    Editops,
    Opcode,
    Opcodes,
    MatchingBlock,
    ScoreAlignment,
)
from . import (
    Levenshtein,
    DamerauLevenshtein,
    Hamming,
    OSA,
    Indel,
    Jaro,
    JaroWinkler,
    LCSseq,
    Prefix,
    Postfix,
)

__all__ = [
    "Levenshtein",
    "DamerauLevenshtein",
    "Hamming",
    "OSA",
    "Indel",
    "Jaro",
    "JaroWinkler",
    "LCSseq",
    "Prefix",
    "Postfix",
    "Editop",
    "Editops",
    "Opcode",
    "Opcodes",
    "MatchingBlock",
    "ScoreAlignment",
]