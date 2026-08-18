"""rapidfuzz.distance.Indel-compatible module."""
from ._common import cutoff_distance, normalized_distance as _normalized
from . import LCSseq


def distance(s1, s2, *, processor=None, score_cutoff=None, score_hint=None):
    del score_hint
    if processor:
        s1, s2 = processor(s1), processor(s2)
    row = [0] * (len(s2) + 1)
    for a in s1:
        diagonal = 0
        for j, b in enumerate(s2, 1):
            old = row[j]
            row[j] = diagonal + 1 if a == b else max(row[j], row[j - 1])
            diagonal = old
    value = len(s1) + len(s2) - 2 * row[-1]
    return cutoff_distance(value, score_cutoff)


def similarity(s1, s2, *, processor=None, score_cutoff=None, score_hint=None):
    del score_hint
    if processor:
        s1, s2 = processor(s1), processor(s2)
    maximum = len(s1) + len(s2)
    value = maximum - distance(s1, s2)
    return value if score_cutoff is None or value >= score_cutoff else 0


def normalized_distance(s1, s2, *, processor=None, score_cutoff=None, score_hint=None):
    del score_hint
    if processor:
        s1, s2 = processor(s1), processor(s2)
    value = _normalized(distance(s1, s2), len(s1) + len(s2))
    return value if score_cutoff is None or value <= score_cutoff else 1.0


def normalized_similarity(s1, s2, *, processor=None, score_cutoff=None, score_hint=None):
    del score_hint
    if processor:
        s1, s2 = processor(s1), processor(s2)
    value = 1.0 - _normalized(distance(s1, s2), len(s1) + len(s2))
    return value if score_cutoff is None or value >= score_cutoff else 0.0


def editops(s1, s2, *, processor=None, score_hint=None):
    """Return Editops describing how to turn s1 into s2."""
    del score_hint
    return LCSseq.editops(s1, s2, processor=processor)


def opcodes(s1, s2, *, processor=None, score_hint=None):
    """Return Opcodes describing how to turn s1 into s2."""
    del score_hint
    return LCSseq.opcodes(s1, s2, processor=processor)
