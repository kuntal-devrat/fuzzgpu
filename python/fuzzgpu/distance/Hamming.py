"""rapidfuzz.distance.Hamming-compatible module."""
from ._common import cutoff_distance, normalized_distance as _normalized


def distance(s1, s2, *, pad=True, processor=None, score_cutoff=None, score_hint=None):
    del score_hint
    if processor:
        s1, s2 = processor(s1), processor(s2)
    if not pad and len(s1) != len(s2):
        raise ValueError("Sequences are not the same length.")
    value = sum(a != b for a, b in zip(s1, s2)) + abs(len(s1) - len(s2))
    return cutoff_distance(value, score_cutoff)


def similarity(s1, s2, *, pad=True, processor=None, score_cutoff=None, score_hint=None):
    del score_hint
    if processor:
        s1, s2 = processor(s1), processor(s2)
    maximum = max(len(s1), len(s2))
    value = maximum - distance(s1, s2, pad=pad)
    return value if score_cutoff is None or value >= score_cutoff else 0


def normalized_distance(s1, s2, *, pad=True, processor=None, score_cutoff=None, score_hint=None):
    del score_hint
    if processor:
        s1, s2 = processor(s1), processor(s2)
    value = _normalized(distance(s1, s2, pad=pad), max(len(s1), len(s2)))
    return value if score_cutoff is None or value <= score_cutoff else 1.0


def normalized_similarity(s1, s2, *, pad=True, processor=None, score_cutoff=None, score_hint=None):
    del score_hint
    if processor:
        s1, s2 = processor(s1), processor(s2)
    value = 1.0 - _normalized(distance(s1, s2, pad=pad), max(len(s1), len(s2)))
    return value if score_cutoff is None or value >= score_cutoff else 0.0
