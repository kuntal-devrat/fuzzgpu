"""rapidfuzz.distance.Hamming-compatible module."""
from ._common import (
    Editop,
    Editops,
    cutoff_distance,
    normalized_distance as _normalized,
)


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


def editops(s1, s2, *, pad=True, processor=None, score_hint=None):
    """Return Editops describing how to turn s1 into s2."""
    del score_hint
    if processor:
        s1, s2 = processor(s1), processor(s2)
    if not pad and len(s1) != len(s2):
        raise ValueError("Sequences are not the same length.")

    ops_list = []
    min_len = min(len(s1), len(s2))
    for i in range(min_len):
        if s1[i] != s2[i]:
            ops_list.append(Editop("replace", i, i))

    for i in range(min_len, len(s1)):
        ops_list.append(Editop("delete", i, len(s2)))

    for i in range(min_len, len(s2)):
        ops_list.append(Editop("insert", len(s1), i))

    # sidestep input validation
    ops = Editops.__new__(Editops)
    ops._src_len = len(s1)
    ops._dest_len = len(s2)
    ops._editops = ops_list
    return ops


def opcodes(s1, s2, *, pad=True, processor=None, score_hint=None):
    """Return Opcodes describing how to turn s1 into s2."""
    del score_hint
    return editops(s1, s2, pad=pad, processor=processor).as_opcodes()
