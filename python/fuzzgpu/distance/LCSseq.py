"""rapidfuzz.distance.LCSseq-compatible module (longest common subsequence)."""

from ._common import Editop, Editops, cutoff_distance, normalized_distance as _normalized


def _common_affix(s1, s2):
    prefix = 0
    for ch1, ch2 in zip(s1, s2):
        if ch1 != ch2:
            break
        prefix += 1
    suffix = 0
    for ch1, ch2 in zip(reversed(s1[prefix:]), reversed(s2[prefix:])):
        if ch1 != ch2:
            break
        suffix += 1
    return prefix, suffix


def _matrix(s1, s2):
    """Myers bit-parallel LCS matrix (exact rapidfuzz port)."""
    if not s1:
        return (0, [])

    S = (1 << len(s1)) - 1
    block = {}
    block_get = block.get
    x = 1
    for ch1 in s1:
        block[ch1] = block_get(ch1, 0) | x
        x <<= 1

    matrix = []
    for ch2 in s2:
        Matches = block_get(ch2, 0)
        u = S & Matches
        S = (S + u) | (S - u)
        matrix.append(S)

    # popcount(~S); breaks for len(s1) == 0
    sim = bin(S)[-len(s1) :].count("0")
    return (sim, matrix)


def similarity(s1, s2, *, processor=None, score_cutoff=None, score_hint=None):
    """Length of the longest common subsequence."""
    del score_hint
    if processor:
        s1, s2 = processor(s1), processor(s2)
    if not s1:
        return 0
    sim, _ = _matrix(s1, s2)
    return sim if score_cutoff is None or sim >= score_cutoff else 0


def distance(s1, s2, *, processor=None, score_cutoff=None, score_hint=None):
    """LCS distance: max(len1, len2) - similarity."""
    del score_hint
    if processor:
        s1, s2 = processor(s1), processor(s2)
    maximum = max(len(s1), len(s2))
    value = maximum - similarity(s1, s2)
    return cutoff_distance(value, score_cutoff)


def normalized_distance(s1, s2, *, processor=None, score_cutoff=None, score_hint=None):
    del score_hint
    if processor:
        s1, s2 = processor(s1), processor(s2)
    if not s1 and not s2:
        return 0
    maximum = max(len(s1), len(s2))
    value = _normalized(distance(s1, s2), maximum)
    return value if score_cutoff is None or value <= score_cutoff else 1.0


def normalized_similarity(s1, s2, *, processor=None, score_cutoff=None, score_hint=None):
    del score_hint
    if processor:
        s1, s2 = processor(s1), processor(s2)
    value = 1.0 - normalized_distance(s1, s2)
    return value if score_cutoff is None or value >= score_cutoff else 0.0


def editops(s1, s2, *, processor=None, score_hint=None):
    """Return Editops describing how to turn s1 into s2."""
    del score_hint
    if processor:
        s1, s2 = processor(s1), processor(s2)
    prefix_len, suffix_len = _common_affix(s1, s2)
    s1_mid = s1[prefix_len : len(s1) - suffix_len]
    s2_mid = s2[prefix_len : len(s2) - suffix_len]
    sim, matrix = _matrix(s1_mid, s2_mid)

    editops = Editops([], 0, 0)
    editops._src_len = len(s1_mid) + prefix_len + suffix_len
    editops._dest_len = len(s2_mid) + prefix_len + suffix_len

    dist = len(s1_mid) + len(s2_mid) - 2 * sim
    if dist == 0:
        return editops

    editop_list = [None] * dist
    col = len(s1_mid)
    row = len(s2_mid)
    while row != 0 and col != 0:
        # deletion
        if matrix[row - 1] & (1 << (col - 1)):
            dist -= 1
            col -= 1
            editop_list[dist] = Editop("delete", col + prefix_len, row + prefix_len)
        else:
            row -= 1

            # insertion
            if row and not (matrix[row - 1] & (1 << (col - 1))):
                dist -= 1
                editop_list[dist] = Editop("insert", col + prefix_len, row + prefix_len)
            # match
            else:
                col -= 1

    while col != 0:
        dist -= 1
        col -= 1
        editop_list[dist] = Editop("delete", col + prefix_len, row + prefix_len)

    while row != 0:
        dist -= 1
        row -= 1
        editop_list[dist] = Editop("insert", col + prefix_len, row + prefix_len)

    editops._editops = editop_list
    return editops


def opcodes(s1, s2, *, processor=None, score_hint=None):
    """Return Opcodes describing how to turn s1 into s2."""
    del score_hint
    return editops(s1, s2, processor=processor).as_opcodes()