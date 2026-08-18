"""rapidfuzz.distance.Levenshtein-compatible module backed by fuzzgpu."""
from .. import fuzzgpu as _native
from ._common import (
    Editop,
    Editops,
    cutoff_distance,
    normalized_distance as _normalized,
)


def distance(s1, s2, *, weights=(1, 1, 1), processor=None, score_cutoff=None, score_hint=None):
    del score_hint
    if processor:
        s1, s2 = processor(s1), processor(s2)
    if weights != (1, 1, 1):
        ins, delete, sub = weights
        row = [j * ins for j in range(len(s2) + 1)]
        for i, a in enumerate(s1, 1):
            prev, row[0] = row[0], i * delete
            for j, b in enumerate(s2, 1):
                old = row[j]
                row[j] = min(prev + (0 if a == b else sub), row[j] + delete, row[j - 1] + ins)
                prev = old
        value = row[-1]
    else:
        value = _native.levenshtein_distance(s1, s2)
    return cutoff_distance(value, score_cutoff)


def similarity(s1, s2, *, processor=None, score_cutoff=None, score_hint=None):
    del score_hint
    if processor:
        s1, s2 = processor(s1), processor(s2)
    # Compute maximum AFTER processing so len() reflects the processed strings.
    maximum = max(len(s1), len(s2))
    value = maximum - distance(s1, s2)
    return value if score_cutoff is None or value >= score_cutoff else 0


def normalized_distance(s1, s2, *, processor=None, score_cutoff=None, score_hint=None):
    del score_hint
    if processor:
        s1, s2 = processor(s1), processor(s2)
    maximum = max(len(s1), len(s2))
    value = _normalized(distance(s1, s2), maximum)
    return value if score_cutoff is None or value <= score_cutoff else 1.0


def normalized_similarity(s1, s2, *, processor=None, score_cutoff=None, score_hint=None):
    del score_hint
    if processor:
        s1, s2 = processor(s1), processor(s2)
    value = 1.0 - _normalized(distance(s1, s2), max(len(s1), len(s2)))
    return value if score_cutoff is None or value >= score_cutoff else 0.0


# ── Alignment helpers ─────────────────────────────────────────────────────────

def _common_affix(s1, s2):
    """Length of the common prefix and suffix of s1 and s2 (exact rapidfuzz
    `common_affix`: the suffix is measured on the post-prefix strings)."""
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
    """Myers bit-parallel distance matrix (exact rapidfuzz port)."""
    if not s1:
        return (len(s2), [], [])

    VP = (1 << len(s1)) - 1
    VN = 0
    currDist = len(s1)
    mask = 1 << (len(s1) - 1)

    block = {}
    block_get = block.get
    x = 1
    for ch1 in s1:
        block[ch1] = block_get(ch1, 0) | x
        x <<= 1

    matrix_VP = []
    matrix_VN = []
    for ch2 in s2:
        # Step 1: Computing D0
        PM_j = block_get(ch2, 0)
        X = PM_j
        D0 = (((X & VP) + VP) ^ VP) | X | VN
        # Step 2: Computing HP and HN
        HP = VN | ~(D0 | VP)
        HN = D0 & VP
        # Step 3: Computing the value D[m,j]
        currDist += (HP & mask) != 0
        currDist -= (HN & mask) != 0
        # Step 4: Computing Vp and VN
        HP = (HP << 1) | 1
        HN = HN << 1
        VP = HN | ~(D0 | HP)
        VN = HP & D0

        matrix_VP.append(VP)
        matrix_VN.append(VN)

    return (currDist, matrix_VP, matrix_VN)


def editops(s1, s2, *, processor=None, score_hint=None):
    """Return Editops describing how to turn s1 into s2."""
    del score_hint
    if processor:
        s1, s2 = processor(s1), processor(s2)
    prefix_len, suffix_len = _common_affix(s1, s2)
    s1_mid = s1[prefix_len : len(s1) - suffix_len]
    s2_mid = s2[prefix_len : len(s2) - suffix_len]
    dist, VP, VN = _matrix(s1_mid, s2_mid)

    editops = Editops([], 0, 0)
    editops._src_len = len(s1_mid) + prefix_len + suffix_len
    editops._dest_len = len(s2_mid) + prefix_len + suffix_len

    if dist == 0:
        return editops

    editop_list = [None] * dist
    col = len(s1_mid)
    row = len(s2_mid)
    while row != 0 and col != 0:
        # deletion
        if VP[row - 1] & (1 << (col - 1)):
            dist -= 1
            col -= 1
            editop_list[dist] = Editop("delete", col + prefix_len, row + prefix_len)
        else:
            row -= 1

            # insertion
            if row and (VN[row - 1] & (1 << (col - 1))):
                dist -= 1
                editop_list[dist] = Editop("insert", col + prefix_len, row + prefix_len)
            else:
                col -= 1

                # replace (Matches are not recorded)
                if s1_mid[col] != s2_mid[row]:
                    dist -= 1
                    editop_list[dist] = Editop("replace", col + prefix_len, row + prefix_len)

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
    return editops(s1, s2, processor=processor, score_hint=score_hint).as_opcodes()
