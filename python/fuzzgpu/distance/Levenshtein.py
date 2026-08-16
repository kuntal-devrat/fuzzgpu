"""rapidfuzz.distance.Levenshtein-compatible module backed by fuzzgpu."""
from .. import fuzzgpu as _native
from ._common import cutoff_distance, normalized_distance as _normalized


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

def _alignment(s1, s2):
    """Return an optimal unit-cost alignment in O(len(s1) * len(s2))."""
    rows, cols = len(s1), len(s2)
    matrix = [list(range(cols + 1))] + [[i] + [0] * cols for i in range(1, rows + 1)]
    for i, a in enumerate(s1, 1):
        for j, b in enumerate(s2, 1):
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,
                matrix[i][j - 1] + 1,
                matrix[i - 1][j - 1] + (a != b),
            )
    steps = []
    i, j = rows, cols
    while i or j:
        if i and j and matrix[i][j] == matrix[i - 1][j - 1] + (s1[i - 1] != s2[j - 1]):
            tag = "equal" if s1[i - 1] == s2[j - 1] else "replace"
            steps.append((tag, i - 1, i, j - 1, j))
            i, j = i - 1, j - 1
        elif i and matrix[i][j] == matrix[i - 1][j] + 1:
            steps.append(("delete", i - 1, i, j, j))
            i -= 1
        else:
            steps.append(("insert", i, i, j - 1, j))
            j -= 1
    steps.reverse()
    return steps


def editops(s1, s2, *, processor=None, score_hint=None):
    """Return a list of (tag, src_pos, dest_pos) edit operations."""
    del score_hint
    if processor:
        s1, s2 = processor(s1), processor(s2)
    return [
        (tag, i1, j1)
        for tag, i1, _i2, j1, _j2 in _alignment(s1, s2)
        if tag != "equal"
    ]


def opcodes(s1, s2, *, processor=None, score_hint=None):
    """Return a list of (tag, i1, i2, j1, j2) opcode blocks."""
    del score_hint
    if processor:
        s1, s2 = processor(s1), processor(s2)
    result = []
    for tag, i1, i2, j1, j2 in _alignment(s1, s2):
        if (
            result
            and result[-1][0] == tag
            and result[-1][2] == i1
            and result[-1][4] == j1
        ):
            old_tag, old_i1, _old_i2, old_j1, _old_j2 = result[-1]
            result[-1] = (old_tag, old_i1, i2, old_j1, j2)
        else:
            result.append((tag, i1, i2, j1, j2))
    return result
