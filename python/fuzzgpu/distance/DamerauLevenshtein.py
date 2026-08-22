"""rapidfuzz.distance.DamerauLevenshtein-compatible module.

Implements the **unrestricted** Damerau-Levenshtein distance (Lowrance &
Wagner 1975), which allows non-adjacent transpositions — the same semantics
as rapidfuzz.distance.DamerauLevenshtein (verified: distance("ca", "abc")
== 2 in both). For the restricted variant use fuzzgpu.distance.OSA, which
matches rapidfuzz.distance.OSA (distance("ca", "abc") == 3 in both).
"""
from .. import fuzzgpu as _native
from ._common import Editop, Editops, cutoff_distance, normalized_distance as _normalized


def distance(s1, s2, *, processor=None, score_cutoff=None, score_hint=None):
    del score_hint
    if processor:
        s1, s2 = processor(s1), processor(s2)
    return cutoff_distance(_native.damerau_levenshtein_distance(s1, s2), score_cutoff)


def similarity(s1, s2, *, processor=None, score_cutoff=None, score_hint=None):
    del score_hint
    if processor:
        s1, s2 = processor(s1), processor(s2)
    maximum = max(len(s1), len(s2))
    value = maximum - distance(s1, s2)
    return value if score_cutoff is None or value >= score_cutoff else 0


def normalized_distance(s1, s2, *, processor=None, score_cutoff=None, score_hint=None):
    del score_hint
    if processor:
        s1, s2 = processor(s1), processor(s2)
    value = _normalized(distance(s1, s2), max(len(s1), len(s2)))
    return value if score_cutoff is None or value <= score_cutoff else 1.0


def normalized_similarity(s1, s2, *, processor=None, score_cutoff=None, score_hint=None):
    del score_hint
    if processor:
        s1, s2 = processor(s1), processor(s2)
    value = 1.0 - _normalized(distance(s1, s2), max(len(s1), len(s2)))
    return value if score_cutoff is None or value >= score_cutoff else 0.0


# ── Alignment helpers ─────────────────────────────────────────────────────────

def _damerau_matrix(s1, s2):
    """Full Lowrance-Wagner DP matrix for traceback.

    Returns the filled (m+2) × (n+2) cost matrix as a list of lists so the
    traceback can reconstruct the edit path.  The layout matches the standard
    Lowrance-Wagner (1975) formulation with the da[] last-seen-position table.
    """
    m, n = len(s1), len(s2)
    INF = m + n + 1

    # d[i][j]: cost to turn s1[:i] into s2[:j]
    d = [[0] * (n + 2) for _ in range(m + 2)]
    d[0][0] = INF
    for i in range(m + 1):
        d[i + 1][0] = INF
        d[i + 1][1] = i
    for j in range(n + 1):
        d[0][j + 1] = INF
        d[1][j + 1] = j

    # da[c] = last row where character c was seen in s1 (1-indexed)
    da = {}

    for i in range(1, m + 1):
        db = 0  # last column where s1[i-1] was seen in s2 (1-indexed)
        for j in range(1, n + 1):
            i1 = da.get(s2[j - 1], 0)
            j1 = db
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            if cost == 0:
                db = j
            d[i + 1][j + 1] = min(
                d[i][j] + cost,           # substitute / match
                d[i + 1][j] + 1,          # insert
                d[i][j + 1] + 1,          # delete
                d[i1][j1] + (i - i1 - 1) + 1 + (j - j1 - 1),  # transpose
            )
        da[s1[i - 1]] = i

    return d


def editops(s1, s2, *, processor=None, score_hint=None):
    """Return Editops describing how to turn s1 into s2 (Lowrance-Wagner).

    The returned edit sequence uses only insert/delete/replace operations —
    transpositions are decomposed into the minimum-cost sequence of those
    three primitive operations, which is how rapidfuzz represents them too.
    """
    del score_hint
    if processor:
        s1, s2 = processor(s1), processor(s2)

    m, n = len(s1), len(s2)
    result = Editops([], 0, 0)
    result._src_len = m
    result._dest_len = n

    if s1 == s2:
        return result

    d = _damerau_matrix(s1, s2)

    # Traceback from d[m+1][n+1] (1-indexed DP, offset by 1 for the INF border)
    ops = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            if d[i + 1][j + 1] == d[i][j] + cost:
                if cost:
                    ops.append(Editop("replace", i - 1, j - 1))
                i -= 1
                j -= 1
                continue
        if j > 0 and d[i + 1][j + 1] == d[i + 1][j] + 1:
            ops.append(Editop("insert", i, j - 1))
            j -= 1
        elif i > 0 and d[i + 1][j + 1] == d[i][j + 1] + 1:
            ops.append(Editop("delete", i - 1, j))
            i -= 1
        else:
            # Transposition or boundary — fall back to delete+insert decomposition
            if i > 0:
                ops.append(Editop("delete", i - 1, j))
                i -= 1
            else:
                ops.append(Editop("insert", i, j - 1))
                j -= 1

    ops.reverse()
    result._editops = ops
    return result


def opcodes(s1, s2, *, processor=None, score_hint=None):
    """Return Opcodes describing how to turn s1 into s2."""
    return editops(s1, s2, processor=processor, score_hint=score_hint).as_opcodes()
