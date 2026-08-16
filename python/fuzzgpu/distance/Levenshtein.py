from .. import fuzzgpu as _native
from ._common import cutoff_distance, normalized_distance as _normalized

def distance(s1, s2, *, weights=(1, 1, 1), processor=None, score_cutoff=None, score_hint=None):
    del score_hint
    if processor: s1, s2 = processor(s1), processor(s2)
    if weights != (1, 1, 1):
        # Weighted DP is deliberately exact, including Unicode code points.
        ins, delete, sub = weights; row = [j * ins for j in range(len(s2) + 1)]
        for i, a in enumerate(s1, 1):
            prev, row[0] = row[0], i * delete
            for j, b in enumerate(s2, 1):
                old = row[j]; row[j] = min(prev + (0 if a == b else sub), row[j] + delete, row[j - 1] + ins); prev = old
        value = row[-1]
    else:
        value = _native.levenshtein_distance(s1, s2)
    return cutoff_distance(value, score_cutoff)

def similarity(s1, s2, *, processor=None, score_cutoff=None, score_hint=None):
    maximum = max(len(s1), len(s2)); value = maximum - distance(s1, s2, processor=processor, score_hint=score_hint)
    return value if score_cutoff is None or value >= score_cutoff else 0

def normalized_distance(s1, s2, *, processor=None, score_cutoff=None, score_hint=None):
    value = _normalized(distance(s1, s2, processor=processor, score_hint=score_hint), max(len(s1), len(s2)))
    return value if score_cutoff is None or value <= score_cutoff else 1.0

def normalized_similarity(s1, s2, *, processor=None, score_cutoff=None, score_hint=None):
    value = 1.0 - normalized_distance(s1, s2, processor=processor, score_hint=score_hint)
    return value if score_cutoff is None or value >= score_cutoff else 0.0

def editops(s1, s2, *, processor=None, score_hint=None):
    if processor: s1, s2 = processor(s1), processor(s2)
    i = j = 0; result = []
    while i < len(s1) or j < len(s2):
        if i < len(s1) and j < len(s2) and s1[i] == s2[j]: i += 1; j += 1
        elif j < len(s2) and (i == len(s1) or distance(s1[i + 1:], s2[j:]) >= distance(s1[i:], s2[j + 1:])): result.append(("insert", i, j)); j += 1
        elif i < len(s1) and (j == len(s2) or distance(s1[i + 1:], s2[j:]) <= distance(s1[i:], s2[j + 1:])): result.append(("delete", i, j)); i += 1
        else: result.append(("replace", i, j)); i += 1; j += 1
    return result

def opcodes(s1, s2, *, processor=None, score_hint=None):
    # A simple opcode form compatible with the edit operation coordinates.
    return [(tag, i, i + (tag != "insert"), j, j + (tag != "delete")) for tag, i, j in editops(s1, s2, processor=processor, score_hint=score_hint)]
