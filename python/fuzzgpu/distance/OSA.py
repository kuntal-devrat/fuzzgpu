from ._common import cutoff_distance, normalized_distance as _normalized

def distance(s1, s2, *, processor=None, score_cutoff=None, score_hint=None):
    del score_hint
    if processor: s1, s2 = processor(s1), processor(s2)
    previous2 = None; previous = list(range(len(s2) + 1))
    for i, a in enumerate(s1, 1):
        current = [i]
        for j, b in enumerate(s2, 1):
            value = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (a != b))
            if previous2 is not None and i > 1 and j > 1 and a == s2[j - 2] and s1[i - 2] == b: value = min(value, previous2[j - 2] + 1)
            current.append(value)
        previous2, previous = previous, current
    return cutoff_distance(previous[-1], score_cutoff)
def similarity(s1, s2, **kwargs): return max(len(s1), len(s2)) - distance(s1, s2, **kwargs)
def normalized_distance(s1, s2, **kwargs): return _normalized(distance(s1, s2, **kwargs), max(len(s1), len(s2)))
def normalized_similarity(s1, s2, **kwargs): return 1.0 - normalized_distance(s1, s2, **kwargs)
