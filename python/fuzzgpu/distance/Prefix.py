"""rapidfuzz.distance.Prefix-compatible module."""

from ._common import normalized_distance as _normalized


def similarity(s1, s2, *, processor=None, score_cutoff=None, score_hint=None):
    del score_hint
    if processor:
        s1, s2 = processor(s1), processor(s2)
    sim = 0
    for ch1, ch2 in zip(s1, s2):
        if ch1 != ch2:
            break
        sim += 1
    return sim if score_cutoff is None or sim >= score_cutoff else 0


def distance(s1, s2, *, processor=None, score_cutoff=None, score_hint=None):
    del score_hint
    if processor:
        s1, s2 = processor(s1), processor(s2)
    maximum = max(len(s1), len(s2))
    value = maximum - similarity(s1, s2)
    from ._common import cutoff_distance

    return cutoff_distance(value, score_cutoff)


def normalized_distance(s1, s2, *, processor=None, score_cutoff=None, score_hint=None):
    del score_hint
    if processor:
        s1, s2 = processor(s1), processor(s2)
    maximum = max(len(s1), len(s2))
    value = distance(s1, s2) / maximum if maximum else 0.0
    return value if score_cutoff is None or value <= score_cutoff else 1.0


def normalized_similarity(s1, s2, *, processor=None, score_cutoff=None, score_hint=None):
    del score_hint
    value = 1.0 - normalized_distance(s1, s2, processor=processor)
    return value if score_cutoff is None or value >= score_cutoff else 0.0