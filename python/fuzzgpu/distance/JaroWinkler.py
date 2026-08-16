"""rapidfuzz.distance.JaroWinkler-compatible module."""
from .. import fuzzgpu as _native


def similarity(s1, s2, *, prefix_weight=0.1, processor=None, score_cutoff=0.0):
    if processor:
        s1, s2 = processor(s1), processor(s2)
    value = _native.jaro_winkler_similarity(s1, s2, prefix_weight)
    return value if value >= score_cutoff else 0.0


def distance(s1, s2, *, prefix_weight=0.1, processor=None, score_cutoff=None):
    if processor:
        s1, s2 = processor(s1), processor(s2)
    value = 1.0 - _native.jaro_winkler_similarity(s1, s2, prefix_weight)
    return value if score_cutoff is None or value <= score_cutoff else 1.0


def normalized_similarity(s1, s2, *, prefix_weight=0.1, processor=None, score_cutoff=0.0):
    return similarity(s1, s2, prefix_weight=prefix_weight,
                      processor=processor, score_cutoff=score_cutoff)


def normalized_distance(s1, s2, *, prefix_weight=0.1, processor=None, score_cutoff=None):
    return distance(s1, s2, prefix_weight=prefix_weight,
                    processor=processor, score_cutoff=score_cutoff)
