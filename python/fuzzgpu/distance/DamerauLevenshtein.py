"""rapidfuzz.distance.DamerauLevenshtein-compatible module.

NOTE: This module implements the **unrestricted** Lowrance-Wagner (1975)
algorithm which allows non-adjacent transpositions.  rapidfuzz's
DamerauLevenshtein uses Optimal String Alignment (OSA) which forbids them.
For OSA-compatible semantics use fuzzgpu.distance.OSA.
Example difference: distance("ca", "abc") == 2 here (unrestricted),
== 3 in rapidfuzz's OSA-based DamerauLevenshtein.
"""
from .. import fuzzgpu as _native
from ._common import cutoff_distance, normalized_distance as _normalized


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
