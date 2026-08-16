from .. import fuzzgpu as _native
from ._common import cutoff_distance, normalized_distance as _normalized

def distance(s1, s2, *, processor=None, score_cutoff=None, score_hint=None):
    del score_hint
    if processor: s1, s2 = processor(s1), processor(s2)
    return cutoff_distance(_native.damerau_levenshtein_distance(s1, s2), score_cutoff)
def similarity(s1, s2, *, processor=None, score_cutoff=None, score_hint=None):
    value = max(len(s1), len(s2)) - distance(s1, s2, processor=processor, score_hint=score_hint); return value if score_cutoff is None or value >= score_cutoff else 0
def normalized_distance(s1, s2, *, processor=None, score_cutoff=None, score_hint=None):
    value = _normalized(distance(s1, s2, processor=processor, score_hint=score_hint), max(len(s1), len(s2))); return value if score_cutoff is None or value <= score_cutoff else 1.0
def normalized_similarity(s1, s2, *, processor=None, score_cutoff=None, score_hint=None):
    value = 1.0 - normalized_distance(s1, s2, processor=processor, score_hint=score_hint); return value if score_cutoff is None or value >= score_cutoff else 0.0
