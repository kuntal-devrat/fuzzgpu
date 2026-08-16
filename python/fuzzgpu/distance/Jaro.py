from .. import fuzzgpu as _native
def similarity(s1, s2, *, processor=None, score_cutoff=0.0):
    if processor: s1, s2 = processor(s1), processor(s2)
    value = _native.jaro_similarity(s1, s2); return value if value >= score_cutoff else 0.0
def distance(s1, s2, **kwargs): return 1.0 - similarity(s1, s2, **kwargs)
def normalized_similarity(s1, s2, **kwargs): return similarity(s1, s2, **kwargs)
def normalized_distance(s1, s2, **kwargs): return distance(s1, s2, **kwargs)
