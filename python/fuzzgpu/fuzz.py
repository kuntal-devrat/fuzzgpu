"""rapidfuzz-compatible fuzzy scorers backed by fuzzgpu's Rust extension."""

from . import fuzzgpu as _native


def _prepare(a, b, processor):
    if processor is not None:
        a, b = processor(a), processor(b)
    return a, b


def _cutoff(score, score_cutoff):
    return score if score >= score_cutoff else 0.0


def ratio(a, b, *, processor=None, score_cutoff=0.0):
    a, b = _prepare(a, b, processor)
    return _cutoff(_native.fuzz_ratio(a, b), score_cutoff)


def partial_ratio(a, b, *, processor=None, score_cutoff=0.0):
    a, b = _prepare(a, b, processor)
    return _cutoff(_native.fuzz_partial_ratio(a, b), score_cutoff)


def partial_ratio_alignment(a, b, *, processor=None):
    """Return (score, src_start, dest_start, length) for the best matching window.

    Compatible with rapidfuzz's ``partial_ratio_alignment``.  ``src_start`` is
    always 0 (the shorter string is the pattern); ``dest_start`` is the char
    offset into the longer string where the best window starts; ``length`` is
    the window width in chars.
    """
    a, b = _prepare(a, b, processor)
    return _native.fuzz_partial_ratio_alignment(a, b)


def token_sort_ratio(a, b, *, processor=None, score_cutoff=0.0):
    a, b = _prepare(a, b, processor)
    return _cutoff(_native.fuzz_token_sort_ratio(a, b), score_cutoff)


def token_set_ratio(a, b, *, processor=None, score_cutoff=0.0):
    a, b = _prepare(a, b, processor)
    return _cutoff(_native.fuzz_token_set_ratio(a, b), score_cutoff)


def partial_token_sort_ratio(s1, s2, *, processor=None, score_cutoff=0.0):
    a, b = _prepare(s1, s2, processor)
    return _cutoff(_native.fuzz_partial_token_sort_ratio(a, b), score_cutoff)


def partial_token_set_ratio(s1, s2, *, processor=None, score_cutoff=0.0):
    a, b = _prepare(s1, s2, processor)
    return _cutoff(_native.fuzz_partial_token_set_ratio(a, b), score_cutoff)


def QRatio(s1, s2, *, processor=None, score_cutoff=0.0):
    """Quick ratio; compatibility alias for ratio (same as in rapidfuzz)."""
    return ratio(s1, s2, processor=processor, score_cutoff=score_cutoff)


def WRatio(a, b, *, processor=None, score_cutoff=0.0):
    a, b = _prepare(a, b, processor)
    return _cutoff(_native.fuzz_wratio(a, b), score_cutoff)


wratio = WRatio


def ratio_batch(query, candidates, *, processor=None, score_cutoff=0.0, workers=None):
    del workers
    if processor is None:
        values = _native.fuzz_ratio_batch(query, candidates)
        return values if score_cutoff <= 0.0 else [_cutoff(v, score_cutoff) for v in values]
    return [ratio(query, c, processor=processor, score_cutoff=score_cutoff) for c in candidates]


def extract(query, choices, score_cutoff=0.0, limit=5, *, scorer=ratio, processor=None,
            score_hint=None, scorer_kwargs=None):
    from .process import extract as _extract
    return _extract(query, choices, scorer=scorer, processor=processor,
                    score_cutoff=score_cutoff, limit=limit, score_hint=score_hint,
                    scorer_kwargs=scorer_kwargs)


def extractOne(query, choices, score_cutoff=0.0, *, scorer=ratio, processor=None,
               score_hint=None, scorer_kwargs=None):
    from .process import extractOne as _extract_one
    return _extract_one(query, choices, scorer=scorer, processor=processor,
                        score_cutoff=score_cutoff, score_hint=score_hint,
                        scorer_kwargs=scorer_kwargs)


extract_one = extractOne
damerau_ratio = _native.damerau_ratio


__all__ = [
    "ratio", "partial_ratio", "partial_ratio_alignment",
    "token_sort_ratio", "token_set_ratio",
    "partial_token_sort_ratio", "partial_token_set_ratio",
    "QRatio", "WRatio", "wratio", "ratio_batch",
    "extract", "extractOne", "extract_one", "damerau_ratio",
]
