"""rapidfuzz.process-compatible helpers."""

from . import fuzz


def _items(choices):
    return choices.items() if hasattr(choices, "items") else enumerate(choices)


def extract_iter(query, choices, *, scorer=fuzz.ratio, processor=None, score_cutoff=0.0, score_hint=None, scorer_kwargs=None):
    del score_hint
    kwargs = scorer_kwargs or {}
    for key, choice in _items(choices):
        if choice is None:
            continue
        score = scorer(query, choice, processor=processor, score_cutoff=score_cutoff, **kwargs)
        if score >= score_cutoff:
            yield (choice, score, key)


def extract(query, choices, score_cutoff=0.0, limit=5, *, scorer=fuzz.ratio, processor=None, score_hint=None, scorer_kwargs=None):
    results = sorted(extract_iter(query, choices, scorer=scorer, processor=processor,
                                  score_cutoff=score_cutoff, score_hint=score_hint,
                                  scorer_kwargs=scorer_kwargs), key=lambda item: item[1], reverse=True)
    return results if limit is None else results[:limit]


def extractOne(query, choices, score_cutoff=0.0, *, scorer=fuzz.ratio, processor=None, score_hint=None, scorer_kwargs=None):
    del score_hint
    kwargs = scorer_kwargs or {}
    best = None
    for key, choice in _items(choices):
        if choice is None:
            continue
        score = scorer(query, choice, processor=processor, score_cutoff=score_cutoff, **kwargs)
        if score >= score_cutoff and (best is None or score > best[1]):
            best = (choice, score, key)
            if score == 100.0:
                break
    return best


extract_one = extractOne


def cdist(queries, choices, *, scorer=fuzz.ratio, processor=None, score_cutoff=0.0, score_hint=None, score_multiplier=1, dtype=None, workers=None, scorer_kwargs=None):
    del score_hint, workers
    kwargs = scorer_kwargs or {}
    matrix = [[scorer(q, c, processor=processor, score_cutoff=score_cutoff, **kwargs) * score_multiplier
               for c in choices] for q in queries]
    if dtype is not None:
        try:
            import numpy as np
            return np.asarray(matrix, dtype=dtype)
        except ImportError:
            pass
    return matrix


__all__ = ["extract", "extractOne", "extract_one", "extract_iter", "cdist"]
