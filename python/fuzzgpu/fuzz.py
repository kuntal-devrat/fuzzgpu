"""rapidfuzz-compatible fuzzy scorers backed by fuzzgpu's Rust extension."""

import os

from . import fuzzgpu as _native
from .distance._common import ScoreAlignment


def _worker_count(workers):
    """Resolve the workers parameter to a concrete thread count."""
    if workers is None or workers == 1:
        return 1
    if workers == -1:
        return os.cpu_count() or 1
    if isinstance(workers, int) and workers > 0:
        return workers
    raise ValueError("workers must be None, -1, or a positive integer")


def _prepare(a, b, processor):
    if processor is not None:
        a, b = processor(a), processor(b)
    return a, b


def _cutoff(score, score_cutoff):
    return score if score >= score_cutoff else 0.0


def ratio(a, b, *, processor=None, score_cutoff=0.0):
    a, b = _prepare(a, b, processor)
    return _native.fuzz_ratio(a, b, score_cutoff)


def partial_ratio(a, b, *, processor=None, score_cutoff=0.0):
    a, b = _prepare(a, b, processor)
    return _native.fuzz_partial_ratio(a, b, score_cutoff)


def partial_ratio_alignment(a, b, *, processor=None, score_cutoff=0.0):
    """Return a ScoreAlignment (score, src_start, src_end, dest_start, dest_end)
    for the best matching window.

    Compatible with rapidfuzz's ``partial_ratio_alignment``.  ``src_*`` fields
    are relative to the first argument, ``dest_*`` to the second; ``score`` is
    already subject to ``score_cutoff`` (0.0 when below it).
    """
    a, b = _prepare(a, b, processor)
    return ScoreAlignment(*_native.fuzz_partial_ratio_alignment(a, b, score_cutoff))


def token_sort_ratio(a, b, *, processor=None, score_cutoff=0.0):
    a, b = _prepare(a, b, processor)
    return _native.fuzz_token_sort_ratio(a, b, score_cutoff)


def token_set_ratio(a, b, *, processor=None, score_cutoff=0.0):
    a, b = _prepare(a, b, processor)
    return _native.fuzz_token_set_ratio(a, b, score_cutoff)


def token_ratio(a, b, *, processor=None, score_cutoff=0.0):
    a, b = _prepare(a, b, processor)
    return _native.fuzz_token_ratio(a, b, score_cutoff)


def partial_token_sort_ratio(s1, s2, *, processor=None, score_cutoff=0.0):
    a, b = _prepare(s1, s2, processor)
    return _native.fuzz_partial_token_sort_ratio(a, b, score_cutoff)


def partial_token_set_ratio(s1, s2, *, processor=None, score_cutoff=0.0):
    a, b = _prepare(s1, s2, processor)
    return _native.fuzz_partial_token_set_ratio(a, b, score_cutoff)


def partial_token_ratio(s1, s2, *, processor=None, score_cutoff=0.0):
    a, b = _prepare(s1, s2, processor)
    return _native.fuzz_partial_token_ratio(a, b, score_cutoff)


def QRatio(s1, s2, *, processor=None, score_cutoff=0.0):
    """Quick ratio (rapidfuzz-compatible: empty strings score 0)."""
    a, b = _prepare(s1, s2, processor)
    return _native.fuzz_qratio(a, b, score_cutoff)


def WRatio(a, b, *, processor=None, score_cutoff=0.0):
    a, b = _prepare(a, b, processor)
    return _native.fuzz_wratio(a, b, score_cutoff)


wratio = WRatio


def ratio_batch(query, candidates, *, processor=None, score_cutoff=0.0, workers=None):
    """Compute ratio(query, c) for every c in candidates.

    When no processor is given the computation runs inside Rust under Rayon
    (all cores); ``workers`` is ignored in that case since Rayon already
    parallelises across cores.  When a Python ``processor`` is supplied the
    work runs in Python and ``workers`` controls the thread pool size
    (``None``/``1`` = single-threaded, ``-1`` = all CPU cores).
    """
    if processor is None:
        values = _native.fuzz_ratio_batch(query, candidates)
        return values if score_cutoff <= 0.0 else [_cutoff(v, score_cutoff) for v in values]
    # processor path: honour workers
    n = _worker_count(workers) if workers is not None and workers != 1 else 1
    if n == 1:
        return [ratio(query, c, processor=processor, score_cutoff=score_cutoff) for c in candidates]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=n) as pool:
        return list(pool.map(
            lambda c: ratio(query, c, processor=processor, score_cutoff=score_cutoff),
            candidates
        ))


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


def cdist(queries, choices, *, scorer=None, processor=None, score_cutoff=None,
          score_hint=None, score_multiplier=1, dtype=None, workers=None,
          scorer_kwargs=None):
    """Pairwise score matrix between all queries and all choices.

    Delegates to ``process.cdist`` with ``scorer`` defaulting to
    ``fuzz.ratio`` (matching rapidfuzz's ``fuzz.cdist``).
    """
    from .process import cdist as _cdist
    return _cdist(
        queries, choices,
        scorer=scorer if scorer is not None else ratio,
        processor=processor,
        score_cutoff=score_cutoff,
        score_hint=score_hint,
        score_multiplier=score_multiplier,
        dtype=dtype,
        workers=workers,
        scorer_kwargs=scorer_kwargs,
    )


__all__ = [
    "ratio", "partial_ratio", "partial_ratio_alignment",
    "token_sort_ratio", "token_set_ratio", "token_ratio",
    "partial_token_sort_ratio", "partial_token_set_ratio", "partial_token_ratio",
    "QRatio", "WRatio", "wratio", "ratio_batch", "cdist",
    "extract", "extractOne", "extract_one", "damerau_ratio",
]
