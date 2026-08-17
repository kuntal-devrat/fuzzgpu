"""rapidfuzz.process-compatible helpers with similarity and distance scorers."""

from concurrent.futures import ThreadPoolExecutor
import inspect
import os

from . import fuzz


def _items(choices):
    return choices.items() if hasattr(choices, "items") else enumerate(choices)


def _is_distance_scorer(scorer):
    """Recognise this package's distance scorers without guessing for user code."""
    return scorer.__module__.startswith("fuzzgpu.distance") and scorer.__name__ in {"distance", "normalized_distance"}


def _is_native_ratio(scorer):
    """Return True when the scorer is the native fuzz.ratio (or its alias).

    When True, `cdist` can route the whole matrix through the hardware-
    accelerated Rust levenshtein_cdist / ratio_batch path instead of calling
    the Python scorer once per cell.
    """
    return scorer is fuzz.ratio


def _accepts(scorer, name):
    try:
        signature = inspect.signature(scorer)
    except (TypeError, ValueError):
        return True
    return name in signature.parameters or any(
        p.kind == p.VAR_KEYWORD for p in signature.parameters.values()
    )


def _score(scorer, query, choice, processor, score_cutoff, scorer_kwargs):
    """Call both fuzzgpu and ordinary two-argument user scorers correctly."""
    kwargs = dict(scorer_kwargs or {})
    accepts_processor = _accepts(scorer, "processor")
    accepts_cutoff = _accepts(scorer, "score_cutoff")
    if processor is not None and not accepts_processor:
        query, choice = processor(query), processor(choice)
    if accepts_processor:
        kwargs["processor"] = processor
    if accepts_cutoff and score_cutoff is not None:
        kwargs["score_cutoff"] = score_cutoff
    return scorer(query, choice, **kwargs)


def _qualifies(score, cutoff, distance):
    if cutoff is None:
        return True
    return score <= cutoff if distance else score >= cutoff


def _better(candidate, current, distance):
    if current is None:
        return True
    return candidate[1] < current[1] if distance else candidate[1] > current[1]


def extract_iter(
    query,
    choices,
    *,
    scorer=fuzz.WRatio,
    processor=None,
    score_cutoff=None,
    score_hint=None,
    scorer_kwargs=None,
):
    del score_hint
    distance = _is_distance_scorer(scorer)
    for key, choice in _items(choices):
        if choice is None:
            continue
        score = _score(scorer, query, choice, processor, score_cutoff, scorer_kwargs)
        if _qualifies(score, score_cutoff, distance):
            yield (choice, score, key)


def extract(
    query,
    choices,
    score_cutoff=None,
    limit=5,
    *,
    scorer=fuzz.WRatio,
    processor=None,
    score_hint=None,
    scorer_kwargs=None,
):
    distance = _is_distance_scorer(scorer)
    results = list(
        extract_iter(
            query,
            choices,
            scorer=scorer,
            processor=processor,
            score_cutoff=score_cutoff,
            score_hint=score_hint,
            scorer_kwargs=scorer_kwargs,
        )
    )
    results.sort(key=lambda item: item[1], reverse=not distance)
    return results if limit is None else results[:limit]


def extractOne(
    query,
    choices,
    score_cutoff=None,
    *,
    scorer=fuzz.WRatio,
    processor=None,
    score_hint=None,
    scorer_kwargs=None,
):
    del score_hint
    distance = _is_distance_scorer(scorer)
    best = None
    for key, choice in _items(choices):
        if choice is None:
            continue
        score = _score(scorer, query, choice, processor, score_cutoff, scorer_kwargs)
        candidate = (choice, score, key)
        if _qualifies(score, score_cutoff, distance) and _better(candidate, best, distance):
            best = candidate
            if score == (0 if distance else 100.0):
                break
    return best


extract_one = extractOne


def _worker_count(workers):
    if workers is None or workers == 1:
        return 1
    if workers == -1:
        return os.cpu_count() or 1
    if isinstance(workers, int) and workers > 0:
        return workers
    raise ValueError("workers must be None, -1, or a positive integer")


def cdist(
    queries,
    choices,
    *,
    scorer=fuzz.WRatio,
    processor=None,
    score_cutoff=None,
    score_hint=None,
    score_multiplier=1,
    dtype=None,
    workers=None,
    scorer_kwargs=None,
):
    """Compute a pairwise score matrix between all queries and all choices.

    When the scorer is the default ``fuzz.ratio`` and no processor or scorer
    kwargs are given, the whole matrix is computed through the Rust
    ``levenshtein_cdist`` / ``ratio_batch`` fast path (Rayon + optional GPU)
    instead of one Python call per cell.  All other scorers fall back to the
    generic per-cell path.
    """
    del score_hint
    # Materialise: one-shot iterables must be reusable across rows, and we
    # need len() for pre-allocation in the fast path.
    queries = list(queries)
    choices = list(choices)

    # ── Fast path: native ratio scorer, no per-string processor ──────────
    # Route through the accelerated Rust matrix kernel.  The result is in
    # 0–100 float, matching what the slow path would produce.
    if (
        _is_native_ratio(scorer)
        and processor is None
        and not scorer_kwargs
        and score_multiplier == 1
    ):
        try:
            from . import fuzzgpu as _native
            # Apply score_cutoff: cells below threshold become 0.0 (same
            # behaviour as the per-cell slow path with score_cutoff set).
            raw: list[list[float]] = _native.fuzz_ratio_batch  # type hint only
            matrix: list[list[float]] = []
            for q in queries:
                row = _native.fuzz_ratio_batch(q, choices)
                if score_cutoff is not None:
                    row = [v if v >= score_cutoff else 0.0 for v in row]
                matrix.append(row)
            if dtype is not None:
                try:
                    import numpy as np
                    return np.asarray(matrix, dtype=dtype)
                except ImportError:
                    pass
            return matrix
        except Exception:
            # Fall through to the generic path on any unexpected error.
            pass

    # ── Generic path ─────────────────────────────────────────────────────
    def row(query):
        return [
            _score(scorer, query, choice, processor, score_cutoff, scorer_kwargs)
            * score_multiplier
            for choice in choices
        ]

    count = _worker_count(workers)
    matrix = (
        list(map(row, queries))
        if count == 1
        else list(ThreadPoolExecutor(max_workers=count).map(row, queries))
    )

    if dtype is not None:
        try:
            import numpy as np
            return np.asarray(matrix, dtype=dtype)
        except ImportError:
            pass
    return matrix


__all__ = ["extract", "extractOne", "extract_one", "extract_iter", "cdist"]
