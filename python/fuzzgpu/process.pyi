from typing import Any, Callable, Iterable, Iterator, Optional, Union
import numpy as np

Scorer = Callable[..., float]
Processor = Callable[[str], str]

def extract_iter(
    query: str,
    choices: Iterable[str],
    *,
    scorer: Scorer = ...,
    processor: Optional[Processor] = ...,
    score_cutoff: Optional[float] = ...,
    score_hint: Optional[float] = ...,
    scorer_kwargs: Optional[dict] = ...,
) -> Iterator[tuple[str, float, int]]: ...

def extract(
    query: str,
    choices: Iterable[str],
    score_cutoff: Optional[float] = ...,
    limit: Optional[int] = ...,
    *,
    scorer: Scorer = ...,
    processor: Optional[Processor] = ...,
    score_hint: Optional[float] = ...,
    scorer_kwargs: Optional[dict] = ...,
) -> list[tuple[str, float, int]]: ...

def extractOne(
    query: str,
    choices: Iterable[str],
    score_cutoff: Optional[float] = ...,
    *,
    scorer: Scorer = ...,
    processor: Optional[Processor] = ...,
    score_hint: Optional[float] = ...,
    scorer_kwargs: Optional[dict] = ...,
) -> Optional[tuple[str, float, int]]: ...

extract_one = extractOne

def cdist(
    queries: Iterable[str],
    choices: Iterable[str],
    *,
    scorer: Scorer = ...,
    processor: Optional[Processor] = ...,
    score_cutoff: Optional[float] = ...,
    score_hint: Optional[float] = ...,
    score_multiplier: float = ...,
    dtype: Optional[Any] = ...,
    workers: Optional[int] = ...,
    scorer_kwargs: Optional[dict] = ...,
) -> Union[list[list[float]], np.ndarray]: ...
