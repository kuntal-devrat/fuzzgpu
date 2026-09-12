from typing import Callable, Iterable, Optional, Sequence, Tuple
Processor = Callable[[str], str]

def ratio(a: str, b: str, *, processor: Optional[Processor] = ..., score_cutoff: float = ...) -> float: ...
def partial_ratio(a: str, b: str, *, processor: Optional[Processor] = ..., score_cutoff: float = ...) -> float: ...
def partial_ratio_alignment(
    a: str, b: str, *, processor: Optional[Processor] = ..., score_cutoff: float = ...
) -> "ScoreAlignment": ...
def token_sort_ratio(a: str, b: str, *, processor: Optional[Processor] = ..., score_cutoff: float = ...) -> float: ...
def token_set_ratio(a: str, b: str, *, processor: Optional[Processor] = ..., score_cutoff: float = ...) -> float: ...
def token_ratio(a: str, b: str, *, processor: Optional[Processor] = ..., score_cutoff: float = ...) -> float: ...
def partial_token_sort_ratio(a: str, b: str, *, processor: Optional[Processor] = ..., score_cutoff: float = ...) -> float: ...
def partial_token_set_ratio(a: str, b: str, *, processor: Optional[Processor] = ..., score_cutoff: float = ...) -> float: ...
def partial_token_ratio(a: str, b: str, *, processor: Optional[Processor] = ..., score_cutoff: float = ...) -> float: ...
def QRatio(a: str, b: str, *, processor: Optional[Processor] = ..., score_cutoff: float = ...) -> float: ...
def WRatio(a: str, b: str, *, processor: Optional[Processor] = ..., score_cutoff: float = ...) -> float: ...
wratio = WRatio
def ratio_batch(query: str, candidates: Sequence[str], *, processor: Optional[Processor] = ..., score_cutoff: float = ..., workers: Optional[int] = ...) -> list[float]: ...
def extract(
    query: str,
    choices: Iterable[str],
    score_cutoff: float = ...,
    limit: int = ...,
    *,
    scorer: Callable = ...,
    processor: Optional[Processor] = ...,
    score_hint: Optional[float] = ...,
    scorer_kwargs: Optional[dict] = ...,
) -> list[Tuple[str, float, int]]: ...
def extractOne(
    query: str,
    choices: Iterable[str],
    score_cutoff: float = ...,
    *,
    scorer: Callable = ...,
    processor: Optional[Processor] = ...,
    score_hint: Optional[float] = ...,
    scorer_kwargs: Optional[dict] = ...,
) -> Optional[Tuple[str, float, int]]: ...
extract_one = extractOne
damerau_ratio: Callable[[str, str], float]

def cdist(
    queries: Iterable[str],
    choices: Iterable[str],
    *,
    scorer: Optional[Callable] = ...,
    processor: Optional[Processor] = ...,
    score_cutoff: Optional[float] = ...,
    score_hint: Optional[float] = ...,
    score_multiplier: float = ...,
    dtype: Optional[object] = ...,
    workers: Optional[int] = ...,
    scorer_kwargs: Optional[dict] = ...,
) -> object: ...

class ScoreAlignment(Tuple[float, int, int, int, int]):
    score: float
    src_start: int
    src_end: int
    dest_start: int
    dest_end: int
