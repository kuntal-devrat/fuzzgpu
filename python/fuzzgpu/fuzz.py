"""rapidfuzz-compatible API for fuzzgpu — delegates to Rust implementations."""

from fuzzgpu.fuzzgpu import (
    fuzz_ratio as ratio,
    fuzz_partial_ratio as partial_ratio,
    fuzz_token_sort_ratio as token_sort_ratio,
    fuzz_token_set_ratio as token_set_ratio,
    fuzz_wratio as wratio,
    fuzz_wratio as WRatio,
    fuzz_ratio_batch as ratio_batch,
    fuzz_extract as extract,
    fuzz_extract_one as extract_one,
    fuzz_extract_one as extractOne,
    damerau_ratio as damerau_ratio,
)

__all__ = [
    "ratio",
    "partial_ratio",
    "token_sort_ratio",
    "token_set_ratio",
    "wratio",
    "WRatio",
    "ratio_batch",
    "extract",
    "extract_one",
    "extractOne",
    "damerau_ratio",
]
