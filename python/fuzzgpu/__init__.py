"""
fuzzgpu - Hardware-Accelerated Fuzzy String Matching & Sequence Alignment.

Cross-platform GPU acceleration via wgpu (Metal, Vulkan, DX12).
No CUDA required. Works on Mac, Linux, Windows.

Usage:
    import fuzzgpu

    # Levenshtein & Damerau-Levenshtein
    dist = fuzzgpu.levenshtein("kitten", "sitting")  # 3
    dam = fuzzgpu.damerau("ab", "ba")  # 1

    # Batch & Cross-product Matrix
    distances = fuzzgpu.levenshtein_batch("hello", ["hallo", "hullo"])
    matrix = fuzzgpu.levenshtein_cdist(["abc", "def"], ["abc", "xyz"])

    # Zero-allocation outputs: write into a preallocated numpy array
    import numpy as np
    out = np.empty(len(candidates), dtype=np.uint32)
    fuzzgpu.levenshtein_batch_into("hello", ["hallo", "hullo"], out)   # fills `out` in place

    # Needleman-Wunsch with linear or affine gap penalty
    score = fuzzgpu.needleman_wunsch("AGTACGCA", "TATGC", 2, -1, -2)
    score_affine = fuzzgpu.needleman_wunsch_affine("AGTACGCA", "TATGC", 2, -1, -3, -1)

    # Jaro & Jaro-Winkler similarity
    sim = fuzzgpu.jaro_winkler("MARTHA", "MARHTA")  # 0.96 (default p=0.1)
    jw_batch = fuzzgpu.jaro_winkler_batch("MARTHA", ["MARHTA", "MATRH"])

    # Utilities
    fuzzgpu.gpu_info()           # Returns active adapter and backend
    fuzzgpu.is_gpu_available()   # Boolean check
    fuzzgpu.set_cpu_only(True)   # Force CPU-only fallback mode
    fuzzgpu.warmup()             # Eager GPU initialization
"""

from fuzzgpu.fuzzgpu import (
    # Levenshtein
    levenshtein_distance,
    levenshtein_batch,
    levenshtein_batch_into,
    levenshtein_cdist,
    levenshtein_cdist_into,
    # Damerau-Levenshtein
    damerau_levenshtein_distance,
    damerau_levenshtein_batch,
    damerau_levenshtein_batch_into,
    damerau_levenshtein_cdist,
    damerau_levenshtein_cdist_into,
    damerau_ratio,
    # Needleman-Wunsch
    needleman_wunsch_score,
    needleman_wunsch_batch_fn,
    needleman_wunsch_affine,
    needleman_wunsch_affine_batch,
    # Jaro-Winkler
    jaro_similarity,
    jaro_winkler_similarity,
    jaro_winkler_batch_fn,
    jaro_winkler_batch_into,
    jaro_winkler_cdist,
    jaro_winkler_cdist_into,
    # Fuzzy matching
    fuzz_ratio,
    fuzz_partial_ratio,
    fuzz_token_sort_ratio,
    fuzz_token_set_ratio,
    fuzz_wratio,
    fuzz_ratio_batch,
    fuzz_extract,
    fuzz_extract_one,
    # Optimized algorithm variants
    levenshtein_myers,
    needleman_wunsch_striped,
    jaro_optimized,
    # Utilities & Control
    set_cpu_only,
    set_gpu_threshold,
    is_gpu_available,
    warmup,
    gpu_info,
    hardware_info,
    __version__,
)

# Intuitive Aliases
levenshtein = levenshtein_distance
damerau_levenshtein = damerau_levenshtein_distance
damerau = damerau_levenshtein_distance
needleman_wunsch = needleman_wunsch_score
needleman_wunsch_batch = needleman_wunsch_batch_fn
jaro = jaro_similarity
jaro_winkler = jaro_winkler_similarity
jaro_winkler_batch = jaro_winkler_batch_fn
ratio = fuzz_ratio
partial_ratio = fuzz_partial_ratio
token_sort_ratio = fuzz_token_sort_ratio
token_set_ratio = fuzz_token_set_ratio
wratio = fuzz_wratio
WRatio = fuzz_wratio
ratio_batch = fuzz_ratio_batch
extract = fuzz_extract
extractOne = fuzz_extract_one
extract_one = fuzz_extract_one

__all__ = [
    # Core API
    "levenshtein_distance", "levenshtein_batch", "levenshtein_batch_into", "levenshtein_cdist", "levenshtein_cdist_into",
    "damerau_levenshtein_distance", "damerau_levenshtein_batch", "damerau_levenshtein_batch_into", "damerau_levenshtein_cdist", "damerau_levenshtein_cdist_into", "damerau_ratio",
    "needleman_wunsch_score", "needleman_wunsch_batch_fn", "needleman_wunsch_affine", "needleman_wunsch_affine_batch",
    "jaro_similarity", "jaro_winkler_similarity", "jaro_winkler_batch_fn", "jaro_winkler_batch_into", "jaro_winkler_cdist", "jaro_winkler_cdist_into",
    "fuzz_ratio", "fuzz_partial_ratio", "fuzz_token_sort_ratio", "fuzz_token_set_ratio",
    "fuzz_wratio", "fuzz_ratio_batch", "fuzz_extract", "fuzz_extract_one",
    "levenshtein_myers", "needleman_wunsch_striped", "jaro_optimized",
    "set_cpu_only", "set_gpu_threshold", "is_gpu_available", "warmup", "gpu_info", "hardware_info", "__version__",
    # Aliases
    "levenshtein", "damerau_levenshtein", "damerau", "needleman_wunsch", "needleman_wunsch_batch",
    "jaro", "jaro_winkler", "jaro_winkler_batch", "ratio", "partial_ratio",
    "token_sort_ratio", "token_set_ratio", "wratio", "WRatio", "ratio_batch",
    "extract", "extractOne", "extract_one",
]
