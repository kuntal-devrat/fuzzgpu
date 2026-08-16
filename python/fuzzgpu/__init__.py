"""
fuzzgpu - GPU-accelerated fuzzy string matching.

Cross-platform GPU acceleration via wgpu (Metal, Vulkan, DX12).
No CUDA required. Works on Mac, Linux, Windows.

Usage:
    import fuzzgpu

    # Levenshtein & Damerau-Levenshtein
    dist = fuzzgpu.levenshtein_distance("kitten", "sitting")  # 3
    dam = fuzzgpu.damerau_levenshtein_distance("ab", "ba")  # 1

    # Batch & Cross-product Matrix
    distances = fuzzgpu.levenshtein_batch("hello", ["hallo", "hullo"])
    matrix = fuzzgpu.levenshtein_cdist(["abc", "def"], ["abc", "xyz"])

    # Needleman-Wunsch with linear or affine gap penalty
    score = fuzzgpu.needleman_wunsch_score("AGTACGCA", "TATGC", 2, -1, -2)
    score_affine = fuzzgpu.needleman_wunsch_affine("AGTACGCA", "TATGC", 2, -1, -3, -1)

    # Jaro-Winkler similarity (GPU-accelerated for batches)
    sim = fuzzgpu.jaro_winkler_similarity("MARTHA", "MARHTA", 0.1)  # 0.96
    jw_batch = fuzzgpu.jaro_winkler_batch("MARTHA", ["MARHTA", "MATRH"], 0.1)

    # Fuzzy matching (rapidfuzz-compatible)
    from fuzzgpu.fuzz import ratio, partial_ratio, token_sort_ratio, token_set_ratio, extract, extractOne
"""

from fuzzgpu.fuzzgpu import (
    # Levenshtein
    levenshtein_distance,
    levenshtein_batch,
    levenshtein_cdist,
    # Damerau-Levenshtein
    damerau_levenshtein_distance,
    damerau_levenshtein_batch,
    damerau_levenshtein_cdist,
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
    jaro_winkler_cdist,
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
    # GPU
    gpu_info,
    __version__,
)

# Re-export with clean names
levenshtein = levenshtein_distance
damerau_levenshtein = damerau_levenshtein_distance
needleman_wunsch = needleman_wunsch_score
needleman_wunsch_batch = needleman_wunsch_batch_fn
jaro_winkler_batch = jaro_winkler_batch_fn
ratio = fuzz_ratio
partial_ratio = fuzz_partial_ratio
token_sort_ratio = fuzz_token_sort_ratio
token_set_ratio = fuzz_token_set_ratio
wratio = fuzz_wratio
ratio_batch = fuzz_ratio_batch
extract = fuzz_extract
extractOne = fuzz_extract_one

__all__ = [
    "levenshtein_distance", "levenshtein_batch", "levenshtein_cdist",
    "damerau_levenshtein_distance", "damerau_levenshtein_batch", "damerau_levenshtein_cdist", "damerau_ratio",
    "needleman_wunsch_score", "needleman_wunsch_batch_fn", "needleman_wunsch_affine", "needleman_wunsch_affine_batch",
    "jaro_similarity", "jaro_winkler_similarity", "jaro_winkler_batch_fn", "jaro_winkler_cdist",
    "fuzz_ratio", "fuzz_partial_ratio", "fuzz_token_sort_ratio", "fuzz_token_set_ratio",
    "fuzz_wratio", "fuzz_ratio_batch", "fuzz_extract", "fuzz_extract_one",
    "levenshtein_myers", "needleman_wunsch_striped", "jaro_optimized",
    "gpu_info", "__version__",
    # Aliases
    "levenshtein", "damerau_levenshtein", "needleman_wunsch", "needleman_wunsch_batch",
    "jaro_winkler_batch", "ratio", "partial_ratio", "token_sort_ratio", "token_set_ratio",
    "wratio", "ratio_batch", "extract", "extractOne",
]
