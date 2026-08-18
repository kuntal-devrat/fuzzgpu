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

NOTE on Damerau-Levenshtein semantics:
    fuzzgpu implements the **unrestricted** Lowrance-Wagner (1975) algorithm,
    which allows non-adjacent transpositions. This differs from rapidfuzz's
    DamerauLevenshtein module (which uses Optimal String Alignment / OSA).
    Example: fuzzgpu.damerau("ca", "abc") == 2 (unrestricted), while
    rapidfuzz.distance.DamerauLevenshtein.distance("ca", "abc") == 3 (OSA).
    Use fuzzgpu.distance.OSA for OSA-semantics compatibility with rapidfuzz.
"""

from fuzzgpu.fuzzgpu import (
    # Levenshtein
    levenshtein_distance,
    levenshtein_batch,
    levenshtein_batch_into,
    levenshtein_cdist,
    levenshtein_cdist_into,
    # Damerau-Levenshtein (unrestricted Lowrance-Wagner)
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
    fuzz_partial_ratio_alignment,
    fuzz_token_sort_ratio,
    fuzz_token_set_ratio,
    fuzz_token_ratio,
    fuzz_wratio,
    fuzz_ratio_batch,
    fuzz_extract,
    fuzz_extract_one,
    fuzz_partial_token_sort_ratio,
    fuzz_partial_token_set_ratio,
    fuzz_partial_token_ratio,
    fuzz_qratio,
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

# ── Intuitive aliases ────────────────────────────────────────
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
token_ratio = fuzz_token_ratio
wratio = fuzz_wratio
WRatio = fuzz_wratio
QRatio = fuzz_qratio
ratio_batch = fuzz_ratio_batch
partial_token_sort_ratio = fuzz_partial_token_sort_ratio
partial_token_set_ratio = fuzz_partial_token_set_ratio
partial_token_ratio = fuzz_partial_token_ratio
extract = fuzz_extract
extractOne = fuzz_extract_one
extract_one = fuzz_extract_one

# ── rapidfuzz-style compatibility namespaces ─────────────────
# These are Python façades over the accelerated extension and accept
# scorer/processor/cutoff keyword arguments.
from . import fuzz, process, distance

# Prefer the richer compatibility signatures at the package root as well.
ratio = fuzz.ratio
partial_ratio = fuzz.partial_ratio
partial_ratio_alignment = fuzz.partial_ratio_alignment
token_sort_ratio = fuzz.token_sort_ratio
token_set_ratio = fuzz.token_set_ratio
token_ratio = fuzz.token_ratio
partial_token_sort_ratio = fuzz.partial_token_sort_ratio
partial_token_set_ratio = fuzz.partial_token_set_ratio
partial_token_ratio = fuzz.partial_token_ratio
wratio = WRatio = fuzz.WRatio
QRatio = fuzz.QRatio
extract = fuzz.extract
extract_one = extractOne = fuzz.extractOne

# Levenshtein alignment helpers (live in distance.Levenshtein, also available
# at the top level for drop-in rapidfuzz compatibility).
from .distance.Levenshtein import editops, opcodes

# rapidfuzz.distance-compatible alignment types at the package root.
from .distance import (
    Editop,
    Editops,
    Opcode,
    Opcodes,
    MatchingBlock,
    ScoreAlignment,
)

__all__ = [
    # Core distance functions
    "levenshtein_distance", "levenshtein_batch", "levenshtein_batch_into",
    "levenshtein_cdist", "levenshtein_cdist_into",
    "damerau_levenshtein_distance", "damerau_levenshtein_batch",
    "damerau_levenshtein_batch_into", "damerau_levenshtein_cdist",
    "damerau_levenshtein_cdist_into", "damerau_ratio",
    "needleman_wunsch_score", "needleman_wunsch_batch_fn",
    "needleman_wunsch_affine", "needleman_wunsch_affine_batch",
    "jaro_similarity", "jaro_winkler_similarity", "jaro_winkler_batch_fn",
    "jaro_winkler_batch_into", "jaro_winkler_cdist", "jaro_winkler_cdist_into",
    # Fuzzy scorers
    "fuzz_ratio", "fuzz_partial_ratio", "fuzz_token_sort_ratio",
    "fuzz_token_set_ratio", "fuzz_token_ratio",
    "fuzz_wratio", "fuzz_ratio_batch",
    "fuzz_extract", "fuzz_extract_one",
    "fuzz_partial_token_sort_ratio", "fuzz_partial_token_set_ratio",
    "fuzz_partial_token_ratio",
    "fuzz_qratio",
    # Optimized variants
    "levenshtein_myers", "needleman_wunsch_striped", "jaro_optimized",
    # Control
    "set_cpu_only", "set_gpu_threshold", "is_gpu_available",
    "warmup", "gpu_info", "hardware_info", "__version__",
    # Aliases
    "levenshtein", "damerau_levenshtein", "damerau",
    "needleman_wunsch", "needleman_wunsch_batch",
    "jaro", "jaro_winkler", "jaro_winkler_batch",
    "ratio", "partial_ratio", "partial_ratio_alignment",
    "token_sort_ratio", "token_set_ratio", "token_ratio",
    "partial_token_sort_ratio", "partial_token_set_ratio", "partial_token_ratio",
    "wratio", "WRatio", "QRatio", "ratio_batch",
    "extract", "extractOne", "extract_one",
    # Alignment helpers
    "editops", "opcodes",
    "Editop", "Editops", "Opcode", "Opcodes", "MatchingBlock", "ScoreAlignment",
    # Submodules
    "fuzz", "process", "distance",
]
