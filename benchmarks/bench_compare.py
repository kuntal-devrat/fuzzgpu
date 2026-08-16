"""Benchmark fuzzgpu vs rapidfuzz vs python-Levenshtein.

All comparisons are fair: batch-vs-batch, single-vs-single.
"""

import time
import random
import string
import sys

import fuzzgpu
import rapidfuzz
import rapidfuzz.distance.Levenshtein as rf_lev
import rapidfuzz.distance.DamerauLevenshtein as rf_dam
from rapidfuzz.distance import JaroWinkler as rf_jw
import Levenshtein as py_lev


def random_string(length=10):
    return "".join(random.choices(string.ascii_lowercase, k=length))


def random_strings(n, length=10):
    return [random_string(length) for _ in range(n)]


def levenshtein_python(a, b):
    """Pure Python Levenshtein."""
    if len(a) < len(b):
        return levenshtein_python(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for ca in a:
        curr = [prev[0] + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def bench(fn, *args, repeats=3):
    """Run fn(*args) `repeats` times, return best time."""
    best = float("inf")
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn(*args)
        elapsed = time.perf_counter() - t0
        best = min(best, elapsed)
    return best, result


def print_tableheader():
    print(f"{'Size':>8} | {'fuzzgpu':>10} | {'rapidfuzz':>10} | {'py-Lev':>10} | {'Python':>10} | {'vs RF':>8} | {'vs py-Lev':>10}")
    print("-" * 95)


def print_tablerow(size, t_gpu, t_rf, t_pylev, t_py):
    rf_vs = t_rf / t_gpu if t_gpu > 0 else float("inf")
    pylev_vs = t_pylev / t_gpu if t_gpu > 0 else float("inf")
    py_str = f"{t_py*1000:>9.2f}ms" if t_py != float("inf") else "       N/A"
    pylev_str = f"{t_pylev*1000:>9.2f}ms" if t_pylev != float("inf") else "       N/A"
    print(f"{size:>8} | {t_gpu*1000:>9.2f}ms | {t_rf*1000:>9.2f}ms | {pylev_str} | {py_str} | {rf_vs:>7.2f}x | {pylev_vs:>9.2f}x")


def benchmark_levenshtein_batch():
    print("=" * 95)
    print("LEVENSHTEIN BATCH: 1 query x N candidates (10-char strings)")
    print("=" * 95)
    print_tableheader()

    query = "hello world"
    candidates = random_strings(50_000, 10)

    for n in [100, 500, 1_000, 5_000, 10_000, 20_000, 50_000]:
        batch = candidates[:n]

        # fuzzgpu (GPU / parallel batch API)
        t_gpu, _ = bench(fuzzgpu.levenshtein_batch, query, batch)

        # rapidfuzz (batch)
        t_rf, _ = bench(lambda: [rf_lev.distance(query, c) for c in batch])

        # python-Levenshtein (batch)
        t_pylev, _ = bench(lambda: [py_lev.distance(query, c) for c in batch])

        # pure python
        if n <= 5_000:
            t_py, _ = bench(lambda: [levenshtein_python(query, c) for c in batch], repeats=1)
        else:
            t_py = float("inf")

        print_tablerow(n, t_gpu, t_rf, t_pylev, t_py)
    print()


def benchmark_damerau_batch():
    print("=" * 95)
    print("DAMERAU-LEVENSHTEIN BATCH: 1 query x N candidates (10-char strings)")
    print("=" * 95)
    print(f"{'Size':>8} | {'fuzzgpu':>10} | {'rapidfuzz':>10} | {'speedup':>8}")
    print("-" * 55)

    query = "hello world"
    candidates = random_strings(50_000, 10)

    for n in [100, 1_000, 5_000, 10_000, 50_000]:
        batch = candidates[:n]

        t_gpu, _ = bench(fuzzgpu.damerau_levenshtein_batch, query, batch)
        t_rf, _ = bench(lambda: [rf_dam.distance(query, c) for c in batch])

        speedup = t_rf / t_gpu if t_gpu > 0 else float("inf")
        print(f"{n:>8} | {t_gpu*1000:>9.2f}ms | {t_rf*1000:>9.2f}ms | {speedup:>7.2f}x")
    print()


def benchmark_levenshtein_cdist():
    print("=" * 95)
    print("LEVENSHTEIN CDIST: N x M matrix (10-char strings)")
    print("=" * 95)
    print_tableheader()

    list_a = random_strings(200, 10)
    list_b = random_strings(200, 10)

    for na, nb in [(10, 10), (50, 50), (100, 100), (200, 200)]:
        a, b = list_a[:na], list_b[:nb]

        t_gpu, _ = bench(fuzzgpu.levenshtein_cdist, a, b)

        def rf_cdist():
            return [[rf_lev.distance(ai, bj) for bj in b] for ai in a]
        t_rf, _ = bench(rf_cdist)

        def pylev_cdist():
            return [[py_lev.distance(ai, bj) for bj in b] for ai in a]
        t_pylev, _ = bench(pylev_cdist)

        total = na * nb
        if total <= 10_000:
            def py_cdist():
                return [[levenshtein_python(ai, bj) for bj in b] for ai in a]
            t_py, _ = bench(py_cdist)
        else:
            t_py = float("inf")

        print_tablerow(f"{na}x{nb}", t_gpu, t_rf, t_pylev, t_py)
    print()


def benchmark_jaro_winkler():
    print("=" * 95)
    print("JARO-WINKLER: 1 query x N candidates (10-char strings)")
    print("=" * 95)
    print_tableheader()

    query = "MARTHA"
    candidates = random_strings(50_000, 10)

    for n in [100, 1_000, 5_000, 10_000, 50_000]:
        batch = candidates[:n]

        t_gpu, _ = bench(fuzzgpu.jaro_winkler_batch_fn, query, batch, 0.1)
        t_rf, _ = bench(lambda: [rf_jw.similarity(query, c) for c in batch])
        t_pylev, _ = bench(lambda: [py_lev.jaro_winkler(query, c) for c in batch])

        if n <= 1_000:
            t_py, _ = bench(lambda: [rf_jw.similarity(query, c) for c in batch])
        else:
            t_py = float("inf")

        print_tablerow(n, t_gpu, t_rf, t_pylev, t_py)
    print()


def benchmark_needleman_affine():
    print("=" * 95)
    print("NEEDLEMAN-WUNSCH (Linear vs Affine Gap): 1 query x N candidates")
    print("=" * 95)
    print(f"{'Size':>8} | {'Linear':>12} | {'Affine':>12}")
    print("-" * 45)

    query = "AGTACGCA"
    candidates = random_strings(50_000, 10)

    for n in [100, 1_000, 5_000, 10_000, 50_000]:
        batch = candidates[:n]

        t_lin, _ = bench(fuzzgpu.needleman_wunsch_batch, query, batch, 2, -1, -2)
        t_aff, _ = bench(fuzzgpu.needleman_wunsch_affine_batch, query, batch, 2, -1, -3, -1)

        print(f"{n:>8} | {t_lin*1000:>11.2f}ms | {t_aff*1000:>11.2f}ms")
    print()


def benchmark_fuzz_ratio():
    print("=" * 95)
    print("FUZZ RATIO BATCH: 1 query x N candidates")
    print("=" * 95)
    print(f"{'Size':>8} | {'fuzzgpu':>10} | {'rapidfuzz':>10} | {'speedup':>8}")
    print("-" * 55)

    query = "hello world this is a test"
    candidates = [f"{random_string(5)} {random_string(5)} {random_string(5)}" for _ in range(50_000)]

    for n in [100, 1_000, 5_000, 10_000, 50_000]:
        batch = candidates[:n]

        t_gpu, _ = bench(fuzzgpu.fuzz_ratio_batch, query, batch)
        t_rf, _ = bench(lambda: [rapidfuzz.fuzz.ratio(query, c) for c in batch])

        speedup = t_rf / t_gpu if t_gpu > 0 else float("inf")
        print(f"{n:>8} | {t_gpu*1000:>9.2f}ms | {t_rf*1000:>9.2f}ms | {speedup:>7.2f}x")
    print()


if __name__ == "__main__":
    print(f"fuzzgpu {fuzzgpu.__version__}")
    print(f"GPU: {fuzzgpu.gpu_info()}")
    print(f"Python: {sys.version}")
    print(f"rapidfuzz: {rapidfuzz.__version__}")
    print()

    benchmark_levenshtein_batch()
    benchmark_damerau_batch()
    benchmark_levenshtein_cdist()
    benchmark_jaro_winkler()
    benchmark_needleman_affine()
    benchmark_fuzz_ratio()

    print("=" * 95)
    print("BENCHMARKS COMPLETED")
    print("=" * 95)
