"""Reproducible cross-library benchmark for fuzzgpu.

Replaces the v0.1.0 script, which timed single un-warmed calls (best-of-3,
no warmup) and compared against pure Python — producing numbers that were
not reproducible on real hardware.

Methodology (all timings are one full library call):
- `warmup()` on fuzzgpu first (engine init + shader compile excluded).
- Each measurement is the **median of `--repeats`** (default 7) runs after a
  warmup call, via `time.perf_counter` around each run.
- Hardware and library versions are printed so the numbers are traceable.
- Columns: fuzzgpu (GPU), fuzzgpu (CPU-only / Rayon), rapidfuzz, and
  python-Levenshtein where it has a comparable API.
  Note: fuzzgpu's CPU path is multi-threaded (Rayon); rapidfuzz and
  python-Levenshtein are single-threaded C/C++ — the speedup column is vs
  rapidfuzz as the standard single-threaded reference.

Run:  python benchmarks/bench_compare.py [--repeats N] [--matrix-max 200]
"""

import argparse
import statistics
import sys
import time

# The × and unicode in table headers must survive Windows' cp1252 console.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import fuzzgpu

USE_CPU_ONLY = "--cpu-only" in sys.argv


def timed(fn, *args, repeats=7):
    """Median wall time (s) of `fn(*args)` over `repeats` runs, one warmup first."""
    fn(*args)  # warmup (also keeps caches hot for the measured runs)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def ms(t):
    return f"{t * 1000:8.2f} ms"


def speedup(base, other):
    if base is None or other is None:
        return "N/A"
    return f"{base / other:6.2f}×"


def batch_header(name, columns):
    print(f"\n### {name}")
    print("| Batch Size | " + " | ".join(columns) + " |")
    print("| :--- | " + " | ".join(":---:" for _ in columns) + " |")


def batch_row(size, row):
    print("| " + " | ".join(str(v) for v in [f"**{size:,}**"] + row) + " |")


def make_candidates(n, length=10, seed=42):
    import random

    rng = random.Random(seed)
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    return ["".join(rng.choices(alphabet, k=length)) for _ in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--matrix-max", type=int, default=200)
    args = ap.parse_args()

    # Engine init + shader compilation must not pollute measurements.
    print(f"fuzzgpu {fuzzgpu.__version__} | gpu_info: {fuzzgpu.gpu_info()}")
    fuzzgpu.warmup()
    fuzzgpu.set_cpu_only(False)
    gpu_available = fuzzgpu.is_gpu_available()
    print(f"GPU available: {gpu_available} | repeats: {args.repeats}")
    if gpu_available:
        print(f"GPU: {fuzzgpu.gpu_info()}")

    try:
        import rapidfuzz
        from rapidfuzz import process as rf_process
        from rapidfuzz.distance import Levenshtein as RFLev
        from rapidfuzz.distance import DamerauLevenshtein as RFDam
        from rapidfuzz.distance import JaroWinkler as RFJW
        rf_version = rapidfuzz.__version__
    except ImportError:
        rf_process = RFLev = RFDam = RFJW = None
        rf_version = "not installed"

    try:
        import Levenshtein as pylev
        pylev_version = pylev.__version__
    except ImportError:
        pylev = None
        pylev_version = "not installed"

    print(f"rapidfuzz {rf_version} | python-Levenshtein {pylev_version}")

    query = "hello world"
    sizes = [100, 1_000, 10_000, 50_000]
    cands = {n: make_candidates(n) for n in sizes}

    def fg_gpu(fn, *a):
        if not gpu_available:
            return None
        fuzzgpu.set_cpu_only(False)
        return timed(fn, *a, repeats=args.repeats)

    def fg_cpu(fn, *a):
        fuzzgpu.set_cpu_only(True)
        return timed(fn, *a, repeats=args.repeats)

    def rf_batch(scorer, query_s, cands_s):
        # rapidfuzz process.cdist: 1 query x N candidates, single-threaded C.
        return timed(lambda: rf_process.cdist([query_s], cands_s, scorer=scorer.distance)[0], repeats=args.repeats)

    # ---- Levenshtein batch -------------------------------------------------
    columns = ["`fuzzgpu` (GPU)", "`fuzzgpu` (CPU)", "`rapidfuzz`", "vs RF (GPU)", "vs RF (CPU)"]
    batch_header("Levenshtein Batch (1 query × N candidates, 10-char strings)", columns)
    for n in sizes:
        c = cands[n]
        tg = fg_gpu(fuzzgpu.levenshtein_batch, query, c)
        tc = fg_cpu(fuzzgpu.levenshtein_batch, query, c)
        tr = rf_batch(RFLev, query, c) if RFLev else None
        batch_row(n, [ms(tg) if tg else "N/A", ms(tc), ms(tr) if tr else "N/A",
                      speedup(tr, tg) if tg else "N/A", speedup(tr, tc) if tc else "N/A"])

    # ---- Damerau-Levenshtein batch -----------------------------------------
    # The GPU kernel exists (Lowrance-Wagner SLM kernel, bit-exact) but the
    # backend-aware routing sends it to CPU on integrated GPUs where the SIMD
    # CPU path wins at every measured scale; the GPU column therefore reports
    # the auto-routed result (which is the CPU path on iGPUs).
    columns = ["`fuzzgpu` (GPU)", "`fuzzgpu` (CPU)", "`rapidfuzz`", "vs RF (GPU)", "vs RF (CPU)"]
    batch_header("Damerau-Levenshtein Batch (1 query × N candidates)", columns)
    for n in sizes:
        c = cands[n]
        tg = fg_gpu(fuzzgpu.damerau_levenshtein_batch, query, c)
        tc = fg_cpu(fuzzgpu.damerau_levenshtein_batch, query, c)
        tr = rf_batch(RFDam, query, c) if RFDam else None
        batch_row(n, [ms(tg) if tg else "N/A", ms(tc), ms(tr) if tr else "N/A",
                      speedup(tr, tg) if tg else "N/A", speedup(tr, tc) if tc else "N/A"])

    # ---- Jaro-Winkler batch -------------------------------------------------
    columns = ["`fuzzgpu` (GPU)", "`fuzzgpu` (CPU)", "`rapidfuzz`", "vs RF (GPU)", "vs RF (CPU)"]
    batch_header("Jaro-Winkler Batch (p = 0.1)", columns)
    for n in sizes:
        c = cands[n]
        tg = fg_gpu(fuzzgpu.jaro_winkler_batch, query, c, 0.1)
        tc = fg_cpu(fuzzgpu.jaro_winkler_batch, query, c, 0.1)
        tr = rf_batch(RFJW, query, c) if RFJW else None
        batch_row(n, [ms(tg) if tg else "N/A", ms(tc), ms(tr) if tr else "N/A",
                      speedup(tr, tg) if tg else "N/A", speedup(tr, tc) if tc else "N/A"])

    # ---- Needleman-Wunsch (affine) batch ------------------------------------
    try:
        from rapidfuzz.distance import NeedlemanWunsch as RFNW
    except ImportError:
        RFNW = None
    columns = ["`fuzzgpu` (GPU)", "`fuzzgpu` (CPU)", "`rapidfuzz`", "vs RF (GPU)", "vs RF (CPU)"]
    batch_header("Needleman-Wunsch Batch (affine, match=1, mismatch=-1, gap_open=-2, gap_extend=-1)", columns)
    for n in sizes:
        c = cands[n]
        tg = fg_gpu(fuzzgpu.needleman_wunsch_affine_batch, query, c, 1, -1, -2, -1)
        tc = fg_cpu(fuzzgpu.needleman_wunsch_affine_batch, query, c, 1, -1, -2, -1)
        tr = rf_batch(RFNW, query, c) if RFNW else None
        batch_row(n, [ms(tg) if tg else "N/A", ms(tc), ms(tr) if tr else "N/A",
                      speedup(tr, tg) if tg else "N/A", speedup(tr, tc) if tc else "N/A"])

    # ---- Levenshtein cdist matrix -------------------------------------------
    matrix_sizes = [(10, 10), (50, 50), (100, 100), (200, 200)]
    print("\n### Levenshtein Cross-Product Matrix (`cdist`)")
    print("| Matrix Size | Total Pairs | `fuzzgpu` (GPU) | `fuzzgpu` (CPU) | `rapidfuzz` | python-Levenshtein | vs RF (GPU) | vs RF (CPU) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for (ra, rb) in matrix_sizes:
        if ra > args.matrix_max:
            continue
        a = make_candidates(ra, length=8, seed=7)
        b = make_candidates(rb, length=8, seed=9)
        total = ra * rb
        tg = fg_gpu(fuzzgpu.levenshtein_cdist, a, b)
        tc = fg_cpu(fuzzgpu.levenshtein_cdist, a, b)
        tr = timed(lambda: rf_process.cdist(a, b, scorer=RFLev.distance), repeats=args.repeats) if RFLev else None
        if pylev:
            tp = timed(lambda: [[pylev.distance(x, y) for y in b] for x in a], repeats=3)
        else:
            tp = None
        print("| " + " | ".join([
            f"**{ra} × {rb}**", f"{total:,}", ms(tg) if tg else "N/A", ms(tc),
            ms(tr) if tr else "N/A", ms(tp) if tp else "N/A",
            speedup(tr, tg) if tg else "N/A", speedup(tr, tc) if tc else "N/A",
        ]) + " |")

    fuzzgpu.set_cpu_only(False)


if __name__ == "__main__":
    main()
