"""Benchmark fuzzgpu vs CPU reference."""

import time
import random
import string
import fuzzgpu


def random_string(length=10):
    return "".join(random.choices(string.ascii_lowercase, k=length))


def levenshtein_cpu(a, b):
    """Pure Python Levenshtein for comparison."""
    if len(a) < len(b):
        return levenshtein_cpu(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + cost))
        prev = curr
    return prev[len(b)]


def benchmark_levenshtein():
    print("=" * 60)
    print("Levenshtein Distance Benchmark: fuzzgpu (GPU) vs pure Python")
    print("=" * 60)

    sizes = [100, 500, 1_000, 5_000, 10_000]
    query = "hello world"
    candidates = [random_string(10) for _ in range(10_000)]

    for n in sizes:
        batch = candidates[:n]

        # GPU
        start = time.perf_counter()
        gpu_results = fuzzgpu.levenshtein_batch(query, batch)
        gpu_time = time.perf_counter() - start

        # CPU
        start = time.perf_counter()
        cpu_results = [levenshtein_cpu(query, c) for c in batch]
        cpu_time = time.perf_counter() - start

        # Verify correctness
        assert gpu_results == cpu_results, f"Mismatch at size {n}!"

        speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")
        print(f"  {n:>6,} pairs: GPU={gpu_time:.4f}s  CPU={cpu_time:.4f}s  speedup={speedup:.1f}x")

    print()


def benchmark_cdist():
    print("=" * 60)
    print("CDist Benchmark")
    print("=" * 60)

    n_a, n_b = 100, 100
    list_a = [random_string(8) for _ in range(n_a)]
    list_b = [random_string(8) for _ in range(n_b)]

    start = time.perf_counter()
    matrix = fuzzgpu.levenshtein_cdist(list_a, list_b)
    elapsed = time.perf_counter() - start
    total = n_a * n_b
    print(f"  {n_a}x{n_b} = {total:,} pairs: {elapsed:.4f}s")
    print()


if __name__ == "__main__":
    print(f"fuzzgpu {fuzzgpu.__version__}")
    print(f"GPU: {fuzzgpu.gpu_info()}")
    print()

    benchmark_levenshtein()
    benchmark_cdist()
