"""High-load stress, scalability, long string, and sustained memory stability testing."""

import os
import random
import string
import time
import pytest
import fuzzgpu
import fuzzgpu.fuzz as fuzz

# Wall-clock budgets are machine-dependent; allow CI/slow runners to disable
# the strict timing asserts (correctness checks below still run).
_PERF_ASSERTS = os.environ.get("FUZZGPU_SKIP_PERF_ASSERTS", "").lower() not in ("1", "true", "yes")


def _assert_perf(elapsed, budget, label):
    if _PERF_ASSERTS:
        assert elapsed < budget, f"{label} took {elapsed:.2f}s (too slow)"


def random_string(length, alphabet=string.ascii_lowercase):
    return "".join(random.choices(alphabet, k=length))


class TestStressAndScalability:
    """Stress test the GPU and CPU engines under large datasets and extreme loads."""

    def test_large_batch_levenshtein_50k(self):
        """Stress test with 50,000 candidate strings in a single batch."""
        random.seed(101)
        query = "database_query_optimizer"
        candidates = [
            f"database_query_{random_string(6)}" for _ in range(50_000)
        ]
        t0 = time.perf_counter()
        results = fuzzgpu.levenshtein_batch(query, candidates)
        elapsed = time.perf_counter() - t0

        assert len(results) == 50_000
        assert all(isinstance(r, int) for r in results[:100])
        # Verify throughput is fast
        _assert_perf(elapsed, 10.0, "50k batch")

    def test_large_batch_jaro_winkler_50k(self):
        """Stress test Jaro-Winkler GPU kernel with 50,000 candidate strings."""
        random.seed(102)
        query = "distributed_gpu_cluster"
        candidates = [
            f"distributed_gpu_{random_string(6)}" for _ in range(50_000)
        ]
        t0 = time.perf_counter()
        results = fuzzgpu.jaro_winkler_batch(query, candidates, 0.1)
        elapsed = time.perf_counter() - t0

        assert len(results) == 50_000
        assert all(0.0 <= r <= 1.0 for r in results[:100])
        _assert_perf(elapsed, 10.0, "50k Jaro batch")

    def test_large_matrix_cdist_500x500(self):
        """Cross-product 500 x 500 matrix = 250,000 distance evaluations on GPU."""
        random.seed(103)
        list_a = [f"item_a_{random_string(8)}" for _ in range(500)]
        list_b = [f"item_b_{random_string(8)}" for _ in range(500)]

        t0 = time.perf_counter()
        matrix = fuzzgpu.levenshtein_cdist(list_a, list_b)
        elapsed = time.perf_counter() - t0

        assert len(matrix) == 500
        assert len(matrix[0]) == 500
        _assert_perf(elapsed, 10.0, "500x500 matrix")

    def test_large_matrix_cdist_1000x1000(self):
        """Cross-product 1000 x 1000 matrix = 1,000,000 distance evaluations on GPU."""
        random.seed(104)
        list_a = [f"row_{random_string(6)}" for _ in range(1000)]
        list_b = [f"col_{random_string(6)}" for _ in range(1000)]

        t0 = time.perf_counter()
        matrix = fuzzgpu.levenshtein_cdist(list_a, list_b)
        elapsed = time.perf_counter() - t0

        assert len(matrix) == 1000
        assert len(matrix[0]) == 1000
        # Spot check corners and center
        assert matrix[0][0] == fuzzgpu.levenshtein(list_a[0], list_b[0])
        assert matrix[500][500] == fuzzgpu.levenshtein(list_a[500], list_b[500])
        assert matrix[999][999] == fuzzgpu.levenshtein(list_a[999], list_b[999])
        _assert_perf(elapsed, 15.0, "1000x1000 matrix")

    def test_ultra_long_strings_cpu_fallback(self):
        """Test very long strings (up to 10,000 chars) for memory and CPU DP stability."""
        random.seed(105)
        long_a = random_string(5000)
        long_b = long_a[:2500] + random_string(100) + long_a[2600:]

        dist = fuzzgpu.levenshtein(long_a, long_b)
        assert isinstance(dist, int)
        assert dist > 0

        # Needleman-Wunsch on 1,000 char strings
        s1 = random_string(1000, alphabet="ACGT")
        s2 = random_string(1000, alphabet="ACGT")
        score = fuzzgpu.needleman_wunsch(s1, s2, 2, -1, -2)
        assert isinstance(score, int)

    def test_sustained_gpu_dispatch_loop(self):
        """Execute 50 rapid consecutive GPU dispatches to test buffer recycling and memory stability."""
        query = "memory_leak_stress_test"
        candidates = ["memory_leak_stress_candidate"] * 1000
        expected = fuzzgpu.levenshtein(query, candidates[0])

        for i in range(50):
            res = fuzzgpu.levenshtein_batch(query, candidates)
            assert len(res) == 1000
            assert res[0] == expected

    def test_large_extract_choices_20k(self):
        """Top-K extraction across 20,000 choices."""
        random.seed(106)
        query = "target_candidate_string"
        choices = [f"candidate_{random_string(10)}" for _ in range(20_000)]
        choices[12_345] = "target_candidate_string"  # exact match inserted

        t0 = time.perf_counter()
        top_k = fuzz.extract(query, choices, score_cutoff=80.0, limit=5)
        elapsed = time.perf_counter() - t0

        assert len(top_k) >= 1
        assert top_k[0][0] == "target_candidate_string"
        assert top_k[0][1] == 100.0
        assert top_k[0][2] == 12_345
        _assert_perf(elapsed, 5.0, "Extract over 20k items")
