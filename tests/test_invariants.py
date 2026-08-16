"""Mathematical invariants, metric properties, and cross-implementation equivalence tests."""

import random
import string
import pytest
import fuzzgpu
import fuzzgpu.fuzz as fuzz


def random_string(min_len=1, max_len=30, alphabet=string.ascii_letters + string.digits):
    length = random.randint(min_len, max_len)
    return "".join(random.choices(alphabet, k=length))


class TestMetricInvariants:
    """Test metric space properties: Identity, Non-negativity, Symmetry, and Triangle Inequality."""

    @pytest.mark.parametrize("seed", range(10))
    def test_triangle_inequality_levenshtein(self, seed):
        random.seed(seed)
        for _ in range(50):
            a = random_string()
            b = random_string()
            c = random_string()

            d_ab = fuzzgpu.levenshtein(a, b)
            d_bc = fuzzgpu.levenshtein(b, c)
            d_ac = fuzzgpu.levenshtein(a, c)

            assert d_ac <= d_ab + d_bc, f"Triangle inequality violated: d({a}, {c})={d_ac} > d({a}, {b})={d_ab} + d({b}, {c})={d_bc}"

    @pytest.mark.parametrize("seed", range(10))
    def test_triangle_inequality_damerau(self, seed):
        random.seed(100 + seed)
        for _ in range(50):
            a = random_string()
            b = random_string()
            c = random_string()

            d_ab = fuzzgpu.damerau_levenshtein(a, b)
            d_bc = fuzzgpu.damerau_levenshtein(b, c)
            d_ac = fuzzgpu.damerau_levenshtein(a, c)

            assert d_ac <= d_ab + d_bc, f"Damerau triangle inequality violated: d({a}, {c})={d_ac} > d({a}, {b})={d_ab} + d({b}, {c})={d_bc}"

    def test_symmetry(self):
        random.seed(42)
        for _ in range(100):
            a = random_string()
            b = random_string()

            assert fuzzgpu.levenshtein(a, b) == fuzzgpu.levenshtein(b, a)
            assert fuzzgpu.damerau_levenshtein(a, b) == fuzzgpu.damerau_levenshtein(b, a)
            assert abs(fuzzgpu.jaro_similarity(a, b) - fuzzgpu.jaro_similarity(b, a)) < 1e-9
            assert abs(fuzzgpu.jaro_winkler_similarity(a, b, 0.1) - fuzzgpu.jaro_winkler_similarity(b, a, 0.1)) < 1e-9
            assert abs(fuzz.ratio(a, b) - fuzz.ratio(b, a)) < 1e-9
            assert abs(fuzz.token_sort_ratio(a, b) - fuzz.token_sort_ratio(b, a)) < 1e-9
            assert abs(fuzz.token_set_ratio(a, b) - fuzz.token_set_ratio(b, a)) < 1e-9

    def test_range_bounds(self):
        random.seed(123)
        for _ in range(100):
            a = random_string()
            b = random_string()

            # Distance bounds
            d_lev = fuzzgpu.levenshtein(a, b)
            assert 0 <= d_lev <= max(len(a), len(b))

            d_dam = fuzzgpu.damerau_levenshtein(a, b)
            assert 0 <= d_dam <= max(len(a), len(b))

            # Ratio bounds [0.0, 100.0]
            for fn in [fuzz.ratio, fuzz.partial_ratio, fuzz.token_sort_ratio, fuzz.token_set_ratio, fuzz.WRatio]:
                score = fn(a, b)
                assert 0.0 <= score <= 100.0, f"Score out of bounds for {fn.__name__}: {score}"

            # Jaro / Jaro-Winkler bounds [0.0, 1.0]
            j = fuzzgpu.jaro_similarity(a, b)
            assert 0.0 <= j <= 1.0
            jw = fuzzgpu.jaro_winkler_similarity(a, b, 0.1)
            assert 0.0 <= jw <= 1.0


class TestCrossAlgorithmEquivalence:
    """Test bit-parallel / optimized / GPU implementations against ground truth CPU implementations."""

    def test_myers_vs_standard_levenshtein(self):
        random.seed(999)
        # Test small, medium, and long strings
        for length in [1, 5, 16, 32, 63, 64, 65, 80, 120]:
            for _ in range(20):
                a = random_string(min_len=length, max_len=length)
                b = random_string(min_len=length, max_len=length)
                std = fuzzgpu.levenshtein(a, b)
                myers = fuzzgpu.levenshtein_myers(a, b)
                assert std == myers, f"Myers mismatch for len={length}: std={std}, myers={myers}"

    def test_jaro_optimized_vs_standard_jaro(self):
        random.seed(888)
        for length in [1, 10, 50, 127, 128, 129, 160]:
            for _ in range(20):
                a = random_string(min_len=length, max_len=length)
                b = random_string(min_len=length, max_len=length)
                std = fuzzgpu.jaro_similarity(a, b)
                opt = fuzzgpu.jaro_optimized(a, b)
                assert abs(std - opt) < 1e-4, f"Jaro mismatch for len={length}: std={std}, opt={opt}"

    def test_needleman_wunsch_striped_vs_standard(self):
        random.seed(777)
        for length in [1, 8, 16, 24, 50, 100]:
            for _ in range(10):
                a = random_string(min_len=length, max_len=length, alphabet="ACGT")
                b = random_string(min_len=length, max_len=length, alphabet="ACGT")
                std = fuzzgpu.needleman_wunsch(a, b, 2, -1, -2)
                striped = fuzzgpu.needleman_wunsch_striped(a, b, 2, -1, -2)
                assert std == striped, f"Needleman-Wunsch mismatch for len={length}: std={std}, striped={striped}"

    def test_levenshtein_batch_gpu_vs_cpu_ground_truth(self):
        random.seed(555)
        query = random_string(min_len=10, max_len=30)
        # Create 600 candidates (triggers GPU acceleration)
        candidates = [random_string(min_len=5, max_len=40) for _ in range(600)]

        gpu_batch_res = fuzzgpu.levenshtein_batch(query, candidates)
        cpu_ground_truth = [fuzzgpu.levenshtein(query, c) for c in candidates]

        assert gpu_batch_res == cpu_ground_truth

    def test_jaro_winkler_batch_gpu_vs_cpu_ground_truth(self):
        random.seed(444)
        query = random_string(min_len=10, max_len=30)
        candidates = [random_string(min_len=5, max_len=30) for _ in range(600)]

        gpu_batch_res = fuzzgpu.jaro_winkler_batch(query, candidates, 0.1)
        cpu_ground_truth = [fuzzgpu.jaro_winkler_similarity(query, c, 0.1) for c in candidates]

        for i, (g, c) in enumerate(zip(gpu_batch_res, cpu_ground_truth)):
            assert abs(g - c) < 1e-4, f"Jaro batch mismatch at index {i}: gpu={g}, cpu={c}"

    def test_levenshtein_cdist_gpu_vs_cpu_ground_truth(self):
        random.seed(333)
        # 30 x 30 = 900 pairs (triggers GPU matrix compute)
        list_a = [random_string(min_len=10, max_len=30) for _ in range(30)]
        list_b = [random_string(min_len=10, max_len=30) for _ in range(30)]

        gpu_matrix = fuzzgpu.levenshtein_cdist(list_a, list_b)
        assert len(gpu_matrix) == 30
        assert len(gpu_matrix[0]) == 30

        for i in range(30):
            for j in range(30):
                expected = fuzzgpu.levenshtein(list_a[i], list_b[j])
                assert gpu_matrix[i][j] == expected, f"Matrix mismatch at ({i}, {j}): gpu={gpu_matrix[i][j]}, cpu={expected}"

    def test_jaro_winkler_cdist_gpu_vs_cpu_ground_truth(self):
        random.seed(222)
        # 30 x 30 = 900 pairs
        list_a = [random_string(min_len=10, max_len=30) for _ in range(30)]
        list_b = [random_string(min_len=10, max_len=30) for _ in range(30)]

        gpu_matrix = fuzzgpu.jaro_winkler_cdist(list_a, list_b, 0.1)
        assert len(gpu_matrix) == 30
        assert len(gpu_matrix[0]) == 30

        for i in range(30):
            for j in range(30):
                expected = fuzzgpu.jaro_winkler_similarity(list_a[i], list_b[j], 0.1)
                assert abs(gpu_matrix[i][j] - expected) < 1e-4, f"Jaro matrix mismatch at ({i}, {j}): gpu={gpu_matrix[i][j]}, cpu={expected}"
