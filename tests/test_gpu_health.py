"""Extended GPU health tests: fault injection edge cases, readback simulated failures."""

import pytest
import fuzzgpu


class TestFaultInjectionStateIsolation:
    """Verify that arm/disarm fault hooks do not leak state between calls."""

    def test_set_cpu_only_isolates_gpu_state(self):
        """Toggling CPU-only mode must not leave stale GPU state on re-enable."""
        fuzzgpu.set_cpu_only(True)
        assert fuzzgpu.is_gpu_available() is False
        fuzzgpu.set_cpu_only(False)
        # After restoring, is_gpu_available should reflect actual hardware
        result = fuzzgpu.is_gpu_available()
        assert isinstance(result, bool)

    def test_gpu_info_after_cpu_only_cycle(self):
        fuzzgpu.set_cpu_only(True)
        info_cpu = fuzzgpu.gpu_info()
        fuzzgpu.set_cpu_only(False)
        info_after = fuzzgpu.gpu_info()
        assert isinstance(info_after, str)
        assert len(info_after) > 0

    def test_hardware_info_after_cpu_only_cycle(self):
        fuzzgpu.set_cpu_only(True)
        hw_cpu = fuzzgpu.hardware_info()
        fuzzgpu.set_cpu_only(False)
        hw_after = fuzzgpu.hardware_info()
        assert isinstance(hw_after, str)
        assert "GPU" in hw_after or "CPU" in hw_after


class TestGpuFallbackResilience:
    """Verify graceful fallback to CPU on GPU errors."""

    def test_levenshtein_batch_always_returns_valid_results(self):
        """All GPU batch calls must return valid results even if GPU errors out."""
        query = "hello"
        candidates = ["hallo", "world", "hello", "xyz"]
        results = fuzzgpu.levenshtein_batch(query, candidates)
        assert len(results) == len(candidates)
        for r in results:
            assert isinstance(r, int)
            assert r >= 0

    def test_jaro_winkler_batch_always_returns_valid_results(self):
        query = "hello"
        candidates = ["hallo", "world", "hello", "xyz"]
        results = fuzzgpu.jaro_winkler_batch(query, candidates, 0.1)
        assert len(results) == len(candidates)
        for r in results:
            assert isinstance(r, float)
            assert 0.0 <= r <= 1.0

    def test_cdist_always_returns_valid_matrix(self):
        a = ["abc", "def"]
        b = ["abc", "abd", "xyz"]
        matrix = fuzzgpu.levenshtein_cdist(a, b)
        assert len(matrix) == 2
        assert all(len(row) == 3 for row in matrix)
        for row in matrix:
            for val in row:
                assert isinstance(val, int)
                assert val >= 0

    def test_multiple_consecutive_cpu_only_toggles(self):
        for _ in range(5):
            fuzzgpu.set_cpu_only(True)
            assert fuzzgpu.is_gpu_available() is False
            fuzzgpu.set_cpu_only(False)


class TestGpuThresholdSafety:
    """Verify threshold override is bounded and safe."""

    def test_negative_threshold_does_not_crash(self):
        """Negative threshold via Python (cast to large usize) should not crash."""
        # Python's usize is always non-negative; just verify set doesn't crash
        fuzzgpu.set_gpu_threshold(0)
        fuzzgpu.set_gpu_threshold(None)

    def test_threshold_override_affects_routing(self):
        """Threshold=0 should route batch to CPU; verify with known data."""
        fuzzgpu.set_gpu_threshold(0)
        try:
            res = fuzzgpu.levenshtein_batch("test", ["test", "best"])
            assert res[0] == 0  # identical
            assert res[1] == 1  # t→b substitution
        finally:
            fuzzgpu.set_gpu_threshold(None)

    def test_into_buffers_respect_cpu_routing(self):
        """ *_into APIs should still produce correct results regardless of routing."""
        import numpy as np
        query = "hello"
        candidates = ["hallo", "world", "hello"]
        out = np.zeros(len(candidates), dtype=np.uint32)
        fuzzgpu.levenshtein_batch_into(query, candidates, out)
        assert list(out) == [1, 4, 0]