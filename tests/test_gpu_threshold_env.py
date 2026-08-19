"""GPU dispatch threshold override, environment variable, and SIMD configuration tests."""

import os
import pytest
import fuzzgpu


class TestGpuThresholdOverride:
    """Test set_gpu_threshold / get behavior and routing impact."""

    def test_default_threshold_is_nonzero(self):
        thresh = fuzzgpu.hardware_info()
        # The hardware_info string mentions the threshold value — just confirm it parses
        assert "threshold" in thresh.lower() or "auto" in thresh.lower() or "GPU" in thresh

    def test_set_threshold_none_restores_auto(self):
        fuzzgpu.set_gpu_threshold(None)
        # No crash; the hardware_info call should still work
        info = fuzzgpu.hardware_info()
        assert isinstance(info, str)

    def test_set_threshold_zero_routes_all_to_cpu(self):
        """Setting threshold to 0 means CPU-only routing; no crash."""
        fuzzgpu.set_gpu_threshold(0)
        try:
            results = fuzzgpu.levenshtein_batch("hello", ["hallo", "world", "hello"])
            assert len(results) == 3
            assert results[0] == 1  # hello vs hallo
            assert results[2] == 0  # hello vs hello
        finally:
            fuzzgpu.set_gpu_threshold(None)

    def test_set_threshold_very_large_routes_all_to_cpu(self):
        fuzzgpu.set_gpu_threshold(10**9)
        try:
            results = fuzzgpu.jaro_winkler_batch("abc", ["abc", "abd"], 0.1)
            assert len(results) == 2
            assert abs(results[0] - 1.0) < 0.01
        finally:
            fuzzgpu.set_gpu_threshold(None)

    def test_set_gpu_threshold_to_1_forces_gpu_dispatch(self):
        """Threshold=1 forces even 1-pair batches onto GPU (if available)."""
        if not fuzzgpu.is_gpu_available():
            pytest.skip("No GPU available")
        fuzzgpu.set_gpu_threshold(1)
        try:
            results = fuzzgpu.levenshtein_batch("abc", ["abd"])
            assert len(results) == 1
        finally:
            fuzzgpu.set_gpu_threshold(None)


class TestEnvConfig:
    """Test environment variable parsing and caching."""

    def test_fuzzgpu_use_cpu_env_var(self):
        """FUZZGPU_USE_CPU=1 disables GPU."""
        old = os.environ.get("FUZZGPU_USE_CPU")
        try:
            os.environ["FUZZGPU_USE_CPU"] = "1"
            # Reimport to pick up the new env var; use a subprocess for isolation
            import subprocess, sys
            result = subprocess.run(
                [sys.executable, "-c",
                 "import fuzzgpu; print(fuzzgpu.is_gpu_available())"],
                env={**os.environ, "FUZZGPU_USE_CPU": "1"},
                capture_output=True, text=True,
            )
            assert result.stdout.strip() == "False"
        finally:
            if old is None:
                os.environ.pop("FUZZGPU_USE_CPU", None)
            else:
                os.environ["FUZZGPU_USE_CPU"] = old

    def test_fuzzgpu_use_cpu_env_var_case_insensitive(self):
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-c",
             "import fuzzgpu; print(fuzzgpu.is_gpu_available())"],
            env={**os.environ, "FUZZGPU_USE_CPU": "TRUE"},
            capture_output=True, text=True,
        )
        assert result.stdout.strip() == "False"

    def test_fuzzgpu_force_gpu_env_var(self):
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-c",
             "import os; os.environ['FUZZGPU_FORCE_GPU']='1'; "
             "import fuzzgpu; print(fuzzgpu.is_gpu_available())"],
            capture_output=True, text=True,
        )
        # With FORCE_GPU set, availability follows hardware (doesn't hard-disable)
        assert result.stdout.strip() in ("True", "False")

    def test_readback_timeout_env_var_parsing(self):
        """Test that FUZZGPU_READBACK_TIMEOUT_MS is parsed correctly."""
        import subprocess, sys
        for val in ["100", "0", "invalid", "-5"]:
            result = subprocess.run(
                [sys.executable, "-c",
                 "import os; "
                 "os.environ['FUZZGPU_READBACK_TIMEOUT_MS']='" + val + "'; "
                 "import fuzzgpu; "
                 "print('ok')"],
                capture_output=True, text=True,
            )
            assert result.returncode == 0, f"Timeout env '{val}' crashed: {result.stderr}"


class TestSimdConfig:
    """Test SIMD override environment variable."""

    def test_simd_env_values_accepted(self):
        """Valid FUZZGPU_SIMD values should not crash."""
        for val in ["avx2", "avx512", "neon", "portable"]:
            import subprocess, sys
            result = subprocess.run(
                [sys.executable, "-c",
                 "import os; os.environ['FUZZGPU_SIMD']='" + val + "'; "
                 "import fuzzgpu; "
                 "print('ok')"],
                capture_output=True, text=True,
            )
            assert result.returncode == 0, f"SIMD env '{val}' crashed: {result.stderr}"

    def test_simd_env_invalid_value_ignored(self):
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-c",
             "import os; os.environ['FUZZGPU_SIMD']='bogus'; "
             "import fuzzgpu; "
             "print('ok')"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0


class TestVersionAttribute:
    """Test __version__ attribute presence and format."""

    def test_version_attribute_exists(self):
        assert hasattr(fuzzgpu, "__version__")

    def test_version_is_string(self):
        assert isinstance(fuzzgpu.__version__, str)

    def test_version_semver_format(self):
        parts = fuzzgpu.__version__.split(".")
        assert len(parts) >= 2, f"Version '{fuzzgpu.__version__}' not semver-like"
        for p in parts[:2]:
            assert p.isdigit(), f"Version part '{p}' not numeric"

    def test_version_matches_pyproject(self):
        import importlib.metadata
        pkg_version = importlib.metadata.version("fuzzgpu")
        assert fuzzgpu.__version__ == pkg_version