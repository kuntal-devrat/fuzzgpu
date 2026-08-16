"""Tests for the zero-allocation ``*_into`` API.

The ``*_into`` functions write results into a caller-supplied preallocated
numpy array instead of boxing every result as a Python int/float. This module
pins three properties:

1. **Parity** — ``*_into`` writes exactly what the plain ``*_batch`` /
   ``*_cdist`` functions return, under both GPU-enabled auto-routing and
   CPU-only mode.
2. **Validation** — wrong length / shape / dtype / read-only arrays fail fast
   (before any compute) with clear exceptions.
3. **In-place semantics** — results land in the caller's buffer; the function
   returns ``None``.
"""

import numpy as np
import pytest

import fuzzgpu


def test_levenshtein_batch_into_matches_batch():
    query = "hello"
    cands = ["hallo", "hullo", "help", "world", "", "hello"]
    out = np.full(len(cands), 99, dtype=np.uint32)
    ret = fuzzgpu.levenshtein_batch_into(query, cands, out)
    assert ret is None
    np.testing.assert_array_equal(
        out, np.array(fuzzgpu.levenshtein_batch(query, cands), dtype=np.uint32)
    )


def test_levenshtein_batch_into_gpu_and_cpu_paths():
    query = "benchmark-query-string"
    cands = ["".join(chr(97 + (i * 7) % 26)) * 8 for i in range(2000)]
    out = np.zeros(len(cands), dtype=np.uint32)
    # GPU-enabled (auto-routed; on an iGPU the CPU SIMD path is used).
    fuzzgpu.levenshtein_batch_into(query, cands, out)
    gpu = out.copy()
    # CPU-only.
    fuzzgpu.set_cpu_only(True)
    fuzzgpu.levenshtein_batch_into(query, cands, out)
    fuzzgpu.set_cpu_only(False)
    np.testing.assert_array_equal(gpu, out)


def test_damerau_batch_into_matches_batch():
    query = "ca"
    cands = ["abc", "ba", "a cat", "sitting"]
    out = np.zeros(len(cands), dtype=np.uint32)
    fuzzgpu.damerau_levenshtein_batch_into(query, cands, out)
    np.testing.assert_array_equal(
        out,
        np.array(fuzzgpu.damerau_levenshtein_batch(query, cands), dtype=np.uint32),
    )
    # Non-adjacent transposition: ca -> abc is 2 (true Damerau, not OSA).
    assert out[0] == 2


def test_jaro_winkler_batch_into_matches_batch():
    query = "MARTHA"
    cands = ["MARHTA", "MARTHA", "dwayne", "", "x"]
    out = np.zeros(len(cands), dtype=np.float64)
    fuzzgpu.jaro_winkler_batch_into(query, cands, out, 0.1)
    expected = np.array(fuzzgpu.jaro_winkler_batch(query, cands, 0.1))
    np.testing.assert_allclose(out, expected, atol=1e-9)


def test_levenshtein_cdist_into_matches_cdist():
    a = ["abc", "def", ""]
    b = ["abc", "xyz", "abcdef"]
    out = np.zeros((len(a), len(b)), dtype=np.uint32)
    fuzzgpu.levenshtein_cdist_into(a, b, out)
    expected = np.array(fuzzgpu.levenshtein_cdist(a, b), dtype=np.uint32)
    np.testing.assert_array_equal(out, expected)


def test_damerau_cdist_into_matches_cdist():
    a = ["ab", "cd"]
    b = ["ba", "dc", "ab"]
    out = np.zeros((len(a), len(b)), dtype=np.uint32)
    fuzzgpu.damerau_levenshtein_cdist_into(a, b, out)
    expected = np.array(fuzzgpu.damerau_levenshtein_cdist(a, b), dtype=np.uint32)
    np.testing.assert_array_equal(out, expected)


def test_jaro_winkler_cdist_into_matches_cdist():
    a = ["MARTHA", "dixon"]
    b = ["MARHTA", "dicksonx", ""]
    out = np.zeros((len(a), len(b)), dtype=np.float64)
    fuzzgpu.jaro_winkler_cdist_into(a, b, out, 0.1)
    expected = np.array(fuzzgpu.jaro_winkler_cdist(a, b, 0.1))
    np.testing.assert_allclose(out, expected, atol=1e-9)


def test_large_batch_into_matches_batch():
    rng = np.random.default_rng(7)
    cands = [
        "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz"), size=12))
        for _ in range(100_000)
    ]
    out = np.zeros(len(cands), dtype=np.uint32)
    fuzzgpu.levenshtein_batch_into("helloworld", cands, out)
    np.testing.assert_array_equal(
        out, np.array(fuzzgpu.levenshtein_batch("helloworld", cands), dtype=np.uint32)
    )


def test_batch_into_wrong_length_raises_value_error():
    out = np.zeros(3, dtype=np.uint32)
    with pytest.raises(ValueError, match="exactly 4 uint32"):
        fuzzgpu.levenshtein_batch_into("ab", ["a", "b", "c", "d"], out)


def test_batch_into_wrong_dtype_raises_type_error():
    out = np.zeros(2, dtype=np.float64)
    with pytest.raises(TypeError):
        fuzzgpu.levenshtein_batch_into("ab", ["a", "b"], out)
    out_u32 = np.zeros(2, dtype=np.uint32)
    with pytest.raises(TypeError):
        fuzzgpu.jaro_winkler_batch_into("ab", ["a", "b"], out_u32)  # wants float64


def test_batch_into_readonly_raises():
    out = np.zeros(4, dtype=np.uint32)
    out.flags.writeable = False
    with pytest.raises(Exception):  # BufferError
        fuzzgpu.levenshtein_batch_into("ab", ["a", "b", "c", "d"], out)


def test_cdist_into_wrong_shape_raises():
    a = ["ab", "cd"]
    b = ["x", "y", "z"]
    out = np.zeros((2, 2), dtype=np.uint32)  # wrong: should be (2, 3)
    with pytest.raises(ValueError, match=r"shape \(2, 3\)"):
        fuzzgpu.levenshtein_cdist_into(a, b, out)


def test_cdist_into_non_contiguous_raises():
    a = ["ab", "cd"]
    b = ["x", "y"]
    out = np.zeros((4, 2), dtype=np.uint32)[::2, :]  # strided view
    with pytest.raises(Exception):
        fuzzgpu.levenshtein_cdist_into(a, b, out)


def test_jaro_batch_into_invalid_p_raises():
    out = np.zeros(1, dtype=np.float64)
    with pytest.raises(ValueError):
        fuzzgpu.jaro_winkler_batch_into("ab", ["a"], out, 0.5)


def test_into_does_not_mutate_on_error():
    query = "ab"
    cands = ["a", "b", "c"]
    out = np.zeros(5, dtype=np.uint32)  # wrong length
    before = out.copy()
    with pytest.raises(ValueError):
        fuzzgpu.levenshtein_batch_into(query, cands, out)
    np.testing.assert_array_equal(out, before)
