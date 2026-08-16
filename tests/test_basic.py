"""Comprehensive tests for fuzzgpu core functionality."""

import fuzzgpu
from fuzzgpu import (
    ratio, partial_ratio, token_sort_ratio, token_set_ratio,
    extract, extractOne, damerau_levenshtein, damerau_ratio,
)


# ── Levenshtein Distance ────────────────────────────────────

class TestLevenshteinDistance:
    def test_basic_cases(self):
        assert fuzzgpu.levenshtein_distance("kitten", "sitting") == 3
        assert fuzzgpu.levenshtein_distance("hello", "hello") == 0
        assert fuzzgpu.levenshtein_distance("abc", "") == 3
        assert fuzzgpu.levenshtein_distance("", "xyz") == 3
        assert fuzzgpu.levenshtein_distance("", "") == 0

    def test_single_character(self):
        assert fuzzgpu.levenshtein_distance("a", "b") == 1
        assert fuzzgpu.levenshtein_distance("a", "a") == 0
        assert fuzzgpu.levenshtein_distance("a", "") == 1

    def test_unicode(self):
        dist = fuzzgpu.levenshtein_distance("café", "cafe")
        assert isinstance(dist, int)
        assert dist > 0

    def test_long_strings(self):
        """Strings > 256 chars: should trigger CPU fallback in GPU mode."""
        a = "a" * 300
        b = "b" * 300
        dist = fuzzgpu.levenshtein_distance(a, b)
        assert dist == 300


class TestLevenshteinBatch:
    def test_basic(self):
        results = fuzzgpu.levenshtein_batch("hello", ["hallo", "hullo", "jello", "hello"])
        assert results == [1, 1, 1, 0]

    def test_empty_candidates(self):
        results = fuzzgpu.levenshtein_batch("hello", [])
        assert results == []

    def test_with_empty_strings(self):
        """Empty strings in batch should not crash or abort GPU dispatch."""
        results = fuzzgpu.levenshtein_batch("hello", ["", "hello", ""])
        assert results == [5, 0, 5]

    def test_large_batch_streaming(self):
        """Large batch > 1000 items to test GPU dispatch & streaming."""
        candidates = ["hello", "world", "help", "helo"] * 300  # 1200 candidates
        results = fuzzgpu.levenshtein_batch("hello", candidates)
        assert len(results) == 1200
        assert results[0] == 0
        assert results[1] == 4
        assert results[2] == 2
        assert results[3] == 1


class TestLevenshteinCdist:
    def test_basic(self):
        matrix = fuzzgpu.levenshtein_cdist(["abc", "def"], ["abc", "axy", "dez"])
        assert matrix[0] == [0, 2, 3]
        assert matrix[1] == [3, 3, 1]


# ── Damerau-Levenshtein ─────────────────────────────────────

class TestDamerauLevenshtein:
    def test_transposition(self):
        # Transposition is 1 in Damerau, but 2 in standard Levenshtein
        assert fuzzgpu.damerau_levenshtein_distance("ab", "ba") == 1
        assert fuzzgpu.levenshtein_distance("ab", "ba") == 2
        assert damerau_levenshtein("ca", "abc") == 2

    def test_identical_and_empty(self):
        assert damerau_levenshtein("", "") == 0
        assert damerau_levenshtein("hello", "hello") == 0
        assert damerau_levenshtein("hello", "") == 5
        assert damerau_levenshtein("", "world") == 5

    def test_batch(self):
        results = fuzzgpu.damerau_levenshtein_batch("ab", ["ba", "ab", "abc", ""])
        assert results == [1, 0, 1, 2]

    def test_cdist(self):
        matrix = fuzzgpu.damerau_levenshtein_cdist(["ab", "cd"], ["ba", "dc"])
        assert matrix[0] == [1, 2]
        assert matrix[1] == [2, 1]

    def test_ratio(self):
        assert damerau_ratio("hello", "hello") == 100.0
        # "ab" vs "ba": len=4, dist=1 -> (4 - 1)/4 * 100 = 75.0%
        r = damerau_ratio("ab", "ba")
        assert abs(r - 75.0) < 0.01


# ── Needleman-Wunsch ────────────────────────────────────────

class TestNeedlemanWunsch:
    def test_basic(self):
        score = fuzzgpu.needleman_wunsch_score("AGTACGCA", "TATGC", 2, -1, -2)
        assert isinstance(score, int)
        assert score == 1

    def test_identical(self):
        """Identical strings should score len × match_score."""
        score = fuzzgpu.needleman_wunsch_score("ACGT", "ACGT", 2, -1, -2)
        assert score == 8  # 4 × 2

    def test_batch(self):
        scores = fuzzgpu.needleman_wunsch_batch_fn("AGTACGCA", ["TATGC", "AGTT"], 2, -1, -2)
        assert len(scores) == 2
        assert all(isinstance(s, int) for s in scores)

    def test_affine(self):
        score_id = fuzzgpu.needleman_wunsch_affine("ACGT", "ACGT", 2, -1, -3, -1)
        assert score_id == 8

        # Affine gap test
        score = fuzzgpu.needleman_wunsch_affine("ACGT", "AT", 2, -1, -3, -1)
        assert isinstance(score, int)

    def test_affine_batch(self):
        scores = fuzzgpu.needleman_wunsch_affine_batch("ACGT", ["ACGT", "AT"], 2, -1, -3, -1)
        assert len(scores) == 2
        assert scores[0] == 8


# ── Jaro / Jaro-Winkler ─────────────────────────────────────

class TestJaro:
    def test_basic(self):
        assert fuzzgpu.jaro_similarity("MARTHA", "MARHTA") > 0.9

    def test_identical(self):
        assert fuzzgpu.jaro_similarity("hello", "hello") == 1.0

    def test_empty(self):
        assert fuzzgpu.jaro_similarity("", "") == 1.0
        assert fuzzgpu.jaro_similarity("hello", "") == 0.0
        assert fuzzgpu.jaro_similarity("", "hello") == 0.0


class TestJaroWinkler:
    def test_basic(self):
        sim = fuzzgpu.jaro_winkler_similarity("MARTHA", "MARHTA", 0.1)
        assert 0.9 < sim <= 1.0

    def test_identical(self):
        assert fuzzgpu.jaro_winkler_similarity("hello", "hello", 0.1) == 1.0

    def test_batch(self):
        results = fuzzgpu.jaro_winkler_batch_fn("MARTHA", ["MARHTA", "MATRH"], 0.1)
        assert len(results) == 2
        assert all(0.0 <= r <= 1.0 for r in results)

    def test_batch_gpu_streaming(self):
        """Large batch > 1000 items to test GPU Jaro-Winkler dispatch."""
        candidates = ["MARHTA", "MATRH", "MARTHA", "OTHER"] * 300  # 1200 items
        results = fuzzgpu.jaro_winkler_batch("MARTHA", candidates, 0.1)
        assert len(results) == 1200
        assert 0.9 < results[0] <= 1.0
        assert results[2] == 1.0

    def test_cdist(self):
        matrix = fuzzgpu.jaro_winkler_cdist(["MARTHA", "HELLO"], ["MARHTA", "WORLD"], 0.1)
        assert len(matrix) == 2
        assert len(matrix[0]) == 2
        assert matrix[0][0] > 0.9


# ── Fuzzy Ratio ──────────────────────────────────────────────

class TestRatio:
    def test_identical(self):
        assert ratio("hello", "hello") == 100.0

    def test_empty(self):
        assert ratio("hello", "") == 0.0
        assert ratio("", "") == 100.0

    def test_sorensen_dice_formula(self):
        """Verify the Sørensen–Dice formula matching RapidFuzz.

        ratio("hello", "hallo"): 4 matching chars out of 5, (2*4)/(5+5) * 100 = 80.0
        """
        r = ratio("hello", "hallo")
        assert abs(r - 80.0) < 0.01, f"Expected 80.0, got {r}"

    def test_asymmetric_formula(self):
        r = ratio("a", "abc")
        assert abs(r - 50.0) < 0.01, f"Expected 50.0, got {r}"


class TestPartialRatio:
    def test_substring_match(self):
        r = partial_ratio("hello", "oh hello there")
        assert r >= 99.9, f"Expected ~100.0, got {r}"

    def test_identical(self):
        assert partial_ratio("hello", "hello") == 100.0

    def test_empty(self):
        assert partial_ratio("hello", "") == 0.0
        assert partial_ratio("", "hello") == 0.0


class TestTokenSortRatio:
    def test_reordered_tokens(self):
        assert token_sort_ratio("b a c", "a b c") == 100.0
        assert token_sort_ratio("hello world", "world hello") == 100.0


class TestTokenSetRatio:
    def test_subset_tokens(self):
        assert token_set_ratio("a b c", "a b c d") == 100.0
        assert token_set_ratio("a b c d", "a b c") == 100.0

    def test_duplicate_tokens(self):
        assert token_set_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear") == 100.0

    def test_partial_overlap(self):
        r = token_set_ratio("a b", "a c")
        assert abs(r - (2.0 / 3.0 * 100.0)) < 0.1, f"Expected 66.67, got {r}"


# ── Extract / ExtractOne ─────────────────────────────────────

class TestExtract:
    def test_basic(self):
        choices = ["apple", "apply", "ape", "banana"]
        results = extract("apple", choices, 50.0, 2)
        assert len(results) == 2
        assert results[0][0] == "apple"
        assert results[0][1] == 100.0

    def test_empty_choices(self):
        results = extract("hello", [], 50.0, 5)
        assert results == []


class TestExtractOne:
    def test_returns_single_tuple(self):
        choices = ["hello", "world", "help"]
        result = extractOne("hellp", choices, 50.0)
        assert result is not None
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 3
        name, score, index = result
        assert isinstance(name, str)
        assert isinstance(score, float)
        assert isinstance(index, int)
        assert name in ("hello", "help")

    def test_returns_none_below_cutoff(self):
        choices = ["zzzzz"]
        result = extractOne("apple", choices, 99.0)
        assert result is None

    def test_returns_best_match(self):
        choices = ["apple", "apply", "ape", "banana"]
        result = extractOne("apple", choices, 50.0)
        assert result is not None
        assert result[0] == "apple"
        assert result[1] == 100.0


# ── GPU Info ─────────────────────────────────────────────────

class TestGpuInfo:
    def test_returns_string(self):
        info = fuzzgpu.gpu_info()
        assert isinstance(info, str)
        assert len(info) > 0


# ── Optimized Algorithm Variants ─────────────────────────────

class TestLevenshteinMyers:
    def test_basic(self):
        assert fuzzgpu.levenshtein_myers("kitten", "sitting") == 3
        assert fuzzgpu.levenshtein_myers("hello", "hello") == 0
        assert fuzzgpu.levenshtein_myers("abc", "") == 3
        assert fuzzgpu.levenshtein_myers("", "xyz") == 3

    def test_consistency_with_standard(self):
        pairs = [
            ("hello", "hallo"),
            ("fuzzgpu", "fizgpu"),
            ("abcdefghij", "jihgfedcba"),
            ("a" * 64, "b" * 64),
        ]
        for a, b in pairs:
            standard = fuzzgpu.levenshtein_distance(a, b)
            myers = fuzzgpu.levenshtein_myers(a, b)
            assert standard == myers, f"Mismatch for ({a!r}, {b!r}): standard={standard}, myers={myers}"


class TestNeedlemanWunschStriped:
    def test_basic(self):
        score = fuzzgpu.needleman_wunsch_striped("AGTACGCA", "TATGC", 2, -1, -2)
        assert isinstance(score, int)


class TestJaroOptimized:
    def test_basic(self):
        sim = fuzzgpu.jaro_optimized("MARTHA", "MARHTA")
        assert 0.9 < sim <= 1.0

    def test_identical(self):
        sim = fuzzgpu.jaro_optimized("hello", "hello")
        assert sim == 1.0


# ── Version ──────────────────────────────────────────────────

class TestVersion:
    def test_version_string(self):
        assert fuzzgpu.__version__ == "0.1.3"
