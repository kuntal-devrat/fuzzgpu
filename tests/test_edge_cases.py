"""Exhaustive edge cases and boundary testing for all fuzzgpu algorithms and kernels."""

import pytest
import fuzzgpu
import fuzzgpu.fuzz as fuzz


class TestEmptyAndSingleCharInputs:
    """Test behavior on completely empty strings, single character strings, and mixed lengths."""

    def test_levenshtein_empty(self):
        assert fuzzgpu.levenshtein("", "") == 0
        assert fuzzgpu.levenshtein("", "a") == 1
        assert fuzzgpu.levenshtein("a", "") == 1
        assert fuzzgpu.levenshtein("", "abcdef") == 6
        assert fuzzgpu.levenshtein("abcdef", "") == 6

    def test_levenshtein_batch_with_empty(self):
        res = fuzzgpu.levenshtein_batch("", ["", "a", "ab", "abc"])
        assert res == [0, 1, 2, 3]

        res2 = fuzzgpu.levenshtein_batch("abc", ["", "abc", ""])
        assert res2 == [3, 0, 3]

    def test_levenshtein_cdist_empty(self):
        assert fuzzgpu.levenshtein_cdist([], []) == []
        assert fuzzgpu.levenshtein_cdist(["a", "b"], []) == []
        assert fuzzgpu.levenshtein_cdist([], ["a", "b"]) == []
        assert fuzzgpu.levenshtein_cdist([""], [""]) == [[0]]
        assert fuzzgpu.levenshtein_cdist(["", "a"], ["", "b"]) == [[0, 1], [1, 1]]

    def test_damerau_empty(self):
        assert fuzzgpu.damerau_levenshtein("", "") == 0
        assert fuzzgpu.damerau_levenshtein("a", "") == 1
        assert fuzzgpu.damerau_levenshtein("", "a") == 1
        assert fuzzgpu.damerau_ratio("", "") == 100.0
        assert fuzzgpu.damerau_ratio("abc", "") == 0.0

    def test_damerau_cdist_empty(self):
        assert fuzzgpu.damerau_levenshtein_cdist([], []) == []
        assert fuzzgpu.damerau_levenshtein_cdist([""], [""]) == [[0]]
        assert fuzzgpu.damerau_levenshtein_cdist(["a", ""], ["", "b"]) == [[1, 1], [0, 1]]

    def test_jaro_and_winkler_empty(self):
        assert fuzzgpu.jaro_similarity("", "") == 1.0
        assert fuzzgpu.jaro_similarity("a", "") == 0.0
        assert fuzzgpu.jaro_similarity("", "a") == 0.0
        assert fuzzgpu.jaro_winkler_similarity("", "", 0.1) == 1.0
        assert fuzzgpu.jaro_winkler_similarity("a", "", 0.1) == 0.0
        assert fuzzgpu.jaro_winkler_similarity("", "a", 0.1) == 0.0

    def test_jaro_batch_and_cdist_empty(self):
        assert fuzzgpu.jaro_winkler_batch("", [], 0.1) == []
        assert fuzzgpu.jaro_winkler_batch("", ["", "a"], 0.1) == [1.0, 0.0]
        assert fuzzgpu.jaro_winkler_cdist([], [], 0.1) == []
        assert fuzzgpu.jaro_winkler_cdist([""], [""], 0.1) == [[1.0]]

    def test_needleman_wunsch_empty(self):
        # Empty vs Empty should score 0
        assert fuzzgpu.needleman_wunsch("", "", 2, -1, -2) == 0
        # Empty vs 3 chars with gap_penalty=-2: 3 * (-2) = -6
        assert fuzzgpu.needleman_wunsch("", "ABC", 2, -1, -2) == -6
        assert fuzzgpu.needleman_wunsch("ABC", "", 2, -1, -2) == -6

        # Affine gap: empty vs empty is 0
        assert fuzzgpu.needleman_wunsch_affine("", "", 2, -1, -3, -1) == 0
        # Affine gap: empty vs 3 chars is gap_open + 3*gap_extend = -3 + 3*(-1) = -6
        assert fuzzgpu.needleman_wunsch_affine("", "ABC", 2, -1, -3, -1) == -6
        assert fuzzgpu.needleman_wunsch_affine("ABC", "", 2, -1, -3, -1) == -6

    def test_fuzzy_ratios_empty(self):
        assert fuzz.ratio("", "") == 100.0
        assert fuzz.ratio("abc", "") == 0.0
        assert fuzz.ratio("", "abc") == 0.0
        assert fuzz.partial_ratio("", "") == 0.0
        assert fuzz.partial_ratio("abc", "") == 0.0
        assert fuzz.partial_ratio("", "abc") == 0.0
        assert fuzz.token_sort_ratio("", "") == 100.0
        assert fuzz.token_set_ratio("", "") == 100.0
        assert fuzz.WRatio("", "") == 100.0

    def test_extract_empty_choices(self):
        assert fuzz.extract("query", [], score_cutoff=0.0, limit=10) == []
        assert fuzz.extractOne("query", [], score_cutoff=0.0) is None
        assert fuzz.extract("", [], score_cutoff=0.0, limit=10) == []
        assert fuzz.extractOne("", [], score_cutoff=0.0) is None

    def test_extract_empty_query(self):
        choices = ["apple", "banana", ""]
        res = fuzz.extract("", choices, score_cutoff=0.0, limit=5)
        assert len(res) == 3
        # Empty string matches empty string with 100.0 score
        assert any(choice == "" and score == 100.0 for choice, score, _ in res)


class TestStringLengthTransitionsAndThresholds:
    """Test boundary transitions for bit-parallel Myers (64 chars), Jaro GPU (128 chars), and Levenshtein GPU (256 chars)."""

    @pytest.mark.parametrize("length", [63, 64, 65, 100])
    def test_myers_bitmask_boundary(self, length):
        """Test exact transitions across the Myers 64-bit threshold."""
        a = "a" * length
        b = "a" * (length - 1) + "b"
        dist_myers = fuzzgpu.levenshtein_myers(a, b)
        dist_std = fuzzgpu.levenshtein(a, b)
        assert dist_myers == dist_std == 1

        # Completely different strings
        c = "x" * length
        assert fuzzgpu.levenshtein_myers(a, c) == fuzzgpu.levenshtein(a, c) == length

    @pytest.mark.parametrize("length", [127, 128, 129, 200])
    def test_jaro_128_char_gpu_threshold(self, length):
        """Test Jaro similarity crossing the GPU 128-char limit."""
        s1 = "MARTHA" * (length // 6) + "A" * (length % 6)
        s2 = "MARHTA" * (length // 6) + "A" * (length % 6)

        sim_cpu = fuzzgpu.jaro_similarity(s1, s2)
        sim_opt = fuzzgpu.jaro_optimized(s1, s2)
        assert abs(sim_cpu - sim_opt) < 1e-4

        # In batch mode (trigger GPU dispatch when candidates >= 500)
        candidates = [s2] * 550
        batch_res = fuzzgpu.jaro_winkler_batch(s1, candidates, 0.1)
        assert len(batch_res) == 550
        assert abs(batch_res[0] - fuzzgpu.jaro_winkler_similarity(s1, s2, 0.1)) < 1e-4

    @pytest.mark.parametrize("length", [255, 256, 257, 350])
    def test_levenshtein_256_char_gpu_threshold(self, length):
        """Test Levenshtein distance crossing the GPU 256-char limit."""
        s1 = "abcdefghij" * (length // 10) + "a" * (length % 10)
        s2 = "abcdefghij" * (length // 10) + "b" * (length % 10)

        dist_cpu = fuzzgpu.levenshtein(s1, s2)
        # GPU Batch dispatch
        batch = [s2] * 550
        res = fuzzgpu.levenshtein_batch(s1, batch)
        assert len(res) == 550
        assert res[0] == dist_cpu

    def test_batch_threshold_transition(self):
        """Test batches below threshold (CPU path) and at/above threshold (GPU path)."""
        query = "kitten"
        candidate = "sitting"
        expected = 3

        # Exactly below threshold: 499
        res_499 = fuzzgpu.levenshtein_batch(query, [candidate] * 499)
        assert len(res_499) == 499
        assert all(d == expected for d in res_499)

        # Exactly at threshold: 500
        res_500 = fuzzgpu.levenshtein_batch(query, [candidate] * 500)
        assert len(res_500) == 500
        assert all(d == expected for d in res_500)

        # Above threshold: 501
        res_501 = fuzzgpu.levenshtein_batch(query, [candidate] * 501)
        assert len(res_501) == 501
        assert all(d == expected for d in res_501)

    def test_matrix_cdist_threshold_transition(self):
        """Test matrix cdist below threshold (<500 pairs) and above threshold (>=500 pairs)."""
        # 10 x 10 = 100 pairs (< 500)
        list_a_small = ["kitten"] * 10
        list_b_small = ["sitting"] * 10
        mat_small = fuzzgpu.levenshtein_cdist(list_a_small, list_b_small)
        assert len(mat_small) == 10
        assert len(mat_small[0]) == 10
        assert mat_small[0][0] == 3

        # 25 x 25 = 625 pairs (>= 500)
        list_a_med = ["kitten"] * 25
        list_b_med = ["sitting"] * 25
        mat_med = fuzzgpu.levenshtein_cdist(list_a_med, list_b_med)
        assert len(mat_med) == 25
        assert len(mat_med[0]) == 25
        assert mat_med[0][0] == 3


class TestUnicodeAndSpecialCharacters:
    """Test full UTF-8 Unicode characters, CJK, Cyrillic, Arabic, Devanagari, Emoji, diacritics, and control chars."""

    def test_cjk_japanese_and_chinese(self):
        a = "こんにちは世界"
        b = "こんばんは世界"
        # Japanese edit distance
        dist = fuzzgpu.levenshtein(a, b)
        assert dist > 0
        r = fuzz.ratio(a, b)
        assert 50.0 < r < 100.0

        # Chinese identical and mutation
        c1 = "自然语言处理与机器学习"
        c2 = "自然语言处理与深度学习"
        assert fuzzgpu.levenshtein(c1, c2) > 0
        assert fuzz.ratio(c1, c1) == 100.0

    def test_cyrillic_and_arabic_and_devanagari(self):
        # Russian / Cyrillic
        assert fuzzgpu.levenshtein("Привет мир", "Привет мир") == 0
        assert fuzzgpu.levenshtein("Привет", "Привит") > 0

        # Arabic
        assert fuzzgpu.levenshtein("مرحبا بالعالم", "مرحبا بالعالم") == 0
        assert fuzz.ratio("مرحبا", "مرحبا") == 100.0

        # Devanagari (Hindi)
        assert fuzzgpu.levenshtein("नमस्ते", "नमस्ते") == 0

    def test_accents_and_diacritics(self):
        assert fuzzgpu.levenshtein("café", "cafe") == 1 or fuzzgpu.levenshtein("café", "cafe") == 2
        assert fuzz.ratio("résumé", "resume") > 50.0
        assert fuzz.ratio("über", "uber") > 50.0
        assert fuzz.ratio("naïve", "naive") > 50.0

    def test_emojis_and_compound_emojis(self):
        e1 = "🚀🌟🎉"
        e2 = "🚀🌟🔥"
        assert fuzzgpu.levenshtein(e1, e1) == 0
        assert fuzzgpu.levenshtein(e1, e2) > 0
        assert fuzz.ratio(e1, e1) == 100.0

        # Compound emoji with zero-width joiners
        fam = "👨‍👩‍👧‍👦"
        assert fuzzgpu.levenshtein(fam, fam) == 0

    def test_complex_whitespace(self):
        s1 = "hello\tworld\nfrom\rfuzzgpu"
        s2 = "hello world from fuzzgpu"
        # token_sort_ratio and token_set_ratio should split on all whitespace identically
        assert fuzz.token_sort_ratio(s1, s2) == 100.0
        assert fuzz.token_set_ratio(s1, s2) == 100.0

        # Multiple consecutive spaces
        s3 = "hello    world"
        s4 = "world   hello"
        assert fuzz.token_sort_ratio(s3, s4) == 100.0

    def test_control_characters_and_null_bytes(self):
        s1 = "hello\x00world"
        s2 = "hello\x00world"
        assert fuzzgpu.levenshtein(s1, s2) == 0
        assert fuzzgpu.levenshtein(s1, "helloworld") > 0


class TestAlgorithmicEdgeScenarios:
    """Test transpositions, homopolymers, Needleman-Wunsch parameters, and extract edge cases."""

    def test_damerau_adjacent_and_non_adjacent_transpositions(self):
        # Adjacent transposition: cost 1
        assert fuzzgpu.damerau_levenshtein("ab", "ba") == 1
        assert fuzzgpu.levenshtein("ab", "ba") == 2

        # Non-adjacent transposition (Wagner-Lowrance): "ca" vs "abc" -> 2
        assert fuzzgpu.damerau_levenshtein("ca", "abc") == 2

    def test_homopolymer_runs(self):
        a = "A" * 500
        b = "A" * 499
        assert fuzzgpu.levenshtein(a, b) == 1
        assert fuzz.ratio(a, b) > 99.5

    def test_needleman_wunsch_extreme_parameters(self):
        # Zero match and zero gap
        assert fuzzgpu.needleman_wunsch("AAAA", "AAAA", 0, 0, 0) == 0

        # Negative match score
        score_neg = fuzzgpu.needleman_wunsch("AAAA", "AAAA", -1, -5, -2)
        assert score_neg == -4

        # Very high gap penalties
        score_high_gap = fuzzgpu.needleman_wunsch("A", "AAAA", 2, -1, -100)
        assert score_high_gap < 0

    def test_jaro_winkler_prefix_scales(self):
        # p = 0.0 (reduces to pure Jaro)
        s1, s2 = "MARTHA", "MARHTA"
        jw_0 = fuzzgpu.jaro_winkler_similarity(s1, s2, 0.0)
        j_pure = fuzzgpu.jaro_similarity(s1, s2)
        assert abs(jw_0 - j_pure) < 1e-6

        # p = 0.25 (maximum recommended prefix scale)
        jw_25 = fuzzgpu.jaro_winkler_similarity(s1, s2, 0.25)
        assert jw_25 >= jw_0

    def test_extract_cutoff_and_limit_bounds(self):
        choices = ["apple", "apply", "ape", "banana", "cherry"]

        # limit = 0 -> returns empty list
        assert fuzz.extract("apple", choices, score_cutoff=0.0, limit=0) == []

        # limit > choices length -> returns all choices matching cutoff
        res_all = fuzz.extract("apple", choices, score_cutoff=0.0, limit=100)
        assert len(res_all) == 5

        # score_cutoff > 100.0 -> returns empty list
        assert fuzz.extract("apple", choices, score_cutoff=101.0, limit=5) == []
        assert fuzz.extractOne("apple", choices, score_cutoff=101.0) is None

        # score_cutoff = 100.0 -> returns only exact matches
        res_exact = fuzz.extract("apple", choices, score_cutoff=100.0, limit=5)
        assert len(res_exact) == 1
        assert res_exact[0][0] == "apple"
