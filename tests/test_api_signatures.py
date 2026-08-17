"""Tests validating API completeness, function aliases, signatures, keyword arguments, and error handling."""

import pytest
import fuzzgpu
import fuzzgpu.fuzz as fuzz


class TestApiSurface:
    """Verify all expected functions, aliases, and attributes exist and are callable."""

    def test_top_level_exports(self):
        expected_exports = [
            "levenshtein_distance", "levenshtein_batch", "levenshtein_cdist",
            "damerau_levenshtein_distance", "damerau_levenshtein_batch", "damerau_levenshtein_cdist", "damerau_ratio",
            "needleman_wunsch_score", "needleman_wunsch_batch_fn", "needleman_wunsch_affine", "needleman_wunsch_affine_batch",
            "jaro_similarity", "jaro_winkler_similarity", "jaro_winkler_batch_fn", "jaro_winkler_cdist",
            "fuzz_ratio", "fuzz_partial_ratio", "fuzz_token_sort_ratio", "fuzz_token_set_ratio",
            "fuzz_wratio", "fuzz_ratio_batch", "fuzz_extract", "fuzz_extract_one",
            "levenshtein_myers", "needleman_wunsch_striped", "jaro_optimized",
            "gpu_info", "__version__",
            # Clean Aliases
            "levenshtein", "damerau_levenshtein", "needleman_wunsch", "needleman_wunsch_batch",
            "jaro_winkler_batch", "ratio", "partial_ratio", "token_sort_ratio", "token_set_ratio",
            "wratio", "ratio_batch", "extract", "extractOne",
        ]
        for name in expected_exports:
            assert hasattr(fuzzgpu, name), f"Missing export in fuzzgpu: {name}"

    def test_fuzz_submodule_exports(self):
        expected_in_fuzz = [
            "ratio", "partial_ratio", "token_sort_ratio", "token_set_ratio",
            "WRatio", "ratio_batch", "extract", "extractOne", "damerau_ratio",
        ]
        for name in expected_in_fuzz:
            assert hasattr(fuzz, name), f"Missing export in fuzzgpu.fuzz: {name}"

    def test_version_validity(self):
        assert isinstance(fuzzgpu.__version__, str)
        parts = fuzzgpu.__version__.split(".")
        assert len(parts) >= 2
        assert all(p.isdigit() for p in parts[:2])

    def test_gpu_info_returns_nonempty_string(self):
        info = fuzzgpu.gpu_info()
        assert isinstance(info, str)
        assert len(info.strip()) > 0


class TestKeywordAndPositionalArgs:
    """Verify functions can be called with keyword arguments and positional arguments."""

    def test_levenshtein(self):
        assert fuzzgpu.levenshtein("kitten", "sitting") == 3
        assert fuzzgpu.levenshtein(a="kitten", b="sitting") == 3
        assert fuzzgpu.levenshtein_distance("kitten", "sitting") == 3

    def test_levenshtein_batch(self):
        res_pos = fuzzgpu.levenshtein_batch("test", ["best", "rest"])
        res_kw = fuzzgpu.levenshtein_batch(query="test", candidates=["best", "rest"])
        assert res_pos == res_kw == [1, 1]

    def test_levenshtein_cdist(self):
        res_pos = fuzzgpu.levenshtein_cdist(["a"], ["b"])
        res_kw = fuzzgpu.levenshtein_cdist(list_a=["a"], list_b=["b"])
        assert res_pos == res_kw == [[1]]

    def test_damerau_levenshtein(self):
        assert fuzzgpu.damerau_levenshtein("ca", "abc") == 2
        assert fuzzgpu.damerau_levenshtein(a="ca", b="abc") == 2
        assert fuzzgpu.damerau_levenshtein_distance("ca", "abc") == 2

    def test_damerau_batch(self):
        res = fuzzgpu.damerau_levenshtein_batch(query="ab", candidates=["ba", "ab"])
        assert res == [1, 0]

    def test_damerau_cdist(self):
        res = fuzzgpu.damerau_levenshtein_cdist(list_a=["ab"], list_b=["ba", "ab"])
        assert res == [[1, 0]]

    def test_damerau_ratio(self):
        res = fuzzgpu.damerau_ratio(a="hello", b="hello")
        assert res == 100.0

    def test_needleman_wunsch(self):
        res_pos = fuzzgpu.needleman_wunsch("ACGT", "ACGT", 2, -1, -2)
        res_kw = fuzzgpu.needleman_wunsch(a="ACGT", b="ACGT", match_score=2, mismatch_score=-1, gap_penalty=-2)
        assert res_pos == res_kw == 8

    def test_needleman_wunsch_batch(self):
        res = fuzzgpu.needleman_wunsch_batch(
            query="ACGT",
            candidates=["ACGT", "AT"],
            match_score=2,
            mismatch_score=-1,
            gap_penalty=-2
        )
        assert res[0] == 8

    def test_needleman_wunsch_affine(self):
        res = fuzzgpu.needleman_wunsch_affine(
            a="ACGT",
            b="ACGT",
            match_score=2,
            mismatch_score=-1,
            gap_open=-3,
            gap_extend=-1
        )
        assert res == 8

    def test_needleman_wunsch_affine_batch(self):
        res = fuzzgpu.needleman_wunsch_affine_batch(
            query="ACGT",
            candidates=["ACGT", "AT"],
            match_score=2,
            mismatch_score=-1,
            gap_open=-3,
            gap_extend=-1
        )
        assert res[0] == 8

    def test_jaro_and_winkler(self):
        assert fuzzgpu.jaro_similarity(a="MARTHA", b="MARHTA") > 0.9
        assert fuzzgpu.jaro_winkler_similarity(a="MARTHA", b="MARHTA", p=0.1) > 0.9
        assert fuzzgpu.jaro_winkler_batch(query="MARTHA", candidates=["MARHTA"], p=0.1)[0] > 0.9
        assert fuzzgpu.jaro_winkler_cdist(list_a=["MARTHA"], list_b=["MARHTA"], p=0.1)[0][0] > 0.9

    def test_fuzzy_matchers(self):
        assert fuzzgpu.ratio(a="apple", b="apple") == 100.0
        assert fuzzgpu.partial_ratio(a="apple", b="apple pie") == 100.0
        assert fuzzgpu.token_sort_ratio(a="apple banana", b="banana apple") == 100.0
        assert fuzzgpu.token_set_ratio(a="apple banana", b="apple banana orange") == 100.0
        assert fuzzgpu.wratio(a="apple banana", b="banana apple") == 95.0
        r_batch = fuzzgpu.ratio_batch(query="apple", candidates=["apple", "aple"])
        assert r_batch[0] == 100.0
        assert abs(r_batch[1] - 88.88888888888889) < 0.01

    def test_extract_and_extract_one(self):
        choices = ["apple", "apply", "banana"]
        ext = fuzzgpu.extract(query="apple", choices=choices, score_cutoff=50.0, limit=2)
        assert len(ext) == 2
        assert ext[0] == ("apple", 100.0, 0)

        ext_one = fuzzgpu.extractOne(query="apple", choices=choices, score_cutoff=50.0)
        assert ext_one == ("apple", 100.0, 0)

    def test_simd_optimized_variants(self):
        assert fuzzgpu.levenshtein_myers(a="kitten", b="sitting") == 3
        assert fuzzgpu.needleman_wunsch_striped(a="ACGT", b="ACGT", match_score=2, mismatch_score=-1, gap_penalty=-2) == 8
        assert fuzzgpu.jaro_optimized(a="MARTHA", b="MARHTA") > 0.9


class TestTypeSafetyAndErrorHandling:
    """Verify that type errors and invalid arguments are caught gracefully without crashes."""

    def test_none_input_raises_type_error(self):
        with pytest.raises(TypeError):
            fuzzgpu.levenshtein(None, "abc")
        with pytest.raises(TypeError):
            fuzzgpu.levenshtein("abc", None)

    def test_numeric_input_raises_type_error(self):
        with pytest.raises(TypeError):
            fuzzgpu.levenshtein(123, "123")

    def test_batch_with_non_iterable_raises_type_error(self):
        with pytest.raises(TypeError):
            fuzzgpu.levenshtein_batch("query", 123)

    def test_extract_with_non_string_choices_raises_type_error(self):
        with pytest.raises(TypeError):
            fuzzgpu.extract("query", [1, 2, 3], score_cutoff=50.0, limit=5)
