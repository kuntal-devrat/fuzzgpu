from fuzzgpu import fuzz, process
from fuzzgpu.distance import Hamming, Indel, Jaro, Levenshtein, OSA


def test_editops_and_opcodes_follow_an_optimal_alignment():
    assert Levenshtein.editops("kitten", "sitting") == [
        ("replace", 0, 0), ("replace", 4, 4), ("insert", 6, 6),
    ]
    assert Levenshtein.opcodes("abcXXXdef", "abcYYYdef") == [
        ("equal", 0, 3, 0, 3), ("replace", 3, 6, 3, 6), ("equal", 6, 9, 6, 9),
    ]


def test_editops_handles_nontrivial_inputs_without_recursive_blowup():
    left, right = "a" * 80 + "b", "a" * 80 + "c"
    assert Levenshtein.editops(left, right) == [("replace", 80, 80)]


def test_process_scorer_processor_and_lazy_results():
    choices = ["New York", "Yorkshire", "Boston"]
    assert process.extractOne("new york", choices, scorer=fuzz.ratio, processor=str.lower) == ("New York", 100.0, 0)
    assert list(process.extract_iter("new york", choices, processor=str.lower, score_cutoff=90)) == [("New York", 100.0, 0)]


def test_additional_distance_metrics():
    assert Hamming.distance("abc", "ab") == 1
    assert OSA.distance("ab", "ba") == 1
    assert Indel.distance("abc", "adc") == 2


def test_process_handles_iterators_distance_scorers_and_plain_callables():
    assert process.cdist(["a", "b"], iter(["a", "b"])) == [[100.0, 0.0], [0.0, 100.0]]
    assert process.extractOne("kitten", ["sitting", "kitten"], scorer=Levenshtein.distance) == ("kitten", 0, 1)
    assert process.extractOne("ABC", ["abc"], scorer=lambda a, b: float(a == b), processor=str.lower) == ("abc", 1.0, 0)


def test_distance_cutoffs_do_not_corrupt_similarity_or_jaro_distance():
    assert Hamming.similarity("abc", "xyz", score_cutoff=1) == 0
    assert OSA.similarity("abc", "xyz", score_cutoff=1) == 0
    assert Indel.similarity("abc", "xyz", score_cutoff=1) == 0
    expected = 1.0 - Jaro.similarity("MARTHA", "MARHTA")
    assert Jaro.distance("MARTHA", "MARHTA", score_cutoff=0.99) == expected
    assert Levenshtein.similarity(" a ", "a", processor=str.strip) == 1
