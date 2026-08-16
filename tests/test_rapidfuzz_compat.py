from fuzzgpu import fuzz, process
from fuzzgpu.distance import Hamming, Indel, Levenshtein, OSA


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
