from fuzzgpu import fuzz, process
from fuzzgpu.distance import (
    Editop,
    Editops,
    Hamming,
    Indel,
    Jaro,
    Levenshtein,
    LCSseq,
    MatchingBlock,
    Opcode,
    Opcodes,
    OSA,
    Postfix,
    Prefix,
    ScoreAlignment,
)


def test_editops_and_opcodes_follow_an_optimal_alignment():
    assert list(Levenshtein.editops("kitten", "sitting")) == [
        ("replace", 0, 0), ("replace", 4, 4), ("insert", 6, 6),
    ]
    assert list(Levenshtein.opcodes("abcXXXdef", "abcYYYdef")) == [
        ("equal", 0, 3, 0, 3), ("replace", 3, 6, 3, 6), ("equal", 6, 9, 6, 9),
    ]


def test_editops_handles_nontrivial_inputs_without_recursive_blowup():
    left, right = "a" * 80 + "b", "a" * 80 + "c"
    assert list(Levenshtein.editops(left, right)) == [("replace", 80, 80)]


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


# ── partial_ratio_alignment: cutoffs, empty, swap, imprecision bound ─────────

def test_partial_ratio_alignment_returns_scorealignment():
    align = fuzz.partial_ratio_alignment("hello", "oh hello there")
    assert isinstance(align, ScoreAlignment)
    assert align.score == 100.0
    assert align.src_start == 0 and align.src_end == 5
    assert align.dest_start == 3 and align.dest_end == 8
    assert tuple(align) == (100.0, 0, 5, 3, 8)


def test_partial_ratio_alignment_empty_sentinel_swap():
    # rapidfuzz returns ScoreAlignment(100, 0, 0, 0, 0) for both empty args
    assert fuzz.partial_ratio_alignment("", "") == ScoreAlignment(100.0, 0, 0, 0, 0)
    # a zero-length arg on either side gives ScoreAlignment(0, 0, len, 0, len)
    a = fuzz.partial_ratio_alignment("abc", "")
    b = fuzz.partial_ratio_alignment("", "abc")
    assert a.score == 0.0 and b.score == 0.0


def test_partial_ratio_imprecision_bound_keeps_ties():
    # NormSim_to_NormDist has a +1e-5 imprecision term that lets branch-and-bound
    # admit ties at exact cutoffs; verify pairs whose best partial score is >= 50
    # stay above the cutoff (rapidfuzz-exact).
    for a, b in [("park", "ba"), ("ba", "park"), ("hallo", "ab")]:
        score0 = fuzz.partial_ratio(a, b, score_cutoff=0.0)
        score50 = fuzz.partial_ratio(a, b, score_cutoff=50.0)
        assert score0 >= 50.0 and score50 == score0, (a, b, score0, score50)


# ── Empty/symmetric parity for the new token functions ───────────────────────

def test_token_ratio_and_partial_token_ratio_empty_semantics():
    # Per rapidfuzz FuzzyWuzzy parity: empty/empty -> 100, anything with empty -> 0
    assert fuzz.token_ratio("", "") == 100.0
    assert fuzz.token_ratio("abc", "") == 0.0
    assert fuzz.partial_token_ratio("", "") == 100.0
    assert fuzz.partial_token_ratio("abc", "") == 0.0
    assert fuzz.WRatio("", "") == 0.0
    assert fuzz.QRatio("", "") == 0.0


def test_token_set_empty_semantics():
    # token_set_ratio returns 0 on any empty per rapidfuzz issue #110
    assert fuzz.token_set_ratio("", "") == 0.0
    assert fuzz.token_set_ratio("abc", "") == 0.0
    assert fuzz.partial_token_set_ratio("", "") == 0.0


# ── process: default scorer is WRatio, matching rapidfuzz ────────────────────

def test_process_default_scorer_is_wratio():
    choices = ["new york mets", "new york yankees", "yankees"]
    out = process.extractOne("new york mets", choices)
    # rapidfuzz WRatio gives 100.0 for this exact substring match
    assert out[0] == "new york mets" and out[1] == 100.0


def test_process_extract_with_wratio_matches_rapidfuzz():
    choices = ["new york mets vs atlanta braves", "atlanta braves vs new york mets"]
    from rapidfuzz import process as rp
    expected = rp.extractOne("new york mets", choices)
    got = process.extractOne("new york mets", choices)
    assert got == expected


# ── Distance modules: LCSseq / Prefix / Postfix values + editops ─────────────

def test_lcsseq_values_match_formulas():
    # LCSseq.distance = max - LCS; similarity = LCS; normalized by max.
    assert LCSseq.distance("abcd", "abxd") == 1
    assert LCSseq.similarity("abcd", "abxd") == 3
    assert LCSseq.distance("abc", "abcde") == 2
    assert LCSseq.distance("abcde", "abc") == 2
    # editops use LCS Myer's bit-parallel matrix: delete checked first
    assert list(LCSseq.editops("abcd", "abxd")) == [
        ("insert", 2, 2), ("delete", 2, 3),
    ]
    assert list(Indel.editops("abcd", "abxd")) == list(LCSseq.editops("abcd", "abxd"))


def test_prefix_postfix_values_and_editops():
    assert Prefix.distance("ab", "abc") == 1
    assert Prefix.similarity("ab", "abc") == 2
    # Use the rapidfuzz-exact float (1 - 1/3 = 0.6666666666666667), not the
    # Python `2/3` literal which differs by 1 ulp from the C++ computation order.
    assert Prefix.normalized_similarity("ab", "abc") == 1.0 - 1.0 / 3.0
    assert Postfix.distance("ba", "cba") == 1
    assert Postfix.similarity("ba", "cba") == 2
    # No editops/opcodes in rapidfuzz Prefix/Postfix (there is no public API for it)


def test_hamming_editops_padding_model():
    # rapidfuzz Hamming.editops: replace per mismatched pos in min_len, then
    # delete(i, len(s2)) for extra s1 chars, insert(len(s1), i) for extra s2 chars.
    assert list(Hamming.editops("abc", "abcx")) == [("insert", 3, 3)]
    assert list(Hamming.editops("abc", "a")) == [("delete", 1, 1), ("delete", 2, 1)]
    assert list(Hamming.editops("a", "abc")) == [("insert", 1, 1), ("insert", 1, 2)]
    assert list(Hamming.editops("abc", "abx")) == [("replace", 2, 2)]


# ── Editops / Opcodes: structural parity with rapidfuzz ───────────────────────

def test_editops_structural_methods():
    ops = Levenshtein.editops("spam", "park")
    assert ops.src_len == 4 and ops.dest_len == 4
    assert list(ops) == list(ops.as_list())
    assert ops.as_list() == ops.inverse().inverse().as_list()
    # rapidfuzz Editops.apply(src, dest) returns the destination string
    assert ops.apply("spam", "park") == "park"
    # as_matching_blocks yields (src_start, dest_start, size) triples
    blocks = list(ops.as_matching_blocks())
    assert all(len(b) == 3 and isinstance(b, MatchingBlock) for b in blocks)
    # as_opcodes is consistent with editops
    codes = ops.as_opcodes()
    assert isinstance(codes, Opcodes)
    assert list(codes.as_editops()) == ops.as_list()


def test_editops_from_opcodes_round_trips():
    codes = Opcodes([
        ("equal", 0, 3, 0, 3),
        ("replace", 3, 6, 3, 6),
        ("equal", 6, 9, 6, 9),
    ], src_len=9, dest_len=9)
    ops = codes.as_editops()
    # replace(3,6,3,6) expands to three per-position replace Editops
    assert ops.as_list() == [
        ("replace", 3, 3), ("replace", 4, 4), ("replace", 5, 5),
    ]
    back = Opcodes.from_editops(ops)
    assert list(back) == list(codes)


def test_editop_and_opcode_unpack_like_rapidfuzz():
    # Editop is a 3-tuple-like namedtuple; Opcode is 5-tuple-like.
    e = Editop("replace", 2, 3)
    assert e[0] == "replace" and e[1] == 2 and e[2] == 3
    assert e.tag == "replace" and e.src_pos == 2 and e.dest_pos == 3
    o = Opcode("replace", 2, 4, 3, 5)
    assert tuple(o) == ("replace", 2, 4, 3, 5)
    assert o.tag == "replace" and o.src_start == 2 and o.src_end == 4


def test_scorealignment_unpack_like_rapidfuzz():
    a = ScoreAlignment(83.33333333333334, 2, 8, 0, 6)
    assert a.score == 83.33333333333334
    assert a[0] == 83.33333333333334
    assert tuple(a) == (83.33333333333334, 2, 8, 0, 6)
    # drop-in parity with rapidfuzz's ScoreAlignment tuple-comparison
    from rapidfuzz.distance import ScoreAlignment as RFA
    rf = RFA(83.33333333333334, 2, 8, 0, 6)
    assert a == rf and tuple(a) == tuple(rf)
