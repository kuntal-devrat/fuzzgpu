"""Randomized differential tests against rapidfuzz (the reference implementation).

rapidfuzz is installed in CI (see .github/workflows/ci.yml), so this suite
genuinely pins fuzzgpu's output to rapidfuzz's across a seeded random corpus —
scorers, score_cutoff semantics, alignments, distance modules, editops, and
process.extract ordering. Any drift from the reference fails here.

Skips cleanly when rapidfuzz is not installed (e.g. a minimal offline env).
"""

import itertools
import random

import pytest

_rapidfuzz = pytest.importorskip("rapidfuzz")
import rapidfuzz.distance as rd  # noqa: E402
import rapidfuzz.process  # noqa: E402
import rapidfuzz.fuzz  # noqa: E402
import fuzzgpu.distance as fd  # noqa: E402
import fuzzgpu.process  # noqa: E402
import fuzzgpu.fuzz  # noqa: E402

random.seed(0xF0F0)

# Corpus covers ASCII, unicode, whitespace-heavy token inputs, empties, and
# length asymmetries (the partial/token/WRatio edge territory).
_CORPUS = [
    "hello", "hallo", "help", "yellow", "mellow", "a", "", "ab", "abc",
    "kitten", "sitting", "saturday", "sunday", "flaw", "lawn",
    "MARTHA", "MARHTA", "DWAYNE", "DUANE", "DIXON", "DICKSONX",
    "new york mets", "new york yankees", "new york", "the quick brown fox",
    "the quick brown fox!", "quick brown fox", "café", "cafe", "naïve", "naive",
    "日本語", "日語本", "🚀🌟", "🌟🚀", "a cat", "an act",
    "this is a longer sentence with many tokens here",
    "tokens many with sentence longer a is this",
]
_PAIRS = list(itertools.product(_CORPUS, _CORPUS))
random.shuffle(_PAIRS)
_PAIRS = _PAIRS[:200]
_CUTOFFS = [0.0, 30.0, 50.0, 70.0, 90.0, 99.0, 100.0]


def _assert_close(a, b, label):
    assert abs(a - b) < 1e-9, f"{label}: rapidfuzz={a} fuzzgpu={b}"


@pytest.mark.parametrize("name", [
    "ratio", "partial_ratio", "token_sort_ratio", "token_set_ratio",
    "token_ratio", "partial_token_sort_ratio", "partial_token_set_ratio",
    "partial_token_ratio", "QRatio", "WRatio",
])
def test_fuzz_scorers_match_rapidfuzz(name):
    rfn = getattr(rapidfuzz.fuzz, name)
    fgn = getattr(fuzzgpu.fuzz, name)
    for a, b in _PAIRS:
        _assert_close(rfn(a, b), fgn(a, b), f"{name}({a!r}, {b!r})")


@pytest.mark.parametrize("name", ["ratio", "partial_ratio", "token_sort_ratio",
                                  "token_set_ratio", "token_ratio", "WRatio", "QRatio"])
def test_score_cutoff_semantics_match_rapidfuzz(name):
    rfn = getattr(rapidfuzz.fuzz, name)
    fgn = getattr(fuzzgpu.fuzz, name)
    for a, b in _PAIRS[:60]:
        for cut in _CUTOFFS:
            _assert_close(
                rfn(a, b, score_cutoff=cut),
                fgn(a, b, score_cutoff=cut),
                f"{name}({a!r}, {b!r}, cutoff={cut})",
            )


def test_partial_ratio_alignment_matches_rapidfuzz():
    for a, b in _PAIRS:
        r = rapidfuzz.fuzz.partial_ratio_alignment(a, b)
        g = fuzzgpu.fuzz.partial_ratio_alignment(a, b)
        assert r.score == pytest.approx(g.score, abs=1e-9), f"score ({a!r}, {b!r})"
        assert (r.src_start, r.src_end) == (g.src_start, g.src_end), f"src ({a!r}, {b!r})"
        assert (r.dest_start, r.dest_end) == (g.dest_start, g.dest_end), f"dest ({a!r}, {b!r})"


_DIST_MODULES = ["Levenshtein", "OSA", "DamerauLevenshtein", "Indel",
                 "Hamming", "Jaro", "JaroWinkler", "LCSseq", "Prefix", "Postfix"]


@pytest.mark.parametrize("mod", _DIST_MODULES)
@pytest.mark.parametrize("fn", ["distance", "similarity",
                                "normalized_distance", "normalized_similarity"])
def test_distance_modules_match_rapidfuzz(mod, fn):
    rmod, gmod = getattr(rd, mod), getattr(fd, mod)
    for a, b in _PAIRS:
        r = getattr(rmod, fn)(a, b)
        g = getattr(gmod, fn)(a, b)
        if isinstance(r, float):
            _assert_close(r, g, f"{mod}.{fn}({a!r}, {b!r})")
        else:
            assert r == g, f"{mod}.{fn}({a!r}, {b!r}): rapidfuzz={r} fuzzgpu={g}"


@pytest.mark.parametrize("mod", ["Levenshtein", "Indel"])
def test_editops_length_matches_rapidfuzz(mod):
    rmod, gmod = getattr(rd, mod), getattr(fd, mod)
    for a, b in _PAIRS:
        rl = len(rmod.editops(a, b))
        gl = len(gmod.editops(a, b))
        assert rl == gl, f"{mod}.editops({a!r}, {b!r}): rapidfuzz={rl} fuzzgpu={gl}"


def test_process_extract_matches_rapidfuzz():
    choices = _CORPUS[:20]
    for cut in [None, 0.0, 30.0, 60.0, 90.0]:
        for limit in [None, 1, 3, 10]:
            r = rapidfuzz.process.extract("new york", choices, score_cutoff=cut, limit=limit)
            g = fuzzgpu.process.extract("new york", choices, score_cutoff=cut, limit=limit)
            rn = [(x[0], round(x[1], 6), x[2]) for x in r]
            gn = [(x[0], round(x[1], 6), x[2]) for x in g]
            assert rn == gn, f"extract(cut={cut}, limit={limit}): rf={rn} fg={gn}"


def test_process_extract_dict_and_ties():
    choices = {"a": "new york mets", "b": "new york yankees", "c": "new york", "d": "mets"}
    r = rapidfuzz.process.extract("new york", choices, limit=10)
    g = fuzzgpu.process.extract("new york", choices, limit=10)
    assert [(x[0], x[2]) for x in r] == [(x[0], x[2]) for x in g]


def test_process_extract_one_matches_rapidfuzz():
    choices = _CORPUS
    r = rapidfuzz.process.extractOne("new york", choices)
    g = fuzzgpu.process.extractOne("new york", choices)
    assert r[0] == g[0] and r[1] == pytest.approx(g[1], abs=1e-9) and r[2] == g[2]


def test_cdist_matches_rapidfuzz_multiplier_cutoff():
    # cdist fast path (native ratio) must match rapidfuzz for score_multiplier
    # (scale after cutoff) and score_cutoff (zero below) in every combination.
    queries = _CORPUS[:8]
    choices = _CORPUS[:8]
    for mult in [1, 2, 0.5]:
        for cut in [None, 30.0, 80.0]:
            r = rapidfuzz.process.cdist(
                queries, choices, scorer=rapidfuzz.fuzz.ratio,
                score_cutoff=cut, score_multiplier=mult,
            )
            g = fuzzgpu.process.cdist(
                queries, choices, scorer=fuzzgpu.fuzz.ratio,
                score_cutoff=cut, score_multiplier=mult,
            )
            # rapidfuzz returns np.float32 by default; fuzzgpu returns Python
            # floats (float64). Compare values with a float32-appropriate
            # tolerance rather than exact equality.
            assert len(g) == len(r) and all(len(gr) == len(rr) for gr, rr in zip(g, r))
            for i in range(len(queries)):
                for j in range(len(choices)):
                    assert g[i][j] == pytest.approx(float(r[i][j]), abs=1e-4), (
                        f"cdist(mult={mult}, cut={cut})[{i}][{j}]: rf={r[i][j]} fg={g[i][j]}"
                    )


def test_cdist_generic_path_zeroes_below_cutoff():
    # Scorer without score_cutoff support: the generic path must zero below the
    # cutoff just like the fast path and rapidfuzz do.
    def plain(a, b):
        return 100.0 if a == b else 0.0

    queries = ["hello", "kitten"]
    choices = ["hello", "world", "kitten"]
    g = fuzzgpu.process.cdist(queries, choices, scorer=plain, score_cutoff=50.0)
    assert g == [[100.0, 0.0, 0.0], [0.0, 0.0, 100.0]]
