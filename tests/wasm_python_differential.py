"""Differential harness: compare wasm JS outputs against the Python bindings.

The JS side (crates/fuzzgpu-wasm/tests/differential_harness.js) computes a
fixed corpus of wasm exports — levenshtein_batch, levenshtein_myers, jaro,
jaro_winkler, ratio, partial_ratio, token_sort_ratio, token_set_ratio, wratio,
damerau_ratio, needleman_wunsch, needleman_wunsch_affine, extract,
extract_one — and writes JSON. This script recomputes the same corpus with
the Python bindings and diffs the two. Both bindings wrap the same Rust
core, so any mismatch is a binding bug — wrong argument order, mangled
f64/u32/i64 fidelity across the FFI boundary, bad tuple layout, off-by-one
index.

The JS side also pins the exact JS-facing type of each result (number vs
bigint, Array vs Uint32Array, null) and records it as `type`; this script
cross-checks it against its own expectation table so the type contract is
guarded on both ends of the JSON round-trip.

Run (after `pip install .` and building the wasm nodejs target):
    python tests/wasm_python_differential.py <js_results.json>

Exits non-zero with a readable diff on the first mismatch.
"""

import json
import math
import sys

import fuzzgpu


def close(a, b, rel_tol=1e-9, abs_tol=1e-12):
    if a == b:
        return True
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


# The JS-facing type each export must produce, mirrored from the JS harness.
# Asserted against the `type` field the harness records per case.
EXPECTED_JS_TYPES = {
    "levenshtein_batch": {"uint32array"},
    "levenshtein_myers": {"number"},
    "jaro": {"number"},
    "jaro_winkler": {"number"},
    "ratio": {"number"},
    "partial_ratio": {"number"},
    "token_sort_ratio": {"number"},
    "token_set_ratio": {"number"},
    "wratio": {"number"},
    "damerau_ratio": {"number"},
    "needleman_wunsch": {"bigint"},
    "needleman_wunsch_affine": {"bigint"},
    "extract": {"array"},
    "extract_one": {"array", "null"},
}


def normalize_extract(v):
    # JS side: extract returns [[choice, score, index], ...], extract_one a
    # flat [choice, score, index] or null; Python side: list of (str, float,
    # int) tuples, a single tuple, or None. Normalize to list-of-triples | None.
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        if len(v) == 0:
            return []
        if isinstance(v[0], (list, tuple)):
            items = v
        else:
            items = [v]  # single triple
    else:
        items = [v]
    return [(str(choice), float(score), int(index)) for choice, score, index in items]


def compare_case(kind, entry, js, py, errors):
    prefix = f"{kind}: {entry}"
    if kind in (
        "jaro", "jaro_winkler", "ratio", "partial_ratio",
        "token_sort_ratio", "token_set_ratio", "wratio", "damerau_ratio",
    ):
        if not close(js, py):
            errors.append(f"{prefix}\n  js={js!r}\n  py={py!r}")
    elif kind in ("levenshtein_batch", "levenshtein_myers"):
        if list(js) != list(py) if isinstance(js, list) else js != py:
            errors.append(f"{prefix}\n  js={js!r}\n  py={py!r}")
    elif kind in ("needleman_wunsch", "needleman_wunsch_affine"):
        # i64 scores cross JSON as decimal strings; Python recomputes exact ints.
        if int(js) != py:
            errors.append(f"{prefix}\n  js={js!r}\n  py={py!r}")
    elif kind in ("extract", "extract_one"):
        js_n, py_n = normalize_extract(js), normalize_extract(py)
        if js_n != py_n:
            errors.append(f"{prefix}\n  js={js_n!r}\n  py={py_n!r}")
    else:
        errors.append(f"{prefix}\n  unknown kind {kind!r}")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    # Error messages embed corpus strings (Unicode, RTL); make printing
    # portable across console encodings (e.g. Windows cp1252).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    with open(sys.argv[1], encoding="utf-8") as f:
        data = json.load(f)

    # Force CPU-only so Python uses exactly the same code paths as the wasm
    # build (the wasm crate compiles without the gpu feature). This keeps the
    # differential purely about binding correctness.
    fuzzgpu.set_cpu_only(True)

    errors = []
    kinds = (
        "levenshtein_batch", "levenshtein_myers",
        "jaro", "jaro_winkler",
        "ratio", "partial_ratio", "token_sort_ratio", "token_set_ratio", "wratio",
        "damerau_ratio",
        "needleman_wunsch", "needleman_wunsch_affine",
        "extract", "extract_one",
    )
    for kind in kinds:
        expected_types = EXPECTED_JS_TYPES[kind]
        for entry in data[kind]:
            # Cross-check the JS-facing type recorded by the harness.
            if entry["type"] not in expected_types:
                errors.append(
                    f"{kind}: JS type mismatch — expected {sorted(expected_types)}, "
                    f"got {entry['type']!r} (entry {entry})"
                )
            if kind == "levenshtein_batch":
                py = fuzzgpu.levenshtein_batch(entry["query"], entry["candidates"])
            elif kind == "levenshtein_myers":
                py = fuzzgpu.levenshtein_myers(entry["a"], entry["b"])
            elif kind == "jaro":
                py = fuzzgpu.jaro(entry["a"], entry["b"])
            elif kind == "jaro_winkler":
                py = fuzzgpu.jaro_winkler(entry["a"], entry["b"], entry["p"])
            elif kind in ("ratio", "partial_ratio", "token_sort_ratio", "token_set_ratio", "wratio", "damerau_ratio"):
                py = getattr(fuzzgpu, kind)(entry["a"], entry["b"])
            elif kind == "needleman_wunsch":
                py = fuzzgpu.needleman_wunsch(
                    entry["a"], entry["b"],
                    int(entry["match_score"]), int(entry["mismatch_score"]), int(entry["gap_penalty"]),
                )
            elif kind == "needleman_wunsch_affine":
                py = fuzzgpu.needleman_wunsch_affine(
                    entry["a"], entry["b"],
                    int(entry["match_score"]), int(entry["mismatch_score"]),
                    int(entry["gap_open"]), int(entry["gap_extend"]),
                )
            elif kind == "extract":
                py = fuzzgpu.extract(
                    entry["query"], entry["choices"], entry["score_cutoff"], entry["limit"]
                )
            else:  # extract_one
                py = fuzzgpu.extract_one(entry["query"], entry["choices"], entry["score_cutoff"])
            compare_case(kind, entry, entry["result"], py, errors)

    if errors:
        print(f"FAIL: {len(errors)} mismatch(es)")
        for e in errors[:10]:
            print(e)
        sys.exit(1)
    total = sum(len(v) for v in data.values())
    print(f"wasm ↔ Python differential: {total} cases match")


if __name__ == "__main__":
    main()
