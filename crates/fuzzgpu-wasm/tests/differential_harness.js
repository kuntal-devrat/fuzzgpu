// Differential harness: exercise wasm exports the other tests don't cover
// (levenshtein_batch, levenshtein_myers, jaro, jaro_winkler, ratio,
// partial_ratio, token_sort_ratio, token_set_ratio, wratio, damerau_ratio,
// needleman_wunsch, needleman_wunsch_affine, extract, extract_one) over a
// fixed corpus, emitting JSON for the Python comparator
// (tests/wasm_python_differential.py) to diff against the Python bindings of
// the same Rust core.
//
// Beyond value equality, the harness pins the *exact JS type* of every result
// (number vs bigint, Array vs Uint32Array, null) BEFORE any normalization, so
// a future wasm-bindgen upgrade that changes the JS-facing type of an export
// fails CI here instead of silently changing the API.
//
// Run (after building the Node target):
//   wasm-pack build --target nodejs --out-dir <dir>
//   node tests/differential_harness.js [<dir>/fuzzgpu_wasm.js] > results.json

const path = require('node:path');

const modulePath =
    process.argv[2] || path.join('target', 'wasm-diff-check', 'fuzzgpu_wasm.js');
const fg = require(path.resolve(modulePath));

// Fail loudly if an expected export is missing — a silently missing export
// would make the differential pass vacuously.
const expected = [
    'levenshtein_batch', 'levenshtein_myers',
    'jaro', 'jaro_winkler',
    'ratio', 'partial_ratio', 'token_sort_ratio', 'token_set_ratio', 'wratio',
    'damerau_ratio',
    'needleman_wunsch', 'needleman_wunsch_affine',
    'extract', 'extract_one',
];
for (const name of expected) {
    if (typeof fg[name] !== 'function') {
        console.error(`missing wasm export: ${name}`);
        process.exit(1);
    }
}

// The exact JS type each export must produce. wasm-bindgen maps:
//   u32/f64 -> number, i64 -> bigint, Vec<u32> -> Uint32Array,
//   serde Vec<T> -> Array, Option<T> -> Array | null.
// Asserted on the RAW result before any normalization below.
const TYPE_EXPECTATIONS = {
    levenshtein_batch: ['uint32array'],
    levenshtein_myers: ['number'],
    jaro: ['number'],
    jaro_winkler: ['number'],
    ratio: ['number'],
    partial_ratio: ['number'],
    token_sort_ratio: ['number'],
    token_set_ratio: ['number'],
    wratio: ['number'],
    damerau_ratio: ['number'],
    needleman_wunsch: ['bigint'],
    needleman_wunsch_affine: ['bigint'],
    extract: ['array'],
    extract_one: ['array', 'null'],
};

function typeOf(v) {
    if (v === null) return 'null';
    if (Array.isArray(v)) return 'array';
    if (v instanceof Uint32Array) return 'uint32array';
    if (typeof v === 'bigint') return 'bigint';
    if (typeof v === 'number') return 'number';
    if (typeof v === 'string') return 'string';
    return typeof v;
}

function assertType(kind, v) {
    const allowed = TYPE_EXPECTATIONS[kind];
    const actual = typeOf(v);
    if (!allowed.includes(actual)) {
        console.error(
            `TYPE MISMATCH ${kind}: expected ${allowed.join(' or ')}, got ${actual}`);
        process.exit(1);
    }
    return actual;
}

// Fixed, deterministic corpus. Deliberately includes Unicode (incl. astral
// and RTL), empty strings, boundary p values, cutoff/limit filtering, empty
// choices, exact-match ties, and scores beyond 2^53 — the shapes that stress
// bindings. i64 score parameters are stored as decimal strings (BigInt can't
// be JSON-serialized) and converted with BigInt() at the call site.
const corpus = {
    levenshtein_batch: [
        { query: 'kitten', candidates: ['sitting', 'kitten', 'SITTING', 'kittens', '', 'abcdefghijklmnopqrstuvwxyz'] },
        { query: '', candidates: ['', 'a', 'abc', '日本語'] },
        { query: '日本語テスト', candidates: ['日本語', 'テスト', '日本', '日本テスト', ''] },
        { query: 'مرحبا بالعالم', candidates: ['مرحبا', 'العالم', 'مرحبا بالعالم', 'hello world'] },
    ],
    jaro_winkler: [
        { a: 'MARTHA', b: 'MARHTA', p: 0.1 },
        { a: 'DWAYNE', b: 'DUANE', p: 0.1 },
        { a: 'dixon', b: 'dicksonx', p: 0.1 },
        { a: 'CRATE', b: 'TRACE', p: 0.1 },
        { a: 'abc', b: 'abc', p: 0.1 },
        { a: '', b: 'abc', p: 0.1 },
        { a: 'a', b: 'b', p: 0.25 },
        { a: 'abcdef', b: 'abcfed', p: 0.0 },
        { a: 'héllo', b: 'hallo', p: 0.1 },
        { a: '日本', b: '日本語', p: 0.1 },
    ],
    ratio: [
        { a: 'fuzzy wuzzy was a bear', b: 'wuzzy fuzzy was a bear' },
        { a: 'hello world', b: 'hello' },
        { a: '', b: '' },
        { a: '', b: 'a' },
        { a: 'a', b: 'b' },
        { a: '日本', b: '日本語' },
        { a: 'this is a test', b: 'this is a test!' },
        { a: 'The quick brown fox jumps over the lazy dog', b: 'the quick brown fox jumps over the lazy dog' },
    ],
    partial_ratio: [
        { a: 'hello', b: 'oh hello there' },
        { a: 'kitten', b: 'sitting' },
        { a: '', b: 'abc' },
        { a: '日本', b: '日本語です' },
        { a: 'this is a test', b: 'this is a test' },
    ],
    token_sort_ratio: [
        { a: 'new york mets', b: 'mets new york' },
        { a: 'fuzzy wuzzy was a bear', b: 'wuzzy fuzzy was a bear' },
        { a: '', b: '' },
        { a: 'a b c', b: 'c b a' },
        { a: '日本 語', b: '語 日本' },
    ],
    token_set_ratio: [
        { a: 'fuzzy was a bear', b: 'fuzzy bear' },
        { a: 'mariners vs angels', b: 'angels vs mariners' },
        { a: 'the quick brown fox', b: 'the brown fox quick' },
        { a: '', b: 'a b' },
        { a: 'a a a', b: 'a' },
    ],
    wratio: [
        { a: 'fuzzy was a bear', b: 'fuzzy was a bear' },
        { a: 'new york mets', b: 'mets new york' },
        { a: 'hello', b: '' },
        { a: '日本', b: '日本語' },
        { a: 'this is a test!', b: 'this is a test' },
    ],
    jaro: [
        { a: 'MARTHA', b: 'MARHTA' },
        { a: 'DWAYNE', b: 'DUANE' },
        { a: 'dixon', b: 'dicksonx' },
        { a: '', b: 'abc' },
        { a: '日本', b: '日本語' },
    ],
    damerau_ratio: [
        { a: 'kitten', b: 'sitting' },
        { a: 'ab', b: 'ba' },
        { a: '', b: '' },
        { a: 'hello', b: 'hello' },
        { a: '日本', b: '日本語' },
    ],
    levenshtein_myers: [
        { a: 'kitten', b: 'sitting' },
        { a: '', b: 'abc' },
        { a: 'abcdefghijklmnopqrstuvwxyz0123456789', b: 'abcdefghijklmnopqrstuvwxyz0123456789!' },
        { a: '日本', b: '日本語' },
        { a: 'a', b: 'a' },
    ],
    needleman_wunsch: [
        { a: 'AGTACGCA', b: 'TATGC', match_score: '2', mismatch_score: '-1', gap_penalty: '-2' },
        { a: 'AAAA', b: 'TTTT', match_score: '2', mismatch_score: '-1', gap_penalty: '-2' },
        // 100 matches * 1e14 = 1e16: beyond 2^53 — BigInt must carry it exactly.
        { a: 'A'.repeat(100), b: 'A'.repeat(100), match_score: '100000000000000', mismatch_score: '-1', gap_penalty: '-2' },
    ],
    needleman_wunsch_affine: [
        { a: 'AGTACGCA', b: 'TATGC', match_score: '2', mismatch_score: '-1', gap_open: '-3', gap_extend: '-1' },
        { a: 'AAAA', b: 'TTTT', match_score: '2', mismatch_score: '-1', gap_open: '-10', gap_extend: '-2' },
        { a: 'A'.repeat(100), b: 'A'.repeat(100), match_score: '100000000000000', mismatch_score: '-1', gap_open: '-10', gap_extend: '-2' },
    ],
    extract: [
        { query: 'appel', choices: ['apple', 'apricot', 'banana', 'appel', 'applesauce'], score_cutoff: 0, limit: 5 },
        { query: 'appel', choices: ['apple', 'apricot', 'banana', 'appel', 'applesauce'], score_cutoff: 60, limit: 2 },
        { query: 'appel', choices: ['apple', 'apricot', 'banana', 'appel', 'applesauce'], score_cutoff: 100, limit: 3 },
        { query: 'xyz', choices: ['abc', 'def'], score_cutoff: 0, limit: 5 },
        { query: 'kitten', choices: ['sitting', 'kitten', ''], score_cutoff: 0, limit: 1 },
        { query: 'appel', choices: [], score_cutoff: 0, limit: 5 },
        { query: 'テスト', choices: ['テスト', 'テス', 'ト', '日本語テスト'], score_cutoff: 0, limit: 3 },
        { query: 'appel', choices: ['apple', 'banana'], score_cutoff: 0, limit: 0 },
    ],
    extract_one: [
        { query: 'appel', choices: ['apple', 'apricot', 'banana'], score_cutoff: 0 },
        { query: 'appel', choices: ['banana', 'apricot'], score_cutoff: 60 },
        { query: 'xyz', choices: ['abc'], score_cutoff: 90 },
        { query: 'テスト', choices: ['テス', 'ト'], score_cutoff: 0 },
    ],
};

const out = {};
for (const [kind, cases] of Object.entries(corpus)) {
    out[kind] = cases.map((c) => {
        let raw;
        let result;
        if (kind === 'levenshtein_batch') {
            raw = fg.levenshtein_batch(c.query, c.candidates);
            // Vec<u32> arrives as a Uint32Array — normalize to a plain array
            // so JSON round-trips to the same shape as Python's list.
            result = Array.from(raw);
        } else if (kind === 'levenshtein_myers') {
            raw = result = fg.levenshtein_myers(c.a, c.b);
        } else if (kind === 'jaro') {
            raw = result = fg.jaro(c.a, c.b);
        } else if (kind === 'jaro_winkler') {
            raw = result = fg.jaro_winkler(c.a, c.b, c.p);
        } else if (['ratio', 'partial_ratio', 'token_sort_ratio', 'token_set_ratio', 'wratio', 'damerau_ratio'].includes(kind)) {
            raw = result = fg[kind](c.a, c.b);
        } else if (kind === 'needleman_wunsch') {
            raw = fg.needleman_wunsch(
                c.a, c.b,
                BigInt(c.match_score), BigInt(c.mismatch_score), BigInt(c.gap_penalty));
            // i64 arrives as BigInt, which JSON cannot serialize — emit the
            // decimal string; Python recomputes the same i64 exactly.
            result = String(raw);
        } else if (kind === 'needleman_wunsch_affine') {
            raw = fg.needleman_wunsch_affine(
                c.a, c.b,
                BigInt(c.match_score), BigInt(c.mismatch_score),
                BigInt(c.gap_open), BigInt(c.gap_extend));
            result = String(raw);
        } else if (kind === 'extract') {
            raw = result = fg.extract(c.query, c.choices, c.score_cutoff, c.limit);
        } else if (kind === 'extract_one') {
            raw = result = fg.extract_one(c.query, c.choices, c.score_cutoff);
        }
        // Pin the raw JS-facing type before any normalization above.
        const type = assertType(kind, raw);
        return { ...c, result, type };
    });
}

console.log(JSON.stringify(out));
