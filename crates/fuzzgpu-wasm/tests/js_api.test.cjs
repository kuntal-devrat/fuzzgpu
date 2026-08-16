// JS-API verification for the fuzzgpu-wasm module.
//
// The Rust #[wasm_bindgen_test]s run inside the wasm module, where i64 is a
// native type and no BigInt conversion happens. Only from JS can we prove the
// actual wasm-bindgen boundary: exported i64s must arrive as `bigint`, round-
// trip exactly (including values past 2^53, the f64 integer limit), and reject
// non-BigInt arguments.
//
// Run after building the Node target:
//   wasm-pack build --target nodejs --out-dir <dir>
//   node tests/js_api.test.cjs [<dir>/fuzzgpu_wasm.js]

const assert = require('node:assert');
const path = require('node:path');

const modulePath =
    process.argv[2] || path.join('target', 'wasm-api-check', 'fuzzgpu_wasm.js');
const fg = require(path.resolve(modulePath));

let passed = 0;
function check(name, fn) {
    fn();
    passed++;
    console.log(`  ok - ${name}`);
}

const A100 = 'A'.repeat(100);

check('needleman_wunsch returns bigint', () => {
    assert.strictEqual(typeof fg.needleman_wunsch('AGTACGCA', 'TATGC', 2n, -1n, -2n), 'bigint');
});

check('small score round-trips exactly', () => {
    assert.strictEqual(fg.needleman_wunsch('AGTACGCA', 'TATGC', 2n, -1n, -2n), 1n);
});

check('score beyond i32 max round-trips as bigint', () => {
    // 3e9 > i32::MAX (~2.1e9): with the old i32 API this wrapped to garbage.
    assert.strictEqual(fg.needleman_wunsch(A100, A100, 30000000n, -1n, -2n), 3000000000n);
});

check('score beyond 2^53 (f64 integer limit) survives exactly', () => {
    // 1e16 > 2^53 ≈ 9.007e15 — a Number-based implementation would lose
    // precision here; BigInt must not. (100 matches * 1e14 = 1e16.)
    assert.strictEqual(fg.needleman_wunsch(A100, A100, 100000000000000n, -1n, -2n), 10000000000000000n);
});

check('negative bigint scores work', () => {
    assert.strictEqual(fg.needleman_wunsch('AAAA', 'TTTT', 2n, -1n, -2n), -4n);
    assert.strictEqual(fg.needleman_wunsch_affine('AAAA', 'TTTT', 2n, -1n, -10n, -2n), -4n);
});

check('affine variant returns bigint with exact large scores', () => {
    assert.strictEqual(typeof fg.needleman_wunsch_affine('AGTACGCA', 'TATGC', 2n, -1n, -10n, -2n), 'bigint');
    assert.strictEqual(fg.needleman_wunsch_affine(A100, A100, 100000000000000n, -1n, -10n, -2n), 10000000000000000n);
});

check('non-bigint arguments are rejected', () => {
    assert.throws(() => fg.needleman_wunsch('AGTACGCA', 'TATGC', 2, -1, -2));
    assert.throws(() => fg.needleman_wunsch_affine('AGTACGCA', 'TATGC', 2, -1, -10, -2));
});

console.log(`\nJS API: ${passed} checks passed`);
