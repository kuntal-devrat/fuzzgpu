// fuzzgpu 2D Damerau-Levenshtein Matrix compute shader.
// Computes the cross-product distance matrix between List A (rows) and
// List B (cols). Each thread runs one (row, col) pair's full Lowrance-Wagner
// DP in workgroup shared memory — identical kernel body to damerau.wgsl, with
// the pair indexed as (gid.y, gid.x) into the two offset/char lists.

struct DamerauMatrixParams {
    rows: u32,
    cols: u32,
}

@group(0) @binding(0) var<storage, read> offsets_a: array<u32>;
@group(0) @binding(1) var<storage, read> chars_a: array<u32>;
@group(0) @binding(2) var<storage, read> offsets_b: array<u32>;
@group(0) @binding(3) var<storage, read> chars_b: array<u32>;
@group(0) @binding(4) var<storage, read_write> matrix: array<u32>;
@group(0) @binding(5) var<uniform> params: DamerauMatrixParams;

const L: u32 = 32u;
const COLS: u32 = L + 2u;
const CELLS: u32 = COLS * COLS;
const THREADS: u32 = 4u;
const DA_STRIDE: u32 = 257u;

var<workgroup> mats: array<u32, THREADS * CELLS>;
var<workgroup> da_tbl: array<u32, THREADS * DA_STRIDE>;

fn min4(a: u32, b: u32, c: u32, d: u32) -> u32 {
    return min(min(a, b), min(c, d));
}

@compute @workgroup_size(4)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let col = gid.x;
    let row = gid.y;
    if (row >= params.rows || col >= params.cols) { return; }
    let out_idx = row * params.cols + col;
    let t = lid.x;

    let a_start = offsets_a[row];
    let a_end = offsets_a[row + 1u];
    let a_len = a_end - a_start;

    let b_start = offsets_b[col];
    let b_end = offsets_b[col + 1u];
    let b_len = b_end - b_start;

    if (a_len > L || b_len > L) {
        matrix[out_idx] = 4294967295u;
        return;
    }
    var ascii_ok = true;
    for (var i = 0u; i < a_len; i++) {
        if (chars_a[a_start + i] > 255u) { ascii_ok = false; }
    }
    for (var j = 0u; j < b_len; j++) {
        if (chars_b[b_start + j] > 255u) { ascii_ok = false; }
    }
    if (!ascii_ok) {
        matrix[out_idx] = 4294967295u;
        return;
    }

    let base = t * CELLS;
    let da_base = t * DA_STRIDE;

    for (var c = 0u; c < 256u; c++) {
        da_tbl[da_base + c] = 0u;
    }

    for (var j = 0u; j <= b_len; j++) {
        mats[base + COLS + (j + 1u)] = j;
    }
    for (var i = 0u; i <= a_len; i++) {
        mats[base + (i + 1u) * COLS + 1u] = i;
    }

    for (var i = 1u; i <= a_len; i++) {
        var db = 0u;
        let ai = chars_a[a_start + (i - 1u)];
        for (var j = 1u; j <= b_len; j++) {
            let bj = chars_b[b_start + (j - 1u)];
            let k = da_tbl[da_base + bj];
            let l = db;
            var cost = 1u;
            if (ai == bj) {
                db = j;
                cost = 0u;
            }

            let sub = mats[base + i * COLS + j] + cost;
            let ins = mats[base + (i + 1u) * COLS + j] + 1u;
            let del = mats[base + i * COLS + (j + 1u)] + 1u;

            var trans = 4294967295u;
            if (k > 0u && l > 0u) {
                trans = mats[base + k * COLS + l]
                      + (i - k - 1u) + 1u + (j - l - 1u);
            }

            mats[base + (i + 1u) * COLS + (j + 1u)] = min4(sub, ins, del, trans);
        }
        da_tbl[da_base + ai] = i;
    }

    matrix[out_idx] = mats[base + (a_len + 1u) * COLS + (b_len + 1u)];
}
