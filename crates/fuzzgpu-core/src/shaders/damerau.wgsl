// fuzzgpu Damerau-Levenshtein compute shader (batch).
// True unrestricted Damerau-Levenshtein (Lowrance & Wagner 1975) — bit-exact
// with the CPU reference (damerau_bytes), including non-adjacent
// transpositions (the "ca" vs "abc" = 2 case that optimal-string-alignment
// gets wrong).
//
// Each invocation runs one pair's full DP. The unrestricted transposition
// term needs arbitrary random access into previously computed rows
// (h[k-1][l-1] with k,l = last occurrences), so the whole (m+2)x(n+2) matrix
// must be kept per pair — it lives in workgroup shared memory (SLM), not
// private memory (a private 34x34 matrix would spill catastrophically, the
// exact problem the Levenshtein short kernel was built to avoid).
//
// SLM budget: workgroup = 4 pairs, each pair 34x34 u32 = 4.6 KiB matrix +
// a 257-entry da table (1 KiB) = ~22.6 KiB total. Requires a device with
// >= 32 KiB of workgroup storage (every dGPU/iGPU of the last decade; the
// engine requests min(adapter, 32 KiB) and the kernel refuses to initialize
// below its budget, routing to CPU instead).
//
// Matrix indexing matches damerau_bytes() exactly: idx(i, j) = (i+1)*COLS +
// (j+1), i in -1..=m, j in -1..=n. The sentinel row/col (-1) is written to
// max_dist by the CPU but never read (sub/ins/del/trans all use indices
// >= 0), so the shader only initializes row 0 and col 0.

struct DamerauParams {
    batch_size: u32,
    max_len: u32,
    offset: u32,
}

@group(0) @binding(0) var<storage, read> offsets_a: array<u32>;
@group(0) @binding(1) var<storage, read> chars_a: array<u32>;
@group(0) @binding(2) var<storage, read> offsets_b: array<u32>;
@group(0) @binding(3) var<storage, read> chars_b: array<u32>;
@group(0) @binding(4) var<storage, read_write> results: array<u32>;
@group(0) @binding(5) var<uniform> params: DamerauParams;

const L: u32 = 32u;            // max string length (chars) per pair
const COLS: u32 = L + 2u;      // 34
const CELLS: u32 = COLS * COLS; // 1156
const THREADS: u32 = 4u;        // pairs per workgroup
const DA_STRIDE: u32 = 257u;    // 256 entries + 1 pad (bank spreading)

var<workgroup> mats: array<u32, THREADS * CELLS>;       // 4624 u32 = 18.5 KiB
var<workgroup> da_tbl: array<u32, THREADS * DA_STRIDE>; // 1028 u32 = 4.1 KiB

fn min4(a: u32, b: u32, c: u32, d: u32) -> u32 {
    return min(min(a, b), min(c, d));
}

@compute @workgroup_size(4)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let pair_idx = params.offset + gid.x;
    if (pair_idx >= params.batch_size) { return; }
    let t = lid.x;

    let a_start = offsets_a[pair_idx];
    let a_end = offsets_a[pair_idx + 1u];
    let a_len = a_end - a_start;

    let b_start = offsets_b[pair_idx];
    let b_end = offsets_b[pair_idx + 1u];
    let b_len = b_end - b_start;

    // Defensive sentinel: > 32 chars (CPU recomputes) or non-ASCII (the da
    // table is keyed by byte value 0..=255, mirroring damerau_bytes' assert).
    if (a_len > L || b_len > L) {
        results[pair_idx] = 4294967295u;
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
        results[pair_idx] = 4294967295u;
        return;
    }

    let base = t * CELLS;
    let da_base = t * DA_STRIDE;

    // Workgroup memory is not guaranteed zero-initialized: explicitly reset
    // the per-thread da table (last-row-seen per byte value).
    for (var c = 0u; c < 256u; c++) {
        da_tbl[da_base + c] = 0u;
    }

    // Row 0: h[0][j] = j. Col 0: h[i][0] = i. (Sentinel row/col -1 unused.)
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
            let k = da_tbl[da_base + bj];   // last row i' < i with a[i'] == bj (0 = none)
            let l = db;                     // last col j' < j with b[j'] == ai
            var cost = 1u;
            if (ai == bj) {
                db = j;
                cost = 0u;
            }

            let sub = mats[base + i * COLS + j] + cost;
            let ins = mats[base + (i + 1u) * COLS + j] + 1u;
            let del = mats[base + i * COLS + (j + 1u)] + 1u;

            var trans = 4294967295u; // u32::MAX ~= infinity (max real dist <= 64)
            if (k > 0u && l > 0u) {
                trans = mats[base + k * COLS + l]
                      + (i - k - 1u) + 1u + (j - l - 1u);
            }

            mats[base + (i + 1u) * COLS + (j + 1u)] = min4(sub, ins, del, trans);
        }
        da_tbl[da_base + ai] = i;
    }

    results[pair_idx] = mats[base + (a_len + 1u) * COLS + (b_len + 1u)];
}
