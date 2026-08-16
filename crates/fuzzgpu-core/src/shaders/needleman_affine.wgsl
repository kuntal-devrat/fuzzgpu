// fuzzgpu affine-gap Needleman-Wunsch (Gotoh 1982) compute shader.
// Each invocation processes one string pair with three single-row DP arrays
// (match/substitution, gap-in-a, gap-in-b) plus scalar diagonals, mirroring
// the CPU implementation's memory layout. A gap of length k costs
// `gap_open + k * gap_extend`.
//
// NOTE: WGSL has no 64-bit integer type, so scores are computed in f32 and the
// Rust wrapper converts to i64. f32 addition of integers is exact while
// |value| < 2^24, which covers the practical affine scoring ranges; the CPU
// path remains the source of truth for extreme ranges. Oversized strings
// (> params.max_len) write the SENTINEL and the wrapper recomputes on CPU.

struct Params {
    batch_size: u32,
    max_len: u32,
    offset: u32,
    match_score: f32,
    mismatch_score: f32,
    gap_open: f32,
    gap_extend: f32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> offsets_a: array<u32>;
@group(0) @binding(1) var<storage, read> chars_a: array<u32>;
@group(0) @binding(2) var<storage, read> offsets_b: array<u32>;
@group(0) @binding(3) var<storage, read> chars_b: array<u32>;
@group(0) @binding(4) var<storage, read_write> results: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

const SENTINEL: f32 = -1e30;
// Rust wrapper caps strings at 128 chars; rows are max_len + 1.
const ROW_CAP: u32 = 129;

// Workgroup 16 keeps local memory (3 x 129 f32 rows per invocation) within the
// 32 KiB workgroup-storage budget of common iGPUs.
@compute @workgroup_size(16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pair_idx = params.offset + gid.x;
    if (pair_idx >= params.batch_size) { return; }

    let a_start = offsets_a[pair_idx];
    let a_end = offsets_a[pair_idx + 1u];
    let a_len = a_end - a_start;
    let b_start = offsets_b[pair_idx];
    let b_end = offsets_b[pair_idx + 1u];
    let b_len = b_end - b_start;

    // Guard: strings longer than max_len get the sentinel (wrapper recomputes
    // on CPU), so the fixed-size rows below can never be overrun.
    if (a_len > params.max_len || b_len > params.max_len) {
        results[pair_idx] = SENTINEL;
        return;
    }

    // WGSL does NOT zero-initialize local arrays — every cell must be written
    // before first read. Mirror the CPU implementation's
    // `vec![NEG_INF; n + 1]` start: all three rows start as SENTINEL, then the
    // first row/column get the affine gap costs.
    var m_row: array<f32, ROW_CAP>;
    var ix_row: array<f32, ROW_CAP>;
    var iy_row: array<f32, ROW_CAP>;
    for (var k = 0u; k < ROW_CAP; k++) {
        m_row[k] = SENTINEL;
        ix_row[k] = SENTINEL;
        iy_row[k] = SENTINEL;
    }

    m_row[0] = 0.0;
    for (var j = 1u; j <= b_len; j++) {
        let gap_cost = params.gap_open + f32(j) * params.gap_extend;
        iy_row[j] = gap_cost;
        m_row[j] = gap_cost;
    }

    for (var i = 1u; i <= a_len; i++) {
        var prev_m = m_row[0];
        var prev_ix = ix_row[0];
        var prev_iy = iy_row[0];

        let gap_cost_i = params.gap_open + f32(i) * params.gap_extend;
        ix_row[0] = gap_cost_i;
        m_row[0] = gap_cost_i;
        iy_row[0] = SENTINEL;

        let a_char = chars_a[a_start + i - 1u];

        for (var j = 1u; j <= b_len; j++) {
            let b_char = chars_b[b_start + j - 1u];
            let sub_score = select(params.mismatch_score, params.match_score, a_char == b_char);

            let new_m = max(prev_m, max(prev_ix, prev_iy)) + sub_score;

            let new_ix = max(ix_row[j] + params.gap_extend,
                max(m_row[j] + params.gap_open + params.gap_extend,
                    iy_row[j] + params.gap_open + params.gap_extend));
            let new_iy = max(iy_row[j - 1u] + params.gap_extend,
                max(m_row[j - 1u] + params.gap_open + params.gap_extend,
                    ix_row[j - 1u] + params.gap_open + params.gap_extend));

            prev_m = m_row[j];
            prev_ix = ix_row[j];
            prev_iy = iy_row[j];

            m_row[j] = new_m;
            ix_row[j] = new_ix;
            iy_row[j] = new_iy;
        }
    }

    results[pair_idx] = max(m_row[b_len], max(ix_row[b_len], iy_row[b_len]));
}
