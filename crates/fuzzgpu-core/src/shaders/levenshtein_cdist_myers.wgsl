// fuzzgpu row-wise Myers (1999) bit-vector Levenshtein cdist shader.
//
// The matrix (list_a x list_b) is computed with one workgroup per ROW of the
// matrix. Workgroup = 64 threads. Thread 0 builds the row query's Peq table
// (byte -> position bitmask) once into shared memory (2 x 128 u32 = 1 KiB
// SLM), then every thread runs the classic Myers recurrence in registers over
// one (or more, strided) texts from list_b. Cost per cell: ~20 bitwise ops per
// text byte, zero DP row — versus O(m·n) for the general matrix kernel.
//
// The 64-bit bit-vector is two u32 words with manual carry propagation (no
// u64, so no SHADER_INT64 feature — works on every WebGPU backend, including
// iGPUs and browsers). Identical arithmetic to shaders/levenshtein_myers.wgsl.
//
// Constraints (enforced by the Rust caller before dispatch):
//   * every string in list_a is non-empty ASCII of <= 64 bytes (the pattern;
//     one Peq per row)
//   * every string in list_b is ASCII (any length; the loop is bounded by the
//     exact packed length)
//   * rows * cols fits the results buffer
//
// Strings are packed linearly with offsets arrays (offsets_x[row] ..= offsets_x[row+1]).

struct Params {
    rows: u32,
    cols: u32,
}

@group(0) @binding(0) var<storage, read> chars_a: array<u32>;    // packed row queries
@group(0) @binding(1) var<storage, read> offsets_a: array<u32>;  // rows + 1 entries
@group(0) @binding(2) var<storage, read> chars_b: array<u32>;    // packed texts
@group(0) @binding(3) var<storage, read> offsets_b: array<u32>;  // cols + 1 entries
@group(0) @binding(4) var<storage, read_write> results: array<u32>;
@group(0) @binding(5) var<uniform> params: Params;

// ASCII alphabet only: 2 x 128 u32 = 1 KiB SLM per workgroup (one row's Peq).
var<workgroup> peq_lo: array<u32, 128>;
var<workgroup> peq_hi: array<u32, 128>;
var<workgroup> wg_m: u32;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wgid.x;
    if (row >= params.rows) {
        return;
    }
    let a_begin = offsets_a[row];
    let a_end = offsets_a[row + 1u];
    let m = a_end - a_begin;

    // Thread 0 builds this row's Peq table; everyone waits at the barrier.
    if (lid.x == 0u) {
        for (var j = 0u; j < 128u; j++) {
            peq_lo[j] = 0u;
            peq_hi[j] = 0u;
        }
        for (var j = 0u; j < m; j++) {
            let c = chars_a[a_begin + j];
            if (c < 128u) {
                if (j < 32u) {
                    peq_lo[c] = peq_lo[c] | (1u << j);
                } else {
                    peq_hi[c] = peq_hi[c] | (1u << (j - 32u));
                }
            }
        }
        wg_m = m;
    }
    workgroupBarrier();
    let m_wg = wg_m;

    // 64-bit mask of m_wg low bits, as (lo, hi); m_wg in 1..=64.
    var mask_lo: u32 = 0u;
    var mask_hi: u32 = 0u;
    if (m_wg < 32u) {
        mask_lo = (1u << m_wg) - 1u;
    } else {
        mask_lo = 0xFFFFFFFFu;
    }
    if (m_wg > 32u) {
        let hb = m_wg - 32u; // bits in the high word: 1..=32
        mask_hi = (1u << (hb - 1u)) * 2u - 1u;
    }

    var col = lid.x;
    while (col < params.cols) {
        let b_begin = offsets_b[col];
        let b_end = offsets_b[col + 1u];
        let n = b_end - b_begin;

        var pv_lo: u32 = mask_lo;
        var pv_hi: u32 = mask_hi;
        var mv_lo: u32 = 0u;
        var mv_hi: u32 = 0u;
        var score: u32 = m_wg;

        for (var i = b_begin; i < b_end; i++) {
            let ch = chars_b[i];
            let eq_lo = peq_lo[ch];
            let eq_hi = peq_hi[ch];

            let xv_lo = eq_lo | mv_lo;
            let xv_hi = eq_hi | mv_hi;

            let a_lo = eq_lo & pv_lo;
            let a_hi = eq_hi & pv_hi;
            let s_lo = a_lo + pv_lo;
            let carry = u32(s_lo < a_lo);
            let s_hi = a_hi + pv_hi + carry;

            let xh_lo = (s_lo ^ pv_lo) | eq_lo;
            let xh_hi = (s_hi ^ pv_hi) | eq_hi;

            let ph_lo = mv_lo | ~(xh_lo | pv_lo);
            let ph_hi = mv_hi | ~(xh_hi | pv_hi);
            let mh_lo = pv_lo & xh_lo;
            let mh_hi = pv_hi & xh_hi;

            if (m_wg > 32u) {
                let bit = m_wg - 33u;
                if ((ph_hi & (1u << bit)) != 0u) { score = score + 1u; }
                if ((mh_hi & (1u << bit)) != 0u) { score = score - 1u; }
            } else {
                let bit = m_wg - 1u;
                if ((ph_lo & (1u << bit)) != 0u) { score = score + 1u; }
                if ((mh_lo & (1u << bit)) != 0u) { score = score - 1u; }
            }

            let ph_s_lo = (ph_lo << 1u) | 1u;
            let ph_s_hi = (ph_hi << 1u) | (ph_lo >> 31u);
            let mh_s_lo = mh_lo << 1u;
            let mh_s_hi = (mh_hi << 1u) | (mh_lo >> 31u);

            pv_lo = (mh_s_lo | ~(xv_lo | ph_s_lo)) & mask_lo;
            pv_hi = (mh_s_hi | ~(xv_hi | ph_s_hi)) & mask_hi;
            mv_lo = (ph_s_lo & xv_lo) & mask_lo;
            mv_hi = (ph_s_hi & xv_hi) & mask_hi;
        }

        results[row * params.cols + col] = score;
        col += 64u;
    }
}
