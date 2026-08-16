// fuzzgpu Myers (1999) bit-vector Levenshtein compute shader.
//
// Fast path for the dominant fuzzy-matching shape: ONE query (pattern, <= 64
// ASCII bytes) vs N candidate texts. Each workgroup = 64 threads = 64 texts,
// all sharing the same pattern. The pattern's Peq table (byte -> position
// bitmask) is built ONCE per workgroup into shared memory (2 x 128 u32 = 1 KiB
// SLM for the ASCII alphabet), then every thread runs the classic Myers
// recurrence in registers: ~20 bitwise ops per text byte, zero inner DP loop,
// no per-thread DP row, no register spills.
//
// The 64-bit bit-vector is implemented as two u32 words (lo/hi) with manual
// carry propagation — deliberately NO `u64` type, so the kernel needs no
// adapter feature (SHADER_INT64 is unavailable on many iGPUs and in browsers)
// and works on every WebGPU backend. Cost: ~5 extra ops per byte.
//
// Constraints (enforced by the Rust caller before dispatch):
//   * pattern length m in 1..=64 (fits one 64-bit bitmask)
//   * all inputs ASCII (Peq is indexed directly by byte value < 128)
//   * all pairs share the same first string (one Peq per workgroup)
//
// Text data is transposed at pack time on the CPU: chars_text[i*B + t] is the
// i-th byte of text t (i in 1..=len, 0 never read); padded entries are never
// read because each thread's loop is bounded by its exact text length.

struct Params {
    batch_size: u32, // total texts (stride of the transposed char matrix)
    max_len: u32,    // pattern length m (the bit-vector width)
    offset: u32,     // first text index handled by this dispatch (0 for a single dispatch)
}

@group(0) @binding(0) var<storage, read> pattern_chars: array<u32>; // linear: pattern[j]
@group(0) @binding(1) var<storage, read> chars_text: array<u32>;    // transposed: chars_text[i*B + t]
@group(0) @binding(2) var<storage, read> text_len: array<u32>;      // per-text byte lengths
@group(0) @binding(3) var<storage, read> unused: array<u32>;
@group(0) @binding(4) var<storage, read_write> results: array<u32>;
@group(0) @binding(5) var<uniform> params: Params;

// ASCII alphabet only: 2 x 128 u32 = 1 KiB SLM per workgroup, shared by all 64
// threads (one pattern per workgroup). The caller gates on ASCII.
var<workgroup> peq_lo: array<u32, 128>;
var<workgroup> peq_hi: array<u32, 128>;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let m = params.max_len;

    // Thread 0 builds the Peq table serially (<= 64 iterations); everyone
    // waits at the barrier. Multiple workgroups rebuild it independently
    // (workgroup-local storage, no cross-workgroup hazard).
    if (lid.x == 0u) {
        for (var j = 0u; j < 128u; j++) {
            peq_lo[j] = 0u;
            peq_hi[j] = 0u;
        }
        for (var j = 0u; j < m; j++) {
            let c = pattern_chars[j];
            if (c < 128u) {
                if (j < 32u) {
                    peq_lo[c] = peq_lo[c] | (1u << j);          // j in 0..=31
                } else {
                    peq_hi[c] = peq_hi[c] | (1u << (j - 32u));  // j in 32..=63
                }
            }
        }
    }
    workgroupBarrier();

    let t = params.offset + gid.x;
    if (t >= params.batch_size) {
        return;
    }
    let n = text_len[t];

    // 64-bit mask of m low bits, as (lo, hi). Uses only safe shifts:
    // m in 1..=64, so (m - 1) and (m - 33) are always in 0..=31.
    var mask_lo: u32 = 0u;
    var mask_hi: u32 = 0u;
    if (m < 32u) {
        mask_lo = (1u << m) - 1u;   // m in 1..=31
    } else {
        mask_lo = 0xFFFFFFFFu;      // m in 32..=64
    }
    if (m > 32u) {
        let hb = m - 32u;           // bits in the high word: 1..=32
        mask_hi = (1u << (hb - 1u)) * 2u - 1u; // *2 wraps to all-ones when hb = 32
    }

    // Myers 1999 init: Pv = mask (positive vertical delta), Mv = 0.
    var pv_lo: u32 = mask_lo;
    var pv_hi: u32 = mask_hi;
    var mv_lo: u32 = 0u;
    var mv_hi: u32 = 0u;
    var score: u32 = m;

    for (var i = 1u; i <= n; i++) {
        let ch = chars_text[i * params.batch_size + t];
        let eq_lo = peq_lo[ch];
        let eq_hi = peq_hi[ch];

        let xv_lo = eq_lo | mv_lo;
        let xv_hi = eq_hi | mv_hi;

        // (eq & pv) + pv as a 64-bit add with carry across the word boundary.
        let a_lo = eq_lo & pv_lo;
        let a_hi = eq_hi & pv_hi;
        let s_lo = a_lo + pv_lo;
        let carry = u32(s_lo < a_lo); // unsigned overflow iff sum < operand
        let s_hi = a_hi + pv_hi + carry;

        let xh_lo = (s_lo ^ pv_lo) | eq_lo;
        let xh_hi = (s_hi ^ pv_hi) | eq_hi;

        let ph_lo = mv_lo | ~(xh_lo | pv_lo);
        let ph_hi = mv_hi | ~(xh_hi | pv_hi);
        let mh_lo = pv_lo & xh_lo;
        let mh_hi = pv_hi & xh_hi;

        // Score update: the m-th bit of Ph / Mh (in the high word for m > 32).
        if (m > 32u) {
            let bit = m - 33u; // 0..=31 within the high word
            if ((ph_hi & (1u << bit)) != 0u) { score = score + 1u; }
            if ((mh_hi & (1u << bit)) != 0u) { score = score - 1u; }
        } else {
            let bit = m - 1u; // 0..=31 within the low word
            if ((ph_lo & (1u << bit)) != 0u) { score = score + 1u; }
            if ((mh_lo & (1u << bit)) != 0u) { score = score - 1u; }
        }

        // Shift Ph and Mh left by one, with the cross-word carry from bit 31.
        let ph_s_lo = (ph_lo << 1u) | 1u;
        let ph_s_hi = (ph_hi << 1u) | (ph_lo >> 31u);
        let mh_s_lo = mh_lo << 1u;
        let mh_s_hi = (mh_hi << 1u) | (mh_lo >> 31u);

        pv_lo = (mh_s_lo | ~(xv_lo | ph_s_lo)) & mask_lo;
        pv_hi = (mh_s_hi | ~(xv_hi | ph_s_hi)) & mask_hi;
        mv_lo = (ph_s_lo & xv_lo) & mask_lo;
        mv_hi = (ph_s_hi & xv_hi) & mask_hi;
    }

    results[t] = score;
}
