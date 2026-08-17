// fuzzgpu Jaro-Winkler compute shader (bitmap matcher, transposed layout).
// Each invocation processes one string pair.
// Max string length: 128 characters.
// Output: per-pair 4×u32 — [matches, transpositions, prefix_len, 0] — at
// results[pair_idx*4 .. pair_idx*4+4]. The host assembles the final f64 score
// (rapidfuzz's exact formula), so GPU and CPU results are bit-identical; the
// f32 score math is deliberately NOT done in the shader (f32 rounding there
// produced ~1e-7 deviations from the f64 CPU reference).
//
// Two performance-critical design choices, both learned from measured
// hardware behavior:
//
// 1. Bitmap matching (no per-thread arrays). The original kernel kept two
//    129-entry u32 match-flag arrays per thread in private memory (~1 KiB),
//    which the driver spills to local memory on iGPUs — that spill dominated
//    the kernel time. Here matched positions are a 128-bit bitmap (4 u32
//    words, no SHADER_INT64 feature needed) in registers: 16 bytes of
//    private state, no spills. Semantics are identical to the CPU reference
//    (jaro_inner_slice / jaro_bitpar): first unmatched j in [i-wd, i+wd],
//    then the ordered k-th-pair transposition walk.
//
// 2. Transposed, pair-major char layout. With per-pair contiguous strings,
//    the O(m·w) window scan issues one random global-memory load per (i, j)
//    comparison (64 threads hitting 64 scattered addresses = a cache line
//    fetch each — measured ~200+ cycle latency, which dominated the kernel:
//    33 ms at 50k pairs). With chars_x[i*B + t] (B = batch size), for a
//    fixed position all 64 threads read consecutive addresses — one
//    coalesced cache-line fetch shared by the whole workgroup. This is the
//    same layout fix that made the Levenshtein short-DP kernel competitive.
//
// The window-scan and transposition semantics are byte-for-byte the CPU
// reference's (jaro_inner_slice).

struct JaroParams {
    batch_size: u32,
    max_len: u32,
    offset: u32,
}

@group(0) @binding(0) var<storage, read> len_a: array<u32>;
@group(0) @binding(1) var<storage, read> chars_a: array<u32>; // chars_a[i*B + t]
@group(0) @binding(2) var<storage, read> len_b: array<u32>;
@group(0) @binding(3) var<storage, read> chars_b: array<u32>; // chars_b[j*B + t]
@group(0) @binding(4) var<storage, read_write> results: array<u32>;
@group(0) @binding(5) var<uniform> params: JaroParams;

// Test bit j of the 128-bit bitmap (4 words, little-endian). Dynamic vector
// indexing (mb[j >> 5]) compiles to a single lane select on all backends.
fn bit_test(mb: vec4<u32>, j: u32) -> bool {
    return ((mb[j >> 5u] >> (j & 31u)) & 1u) != 0u;
}

// Return the bitmap with bit j set.
fn bit_set(mb: vec4<u32>, j: u32) -> vec4<u32> {
    var v = mb;
    v[j >> 5u] = v[j >> 5u] | (1u << (j & 31u));
    return v;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pair_idx = params.offset + gid.x;
    if (pair_idx >= params.batch_size) { return; }

    let a_len = len_a[pair_idx];
    let b_len = len_b[pair_idx];
    let out = pair_idx * 4u;

    // Sentinel: matches = 0xFFFFFFFF signals "CPU recompute" (Rust side).
    if (a_len > 128u || b_len > 128u) {
        results[out] = 0xFFFFFFFFu;
        results[out + 1u] = 0u;
        results[out + 2u] = 0u;
        results[out + 3u] = 0u;
        return;
    }

    // Edge cases
    if (a_len == 0u && b_len == 0u) {
        results[out] = 0xFFFFFFFFu;
        results[out + 1u] = 0u;
        results[out + 2u] = 0u;
        results[out + 3u] = 0u;
        return;
    }
    if (a_len == 0u || b_len == 0u) {
        results[out] = 0xFFFFFFFFu;
        results[out + 1u] = 0u;
        results[out + 2u] = 0u;
        results[out + 3u] = 0u;
        return;
    }

    // Match distance: wd = max(m, n) / 2 - 1 (saturating).
    let half = max(a_len, b_len) / 2u;
    var match_distance = 0u;
    if (half > 0u) {
        match_distance = half - 1u;
    }

    var matched_b = vec4<u32>(0u);
    var matched_a = vec4<u32>(0u);
    var matches = 0u;

    // Matching pass: for each i in a, the first unmatched j in the window
    // [i - wd, i + wd] with a[i] == b[j]. Coalesced transposed reads.
    for (var i = 0u; i < a_len; i++) {
        var lo = 0u;
        if (i > match_distance) {
            lo = i - match_distance;
        }
        let hi = min(i + match_distance + 1u, b_len);
        let ai = chars_a[i * params.batch_size + pair_idx];

        for (var j = lo; j < hi; j++) {
            if (bit_test(matched_b, j)) { continue; }
            if (ai != chars_b[j * params.batch_size + pair_idx]) { continue; }
            matched_b = bit_set(matched_b, j);
            matched_a = bit_set(matched_a, i);
            matches += 1u;
            break;
        }
    }

    if (matches == 0u) {
        results[out] = 0u;
        results[out + 1u] = 0u;
        results[out + 2u] = 0u;
        results[out + 3u] = 0u;
        return;
    }

    // Ordered k-th-pair transposition walk (mirrors the reference's
    // `while !b_matches[k]` loop: the k-th matched position of a is paired
    // with the k-th matched position of b, in order).
    var transpositions = 0u;
    var k = 0u;
    for (var i = 0u; i < a_len; i++) {
        if (!bit_test(matched_a, i)) { continue; }
        while (!bit_test(matched_b, k)) { k += 1u; }
        if (chars_a[i * params.batch_size + pair_idx] != chars_b[k * params.batch_size + pair_idx]) {
            transpositions += 1u;
        }
        k += 1u;
    }

    // Winkler prefix length (host applies the f64 boost).
    var prefix_len = 0u;
    let max_prefix = min(min(a_len, b_len), 4u);
    for (var i = 0u; i < max_prefix; i++) {
        if (chars_a[i * params.batch_size + pair_idx] == chars_b[i * params.batch_size + pair_idx]) {
            prefix_len += 1u;
        } else {
            break;
        }
    }

    results[out] = matches;
    results[out + 1u] = transpositions;
    results[out + 2u] = prefix_len;
    results[out + 3u] = 0u;
}
