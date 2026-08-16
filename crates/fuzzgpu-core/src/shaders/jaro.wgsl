// fuzzgpu Jaro-Winkler compute shader
// Each invocation processes one string pair.
// Max string length: 128 characters (match flag arrays).
// Output: Jaro-Winkler similarity as f32 bitcast to u32.

struct JaroParams {
    batch_size: u32,
    max_len: u32,
    offset: u32,
    winkler_p_bits: u32,  // f32 Winkler prefix weight, bitcast to u32
}

@group(0) @binding(0) var<storage, read> offsets_a: array<u32>;
@group(0) @binding(1) var<storage, read> chars_a: array<u32>;
@group(0) @binding(2) var<storage, read> offsets_b: array<u32>;
@group(0) @binding(3) var<storage, read> chars_b: array<u32>;
@group(0) @binding(4) var<storage, read_write> results: array<u32>;
@group(0) @binding(5) var<uniform> params: JaroParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pair_idx = params.offset + gid.x;
    if (pair_idx >= params.batch_size) { return; }

    let a_start = offsets_a[pair_idx];
    let a_end = offsets_a[pair_idx + 1u];
    let a_len = a_end - a_start;

    let b_start = offsets_b[pair_idx];
    let b_end = offsets_b[pair_idx + 1u];
    let b_len = b_end - b_start;

    // Sentinel for strings > 128 chars (Rust recomputes on CPU)
    if (a_len > 128u || b_len > 128u) {
        results[pair_idx] = bitcast<u32>(-1.0f);
        return;
    }

    // Edge cases
    if (a_len == 0u && b_len == 0u) {
        results[pair_idx] = bitcast<u32>(1.0f);
        return;
    }
    if (a_len == 0u || b_len == 0u) {
        results[pair_idx] = bitcast<u32>(0.0f);
        return;
    }

    // Match distance
    let half = max(a_len, b_len) / 2u;
    var match_distance = 0u;
    if (half > 0u) {
        match_distance = half - 1u;
    }

    // Match flags (0 = unmatched, 1 = matched)
    var a_matched: array<u32, 129>;
    var b_matched: array<u32, 129>;
    for (var i = 0u; i < a_len; i++) { a_matched[i] = 0u; }
    for (var i = 0u; i < b_len; i++) { b_matched[i] = 0u; }

    var matches = 0u;

    // Find matches within the match window
    for (var i = 0u; i < a_len; i++) {
        var lo = 0u;
        if (i > match_distance) {
            lo = i - match_distance;
        }
        let hi = min(i + match_distance + 1u, b_len);
        let ai = chars_a[a_start + i];

        for (var j = lo; j < hi; j++) {
            if (b_matched[j] != 0u) { continue; }
            if (ai != chars_b[b_start + j]) { continue; }
            a_matched[i] = 1u;
            b_matched[j] = 1u;
            matches += 1u;
            break;
        }
    }

    if (matches == 0u) {
        results[pair_idx] = bitcast<u32>(0.0f);
        return;
    }

    // Count transpositions
    var transpositions = 0u;
    var k = 0u;
    for (var i = 0u; i < a_len; i++) {
        if (a_matched[i] == 0u) { continue; }
        while (b_matched[k] == 0u) { k += 1u; }
        if (chars_a[a_start + i] != chars_b[b_start + k]) {
            transpositions += 1u;
        }
        k += 1u;
    }

    // Jaro similarity
    let m_f = f32(matches);
    let a_f = f32(a_len);
    let b_f = f32(b_len);
    let t_f = f32(transpositions);
    let jaro = (m_f / a_f + m_f / b_f + (m_f - t_f / 2.0) / m_f) / 3.0;

    // Winkler prefix bonus
    let p = bitcast<f32>(params.winkler_p_bits);
    var prefix_len = 0u;
    let max_prefix = min(min(a_len, b_len), 4u);
    for (var i = 0u; i < max_prefix; i++) {
        if (chars_a[a_start + i] == chars_b[b_start + i]) {
            prefix_len += 1u;
        } else {
            break;
        }
    }

    let jw = jaro + f32(prefix_len) * p * (1.0 - jaro);
    results[pair_idx] = bitcast<u32>(jw);
}
