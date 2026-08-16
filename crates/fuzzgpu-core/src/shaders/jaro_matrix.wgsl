// fuzzgpu 2D Jaro-Winkler Matrix compute shader (bitmap matcher, transposed).
// Computes the cross-product similarity matrix between List A (rows) and
// List B (cols); one thread per (row, col) cell. Same bitmap matching as the
// batch kernel (jaro.wgsl), with the transposed layout per axis: chars_a are
// indexed [i * rows + row], chars_b are [j * cols + col], so threads in a
// workgroup row read consecutive addresses (coalesced).

struct JaroMatrixParams {
    rows: u32,
    cols: u32,
    winkler_p_bits: u32,
}

@group(0) @binding(0) var<storage, read> len_a: array<u32>;
@group(0) @binding(1) var<storage, read> chars_a: array<u32>;
@group(0) @binding(2) var<storage, read> len_b: array<u32>;
@group(0) @binding(3) var<storage, read> chars_b: array<u32>;
@group(0) @binding(4) var<storage, read_write> matrix: array<u32>;
@group(0) @binding(5) var<uniform> params: JaroMatrixParams;

fn bit_test(mb: vec4<u32>, j: u32) -> bool {
    return ((mb[j >> 5u] >> (j & 31u)) & 1u) != 0u;
}

fn bit_set(mb: vec4<u32>, j: u32) -> vec4<u32> {
    var v = mb;
    v[j >> 5u] = v[j >> 5u] | (1u << (j & 31u));
    return v;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;

    if (row >= params.rows || col >= params.cols) {
        return;
    }

    let out_idx = row * params.cols + col;

    let a_len = len_a[row];
    let b_len = len_b[col];

    // Edge cases
    if (a_len == 0u && b_len == 0u) {
        matrix[out_idx] = bitcast<u32>(1.0f);
        return;
    }
    if (a_len == 0u || b_len == 0u) {
        matrix[out_idx] = bitcast<u32>(0.0f);
        return;
    }

    // Sentinel for strings > 128 chars (CPU recomputes)
    if (a_len > 128u || b_len > 128u) {
        matrix[out_idx] = bitcast<u32>(-1.0f);
        return;
    }

    let half = max(a_len, b_len) / 2u;
    var match_distance = 0u;
    if (half > 0u) {
        match_distance = half - 1u;
    }

    var matched_b = vec4<u32>(0u);
    var matched_a = vec4<u32>(0u);
    var matches = 0u;

    for (var i = 0u; i < a_len; i++) {
        var lo = 0u;
        if (i > match_distance) {
            lo = i - match_distance;
        }
        let hi = min(i + match_distance + 1u, b_len);
        let ai = chars_a[i * params.rows + row];

        for (var j = lo; j < hi; j++) {
            if (bit_test(matched_b, j)) { continue; }
            if (ai != chars_b[j * params.cols + col]) { continue; }
            matched_b = bit_set(matched_b, j);
            matched_a = bit_set(matched_a, i);
            matches += 1u;
            break;
        }
    }

    if (matches == 0u) {
        matrix[out_idx] = bitcast<u32>(0.0f);
        return;
    }

    var transpositions = 0u;
    var k = 0u;
    for (var i = 0u; i < a_len; i++) {
        if (!bit_test(matched_a, i)) { continue; }
        while (!bit_test(matched_b, k)) { k += 1u; }
        if (chars_a[i * params.rows + row] != chars_b[k * params.cols + col]) {
            transpositions += 1u;
        }
        k += 1u;
    }

    let m_f = f32(matches);
    let a_f = f32(a_len);
    let b_f = f32(b_len);
    let t_f = f32(transpositions);
    let jaro = (m_f / a_f + m_f / b_f + (m_f - t_f / 2.0) / m_f) / 3.0;

    var jw = jaro;
    if (jaro >= 0.7) {
        let p = bitcast<f32>(params.winkler_p_bits);
        var prefix_len = 0u;
        let max_prefix = min(min(a_len, b_len), 4u);
        for (var i = 0u; i < max_prefix; i++) {
            if (chars_a[i * params.rows + row] == chars_b[i * params.cols + col]) {
                prefix_len += 1u;
            } else {
                break;
            }
        }
        jw = min(jaro + f32(prefix_len) * p * (1.0 - jaro), 1.0);
    }
    matrix[out_idx] = bitcast<u32>(jw);
}
