// fuzzgpu 2D Jaro-Winkler Matrix compute shader (bitmap matcher, transposed).
// Computes the cross-product similarity matrix between List A (rows) and
// List B (cols); one thread per (row, col) cell. Same bitmap matching as the
// batch kernel (jaro.wgsl), with the transposed layout per axis: chars_a are
// indexed [i * rows + row], chars_b are [j * cols + col], so threads in a
// workgroup row read consecutive addresses (coalesced).
//
// Output: per-cell 4×u32 — [matches, transpositions, prefix_len, 0] — at
// matrix[cell*4 .. cell*4+4]. The host assembles the final f64 score so GPU
// and CPU results are bit-identical (no f32 rounding in the shader).

struct JaroMatrixParams {
    rows: u32,
    cols: u32,
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
    let out = out_idx * 4u;

    let a_len = len_a[row];
    let b_len = len_b[col];

    // Edge cases (host recomputes empty pairs on CPU)
    if (a_len == 0u && b_len == 0u) {
        matrix[out] = 0xFFFFFFFFu;
        matrix[out + 1u] = 0u;
        matrix[out + 2u] = 0u;
        matrix[out + 3u] = 0u;
        return;
    }
    if (a_len == 0u || b_len == 0u) {
        matrix[out] = 0xFFFFFFFFu;
        matrix[out + 1u] = 0u;
        matrix[out + 2u] = 0u;
        matrix[out + 3u] = 0u;
        return;
    }

    // Sentinel for strings > 128 chars (CPU recomputes)
    if (a_len > 128u || b_len > 128u) {
        matrix[out] = 0xFFFFFFFFu;
        matrix[out + 1u] = 0u;
        matrix[out + 2u] = 0u;
        matrix[out + 3u] = 0u;
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
        matrix[out] = 0u;
        matrix[out + 1u] = 0u;
        matrix[out + 2u] = 0u;
        matrix[out + 3u] = 0u;
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

    // Winkler prefix length (host applies the f64 boost).
    var prefix_len = 0u;
    let max_prefix = min(min(a_len, b_len), 4u);
    for (var i = 0u; i < max_prefix; i++) {
        if (chars_a[i * params.rows + row] == chars_b[i * params.cols + col]) {
            prefix_len += 1u;
        } else {
            break;
        }
    }

    matrix[out] = matches;
    matrix[out + 1u] = transpositions;
    matrix[out + 2u] = prefix_len;
    matrix[out + 3u] = 0u;
}
