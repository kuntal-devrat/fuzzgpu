// fuzzgpu 2D Levenshtein Matrix compute shader
// Computes cross-product distance matrix between List A (rows) and List B (cols).
// Avoids O(N*M) string data replication by storing List A and List B separately.

struct MatrixParams {
    rows: u32,
    cols: u32,
}

@group(0) @binding(0) var<storage, read> offsets_a: array<u32>;
@group(0) @binding(1) var<storage, read> chars_a: array<u32>;
@group(0) @binding(2) var<storage, read> offsets_b: array<u32>;
@group(0) @binding(3) var<storage, read> chars_b: array<u32>;
@group(0) @binding(4) var<storage, read_write> matrix: array<u32>;
@group(0) @binding(5) var<uniform> params: MatrixParams;

fn min3(a: u32, b: u32, c: u32) -> u32 {
    return min(a, min(b, c));
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;

    if (row >= params.rows || col >= params.cols) {
        return;
    }

    let out_idx = row * params.cols + col;

    let a_start = offsets_a[row];
    let a_end = offsets_a[row + 1u];
    let a_len = a_end - a_start;

    let b_start = offsets_b[col];
    let b_end = offsets_b[col + 1u];
    let b_len = b_end - b_start;

    // Fast paths
    if (a_len == 0u) {
        matrix[out_idx] = b_len;
        return;
    }
    if (b_len == 0u) {
        matrix[out_idx] = a_len;
        return;
    }

    // Guard: strings longer than 256 chars get sentinel (CPU recomputes)
    if (a_len > 256u || b_len > 256u) {
        matrix[out_idx] = 0xFFFFFFFFu;
        return;
    }

    // Single-row DP + scalar diagonal
    var dp: array<u32, 257>;
    let num_cols = b_len + 1u;

    for (var j = 0u; j < num_cols; j++) {
        dp[j] = j;
    }

    for (var i = 1u; i <= a_len; i++) {
        var prev_diag = dp[0];
        dp[0] = i;
        let a_char = chars_a[a_start + i - 1u];

        for (var j = 1u; j <= b_len; j++) {
            let b_char = chars_b[b_start + j - 1u];
            let old = dp[j];
            let cost = select(1u, 0u, a_char == b_char);
            let del = dp[j] + 1u;
            let ins = dp[j - 1u] + 1u;
            let sub = prev_diag + cost;
            dp[j] = min3(del, ins, sub);
            prev_diag = old;
        }
    }

    matrix[out_idx] = dp[b_len];
}
