// fuzzgpu Levenshtein distance compute shader
// Each invocation processes one string pair using single-row DP.
// Max string length: 256 characters (enforced by array size).

struct Params {
    batch_size: u32,
    max_len: u32,
    offset: u32,
}

@group(0) @binding(0) var<storage, read> offsets_a: array<u32>;
@group(0) @binding(1) var<storage, read> chars_a: array<u32>;
@group(0) @binding(2) var<storage, read> offsets_b: array<u32>;
@group(0) @binding(3) var<storage, read> chars_b: array<u32>;
@group(0) @binding(4) var<storage, read_write> results: array<u32>;
@group(0) @binding(5) var<uniform> params: Params;

fn min3(a: u32, b: u32, c: u32) -> u32 {
    return min(a, min(b, c));
}

// Reduced workgroup size from 256 to 64 to lower register pressure
// on integrated GPUs (Intel Iris Xe, Apple M-series).
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

    // Guard: strings longer than 256 get sentinel (Rust side recomputes on CPU)
    if (a_len > 256u || b_len > 256u) {
        results[pair_idx] = 0xFFFFFFFFu;
        return;
    }

    // Single-row DP + scalar diagonal variable.
    // Only one array of 257 elements instead of two arrays of 1025.
    // This halves register/local-memory usage.
    var row: array<u32, 257>;

    let cols = b_len + 1u;

    for (var j = 0u; j < cols; j++) {
        row[j] = j;
    }

    for (var i = 1u; i <= a_len; i++) {
        var prev_diag = row[0];
        row[0] = i;
        let a_char = chars_a[a_start + i - 1u];

        for (var j = 1u; j <= b_len; j++) {
            let b_char = chars_b[b_start + j - 1u];
            let old = row[j];
            let cost = select(1u, 0u, a_char == b_char);
            let del = row[j] + 1u;
            let ins = row[j - 1u] + 1u;
            let sub = prev_diag + cost;
            row[j] = min3(del, ins, sub);
            prev_diag = old;
        }
    }

    results[pair_idx] = row[b_len];
}
