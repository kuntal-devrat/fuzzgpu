// fuzzgpu short-string Levenshtein compute shader.
// Handles pairs where both strings are <= 64 characters (SHORT_MAX_LEN).
//
// Why this kernel exists (measured): the general kernel keeps a 257-entry DP
// row per thread in private memory, which the driver spills to local memory on
// iGPUs (Intel Iris Xe: 128 regs/thread ~= 512 B, the row is 1 KiB) and that
// spill dominates the kernel time. This kernel stores each thread's row in
// workgroup shared memory (SLM) instead — no per-thread spill, low register
// pressure — and reads char data in a transposed, pair-major layout so that
// for a fixed DP column, consecutive threads read consecutive addresses
// (coalesced loads).
//
// Layout: workgroup = 64 threads = 64 pairs; each thread runs a serial
// single-row DP entirely inside its own slice of `rows`. Slices are stride-65,
// so for a fixed column every thread hits a distinct shared-memory bank
// (65 % 32 == 1, no bank conflicts).
//
// Char data is transposed at pack time on the CPU: chars_a[i*B + t] is the
// i-th char of pair t (i in 1..=len, 0 never read). Padded entries past a
// pair's length are never read (loops are bounded by the exact per-pair
// lengths), so the CPU packer only writes valid entries.

struct Params {
    batch_size: u32,
    max_len: u32,
    offset: u32,
}

@group(0) @binding(0) var<storage, read> chars_a: array<u32>; // transposed: chars_a[i*B + t]
@group(0) @binding(1) var<storage, read> chars_b: array<u32>; // transposed
@group(0) @binding(2) var<storage, read> len_a: array<u32>;   // per-pair lengths
@group(0) @binding(3) var<storage, read> len_b: array<u32>;
@group(0) @binding(4) var<storage, read_write> results: array<u32>;
@group(0) @binding(5) var<uniform> params: Params;

// 64 pairs x (64 columns + 1) x 4 bytes = 16 KiB SLM per workgroup.
var<workgroup> rows: array<u32, 64 * 65>;

fn min3(a: u32, b: u32, c: u32) -> u32 {
    return min(a, min(b, c));
}

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let t = params.offset + gid.x;
    if (t >= params.batch_size) { return; }
    let base = lid.x * 65u;

    let a_len = len_a[t];
    let b_len = len_b[t];

    // The exact per-pair lengths bound every loop, so the transposed char
    // matrices are only ever read at valid indices (never the padded region).
    for (var j = 0u; j <= b_len; j++) {
        rows[base + j] = j;
    }

    for (var i = 1u; i <= a_len; i++) {
        let a_char = chars_a[i * params.batch_size + t];
        var prev_diag = rows[base];
        rows[base] = i;
        for (var j = 1u; j <= b_len; j++) {
            let b_char = chars_b[j * params.batch_size + t];
            let old = rows[base + j];
            let cost = select(1u, 0u, a_char == b_char);
            let del = rows[base + j] + 1u;
            let ins = rows[base + j - 1u] + 1u;
            let sub = prev_diag + cost;
            rows[base + j] = min3(del, ins, sub);
            prev_diag = old;
        }
    }

    results[t] = rows[base + b_len];
}
