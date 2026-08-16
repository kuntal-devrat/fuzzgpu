// fuzzgpu affine-gap Needleman-Wunsch (Gotoh 1982) — anti-diagonal wavefront
// compute shader.
//
// Each workgroup processes ONE string pair (one workgroup per pair, so the
// pair index is the WORKGROUP id, not the global invocation id). Instead of
// one thread walking the whole DP matrix serially (the per-thread 3x129 f32
// rows of the general shader spill on iGPUs), the 128 threads of the
// workgroup cooperatively compute the DP matrix anti-diagonal by
// anti-diagonal: every cell on diagonal k = i + j depends only on cells on
// diagonals k-1 and k-2, so all cells of a diagonal are independent and can
// be computed in parallel. This turns the serial O(m*n) step count into
// O(m+n) parallel steps — the GASAL/CUDA-aligner wavefront idea — and is
// where long-string alignment beats both the serial shader and (at scale)
// Rayon.
//
// Diagonal storage: for diagonal k we keep a `State { m, ix, iy }` per cell,
// indexed by column j (i = k - j). Three buffers ping-pong (k-2, k-1, k).
// 3 x 129 x 12 B = 4.6 KiB SLM per workgroup; all 128 threads share it.
//
// Boundary cells: (0,0) -> M=0; (0,j) -> M = Ix = open + j*ext; (i,0) ->
// M = Iy = open + i*ext. Same semantics as the CPU Gotoh implementation.
//
// NOTE: WGSL has no i64, so scores are f32 (exact while |v| < 2^24), matching
// the general affine shader's precision contract; the wrapper converts to i64.

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

const NEG: f32 = -1e30;
// Rust wrapper routes pairs 65..=128 chars here; max diagonal length is
// min(m,n) + 1 <= 129.
const CAP: u32 = 129;

struct State {
    m: f32,
    ix: f32,
    iy: f32,
}

var<workgroup> diags: array<array<State, CAP>, 3>;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    // One workgroup per pair: the pair index is the workgroup id.
    let pair_idx = params.offset + wgid.x;
    if (pair_idx >= params.batch_size) { return; }

    let a_start = offsets_a[pair_idx];
    let m = offsets_a[pair_idx + 1u] - a_start;
    let b_start = offsets_b[pair_idx];
    let n = offsets_b[pair_idx + 1u] - b_start;

    if (m > params.max_len || n > params.max_len) {
        results[pair_idx] = NEG;
        return;
    }

    // Seed diagonal 0 (cell (0,0)) and diagonal 1 (cells (1,0) and (0,1)).
    // Cells that do not exist for degenerate pairs are never read: the only
    // reader of diag k column j is cell (i+1, j+1) on diagonal k+2, which
    // exists only when i+1 <= m and j+1 <= n.
    //
    // FIX: Use a single workgroup barrier after ALL seed writes rather than
    // two separate if-branches with one barrier at the end. The WGSL memory
    // model does NOT guarantee cross-thread visibility between writes from
    // lid.x==0 and lid.x==1 without an explicit barrier separating them.
    // Assigning both cells from lid.x==0 and protecting with one barrier
    // is the simplest correct pattern (both writes are in the same thread).
    if (lid.x == 0u) {
        diags[0][0] = State(0.0, NEG, NEG);
        // Diagonal 1, column 0: cell (1, 0) — M = Iy = gap_open + gap_extend
        diags[1][0] = State(params.gap_open + params.gap_extend, NEG, params.gap_open + params.gap_extend);
        // Diagonal 1, column 1: cell (0, 1) — M = Ix = gap_open + gap_extend
        diags[1][1] = State(params.gap_open + params.gap_extend, params.gap_open + params.gap_extend, NEG);
    }
    workgroupBarrier();

    for (var k = 2u; k <= m + n; k++) {
        // Active cells on diagonal k: (i, j) with i = k - j, 0 <= i <= m, 0 <= j <= n.
        let j_start = max(k, m) - m; // j >= k - m (i <= m); 0 when k <= m
        let j_end = min(n, k);       // j <= n and j <= k (i >= 0)
        let count = j_end - j_start + 1u;

        let t = lid.x;
        if (t < count) {
            let j = j_start + t;
            let i = k - j;

            let prev = diags[(k - 1u) % 3u];
            let prev2 = diags[(k - 2u) % 3u];

            var cell: State;
            if (i == 0u) {
                // Cell (0, j): M = Ix = open + j*ext.
                let gap = params.gap_open + f32(j) * params.gap_extend;
                cell = State(gap, gap, NEG);
            } else if (j == 0u) {
                // Cell (i, 0): M = Iy = open + i*ext.
                let gap = params.gap_open + f32(i) * params.gap_extend;
                cell = State(gap, NEG, gap);
            } else {
                let a_char = chars_a[a_start + i - 1u];
                let b_char = chars_b[b_start + j - 1u];
                let sub = select(params.mismatch_score, params.match_score, a_char == b_char);

                // M(i,j) = max(M,Ix,Iy)(i-1,j-1) + sub   -- diagonal k-2.
                let best_diag = max(prev2[j - 1u].m, max(prev2[j - 1u].ix, prev2[j - 1u].iy));
                let cell_m = best_diag + sub;

                // Ix(i,j) from (i,j-1) -- diagonal k-1, column j-1.
                let cell_ix = max(prev[j - 1u].ix + params.gap_extend,
                    max(prev[j - 1u].m + params.gap_open + params.gap_extend,
                        prev[j - 1u].iy + params.gap_open + params.gap_extend));

                // Iy(i,j) from (i-1,j) -- diagonal k-1, column j.
                let cell_iy = max(prev[j].iy + params.gap_extend,
                    max(prev[j].m + params.gap_open + params.gap_extend,
                        prev[j].ix + params.gap_open + params.gap_extend));

                cell = State(cell_m, cell_ix, cell_iy);
            }
            diags[k % 3u][j] = cell;
        }
        workgroupBarrier();
    }

    // Final cell (m, n) sits on diagonal m+n at column n.
    if (lid.x == 0u) {
        let fin = diags[(m + n) % 3u][n];
        results[pair_idx] = max(fin.m, max(fin.ix, fin.iy));
    }
}
