#![allow(
    clippy::manual_clamp,
    clippy::manual_div_ceil,
    clippy::needless_range_loop
)]

use crate::{sat_add, sat_mul};
use rayon::prelude::*;

/// Needleman-Wunsch global alignment score with linear gap penalty.
///
/// Uses single-row DP + scalar diagonal for minimal memory.
/// Supports both ASCII fast-path and full Unicode characters.
/// All scores use `i64` with saturating arithmetic to prevent integer overflow on long sequences.
pub fn needleman_wunsch(
    a: &str,
    b: &str,
    match_score: i64,
    mismatch_score: i64,
    gap_penalty: i64,
) -> i64 {
    if a.is_ascii() && b.is_ascii() {
        needleman_wunsch_bytes(
            a.as_bytes(),
            b.as_bytes(),
            match_score,
            mismatch_score,
            gap_penalty,
        )
    } else {
        let a_chars: Vec<char> = a.chars().collect();
        let b_chars: Vec<char> = b.chars().collect();
        needleman_wunsch_chars(&a_chars, &b_chars, match_score, mismatch_score, gap_penalty)
    }
}

fn needleman_wunsch_bytes(
    a: &[u8],
    b: &[u8],
    match_score: i64,
    mismatch_score: i64,
    gap_penalty: i64,
) -> i64 {
    needleman_wunsch_slice(a, b, match_score, mismatch_score, gap_penalty)
}

fn needleman_wunsch_chars(
    a: &[char],
    b: &[char],
    match_score: i64,
    mismatch_score: i64,
    gap_penalty: i64,
) -> i64 {
    needleman_wunsch_slice(a, b, match_score, mismatch_score, gap_penalty)
}

fn needleman_wunsch_slice<T: PartialEq>(
    a: &[T],
    b: &[T],
    match_score: i64,
    mismatch_score: i64,
    gap_penalty: i64,
) -> i64 {
    let (m, n) = (a.len(), b.len());

    if m == 0 {
        return sat_mul(n as i64, gap_penalty);
    }
    if n == 0 {
        return sat_mul(m as i64, gap_penalty);
    }
    if a == b {
        return sat_mul(m as i64, match_score);
    }

    let mut row = vec![0i64; n + 1];
    for (j, item) in row.iter_mut().enumerate() {
        *item = sat_mul(j as i64, gap_penalty);
    }

    for i in 1..=m {
        let mut prev_diag = row[0];
        row[0] = sat_mul(i as i64, gap_penalty);
        let ai = &a[i - 1];
        for j in 1..=n {
            let old = row[j];
            let score = if ai == &b[j - 1] {
                match_score
            } else {
                mismatch_score
            };
            row[j] = sat_add(prev_diag, score)
                .max(sat_add(row[j], gap_penalty))
                .max(sat_add(row[j - 1], gap_penalty));
            prev_diag = old;
        }
    }
    row[n]
}

/// Batch Needleman-Wunsch with linear gap penalty.
pub fn needleman_wunsch_batch(
    query: &str,
    candidates: &[&str],
    match_score: i64,
    mismatch_score: i64,
    gap_penalty: i64,
) -> Vec<i64> {
    candidates
        .par_iter()
        .map(|c| needleman_wunsch(query, c, match_score, mismatch_score, gap_penalty))
        .collect()
}

const NEG_INF: i64 = -1_000_000_000_000_000_000;

/// Needleman-Wunsch global alignment score with affine gap penalties (Gotoh 1982 algorithm).
///
/// Affine model: gap of length k costs `gap_open + k * gap_extend`.
pub fn needleman_wunsch_affine(
    a: &str,
    b: &str,
    match_score: i64,
    mismatch_score: i64,
    gap_open: i64,
    gap_extend: i64,
) -> i64 {
    if a.is_ascii() && b.is_ascii() {
        needleman_wunsch_affine_slice(
            a.as_bytes(),
            b.as_bytes(),
            match_score,
            mismatch_score,
            gap_open,
            gap_extend,
        )
    } else {
        let a_chars: Vec<char> = a.chars().collect();
        let b_chars: Vec<char> = b.chars().collect();
        needleman_wunsch_affine_slice(
            &a_chars,
            &b_chars,
            match_score,
            mismatch_score,
            gap_open,
            gap_extend,
        )
    }
}

fn needleman_wunsch_affine_slice<T: PartialEq>(
    a: &[T],
    b: &[T],
    match_score: i64,
    mismatch_score: i64,
    gap_open: i64,
    gap_extend: i64,
) -> i64 {
    let (m, n) = (a.len(), b.len());

    if m == 0 && n == 0 {
        return 0;
    }
    if m == 0 {
        return sat_add(gap_open, sat_mul(n as i64, gap_extend));
    }
    if n == 0 {
        return sat_add(gap_open, sat_mul(m as i64, gap_extend));
    }
    if a == b {
        return sat_mul(m as i64, match_score);
    }

    let mut m_row = vec![NEG_INF; n + 1];
    let mut ix_row = vec![NEG_INF; n + 1];
    let mut iy_row = vec![NEG_INF; n + 1];

    m_row[0] = 0;
    for j in 1..=n {
        let gap_cost = sat_add(gap_open, sat_mul(j as i64, gap_extend));
        iy_row[j] = gap_cost;
        m_row[j] = gap_cost;
    }

    for i in 1..=m {
        let mut prev_m_diag = m_row[0];
        let mut prev_ix_diag = ix_row[0];
        let mut prev_iy_diag = iy_row[0];

        let gap_cost_i = sat_add(gap_open, sat_mul(i as i64, gap_extend));
        ix_row[0] = gap_cost_i;
        m_row[0] = gap_cost_i;
        iy_row[0] = NEG_INF;

        let ai = &a[i - 1];

        for j in 1..=n {
            let bj = &b[j - 1];
            let sub_score = if ai == bj {
                match_score
            } else {
                mismatch_score
            };

            let prev_diag_best = prev_m_diag.max(prev_ix_diag).max(prev_iy_diag);
            let new_m = sat_add(prev_diag_best, sub_score);

            let new_ix = sat_add(ix_row[j], gap_extend)
                .max(sat_add(sat_add(m_row[j], gap_open), gap_extend))
                .max(sat_add(sat_add(iy_row[j], gap_open), gap_extend));
            let new_iy = sat_add(iy_row[j - 1], gap_extend)
                .max(sat_add(sat_add(m_row[j - 1], gap_open), gap_extend))
                .max(sat_add(sat_add(ix_row[j - 1], gap_open), gap_extend));

            prev_m_diag = m_row[j];
            prev_ix_diag = ix_row[j];
            prev_iy_diag = iy_row[j];

            m_row[j] = new_m;
            ix_row[j] = new_ix;
            iy_row[j] = new_iy;
        }
    }

    m_row[n].max(ix_row[n]).max(iy_row[n])
}

/// Batch Needleman-Wunsch with affine gap penalty.
pub fn needleman_wunsch_affine_batch(
    query: &str,
    candidates: &[&str],
    match_score: i64,
    mismatch_score: i64,
    gap_open: i64,
    gap_extend: i64,
) -> Vec<i64> {
    candidates
        .par_iter()
        .map(|c| {
            needleman_wunsch_affine(query, c, match_score, mismatch_score, gap_open, gap_extend)
        })
        .collect()
}

#[cfg(feature = "gpu")]
pub mod gpu_ext {
    use super::*;
    use crate::gpu::{
        BufferPool, FuzzGpuError, GpuEngine, Result, SLOT_CHARS_A, SLOT_CHARS_B, SLOT_OFFSETS_A,
        SLOT_OFFSETS_B, SLOT_PARAMS, SLOT_RESULTS, SLOT_STAGING,
    };
    use bytemuck::{Pod, Zeroable};
    use std::sync::OnceLock;

    const SHADER_SRC: &str = include_str!("shaders/needleman_affine.wgsl");
    // Anti-diagonal wavefront kernel (one workgroup per pair, all cells of a
    // diagonal in parallel — see shaders/needleman_wavefront.wgsl). Kept as an
    // explicit API + differential-tested, but NOT the default routing: on an
    // iGPU the per-pair barrier cost makes it ~140x slower than the serial
    // shader for batches (see [`GpuNeedlemanAffineKernel::compute_gpu_wavefront`]).
    const WAVEFRONT_SHADER_SRC: &str = include_str!("shaders/needleman_wavefront.wgsl");
    const GPU_MAX_STRING_LEN: usize = 128;
    const MAX_DISPATCH: u32 = 65535;
    const MAX_DESIRED_CHUNK_PAIRS: usize = 500_000;
    // The shader writes this sentinel for oversized strings; no legitimate
    // score reaches it for the f32 ranges this kernel accepts.
    const SENTINEL_THRESHOLD: f32 = -1e29;

    #[repr(C)]
    #[derive(Copy, Clone, Pod, Zeroable)]
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

    pub struct GpuNeedlemanAffineKernel {
        engine: std::sync::Arc<GpuEngine>,
        pipeline: wgpu::ComputePipeline,
        wavefront_pipeline: wgpu::ComputePipeline,
        bind_group_layout: wgpu::BindGroupLayout,
        // Persistent buffer arena (see gpu::BufferPool) — removes the per-call
        // `create_buffer` cost that dominated small-batch dispatches.
        pool: std::sync::Mutex<BufferPool>,
    }

    static GLOBAL_GPU_KERNEL: OnceLock<GpuNeedlemanAffineKernel> = OnceLock::new();

    impl GpuNeedlemanAffineKernel {
        pub fn get() -> Result<&'static Self> {
            if let Some(k) = GLOBAL_GPU_KERNEL.get() {
                return Ok(k);
            }
            let engine = GpuEngine::get()?;
            let kernel = Self::new_inner(engine)?;
            let _ = GLOBAL_GPU_KERNEL.set(kernel);
            GLOBAL_GPU_KERNEL.get().ok_or_else(|| {
                FuzzGpuError::NoDevice("Needleman kernel unexpectedly absent after init".into())
            })
        }

        fn new_inner(engine: std::sync::Arc<GpuEngine>) -> Result<Self> {
            let bind_group_layout =
                engine
                    .device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("needleman-affine bgl"),
                        entries: &[
                            bg_entry(0, wgpu::BufferBindingType::Storage { read_only: true }),
                            bg_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                            bg_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                            bg_entry(3, wgpu::BufferBindingType::Storage { read_only: true }),
                            bg_entry(4, wgpu::BufferBindingType::Storage { read_only: false }),
                            bg_entry(5, wgpu::BufferBindingType::Uniform),
                        ],
                    });
            let layout = engine
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    // wgpu 30 wraps bind group layouts in `Option` and replaces
                    // `push_constant_ranges` with `immediate_size` (0 = none).
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });
            let pipeline = engine.build_compute_pipeline(
                "needleman-affine pipeline",
                &crate::gpu::effective_shader_source(SHADER_SRC),
                &layout,
            )?;
            let wavefront_pipeline = engine.build_compute_pipeline(
                "needleman-affine wavefront pipeline",
                &crate::gpu::effective_shader_source(WAVEFRONT_SHADER_SRC),
                &layout,
            )?;
            Ok(Self {
                engine,
                pipeline,
                wavefront_pipeline,
                bind_group_layout,
                pool: std::sync::Mutex::new(BufferPool::new()),
            })
        }

        /// GPU/CPU batch for affine-gap Needleman-Wunsch (Gotoh).
        ///
        /// Strings up to `GPU_MAX_STRING_LEN` (128) chars run on the GPU
        /// shader; longer ones route to the CPU implementation. Scores are f32
        /// on the GPU (WGSL has no i64) and exact for the practical scoring
        /// ranges — see the shader note. Below the engine's (auto or user-set)
        /// threshold the whole batch routes to CPU, mirroring the Levenshtein
        /// kernel.
        ///
        /// The general shader serves the whole <= 128-char range. An
        /// anti-diagonal wavefront kernel exists (see
        /// [`Self::compute_gpu_wavefront`]) but is NOT the default: measured on
        /// an iGPU it loses ~140x to the serial shader because per-pair
        /// workgroup barriers (~m+n per pair) dominate — the parallelism in
        /// fuzzy-matching batches is *across* thousands of independent pairs,
        /// which the serial shader (one thread per pair) already exploits.
        pub fn compute_batch(
            &self,
            pairs: &[(&str, &str)],
            match_score: i64,
            mismatch_score: i64,
            gap_open: i64,
            gap_extend: i64,
        ) -> Result<Vec<i64>> {
            // Serialize GPU dispatch across threads (gfx-rs/wgpu#10085).
            let _dispatch = self.engine.dispatch_lock();
            let n = pairs.len();
            if n == 0 {
                return Ok(vec![]);
            }

            // f32 precision guard: the GPU shader computes scores in f32 (WGSL
            // has no i64).  f32 can represent every integer exactly up to 2^24
            // (16,777,216).  A worst-case score is max(|match|, |mismatch|,
            // |gap_open|, |gap_extend|) * max_string_len; if that exceeds 2^24
            // the GPU would silently lose precision.  Route the whole batch to
            // CPU when any scoring parameter is out of the safe range.
            const F32_EXACT_MAX: i64 = 1 << 24; // 16,777,216
            let max_score_magnitude = [match_score, mismatch_score, gap_open, gap_extend]
                .iter()
                .map(|&v| v.unsigned_abs())
                .max()
                .unwrap_or(0);
            let worst_case = max_score_magnitude.saturating_mul(GPU_MAX_STRING_LEN as u64);
            if worst_case > F32_EXACT_MAX as u64 {
                // Scoring parameters exceed f32 exact range — use CPU for the
                // whole batch (same semantics, no precision loss).
                return Ok(pairs
                    .par_iter()
                    .map(|(a, b)| {
                        needleman_wunsch_affine(
                            a,
                            b,
                            match_score,
                            mismatch_score,
                            gap_open,
                            gap_extend,
                        )
                    })
                    .collect());
            }

            let mut results = vec![0i64; n];
            let mut gpu_indices: Vec<usize> = Vec::with_capacity(n);
            let mut cpu_indices: Vec<usize> = Vec::new();

            for (i, (a, b)) in pairs.iter().enumerate() {
                if a.chars().count() > GPU_MAX_STRING_LEN || b.chars().count() > GPU_MAX_STRING_LEN
                {
                    cpu_indices.push(i);
                } else {
                    gpu_indices.push(i);
                }
            }

            if !cpu_indices.is_empty() {
                let cpu_results: Vec<i64> = cpu_indices
                    .par_iter()
                    .map(|&i| {
                        needleman_wunsch_affine(
                            pairs[i].0,
                            pairs[i].1,
                            match_score,
                            mismatch_score,
                            gap_open,
                            gap_extend,
                        )
                    })
                    .collect();
                for (idx, &orig_i) in cpu_indices.iter().enumerate() {
                    results[orig_i] = cpu_results[idx];
                }
            }

            if gpu_indices.is_empty() || gpu_indices.len() < self.engine.effective_gpu_threshold() {
                // Below the (auto or user-set) threshold: CPU is cheaper.
                crate::gpu::GpuEngine::record_routing(0, n);
                let cpu_results: Vec<i64> = gpu_indices
                    .par_iter()
                    .map(|&i| {
                        needleman_wunsch_affine(
                            pairs[i].0,
                            pairs[i].1,
                            match_score,
                            mismatch_score,
                            gap_open,
                            gap_extend,
                        )
                    })
                    .collect();
                for (idx, &orig_i) in gpu_indices.iter().enumerate() {
                    results[orig_i] = cpu_results[idx];
                }
                return Ok(results);
            }
            crate::gpu::GpuEngine::record_routing(gpu_indices.len(), cpu_indices.len());

            // Chunk sizing from device limits, mirroring the Levenshtein kernel.
            let max_allowed_binding = self.engine.max_storage_buffer_binding_size as usize;
            let bytes_per_pair = (GPU_MAX_STRING_LEN * 4 * 2 + 8).max(128);
            let dynamic_chunk_size = (max_allowed_binding / bytes_per_pair)
                .min(MAX_DESIRED_CHUNK_PAIRS)
                .max(512);

            for stream_chunk in gpu_indices.chunks(dynamic_chunk_size) {
                let chunk_results = self.compute_gpu_chunk(
                    pairs,
                    stream_chunk,
                    match_score,
                    mismatch_score,
                    gap_open,
                    gap_extend,
                )?;
                for (idx, &orig_i) in stream_chunk.iter().enumerate() {
                    if chunk_results[idx] < SENTINEL_THRESHOLD {
                        results[orig_i] = needleman_wunsch_affine(
                            pairs[orig_i].0,
                            pairs[orig_i].1,
                            match_score,
                            mismatch_score,
                            gap_open,
                            gap_extend,
                        );
                    } else {
                        results[orig_i] = chunk_results[idx] as i64;
                    }
                }
            }

            Ok(results)
        }

        fn compute_gpu_chunk(
            &self,
            pairs: &[(&str, &str)],
            indices: &[usize],
            match_score: i64,
            mismatch_score: i64,
            gap_open: i64,
            gap_extend: i64,
        ) -> Result<Vec<f32>> {
            let batch_size = indices.len() as u32;

            let mut offsets_a: Vec<u32> = Vec::with_capacity(indices.len() + 1);
            let mut chars_a: Vec<u32> = Vec::new();
            let mut offsets_b: Vec<u32> = Vec::with_capacity(indices.len() + 1);
            let mut chars_b: Vec<u32> = Vec::new();
            offsets_a.push(0);
            offsets_b.push(0);

            for &i in indices {
                let (a, b) = pairs[i];
                chars_a.extend(a.chars().map(|c| c as u32));
                offsets_a.push(chars_a.len() as u32);
                chars_b.extend(b.chars().map(|c| c as u32));
                offsets_b.push(chars_b.len() as u32);
            }

            if chars_a.is_empty() {
                chars_a.push(0);
            }
            if chars_b.is_empty() {
                chars_b.push(0);
            }

            // Validate total buffer allocations against hardware max binding size.
            let chars_a_bytes = (chars_a.len() * 4) as u64;
            let chars_b_bytes = (chars_b.len() * 4) as u64;
            let results_size = (batch_size as u64) * 4;
            if chars_a_bytes > self.engine.max_buffer_size_effective()
                || chars_b_bytes > self.engine.max_buffer_size_effective()
                || results_size > self.engine.max_buffer_size_effective()
            {
                return Err(FuzzGpuError::BufferError(
                    "Buffer size exceeds device max_buffer_size".into(),
                ));
            }

            // Persistent buffers (see gpu::BufferPool) — same arena pattern as
            // the other kernels: ensure capacity, upload once, reuse.
            let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
            let offsets_bytes = ((offsets_a.len() * 4) as u64).max(results_size);
            pool.ensure(
                &self.engine.device,
                SLOT_OFFSETS_A,
                offsets_bytes,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                "oa",
            );
            pool.ensure(
                &self.engine.device,
                SLOT_OFFSETS_B,
                offsets_bytes,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                "ob",
            );
            pool.ensure(
                &self.engine.device,
                SLOT_CHARS_A,
                chars_a_bytes,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                "ca",
            );
            pool.ensure(
                &self.engine.device,
                SLOT_CHARS_B,
                chars_b_bytes,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                "cb",
            );
            pool.ensure(
                &self.engine.device,
                SLOT_RESULTS,
                results_size,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                "res",
            );
            pool.ensure(
                &self.engine.device,
                SLOT_STAGING,
                results_size,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                "stg",
            );
            pool.ensure(
                &self.engine.device,
                SLOT_PARAMS,
                std::mem::size_of::<Params>() as u64,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                "p",
            );

            pool.write(
                &self.engine.queue,
                SLOT_OFFSETS_A,
                bytemuck::cast_slice(&offsets_a),
            );
            pool.write(
                &self.engine.queue,
                SLOT_CHARS_A,
                bytemuck::cast_slice(&chars_a),
            );
            pool.write(
                &self.engine.queue,
                SLOT_OFFSETS_B,
                bytemuck::cast_slice(&offsets_b),
            );
            pool.write(
                &self.engine.queue,
                SLOT_CHARS_B,
                bytemuck::cast_slice(&chars_b),
            );

            let buf_offsets_a = pool.get(SLOT_OFFSETS_A);
            let buf_chars_a = pool.get(SLOT_CHARS_A);
            let buf_offsets_b = pool.get(SLOT_OFFSETS_B);
            let buf_chars_b = pool.get(SLOT_CHARS_B);
            let buf_results = pool.get(SLOT_RESULTS);
            let buf_staging = pool.get(SLOT_STAGING);
            let buf_params = pool.get(SLOT_PARAMS);

            // Per-chunk submit + readback (see the Levenshtein kernel: one
            // shared submit would give every dispatch the LAST chunk's offset).
            let mut gpu_results: Vec<f32> = Vec::with_capacity(batch_size as usize);
            let mut remaining = batch_size;
            let mut offset = 0u32;
            while remaining > 0 {
                let chunk = remaining.min(MAX_DISPATCH);
                let params = Params {
                    batch_size,
                    max_len: GPU_MAX_STRING_LEN as u32,
                    offset,
                    match_score: match_score as f32,
                    mismatch_score: mismatch_score as f32,
                    gap_open: gap_open as f32,
                    gap_extend: gap_extend as f32,
                    _pad: 0,
                };
                pool.write(&self.engine.queue, SLOT_PARAMS, bytemuck::bytes_of(&params));

                let bg = self
                    .engine
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: buf_offsets_a.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: buf_chars_a.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: buf_offsets_b.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: buf_chars_b.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: buf_results.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: buf_params.as_entire_binding(),
                            },
                        ],
                    });

                let mut encoder =
                    self.engine
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("affine enc"),
                        });
                let workgroups = chunk.div_ceil(16);
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.pipeline);
                    pass.set_bind_group(0, &bg, &[]);
                    pass.dispatch_workgroups(workgroups, 1, 1);
                }
                let chunk_bytes = (chunk as u64) * 4;
                encoder.copy_buffer_to_buffer(buf_results, 0, buf_staging, 0, chunk_bytes);
                let bytes = self.engine.readback(encoder, &pool, chunk_bytes)?;
                gpu_results.extend_from_slice(bytemuck::cast_slice(&bytes));

                remaining -= chunk;
                offset += chunk;
            }
            Ok(gpu_results)
        }

        /// Anti-diagonal wavefront dispatch for 65..=128-char pairs
        /// (needleman_wavefront.wgsl): one workgroup per pair, cells of each
        /// diagonal computed in parallel, O(m+n) sequential steps instead of
        /// O(m*n) on a single thread.
        ///
        /// Pairs are packed with the standard offsets+chars layout (identical
        /// to the general shader), so the same bind group layout serves both
        /// pipelines. Chunks beyond MAX_DISPATCH workgroups get their own
        /// submit+readback (each chunk's params buffer must be written before
        /// its own submission — the shared pool slot cannot hold two offsets
        /// for two dispatches in one encoder).
        /// Anti-diagonal wavefront dispatch for 65..=128-char pairs
        /// (needleman_wavefront.wgsl): one workgroup per pair, cells of each
        /// diagonal computed in parallel, O(m+n) sequential steps instead of
        /// O(m*n) on a single thread.
        ///
        /// Exposed explicitly (not the default routing) so advanced callers can
        /// opt in and the correctness differential can pin it against the CPU
        /// Gotoh oracle. Measured reality on an Intel iGPU: the ~m+n
        /// workgroup barriers per pair cost ~140x more than the serial shader
        /// for 1000 x 80-char pairs, because batch parallelism comes from
        /// thousands of independent pairs, not from intra-pair diagonals.
        #[doc(hidden)]
        pub fn compute_gpu_wavefront(
            &self,
            pairs: &[(&str, &str)],
            indices: &[usize],
            match_score: i64,
            mismatch_score: i64,
            gap_open: i64,
            gap_extend: i64,
        ) -> Result<Vec<f32>> {
            let batch_size = indices.len() as u32;
            if batch_size == 0 {
                return Ok(vec![]);
            }

            let mut offsets_a: Vec<u32> = Vec::with_capacity(indices.len() + 1);
            let mut chars_a: Vec<u32> = Vec::new();
            let mut offsets_b: Vec<u32> = Vec::with_capacity(indices.len() + 1);
            let mut chars_b: Vec<u32> = Vec::new();
            offsets_a.push(0);
            offsets_b.push(0);

            for &i in indices {
                let (a, b) = pairs[i];
                chars_a.extend(a.chars().map(|c| c as u32));
                offsets_a.push(chars_a.len() as u32);
                chars_b.extend(b.chars().map(|c| c as u32));
                offsets_b.push(chars_b.len() as u32);
            }

            if chars_a.is_empty() {
                chars_a.push(0);
            }
            if chars_b.is_empty() {
                chars_b.push(0);
            }

            let chars_a_bytes = (chars_a.len() * 4) as u64;
            let chars_b_bytes = (chars_b.len() * 4) as u64;
            let results_size = (batch_size as u64) * 4;
            let limit = self.engine.max_buffer_size_effective();
            if chars_a_bytes > limit || chars_b_bytes > limit || results_size > limit {
                return Err(FuzzGpuError::BufferError(
                    "Wavefront buffer size exceeds device max_buffer_size".into(),
                ));
            }

            let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
            let offsets_bytes = ((offsets_a.len() * 4) as u64).max(results_size);
            pool.ensure(
                &self.engine.device,
                SLOT_OFFSETS_A,
                offsets_bytes,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                "wfoa",
            );
            pool.ensure(
                &self.engine.device,
                SLOT_OFFSETS_B,
                offsets_bytes,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                "wfob",
            );
            pool.ensure(
                &self.engine.device,
                SLOT_CHARS_A,
                chars_a_bytes,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                "wfca",
            );
            pool.ensure(
                &self.engine.device,
                SLOT_CHARS_B,
                chars_b_bytes,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                "wfcb",
            );
            pool.ensure(
                &self.engine.device,
                SLOT_RESULTS,
                results_size,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                "wfres",
            );
            pool.ensure(
                &self.engine.device,
                SLOT_STAGING,
                results_size,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                "wfstg",
            );
            pool.ensure(
                &self.engine.device,
                SLOT_PARAMS,
                std::mem::size_of::<Params>() as u64,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                "wfprm",
            );

            pool.write(
                &self.engine.queue,
                SLOT_OFFSETS_A,
                bytemuck::cast_slice(&offsets_a),
            );
            pool.write(
                &self.engine.queue,
                SLOT_CHARS_A,
                bytemuck::cast_slice(&chars_a),
            );
            pool.write(
                &self.engine.queue,
                SLOT_OFFSETS_B,
                bytemuck::cast_slice(&offsets_b),
            );
            pool.write(
                &self.engine.queue,
                SLOT_CHARS_B,
                bytemuck::cast_slice(&chars_b),
            );

            let buf_offsets_a = pool.get(SLOT_OFFSETS_A);
            let buf_chars_a = pool.get(SLOT_CHARS_A);
            let buf_offsets_b = pool.get(SLOT_OFFSETS_B);
            let buf_chars_b = pool.get(SLOT_CHARS_B);
            let buf_results = pool.get(SLOT_RESULTS);
            let buf_params = pool.get(SLOT_PARAMS);

            let mut out: Vec<f32> = Vec::with_capacity(batch_size as usize);
            let mut remaining = batch_size;
            let mut offset = 0u32;

            // Each chunk: fresh params write -> own encoder -> own readback.
            while remaining > 0 {
                let chunk = remaining.min(MAX_DISPATCH);
                let params = Params {
                    batch_size,
                    max_len: GPU_MAX_STRING_LEN as u32,
                    offset,
                    match_score: match_score as f32,
                    mismatch_score: mismatch_score as f32,
                    gap_open: gap_open as f32,
                    gap_extend: gap_extend as f32,
                    _pad: 0,
                };
                pool.write(&self.engine.queue, SLOT_PARAMS, bytemuck::bytes_of(&params));

                let bg = self
                    .engine
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: buf_offsets_a.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: buf_chars_a.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: buf_offsets_b.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: buf_chars_b.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: buf_results.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: buf_params.as_entire_binding(),
                            },
                        ],
                    });

                let mut encoder =
                    self.engine
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("wavefront enc"),
                        });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.wavefront_pipeline);
                    pass.set_bind_group(0, &bg, &[]);
                    pass.dispatch_workgroups(chunk, 1, 1);
                }
                encoder.copy_buffer_to_buffer(
                    buf_results,
                    0,
                    pool.get(SLOT_STAGING),
                    0,
                    results_size,
                );
                let bytes = self.engine.readback(encoder, &pool, results_size)?;
                let flat: &[f32] = bytemuck::cast_slice(&bytes);
                out.extend_from_slice(&flat[..chunk as usize]);

                remaining -= chunk;
                offset += chunk;
            }
            Ok(out)
        }

        /// Create a batched dispatch: enqueue several pair-lists, then
        /// [`GpuNeedlemanAffineBatch::execute`] once. All GPU-eligible pairs
        /// across every enqueued op share one command encoder and one readback,
        /// amortizing the per-call sync round-trip. Scoring parameters are fixed
        /// for the whole batch, matching [`Self::compute_batch`].
        pub fn batch(
            &self,
            match_score: i64,
            mismatch_score: i64,
            gap_open: i64,
            gap_extend: i64,
        ) -> GpuNeedlemanAffineBatch<'_> {
            GpuNeedlemanAffineBatch {
                kernel: self,
                match_score,
                mismatch_score,
                gap_open,
                gap_extend,
                ops: Vec::new(),
            }
        }
    }

    /// A queued set of affine Needleman-Wunsch batch operations executed with a
    /// single GPU dispatch + readback. Each enqueued op returns its own
    /// `Vec<i64>` of scores, with the same semantics as
    /// [`GpuNeedlemanAffineKernel::compute_batch`] (f32 GPU precision, sentinel
    /// recompute) applied per op.
    pub struct GpuNeedlemanAffineBatch<'k> {
        kernel: &'k GpuNeedlemanAffineKernel,
        match_score: i64,
        mismatch_score: i64,
        gap_open: i64,
        gap_extend: i64,
        ops: Vec<Vec<(&'k str, &'k str)>>,
    }

    impl<'k> GpuNeedlemanAffineBatch<'k> {
        /// Enqueue one operation (a list of pairs) into this batch.
        pub fn add(&mut self, pairs: &[(&'k str, &'k str)]) {
            self.ops.push(pairs.to_vec());
        }

        /// Number of enqueued operations.
        pub fn len(&self) -> usize {
            self.ops.len()
        }

        pub fn is_empty(&self) -> bool {
            self.ops.is_empty()
        }

        /// Execute all enqueued operations in one dispatch + readback, returning
        /// one result vector per op.
        pub fn execute(self) -> Result<Vec<Vec<i64>>> {
            // Serialize GPU dispatch across threads (gfx-rs/wgpu#10085).
            let _dispatch = self.kernel.engine.dispatch_lock();
            let n_ops = self.ops.len();
            if n_ops == 0 {
                return Ok(vec![]);
            }

            // f32 precision guard — same as compute_batch: if scoring params
            // exceed the f32 exact integer range (2^24 = 16,777,216) the shader
            // would silently lose precision; route the whole batch to CPU.
            const F32_EXACT_MAX: i64 = 1 << 24;
            let max_score_magnitude = [
                self.match_score,
                self.mismatch_score,
                self.gap_open,
                self.gap_extend,
            ]
            .iter()
            .map(|&v| v.unsigned_abs())
            .max()
            .unwrap_or(0);
            let worst_case = max_score_magnitude.saturating_mul(GPU_MAX_STRING_LEN as u64);
            if worst_case > F32_EXACT_MAX as u64 {
                let mut out: Vec<Vec<i64>> = Vec::with_capacity(n_ops);
                for pairs in &self.ops {
                    let row = pairs
                        .par_iter()
                        .map(|(a, b)| {
                            needleman_wunsch_affine(
                                a,
                                b,
                                self.match_score,
                                self.mismatch_score,
                                self.gap_open,
                                self.gap_extend,
                            )
                        })
                        .collect();
                    out.push(row);
                }
                return Ok(out);
            }

            // Classify + pack every pair across ops (mirrors compute_batch:
            // only >128-char pairs route to CPU; no empty/identical
            // short-circuits — the shader handles all lengths <= 128).
            let mut out: Vec<Vec<i64>> = Vec::with_capacity(n_ops);
            let mut gpu_ranges: Vec<(u32, u32)> = Vec::with_capacity(n_ops);
            let mut op_gpu_to_pair: Vec<Vec<usize>> = Vec::with_capacity(n_ops);
            let mut cpu_oversized: Vec<(usize, usize)> = Vec::new();
            let mut offsets_a: Vec<u32> = vec![0];
            let mut chars_a: Vec<u32> = Vec::new();
            let mut offsets_b: Vec<u32> = vec![0];
            let mut chars_b: Vec<u32> = Vec::new();
            let mut gpu_global: u32 = 0;

            for (op_i, pairs) in self.ops.iter().enumerate() {
                let op_results = vec![0i64; pairs.len()];
                let mut op_gpu: Vec<usize> = Vec::new();
                for (j, (a, b)) in pairs.iter().enumerate() {
                    let a_len = a.chars().count();
                    let b_len = b.chars().count();
                    if a_len > GPU_MAX_STRING_LEN || b_len > GPU_MAX_STRING_LEN {
                        cpu_oversized.push((op_i, j));
                    } else {
                        chars_a.extend(a.chars().map(|c| c as u32));
                        offsets_a.push(chars_a.len() as u32);
                        chars_b.extend(b.chars().map(|c| c as u32));
                        offsets_b.push(chars_b.len() as u32);
                        op_gpu.push(j);
                    }
                }
                let start = gpu_global;
                gpu_global += op_gpu.len() as u32;
                gpu_ranges.push((start, op_gpu.len() as u32));
                op_gpu_to_pair.push(op_gpu);
                out.push(op_results);
            }

            let total_gpu = gpu_global as usize;

            if !cpu_oversized.is_empty() {
                let cpu_res: Vec<i64> = cpu_oversized
                    .par_iter()
                    .map(|&(op, j)| {
                        needleman_wunsch_affine(
                            self.ops[op][j].0,
                            self.ops[op][j].1,
                            self.match_score,
                            self.mismatch_score,
                            self.gap_open,
                            self.gap_extend,
                        )
                    })
                    .collect();
                for (k, &(op, j)) in cpu_oversized.iter().enumerate() {
                    out[op][j] = cpu_res[k];
                }
            }

            // Below the (auto or user-set) GPU threshold the whole batch is
            // cheaper on CPU.
            if total_gpu < self.kernel.engine.effective_gpu_threshold() {
                crate::gpu::GpuEngine::record_routing(0, total_gpu);
                let mut gpu_op_pair: Vec<(usize, usize)> = Vec::with_capacity(total_gpu);
                for (op, &(_, count)) in gpu_ranges.iter().enumerate() {
                    for k in 0..count as usize {
                        gpu_op_pair.push((op, op_gpu_to_pair[op][k]));
                    }
                }
                let cpu_res: Vec<i64> = gpu_op_pair
                    .par_iter()
                    .map(|&(op, j)| {
                        needleman_wunsch_affine(
                            self.ops[op][j].0,
                            self.ops[op][j].1,
                            self.match_score,
                            self.mismatch_score,
                            self.gap_open,
                            self.gap_extend,
                        )
                    })
                    .collect();
                for (idx, &(op, j)) in gpu_op_pair.iter().enumerate() {
                    out[op][j] = cpu_res[idx];
                }
                return Ok(out);
            }
            crate::gpu::GpuEngine::record_routing(total_gpu, 0);

            if chars_a.is_empty() {
                chars_a.push(0);
            }
            if chars_b.is_empty() {
                chars_b.push(0);
            }

            let chars_a_bytes = (chars_a.len() * 4) as u64;
            let chars_b_bytes = (chars_b.len() * 4) as u64;
            let results_size = (total_gpu as u64) * 4;
            let limit = self.kernel.engine.max_buffer_size_effective();
            if chars_a_bytes > limit || chars_b_bytes > limit || results_size > limit {
                return Err(FuzzGpuError::BufferError(
                    "Batch buffer size exceeds device max_buffer_size".into(),
                ));
            }

            // Single submit: all chunks recorded into one encoder, read back once.
            let mut pool = self.kernel.pool.lock().unwrap_or_else(|e| e.into_inner());
            let offsets_bytes = ((offsets_a.len() * 4) as u64).max(results_size);
            pool.ensure(
                &self.kernel.engine.device,
                SLOT_OFFSETS_A,
                offsets_bytes,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                "bnoa",
            );
            pool.ensure(
                &self.kernel.engine.device,
                SLOT_OFFSETS_B,
                offsets_bytes,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                "bnob",
            );
            pool.ensure(
                &self.kernel.engine.device,
                SLOT_CHARS_A,
                chars_a_bytes,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                "bnca",
            );
            pool.ensure(
                &self.kernel.engine.device,
                SLOT_CHARS_B,
                chars_b_bytes,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                "bncb",
            );
            pool.ensure(
                &self.kernel.engine.device,
                SLOT_RESULTS,
                results_size,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                "bnres",
            );
            pool.ensure(
                &self.kernel.engine.device,
                SLOT_STAGING,
                results_size,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                "bnstg",
            );
            pool.ensure(
                &self.kernel.engine.device,
                SLOT_PARAMS,
                std::mem::size_of::<Params>() as u64,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                "bnp",
            );

            pool.write(
                &self.kernel.engine.queue,
                SLOT_OFFSETS_A,
                bytemuck::cast_slice(&offsets_a),
            );
            pool.write(
                &self.kernel.engine.queue,
                SLOT_CHARS_A,
                bytemuck::cast_slice(&chars_a),
            );
            pool.write(
                &self.kernel.engine.queue,
                SLOT_OFFSETS_B,
                bytemuck::cast_slice(&offsets_b),
            );
            pool.write(
                &self.kernel.engine.queue,
                SLOT_CHARS_B,
                bytemuck::cast_slice(&chars_b),
            );

            let buf_offsets_a = pool.get(SLOT_OFFSETS_A);
            let buf_chars_a = pool.get(SLOT_CHARS_A);
            let buf_offsets_b = pool.get(SLOT_OFFSETS_B);
            let buf_chars_b = pool.get(SLOT_CHARS_B);
            let buf_results = pool.get(SLOT_RESULTS);
            let buf_params = pool.get(SLOT_PARAMS);

            // Per-chunk submit + readback (see the Levenshtein kernel: one
            // shared submit would give every dispatch the LAST chunk's offset).
            let mut raw: Vec<f32> = Vec::with_capacity(total_gpu);
            let mut remaining = total_gpu as u32;
            let mut offset = 0u32;
            while remaining > 0 {
                let chunk = remaining.min(MAX_DISPATCH);
                let params = Params {
                    batch_size: total_gpu as u32,
                    max_len: GPU_MAX_STRING_LEN as u32,
                    offset,
                    match_score: self.match_score as f32,
                    mismatch_score: self.mismatch_score as f32,
                    gap_open: self.gap_open as f32,
                    gap_extend: self.gap_extend as f32,
                    _pad: 0,
                };
                pool.write(
                    &self.kernel.engine.queue,
                    SLOT_PARAMS,
                    bytemuck::bytes_of(&params),
                );

                let bg = self
                    .kernel
                    .engine
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.kernel.bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: buf_offsets_a.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: buf_chars_a.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: buf_offsets_b.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: buf_chars_b.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: buf_results.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: buf_params.as_entire_binding(),
                            },
                        ],
                    });

                let mut encoder = self.kernel.engine.device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor {
                        label: Some("needleman batch encoder"),
                    },
                );
                let workgroups = chunk.div_ceil(16);
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.kernel.pipeline);
                    pass.set_bind_group(0, &bg, &[]);
                    pass.dispatch_workgroups(workgroups, 1, 1);
                }
                let chunk_bytes = (chunk as u64) * 4;
                encoder.copy_buffer_to_buffer(
                    buf_results,
                    0,
                    pool.get(SLOT_STAGING),
                    0,
                    chunk_bytes,
                );
                let bytes = self.kernel.engine.readback(encoder, &pool, chunk_bytes)?;
                raw.extend_from_slice(bytemuck::cast_slice(&bytes));

                remaining -= chunk;
                offset += chunk;
            }

            // Split the flat result range back into per-op vectors (f32 -> i64).
            for (op, &(start, count)) in gpu_ranges.iter().enumerate() {
                for k in 0..count as usize {
                    let j = op_gpu_to_pair[op][k];
                    let v = raw[(start as usize) + k];
                    if v < SENTINEL_THRESHOLD {
                        out[op][j] = needleman_wunsch_affine(
                            self.ops[op][j].0,
                            self.ops[op][j].1,
                            self.match_score,
                            self.mismatch_score,
                            self.gap_open,
                            self.gap_extend,
                        );
                    } else {
                        out[op][j] = v as i64;
                    }
                }
            }
            Ok(out)
        }
    }

    fn bg_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Unit tests for the CPU paths ─────────────────────────────────────────

    #[test]
    fn test_linear_identical() {
        assert_eq!(needleman_wunsch("ACGT", "ACGT", 2, -1, -2), 8);
        assert_eq!(needleman_wunsch("", "", 2, -1, -2), 0);
        assert_eq!(needleman_wunsch("a", "a", 1, -1, -2), 1);
    }

    #[test]
    fn test_linear_symmetry() {
        for (a, b) in &[("AGTACGCA", "TATGC"), ("hello", "world"), ("", "x")] {
            let ab = needleman_wunsch(a, b, 2, -1, -2);
            let ba = needleman_wunsch(b, a, 2, -1, -2);
            assert_eq!(ab, ba, "NW linear must be symmetric for ({a:?}, {b:?})");
        }
    }

    #[test]
    fn test_linear_known_value() {
        // Cross-checked against EMBOSS Needle (match=2, mismatch=-1, gap=-2).
        assert_eq!(needleman_wunsch("AGTACGCA", "TATGC", 2, -1, -2), 1);
    }

    #[test]
    fn test_linear_batch_matches_serial() {
        let query = "AGTACGCA";
        let cands = vec!["TATGC", "AGTACGCA", "", "AAAA", "TTTT"];
        let batch = needleman_wunsch_batch(query, &cands, 2, -1, -2);
        for (i, c) in cands.iter().enumerate() {
            let expected = needleman_wunsch(query, c, 2, -1, -2);
            assert_eq!(batch[i], expected, "batch[{i}] mismatch for {c:?}");
        }
    }

    #[test]
    fn test_affine_gap() {
        let s1 = "AGCT";
        let s2 = "AGCT";
        assert_eq!(needleman_wunsch_affine(s1, s2, 2, -1, -3, -1), 8);

        let score = needleman_wunsch_affine("ACGT", "AT", 2, -1, -3, -1);
        assert!(score < 8);
    }

    #[test]
    fn test_linear_empty_strings() {
        assert_eq!(needleman_wunsch("", "", 2, -1, -2), 0);
        assert_eq!(needleman_wunsch("abc", "", 2, -1, -2), -6);
        assert_eq!(needleman_wunsch("", "abc", 2, -1, -2), -6);
    }

    #[test]
    fn test_affine_empty_strings() {
        // gap_open=-3, gap_extend=-1: gap of length k costs -3 + k*(-1) = -(3 + k)
        assert_eq!(needleman_wunsch_affine("", "", 2, -1, -3, -1), 0);
        assert_eq!(needleman_wunsch_affine("abc", "", 2, -1, -3, -1), -6);
        assert_eq!(needleman_wunsch_affine("", "abc", 2, -1, -3, -1), -6);
    }

    /// Confirm that large scoring parameters that would exceed f32 precision
    /// (2^24 = 16,777,216) are correctly detected and routed to CPU even when
    /// the GPU feature is compiled in.  The CPU result must be exact.
    #[test]
    fn test_large_scoring_params_exact() {
        // match_score = 1_000_000; for 4 identical chars the score = 4_000_000
        // which is below 2^24, but match_score * 128 (GPU_MAX_STRING_LEN) =
        // 128_000_000 which is above 2^24.  The guard must route to CPU.
        let score = needleman_wunsch_affine("AGCT", "AGCT", 1_000_000, -1, -3, -1);
        assert_eq!(
            score, 4_000_000,
            "large match_score must compute exactly on CPU"
        );
    }

    #[test]
    fn test_unicode_alignment() {
        // Basic Unicode: café vs cafe — one substitution (é→e).
        let score_sub = needleman_wunsch_affine("café", "cafe", 2, -1, -3, -1);
        let score_same = needleman_wunsch_affine("café", "café", 2, -1, -3, -1);
        assert!(score_sub < score_same, "substitution must reduce score");
    }

    #[test]
    fn test_affine_batch_matches_serial() {
        let query = "AGTACGCA";
        let cands = vec!["TATGC", "AGTACGCA", "", "AAAA", "TTTT"];
        let batch = needleman_wunsch_affine_batch(query, &cands, 2, -1, -3, -1);
        for (i, c) in cands.iter().enumerate() {
            let expected = needleman_wunsch_affine(query, c, 2, -1, -3, -1);
            assert_eq!(batch[i], expected, "affine batch[{i}] mismatch for {c:?}");
        }
    }

    #[test]
    fn test_affine_open_zero_degenerates_to_linear() {
        // gap_open=0: affine cost = 0 + k*gap_extend = k*gap_extend = linear model.
        for (a, b) in &[("AGCT", "AGCT"), ("AGCT", "TGCA"), ("hello", "world")] {
            let linear = needleman_wunsch(a, b, 2, -1, -1);
            let affine = needleman_wunsch_affine(a, b, 2, -1, 0, -1);
            assert_eq!(
                affine, linear,
                "affine(open=0) must equal linear for ({a:?}, {b:?})"
            );
        }
    }

    #[test]
    fn test_affine_symmetry() {
        for (a, b) in &[("AGTACGCA", "TATGC"), ("café", "cafe"), ("", "x")] {
            let ab = needleman_wunsch_affine(a, b, 2, -1, -3, -1);
            let ba = needleman_wunsch_affine(b, a, 2, -1, -3, -1);
            assert_eq!(ab, ba, "NW affine must be symmetric for ({a:?}, {b:?})");
        }
    }

    #[test]
    fn test_sat_helpers_used_in_needleman() {
        // Very long strings with high scores: must not overflow or panic in release.
        let a = "A".repeat(1000);
        let b = "A".repeat(1000);
        let score = needleman_wunsch_affine(&a, &b, 1_000_000, -1, -3, -1);
        assert_eq!(
            score, 1_000_000_000,
            "1000 matches × 1_000_000 = 1_000_000_000"
        );
    }

    #[cfg(feature = "gpu")]
    mod gpu_tests {
        use super::*;
        use crate::needleman::gpu_ext::GpuNeedlemanAffineKernel;

        fn gpu_kernel_or_skip() -> Option<&'static GpuNeedlemanAffineKernel> {
            match GpuNeedlemanAffineKernel::get() {
                Ok(k) => Some(k),
                Err(e) => {
                    if crate::gpu::require_gpu() {
                        panic!("FUZZGPU_REQUIRE_GPU is set but no GPU device: {e}");
                    }
                    eprintln!("skipping GPU NW test (no device): {e}");
                    None
                }
            }
        }

        /// f32 precision guard: scoring params that exceed 2^24 * max_len must
        /// route to CPU, producing an exact i64 result.
        #[test]
        fn test_gpu_f32_precision_guard_routes_to_cpu() {
            let _gpu_guard = crate::gpu::gpu_test_lock();
            let Some(kernel) = gpu_kernel_or_skip() else {
                return;
            };

            // match_score = 200_000 * 128 = 25_600_000 > 2^24.
            let pairs = vec![("AGCT", "AGCT"), ("hello", "world")];
            let gpu_res = kernel
                .compute_batch(&pairs, 200_000, -1, -3, -1)
                .expect("compute_batch should succeed (routes to CPU)");
            let cpu_res: Vec<i64> = pairs
                .iter()
                .map(|(a, b)| needleman_wunsch_affine(a, b, 200_000, -1, -3, -1))
                .collect();
            assert_eq!(
                gpu_res, cpu_res,
                "large scoring params must be exact (CPU path)"
            );
        }

        /// Normal range scoring params should still match CPU exactly on GPU.
        #[test]
        fn test_gpu_normal_scoring_matches_cpu() {
            let _gpu_guard = crate::gpu::gpu_test_lock();
            let Some(kernel) = gpu_kernel_or_skip() else {
                return;
            };
            // RAII: holds the threshold lock so no concurrent test can steal
            // the override, and restores `None` on drop.
            let _force = crate::gpu::force_gpu_threshold(1);

            let mut state: u64 = 0xDEAD_CAFE;
            let mut pairs: Vec<(String, String)> = Vec::with_capacity(500);
            for _ in 0..500 {
                let la = 1 + (state % 60) as usize;
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let lb = 1 + (state % 60) as usize;
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let a: String = (0..la)
                    .map(|_| {
                        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                        (b'a' + (state >> 33) as u8 % 26) as char
                    })
                    .collect();
                let b: String = (0..lb)
                    .map(|_| {
                        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                        (b'a' + (state >> 33) as u8 % 26) as char
                    })
                    .collect();
                pairs.push((a, b));
            }
            let refs: Vec<(&str, &str)> = pairs
                .iter()
                .map(|(a, b)| (a.as_str(), b.as_str()))
                .collect();
            let gpu = kernel
                .compute_batch(&refs, 2, -1, -3, -1)
                .expect("GPU NW batch");
            let cpu: Vec<i64> = refs
                .iter()
                .map(|(a, b)| needleman_wunsch_affine(a, b, 2, -1, -3, -1))
                .collect();
            assert_eq!(gpu, cpu, "GPU NW must match CPU for normal scoring range");
        }
    }
}
