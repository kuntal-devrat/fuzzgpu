use rayon::prelude::*;
use std::collections::HashMap;

/// True unrestricted Damerau-Levenshtein distance (Lowrance & Wagner 1975).
/// Computes edit distance allowing insertions, deletions, substitutions, and transpositions of any characters
/// (including non-adjacent transpositions where characters were inserted/deleted in between).
/// Supports both ASCII (fast array path) and full Unicode characters.
pub fn damerau_levenshtein_distance(a: &str, b: &str) -> u32 {
    if a.is_ascii() && b.is_ascii() {
        damerau_bytes(a.as_bytes(), b.as_bytes())
    } else {
        let a_chars: Vec<char> = a.chars().collect();
        let b_chars: Vec<char> = b.chars().collect();
        damerau_chars(&a_chars, &b_chars)
    }
}

fn damerau_bytes(a: &[u8], b: &[u8]) -> u32 {
    // Safety invariant: this function indexes `da` by raw byte value (0..=255),
    // so non-ASCII bytes would silently produce wrong distances due to byte-level
    // collisions between UTF-8 continuation bytes and ASCII characters.
    // This assert! fires in both debug AND release builds (PyPI wheels ship release).
    assert!(a.is_ascii() && b.is_ascii(), "damerau_bytes requires ASCII inputs");
    let (m, n) = (a.len(), b.len());
    if m == 0 { return n as u32; }
    if n == 0 { return m as u32; }
    if a == b { return 0; }

    let max_dist = (m + n) as u32;
    let cols = n + 2;
    let mut h = vec![0u32; (m + 2) * cols];

    let idx = |i: isize, j: isize| -> usize {
        ((i + 1) as usize) * cols + ((j + 1) as usize)
    };

    h[idx(-1, -1)] = max_dist;
    for i in 0..=m {
        h[idx(i as isize, -1)] = max_dist;
        h[idx(i as isize, 0)] = i as u32;
    }
    for j in 0..=n {
        h[idx(-1, j as isize)] = max_dist;
        h[idx(0, j as isize)] = j as u32;
    }

    let mut da = [0usize; 256];

    for i in 1..=m {
        let mut db = 0usize;
        let ai = a[i - 1];

        for j in 1..=n {
            let bj = b[j - 1];
            let k = da[bj as usize];
            let l = db;

            let cost = if ai == bj {
                db = j;
                0u32
            } else {
                1u32
            };

            let sub = h[idx((i - 1) as isize, (j - 1) as isize)] + cost;
            let ins = h[idx(i as isize, (j - 1) as isize)] + 1;
            let del = h[idx((i - 1) as isize, j as isize)] + 1;

            let trans = if k > 0 && l > 0 {
                h[idx((k - 1) as isize, (l - 1) as isize)] + ((i - k - 1) as u32) + 1 + ((j - l - 1) as u32)
            } else {
                max_dist
            };

            h[idx(i as isize, j as isize)] = sub.min(ins).min(del).min(trans);
        }

        da[ai as usize] = i;
    }

    h[idx(m as isize, n as isize)]
}

fn damerau_chars(a: &[char], b: &[char]) -> u32 {
    let (m, n) = (a.len(), b.len());
    if m == 0 { return n as u32; }
    if n == 0 { return m as u32; }
    if a == b { return 0; }

    let max_dist = (m + n) as u32;
    let cols = n + 2;
    let mut h = vec![0u32; (m + 2) * cols];

    let idx = |i: isize, j: isize| -> usize {
        ((i + 1) as usize) * cols + ((j + 1) as usize)
    };

    h[idx(-1, -1)] = max_dist;
    for i in 0..=m {
        h[idx(i as isize, -1)] = max_dist;
        h[idx(i as isize, 0)] = i as u32;
    }
    for j in 0..=n {
        h[idx(-1, j as isize)] = max_dist;
        h[idx(0, j as isize)] = j as u32;
    }

    let mut da: HashMap<char, usize> = HashMap::with_capacity(a.len());

    for i in 1..=m {
        let mut db = 0usize;
        let ai = a[i - 1];

        for j in 1..=n {
            let bj = b[j - 1];
            let k = da.get(&bj).copied().unwrap_or(0);
            let l = db;

            let cost = if ai == bj {
                db = j;
                0u32
            } else {
                1u32
            };

            let sub = h[idx((i - 1) as isize, (j - 1) as isize)] + cost;
            let ins = h[idx(i as isize, (j - 1) as isize)] + 1;
            let del = h[idx((i - 1) as isize, j as isize)] + 1;

            let trans = if k > 0 && l > 0 {
                h[idx((k - 1) as isize, (l - 1) as isize)] + ((i - k - 1) as u32) + 1 + ((j - l - 1) as u32)
            } else {
                max_dist
            };

            h[idx(i as isize, j as isize)] = sub.min(ins).min(del).min(trans);
        }

        da.insert(ai, i);
    }

    h[idx(m as isize, n as isize)]
}

/// Batch Damerau-Levenshtein: one query vs many candidates.
pub fn damerau_levenshtein_batch(query: &str, candidates: &[&str]) -> Vec<u32> {
    candidates.par_iter().map(|c| damerau_levenshtein_distance(query, c)).collect()
}

/// Cross-product matrix for Damerau-Levenshtein.
pub fn damerau_levenshtein_cdist(list_a: &[&str], list_b: &[&str]) -> Vec<Vec<u32>> {
    if list_a.is_empty() || list_b.is_empty() {
        return vec![];
    }
    list_a.par_iter().map(|a| {
        list_b.iter().map(|b| damerau_levenshtein_distance(a, b)).collect()
    }).collect()
}

/// Damerau-Levenshtein normalized ratio (0.0 to 100.0) based on standard edit distance similarity formula:
/// `((total_len - dist) / total_len) * 100.0`.
pub fn damerau_ratio(s1: &str, s2: &str) -> f64 {
    let len_a = if s1.is_ascii() { s1.len() } else { s1.chars().count() };
    let len_b = if s2.is_ascii() { s2.len() } else { s2.chars().count() };
    let total = len_a + len_b;
    if total == 0 { return 100.0; }
    let dist = damerau_levenshtein_distance(s1, s2) as f64;
    ((total as f64 - dist) / total as f64) * 100.0
}

#[cfg(feature = "gpu")]
pub mod gpu_ext {
    use super::*;
    use bytemuck::{Pod, Zeroable};
    use std::sync::OnceLock;
    use crate::gpu::{
        BufferPool, FuzzGpuError, GpuEngine, Result, SLOT_CHARS_A, SLOT_CHARS_B, SLOT_OFFSETS_A,
        SLOT_OFFSETS_B, SLOT_PARAMS, SLOT_RESULTS, SLOT_STAGING,
    };

    const SHADER_SRC: &str = include_str!("shaders/damerau.wgsl");
    const MATRIX_SHADER_SRC: &str = include_str!("shaders/damerau_matrix.wgsl");

    /// Max string length the GPU kernel handles (chars, ASCII). The full
    /// Lowrance-Wagner matrix per pair lives in workgroup shared memory:
    /// 4 pairs × 34×34 u32 + per-pair 257-entry da tables ≈ 22.6 KiB.
    const GPU_MAX_STRING_LEN: usize = 32;
    const MAX_DISPATCH: u32 = 65535;
    const MAX_DESIRED_CHUNK_PAIRS: usize = 500_000;

    /// Scales the discrete-GPU auto threshold (64) for the Lowrance-Wagner
    /// SLM kernel: the full-matrix DP is ~10× heavier per pair than the Myers
    /// bit-vector, so even on a dGPU it needs a much larger batch to win
    /// (64 × 32 = 2048 pairs). On integrated GPUs the SIMD CPU path wins at
    /// every measured scale (Iris Xe: ~6× at 50k pairs), so auto-routing
    /// never dispatches there (see `GpuEngine::metric_gpu_threshold`).
    const DAMERAU_DISCRETE_FACTOR: usize = 32;

    /// Workgroup-storage budget of the shaders (u32 words → bytes).
    const SLM_BYTES: u32 = (4 * 34 * 34 + 4 * 257) * 4;

    #[repr(C)]
    #[derive(Copy, Clone, Pod, Zeroable)]
    struct DamerauParams {
        batch_size: u32,
        max_len: u32,
        offset: u32,
    }

    #[repr(C)]
    #[derive(Copy, Clone, Pod, Zeroable)]
    struct DamerauMatrixParams {
        rows: u32,
        cols: u32,
    }

    pub struct GpuDamerauKernel {
        engine: std::sync::Arc<GpuEngine>,
        pipeline: wgpu::ComputePipeline,
        matrix_pipeline: wgpu::ComputePipeline,
        bind_group_layout: wgpu::BindGroupLayout,
        // Persistent buffer arena (see gpu::BufferPool) — removes the per-call
        // `create_buffer` cost that dominated small-batch dispatches.
        pool: std::sync::Mutex<BufferPool>,
    }

    static GLOBAL_GPU_DAMERAU_KERNEL: OnceLock<GpuDamerauKernel> = OnceLock::new();

    impl GpuDamerauKernel {
        pub fn get() -> Result<&'static Self> {
            if let Some(k) = GLOBAL_GPU_DAMERAU_KERNEL.get() {
                return Ok(k);
            }
            let engine = GpuEngine::get()?;
            let kernel = Self::new_inner(engine)?;
            let _ = GLOBAL_GPU_DAMERAU_KERNEL.set(kernel);
            Ok(GLOBAL_GPU_DAMERAU_KERNEL.get().unwrap())
        }

        fn new_inner(engine: std::sync::Arc<GpuEngine>) -> Result<Self> {
            // The kernel keeps a 34×34 u32 matrix + 257-entry da table per
            // pair in workgroup shared memory (~22.6 KiB at workgroup size 4).
            // Devices below that budget cannot run it — refuse to initialize
            // with a clear error so callers route to CPU instead of failing at
            // pipeline creation with a cryptic validation error.
            let wg_limit = engine.device.limits().max_compute_workgroup_storage_size;
            if wg_limit < SLM_BYTES {
                return Err(FuzzGpuError::NoDevice(format!(
                    "Damerau GPU kernel needs {SLM_BYTES} B of workgroup storage, device has {wg_limit} B"
                )));
            }

            let bind_group_layout = engine.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("damerau bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                    wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                    wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                    wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                    wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                    wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                ],
            });

            let layout = engine.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });

            let pipeline = engine.build_compute_pipeline(
                "damerau pipeline",
                &crate::gpu::effective_shader_source(SHADER_SRC),
                &layout,
            )?;
            let matrix_pipeline = engine.build_compute_pipeline(
                "damerau matrix pipeline",
                &crate::gpu::effective_shader_source(MATRIX_SHADER_SRC),
                &layout,
            )?;

            Ok(Self { engine, pipeline, matrix_pipeline, bind_group_layout, pool: std::sync::Mutex::new(BufferPool::new()) })
        }

        /// Smart streaming GPU/CPU dispatch for batch Damerau-Levenshtein with
        /// dynamic chunk sizing. Unrestricted Lowrance-Wagner semantics,
        /// bit-exact with [`damerau_levenshtein_distance`] on ASCII inputs:
        /// identical pairs short-circuit to 0, empties to the other's length,
        /// non-ASCII or > 32-char pairs route to CPU, and the shader's
        /// u32::MAX sentinel triggers a CPU recompute as a backstop.
        pub fn compute_batch(&self, pairs: &[(&str, &str)]) -> Result<Vec<u32>> {
            // Serialize GPU dispatch across threads (gfx-rs/wgpu#10085).
            let _dispatch = self.engine.dispatch_lock();
            let n = pairs.len();
            if n == 0 {
                return Ok(vec![]);
            }

            let mut results = vec![0u32; n];
            let mut gpu_indices: Vec<usize> = Vec::with_capacity(n);
            let mut cpu_indices: Vec<usize> = Vec::new();

            for (i, (a, b)) in pairs.iter().enumerate() {
                if *a == *b {
                    results[i] = 0;
                } else if a.is_empty() {
                    results[i] = b.chars().count() as u32;
                } else if b.is_empty() {
                    results[i] = a.chars().count() as u32;
                } else if a.is_ascii()
                    && b.is_ascii()
                    && a.len() <= GPU_MAX_STRING_LEN
                    && b.len() <= GPU_MAX_STRING_LEN
                {
                    gpu_indices.push(i);
                } else {
                    cpu_indices.push(i);
                }
            }

            if !cpu_indices.is_empty() {
                let cpu_results: Vec<u32> = cpu_indices.par_iter()
                    .map(|&i| damerau_levenshtein_distance(pairs[i].0, pairs[i].1))
                    .collect();
                for (idx, &orig_i) in cpu_indices.iter().enumerate() {
                    results[orig_i] = cpu_results[idx];
                }
            }

            if gpu_indices.len() < self.engine.metric_gpu_threshold(DAMERAU_DISCRETE_FACTOR) {
                // Below the (auto or user-set) threshold: CPU is cheaper.
                crate::gpu::GpuEngine::record_routing(0, n);
                let cpu_results: Vec<u32> = gpu_indices.par_iter()
                    .map(|&i| damerau_levenshtein_distance(pairs[i].0, pairs[i].1))
                    .collect();
                for (idx, &orig_i) in gpu_indices.iter().enumerate() {
                    results[orig_i] = cpu_results[idx];
                }
                return Ok(results);
            }
            crate::gpu::GpuEngine::record_routing(gpu_indices.len(), cpu_indices.len());

            let max_allowed_binding = self.engine.max_storage_buffer_binding_size as usize;
            let bytes_per_pair = (GPU_MAX_STRING_LEN * 4 * 2 + 8).max(128);
            let dynamic_chunk_size = (max_allowed_binding / bytes_per_pair)
                .min(MAX_DESIRED_CHUNK_PAIRS)
                .max(512);

            for stream_chunk in gpu_indices.chunks(dynamic_chunk_size) {
                let gpu_results = self.compute_gpu_subset(pairs, stream_chunk)?;
                for (idx, &orig_i) in stream_chunk.iter().enumerate() {
                    let d = gpu_results[idx];
                    if d == u32::MAX {
                        results[orig_i] =
                            damerau_levenshtein_distance(pairs[orig_i].0, pairs[orig_i].1);
                    } else {
                        results[orig_i] = d;
                    }
                }
            }

            Ok(results)
        }

        fn compute_gpu_subset(&self, pairs: &[(&str, &str)], indices: &[usize]) -> Result<Vec<u32>> {
            let batch_size = indices.len() as u32;

            // ASCII byte packing (mirrors damerau_bytes; the classification
            // already gated every pair here to ASCII ≤ 32 chars).
            let mut offsets_a: Vec<u32> = Vec::with_capacity(indices.len() + 1);
            let mut chars_a: Vec<u32> = Vec::new();
            let mut offsets_b: Vec<u32> = Vec::with_capacity(indices.len() + 1);
            let mut chars_b: Vec<u32> = Vec::new();
            offsets_a.push(0);
            offsets_b.push(0);

            let mut max_len = 0u32;
            for &i in indices {
                let (a, b) = pairs[i];
                chars_a.extend(a.as_bytes().iter().map(|&c| c as u32));
                offsets_a.push(chars_a.len() as u32);
                chars_b.extend(b.as_bytes().iter().map(|&c| c as u32));
                offsets_b.push(chars_b.len() as u32);
                max_len = max_len.max((a.len() as u32).max(b.len() as u32));
            }

            if chars_a.is_empty() { chars_a.push(0); }
            if chars_b.is_empty() { chars_b.push(0); }

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
            pool.ensure(&self.engine.device, SLOT_OFFSETS_A, offsets_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, "doa");
            pool.ensure(&self.engine.device, SLOT_OFFSETS_B, offsets_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, "dob");
            pool.ensure(&self.engine.device, SLOT_CHARS_A, chars_a_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, "dca");
            pool.ensure(&self.engine.device, SLOT_CHARS_B, chars_b_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, "dcb");
            pool.ensure(&self.engine.device, SLOT_RESULTS, results_size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, "dres");
            pool.ensure(&self.engine.device, SLOT_STAGING, results_size, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, "dstg");
            pool.ensure(&self.engine.device, SLOT_PARAMS, std::mem::size_of::<DamerauParams>() as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, "dp");

            pool.write(&self.engine.queue, SLOT_OFFSETS_A, bytemuck::cast_slice(&offsets_a));
            pool.write(&self.engine.queue, SLOT_CHARS_A, bytemuck::cast_slice(&chars_a));
            pool.write(&self.engine.queue, SLOT_OFFSETS_B, bytemuck::cast_slice(&offsets_b));
            pool.write(&self.engine.queue, SLOT_CHARS_B, bytemuck::cast_slice(&chars_b));

            let buf_offsets_a = pool.get(SLOT_OFFSETS_A);
            let buf_chars_a = pool.get(SLOT_CHARS_A);
            let buf_offsets_b = pool.get(SLOT_OFFSETS_B);
            let buf_chars_b = pool.get(SLOT_CHARS_B);
            let buf_results = pool.get(SLOT_RESULTS);
            let buf_staging = pool.get(SLOT_STAGING);
            let buf_params = pool.get(SLOT_PARAMS);

            // Per-chunk submit + readback (the params buffer is written through
            // the queue, so one shared submit would give every dispatch the
            // LAST chunk's offset — the multi-chunk correctness bug fixed
            // across all kernels).
            let mut gpu_results: Vec<u32> = Vec::with_capacity(batch_size as usize);
            let mut remaining = batch_size;
            let mut offset = 0u32;

            while remaining > 0 {
                let chunk = remaining.min(MAX_DISPATCH);
                let params = DamerauParams { batch_size, max_len, offset };
                pool.write(&self.engine.queue, SLOT_PARAMS, bytemuck::bytes_of(&params));

                let bg = self.engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None, layout: &self.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: buf_offsets_a.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: buf_chars_a.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: buf_offsets_b.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: buf_chars_b.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: buf_results.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 5, resource: buf_params.as_entire_binding() },
                    ],
                });

                let mut encoder = self.engine.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("damerau encoder") });
                // Workgroup size 4 (SLM budget): one workgroup per 4 pairs.
                let workgroups = chunk.div_ceil(4);
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                    pass.set_pipeline(&self.pipeline);
                    pass.set_bind_group(0, &bg, &[]);
                    pass.dispatch_workgroups(workgroups, 1, 1);
                }
                let chunk_bytes = (chunk as u64) * 4;
                encoder.copy_buffer_to_buffer(buf_results, 0, buf_staging, 0, chunk_bytes);
                let bytes = self.engine.readback(encoder, &pool, chunk_bytes)?;
                let raw: &[u32] = bytemuck::cast_slice(&bytes);
                gpu_results.extend_from_slice(raw);

                remaining -= chunk;
                offset += chunk;
            }
            Ok(gpu_results)
        }

        /// 2D Damerau-Levenshtein matrix: O(N + M) upload instead of O(N * M).
        /// Every cell runs the same Lowrance-Wagner DP on the GPU; pairs with
        /// non-ASCII or > 32-char strings fall back to the CPU cdist.
        pub fn compute_matrix(&self, list_a: &[&str], list_b: &[&str]) -> Result<Vec<Vec<u32>>> {
            // Serialize GPU dispatch across threads (gfx-rs/wgpu#10085).
            let _dispatch = self.engine.dispatch_lock();
            let rows = list_a.len();
            let cols = list_b.len();
            if rows == 0 || cols == 0 {
                return Ok(vec![]);
            }

            let total_pairs = rows * cols;
            if total_pairs < self.engine.metric_gpu_threshold(DAMERAU_DISCRETE_FACTOR) {
                crate::gpu::GpuEngine::record_routing(0, total_pairs);
                return Ok(damerau_levenshtein_cdist(list_a, list_b));
            }
            crate::gpu::GpuEngine::record_routing(total_pairs, 0);

            let has_oversized = list_a.iter().any(|s| !s.is_ascii() || s.len() > GPU_MAX_STRING_LEN)
                || list_b.iter().any(|s| !s.is_ascii() || s.len() > GPU_MAX_STRING_LEN);

            let matrix_size = (total_pairs as u64) * 4;
            if has_oversized || matrix_size > self.engine.max_buffer_size_effective() {
                return Ok(damerau_levenshtein_cdist(list_a, list_b));
            }

            // Pack List A (rows)
            let mut offsets_a: Vec<u32> = Vec::with_capacity(rows + 1);
            let mut chars_a: Vec<u32> = Vec::new();
            offsets_a.push(0);
            for a in list_a {
                chars_a.extend(a.as_bytes().iter().map(|&c| c as u32));
                offsets_a.push(chars_a.len() as u32);
            }

            // Pack List B (cols)
            let mut offsets_b: Vec<u32> = Vec::with_capacity(cols + 1);
            let mut chars_b: Vec<u32> = Vec::new();
            offsets_b.push(0);
            for b in list_b {
                chars_b.extend(b.as_bytes().iter().map(|&c| c as u32));
                offsets_b.push(chars_b.len() as u32);
            }

            if chars_a.is_empty() { chars_a.push(0); }
            if chars_b.is_empty() { chars_b.push(0); }

            let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
            let offsets_bytes = ((offsets_a.len() * 4) as u64).max(matrix_size);
            pool.ensure(&self.engine.device, SLOT_OFFSETS_A, offsets_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, "dmoa");
            pool.ensure(&self.engine.device, SLOT_OFFSETS_B, offsets_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, "dmob");
            pool.ensure(&self.engine.device, SLOT_CHARS_A, (chars_a.len() * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, "dmca");
            pool.ensure(&self.engine.device, SLOT_CHARS_B, (chars_b.len() * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, "dmcb");
            pool.ensure(&self.engine.device, SLOT_RESULTS, matrix_size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, "dmres");
            pool.ensure(&self.engine.device, SLOT_STAGING, matrix_size, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, "dmstg");
            pool.ensure(&self.engine.device, SLOT_PARAMS, std::mem::size_of::<DamerauMatrixParams>() as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, "dmp");

            pool.write(&self.engine.queue, SLOT_OFFSETS_A, bytemuck::cast_slice(&offsets_a));
            pool.write(&self.engine.queue, SLOT_CHARS_A, bytemuck::cast_slice(&chars_a));
            pool.write(&self.engine.queue, SLOT_OFFSETS_B, bytemuck::cast_slice(&offsets_b));
            pool.write(&self.engine.queue, SLOT_CHARS_B, bytemuck::cast_slice(&chars_b));

            let params = DamerauMatrixParams { rows: rows as u32, cols: cols as u32 };
            pool.write(&self.engine.queue, SLOT_PARAMS, bytemuck::bytes_of(&params));

            let buf_offsets_a = pool.get(SLOT_OFFSETS_A);
            let buf_chars_a = pool.get(SLOT_CHARS_A);
            let buf_offsets_b = pool.get(SLOT_OFFSETS_B);
            let buf_chars_b = pool.get(SLOT_CHARS_B);
            let buf_matrix = pool.get(SLOT_RESULTS);
            let buf_staging = pool.get(SLOT_STAGING);
            let buf_params = pool.get(SLOT_PARAMS);

            let bg = self.engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None, layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: buf_offsets_a.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: buf_chars_a.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: buf_offsets_b.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: buf_chars_b.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: buf_matrix.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: buf_params.as_entire_binding() },
                ],
            });

            let mut encoder = self.engine.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("damerau matrix encoder") },
            );

            let workgroups_x = (cols as u32).div_ceil(4);
            let workgroups_y = rows as u32;

            {
                let mut pass = encoder.begin_compute_pass(
                    &wgpu::ComputePassDescriptor { label: None, timestamp_writes: None },
                );
                pass.set_pipeline(&self.matrix_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
            }

            encoder.copy_buffer_to_buffer(buf_matrix, 0, buf_staging, 0, matrix_size);
            self.engine.submit(encoder);

            let slice = buf_staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            self.engine.map_readback(&slice, move |r| { let _ = tx.send(r); });
            self.engine.poll();

            rx.recv_timeout(GpuEngine::readback_timeout())
                .map_err(|_| FuzzGpuError::Timeout("GPU Damerau matrix readback timed out after 10s".into()))?
                .map_err(|e| FuzzGpuError::BufferError(format!("GPU buffer map failed: {}", e)))?;

            let data = slice
                .get_mapped_range()
                .map_err(|e| FuzzGpuError::BufferError(format!("GPU buffer map range failed: {e}")))?;
            let raw: &[u32] = bytemuck::cast_slice(&data);

            let mut matrix: Vec<Vec<u32>> = Vec::with_capacity(rows);
            for i in 0..rows {
                let start = i * cols;
                let end = start + cols;
                let mut row: Vec<u32> = raw[start..end].to_vec();
                // Backstop: u32::MAX sentinel cells (mis-routed) recomputed on CPU.
                for (j, cell) in row.iter_mut().enumerate() {
                    if *cell == u32::MAX {
                        *cell = damerau_levenshtein_distance(list_a[i], list_b[j]);
                    }
                }
                matrix.push(row);
            }

            drop(data);
            buf_staging.unmap();
            Ok(matrix)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// Deterministic pseudo-random ASCII strings (LCG), lengths 1..=16.
        fn gen_strings(count: usize, seed: u64) -> Vec<String> {
            let mut state = seed;
            let mut out = Vec::with_capacity(count);
            for _ in 0..count {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let len = 1 + ((state >> 33) as usize % 16);
                let mut s = String::with_capacity(len);
                for _ in 0..len {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    s.push((b'a' + ((state >> 33) as u8 % 26)) as char);
                }
                out.push(s);
            }
            out
        }

        fn gpu_kernel_or_skip() -> Option<&'static GpuDamerauKernel> {
            match GpuDamerauKernel::get() {
                Ok(k) => Some(k),
                Err(e) => {
                    if crate::gpu::require_gpu() {
                        panic!("FUZZGPU_REQUIRE_GPU is set but no usable GPU device: {}", e);
                    }
                    eprintln!("skipping GPU test (no usable device): {}", e);
                    None
                }
            }
        }

        /// Force GPU dispatch regardless of metric/auto routing (which sends
        /// Damerau to CPU on integrated GPUs where the SIMD path wins). RAII
        /// guard restores the override on drop; tests hold the GPU lock.
        struct ForceGpu;
        impl ForceGpu {
            fn new() -> Self {
                GpuEngine::set_gpu_threshold(Some(1));
                ForceGpu
            }
        }
        impl Drop for ForceGpu {
            fn drop(&mut self) {
                GpuEngine::set_gpu_threshold(None);
            }
        }

        /// Exercises shader compilation, buffer sizing/allocation, dispatch,
        /// and readback end-to-end against the CPU reference — including the
        /// transposition-heavy cases (adjacent AND non-adjacent, the "ca" vs
        /// "abc" = 2 case that OSA gets wrong) plus > 32-char pairs that must
        /// route to CPU.
        #[test]
        fn test_gpu_batch_matches_cpu() {
            let _gpu_guard = crate::gpu::gpu_test_lock();
            let Some(kernel) = gpu_kernel_or_skip() else { return; };
            let _force = ForceGpu::new();

            let a = gen_strings(1000, 0xD00DCAFE);
            let b = gen_strings(1000, 0x0DDBA11);
            let mut pairs: Vec<(&str, &str)> =
                a.iter().zip(b.iter()).map(|(x, y)| (x.as_str(), y.as_str())).collect();
            pairs[0] = ("", "");
            pairs[1] = ("", "xyz");
            pairs[2] = ("hello", "hello");
            pairs[3] = ("ab", "ba");           // adjacent transposition
            pairs[4] = ("ca", "abc");          // non-adjacent transposition (2)
            pairs[5] = ("a cat", "an act");    // classic multi-transposition
            pairs[6] = ("sitting", "kitten");
            // > 32 chars must route to CPU inside compute_batch.
            let long_a = "a".repeat(40);
            let long_b = "b".repeat(40);
            let long_c = "x".repeat(33);
            let long_d = "y".repeat(35);
            pairs.push((&long_a, &long_b));
            pairs.push((&long_c, &long_d));

            let gpu = kernel.compute_batch(&pairs).expect("GPU batch should succeed");
            let cpu: Vec<u32> = pairs.iter()
                .map(|(x, y)| damerau_levenshtein_distance(x, y))
                .collect();
            assert_eq!(gpu, cpu, "GPU Damerau batch must match CPU");
            // The u32::MAX sentinel must never leak.
            assert!(gpu.iter().all(|&d| d != u32::MAX), "u32::MAX sentinel leaked");
        }

        /// 2D matrix path vs the CPU cdist reference.
        #[test]
        fn test_gpu_matrix_matches_cpu() {
            let _gpu_guard = crate::gpu::gpu_test_lock();
            let Some(kernel) = gpu_kernel_or_skip() else { return; };
            let _force = ForceGpu::new();

            let a = gen_strings(30, 0x13579BDF);
            let b = gen_strings(30, 0x2468ACE0);
            let refs_a: Vec<&str> = a.iter().map(|s| s.as_str()).collect();
            let refs_b: Vec<&str> = b.iter().map(|s| s.as_str()).collect();

            let gpu = kernel.compute_matrix(&refs_a, &refs_b).expect("GPU matrix should succeed");
            let cpu = damerau_levenshtein_cdist(&refs_a, &refs_b);
            assert_eq!(gpu, cpu, "GPU Damerau matrix must match CPU");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// The ASCII byte-array path and the Unicode char path are two
        /// independent implementations of Lowrance-Wagner; they must agree
        /// exactly on ASCII input.
        #[test]
        fn ascii_byte_and_char_paths_agree(
            a in prop::collection::vec(prop::char::range('a', 'z'), 0..=60usize),
            b in prop::collection::vec(prop::char::range('a', 'z'), 0..=60usize),
        ) {
            let a: String = a.into_iter().collect();
            let b: String = b.into_iter().collect();
            let a_chars: Vec<char> = a.chars().collect();
            let b_chars: Vec<char> = b.chars().collect();
            prop_assert_eq!(
                damerau_bytes(a.as_bytes(), b.as_bytes()),
                damerau_chars(&a_chars, &b_chars),
                "byte path {} != char path for {:?} vs {:?}",
                damerau_bytes(a.as_bytes(), b.as_bytes()),
                a, b
            );
        }
    }

    #[test]
    fn test_damerau_transposition() {
        assert_eq!(damerau_levenshtein_distance("ab", "ba"), 1);
        assert_eq!(damerau_levenshtein_distance("ca", "abc"), 2);
        assert_eq!(damerau_levenshtein_distance("kitten", "sitting"), 3);
    }

    #[test]
    fn test_damerau_unicode() {
        assert_eq!(damerau_levenshtein_distance("café", "cafe"), 1);
        assert_eq!(damerau_levenshtein_distance("naïve", "naive"), 1);
        assert_eq!(damerau_levenshtein_distance("🚀", ""), 1);
        assert_eq!(damerau_levenshtein_distance("中文", "中问"), 1);
    }

    #[test]
    fn test_damerau_identical_and_empty() {
        assert_eq!(damerau_levenshtein_distance("", ""), 0);
        assert_eq!(damerau_levenshtein_distance("hello", "hello"), 0);
        assert_eq!(damerau_levenshtein_distance("hello", ""), 5);
        assert_eq!(damerau_levenshtein_distance("", "world"), 5);
    }

    // The ASCII gate is a safety invariant (`assert!`, not `debug_assert!`):
    // it must fire in both debug and release builds.
    #[test]
    #[should_panic(expected = "damerau_bytes requires ASCII inputs")]
    fn test_damerau_bytes_rejects_non_ascii() {
        let _ = damerau_bytes("café".as_bytes(), b"cafe");
    }

    #[cfg(not(debug_assertions))]
    #[test]
    #[should_panic(expected = "damerau_bytes requires ASCII inputs")]
    fn test_damerau_bytes_rejects_non_ascii_release() {
        // Verify the assert! (not debug_assert!) fires in release builds too.
        let _ = damerau_bytes("café".as_bytes(), b"cafe");
    }
}
