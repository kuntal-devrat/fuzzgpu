use rayon::prelude::*;

/// Levenshtein distance with the Myers (1999) bit-vector fast path.
///
/// ASCII inputs route through `levenshtein_myers`, which is ~30x faster than
/// single-row DP for patterns <= 64 chars (zero inner loop) and falls back to
/// the identical single-row DP for longer strings. Unicode inputs use the
/// scalar-value DP path. This is the CPU engine behind every batch/matrix
/// call, so the fast path applies everywhere — not just the GPU kernels.
pub fn levenshtein_distance_raw(a: &str, b: &str) -> u32 {
    if a.is_ascii() && b.is_ascii() {
        crate::simd::levenshtein_myers(a.as_bytes(), b.as_bytes())
    } else {
        let a_chars: Vec<char> = a.chars().collect();
        let b_chars: Vec<char> = b.chars().collect();
        levenshtein_distance_slice(&a_chars, &b_chars)
    }
}

fn levenshtein_distance_slice<T: PartialEq>(a: &[T], b: &[T]) -> u32 {
    let (m, n) = (a.len(), b.len());
    if m == 0 { return u32::try_from(n).unwrap_or(u32::MAX); }
    if n == 0 { return u32::try_from(m).unwrap_or(u32::MAX); }
    if a == b { return 0; }

    // Single-row + diagonal optimization: halves memory vs two-row.
    let mut row = vec![0u32; n + 1];
    for (j, item) in row.iter_mut().enumerate() {
        *item = u32::try_from(j).unwrap_or(u32::MAX);
    }

    for i in 1..=m {
        let mut prev_diag = row[0];
        row[0] = u32::try_from(i).unwrap_or(u32::MAX);
        let ai = &a[i - 1];
        for j in 1..=n {
            let old = row[j];
            let cost = if ai == &b[j - 1] { 0 } else { 1 };
            row[j] = prev_diag.saturating_add(cost)
                .min(row[j].saturating_add(1))
                .min(row[j - 1].saturating_add(1));
            prev_diag = old;
        }
    }
    row[n]
}

/// Cross-product matrix computed on CPU via Rayon, with a per-row 4-way SIMD
/// fast path: when a row's query is ASCII ≤ 64 bytes and every text in the row
/// is ASCII, four distances are produced per AVX2 vector (see
/// [`crate::simd::levenshtein_myers_4way`]).
pub fn levenshtein_cdist_cpu(list_a: &[&str], list_b: &[&str]) -> Vec<Vec<u32>> {
    if list_a.is_empty() || list_b.is_empty() {
        return vec![];
    }
    list_a.par_iter().map(|a| {
        let mut row = vec![0u32; list_b.len()];
        if let Some(qb) = ascii_short_pattern(a) {
            if list_b.iter().all(|b| b.is_ascii()) {
                // Build the Myers pattern state once per row, then run the
                // widest SIMD kernel available (AVX512 8-way / AVX2 4-way /
                // NEON 2-way / scalar).
                let pat = crate::simd::MyersPattern::new(qb);
                let w = crate::simd::myers_simd_width();
                let mut k = 0;
                // Use a stack-allocated buffer (max 8 slots, matching the
                // widest SIMD width) instead of a per-group heap Vec.
                let mut group_buf: [&[u8]; 8] = [b""; 8];
                while k + w <= row.len() {
                    for t in 0..w {
                        group_buf[t] = list_b[k + t].as_bytes();
                    }
                    let res = crate::simd::levenshtein_myers_width(&pat, &group_buf[..w]);
                    for (t, v) in res.into_iter().enumerate() {
                        row[k + t] = v;
                    }
                    k += w;
                }
                while k < row.len() {
                    row[k] = crate::simd::levenshtein_myers_pattern(&pat, list_b[k].as_bytes());
                    k += 1;
                }
                return row;
            }
        }
        for (j, b) in list_b.iter().enumerate() {
            row[j] = levenshtein_distance_raw(a, b);
        }
        row
    }).collect()
}

/// Non-empty ASCII ≤ 64-byte prefix usable as a 4-way Myers pattern.
fn ascii_short_pattern(s: &str) -> Option<&[u8]> {
    let b = s.as_bytes();
    if !b.is_empty() && b.len() <= 64 && b.is_ascii() {
        Some(b)
    } else {
        None
    }
}

/// Batch distance over pairs, with the shared-query SIMD fast path.
///
/// When every pair shares one ASCII query of ≤ 64 bytes (the fuzzy-matching
/// shape), texts are processed four-at-a-time by the AVX2 Myers kernel;
/// otherwise the per-pair Rayon path is used.
pub fn levenshtein_batch_auto(pairs: &[(&str, &str)]) -> Vec<u32> {
    let q = match shared_query(pairs) {
        Some(q) => q,
        None => return pairs.par_iter().map(|(a, b)| levenshtein_distance_raw(a, b)).collect(),
    };
    let qb = q.as_bytes();
    // Build the Myers pattern state exactly once for the whole batch (not per
    // text or per 4-text group) — this is what makes the fast path cheap at
    // 100k-pair scale.
    let pat = crate::simd::MyersPattern::new(qb);
    let w = crate::simd::myers_simd_width();
    let mut out = vec![0u32; pairs.len()];
    // Chunks of 4096 texts per Rayon task; each task runs the widest SIMD
    // kernel available (AVX512 8-way / AVX2 4-way / NEON 2-way / scalar).
    const CHUNK: usize = 4096;
    out.par_chunks_mut(CHUNK).enumerate().for_each(|(ci, chunk_out)| {
        let start = ci * CHUNK;
        let mut k = 0;
        // Stack-allocated buffer avoids a heap Vec per SIMD group.
        let mut group_buf: [&[u8]; 8] = [b""; 8];
        while k + w <= chunk_out.len() {
            for t in 0..w {
                group_buf[t] = pairs[start + k + t].1.as_bytes();
            }
            let res = crate::simd::levenshtein_myers_width(&pat, &group_buf[..w]);
            for (t, v) in res.into_iter().enumerate() {
                chunk_out[k + t] = v;
            }
            k += w;
        }
        while k < chunk_out.len() {
            chunk_out[k] = crate::simd::levenshtein_myers_pattern(&pat, pairs[start + k].1.as_bytes());
            k += 1;
        }
    });
    out
}

/// If every pair shares the same non-empty ASCII first string of ≤ 64 bytes,
/// return it — the precondition for the shared-query 4-way SIMD path.
fn shared_query<'a>(pairs: &[(&'a str, &'a str)]) -> Option<&'a str> {
    let q = pairs.first()?.0;
    if q.is_empty() || !q.is_ascii() || q.len() > 64 {
        return None;
    }
    if pairs.iter().any(|(a, b)| *a != q || !b.is_ascii()) {
        return None;
    }
    Some(q)
}

/// CPU-parallel Levenshtein kernel using Rayon.
pub struct LevenshteinKernel;

impl LevenshteinKernel {
    pub fn compute(&self, pairs: &[(&str, &str)]) -> crate::Result<Vec<u32>> {
        Ok(levenshtein_batch_auto(pairs))
    }
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

    const SHADER_SRC: &str = include_str!("shaders/levenshtein.wgsl");
    const MATRIX_SHADER_SRC: &str = include_str!("shaders/levenshtein_matrix.wgsl");
    // Short-string kernel: workgroup-shared DP rows + coalesced transposed
    // loads (see shaders/levenshtein_short.wgsl). Pairs with both strings
    // <= SHORT_MAX_LEN chars take this path; longer pairs (<= 256) keep the
    // general kernel; anything longer routes to CPU.
    const SHORT_SHADER_SRC: &str = include_str!("shaders/levenshtein_short.wgsl");
    // Myers (1999) bit-vector kernel for the shared-query ASCII case (one Peq
    // table per workgroup, bit-vector in registers — see the shader header).
    const MYERS_SHADER_SRC: &str = include_str!("shaders/levenshtein_myers.wgsl");
    // Row-wise Myers cdist kernel: one workgroup per matrix row, shared Peq per
    // row in SLM, 64 threads strided over the columns (see the shader header).
    const MYERS_CDIST_SHADER_SRC: &str = include_str!("shaders/levenshtein_cdist_myers.wgsl");
    const SHORT_MAX_LEN: usize = 64;

    const GPU_MAX_STRING_LEN: usize = 256;
    const MAX_DISPATCH: u32 = 65535;
    const MAX_DESIRED_CHUNK_PAIRS: usize = 500_000;

    /// If every pair shares the same non-empty ASCII first string of at most
    /// `SHORT_MAX_LEN` bytes, return it — the precondition for the shared-Peq
    /// Myers kernel (one pattern per workgroup, Peq indexed by byte value).
    fn shared_ascii_query<'a>(pairs: &[(&'a str, &'a str)], indices: &[usize]) -> Option<&'a str> {
        let q = pairs[indices[0]].0;
        if q.is_empty() || !q.is_ascii() || q.len() > SHORT_MAX_LEN {
            return None;
        }
        for &i in indices {
            let (a, b) = pairs[i];
            if a != q || !b.is_ascii() {
                return None;
            }
        }
        Some(q)
    }

    #[repr(C)]
    #[derive(Copy, Clone, Pod, Zeroable)]
    struct Params {
        batch_size: u32,
        max_len: u32,
        offset: u32,
    }

    #[repr(C)]
    #[derive(Copy, Clone, Pod, Zeroable)]
    struct MatrixParams {
        rows: u32,
        cols: u32,
    }

    pub struct GpuLevenshteinKernel {
        engine: std::sync::Arc<GpuEngine>,
        pipeline: wgpu::ComputePipeline,
        short_pipeline: wgpu::ComputePipeline,
        myers_pipeline: wgpu::ComputePipeline,
        myers_cdist_pipeline: wgpu::ComputePipeline,
        matrix_pipeline: wgpu::ComputePipeline,
        bind_group_layout: wgpu::BindGroupLayout,
        // Persistent buffer arena: removes the per-call `create_buffer` cost
        // that dominated small-batch dispatches (see gpu::BufferPool).
        pool: std::sync::Mutex<BufferPool>,
    }

    static GLOBAL_GPU_KERNEL: OnceLock<GpuLevenshteinKernel> = OnceLock::new();

    impl GpuLevenshteinKernel {
        pub fn get() -> Result<&'static Self> {
            if let Some(k) = GLOBAL_GPU_KERNEL.get() { return Ok(k); }
            let engine = GpuEngine::get()?;
            let kernel = Self::new_inner(engine)?;
            let _ = GLOBAL_GPU_KERNEL.set(kernel);
            Ok(GLOBAL_GPU_KERNEL.get().unwrap())
        }

        fn new_inner(engine: std::sync::Arc<GpuEngine>) -> Result<Self> {
            // Both kernels register through the public kernel-registration API
            // (`GpuEngine::build_compute_pipeline`), so shader/pipeline
            // validation failures surface as `FuzzGpuError::ShaderError` instead
            // of panicking. Test builds can fault-inject invalid WGSL via
            // `effective_shader_source`.
            let bind_group_layout = engine.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("levenshtein bgl"),
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
                // wgpu 30 wraps bind group layouts in `Option` and replaces
                // `push_constant_ranges` with `immediate_size` (0 = none).
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });

            let pipeline = engine.build_compute_pipeline(
                "levenshtein pipeline",
                &crate::gpu::effective_shader_source(SHADER_SRC),
                &layout,
            )?;
            let matrix_pipeline = engine.build_compute_pipeline(
                "levenshtein matrix pipeline",
                &crate::gpu::effective_shader_source(MATRIX_SHADER_SRC),
                &layout,
            )?;

            let short_pipeline = engine.build_compute_pipeline(
                "levenshtein short pipeline",
                &crate::gpu::effective_shader_source(SHORT_SHADER_SRC),
                &layout,
            )?;

            let myers_pipeline = engine.build_compute_pipeline(
                "levenshtein myers pipeline",
                &crate::gpu::effective_shader_source(MYERS_SHADER_SRC),
                &layout,
            )?;

            let myers_cdist_pipeline = engine.build_compute_pipeline(
                "levenshtein myers cdist pipeline",
                &crate::gpu::effective_shader_source(MYERS_CDIST_SHADER_SRC),
                &layout,
            )?;

            Ok(Self { engine, pipeline, short_pipeline, myers_pipeline, myers_cdist_pipeline, matrix_pipeline, bind_group_layout, pool: std::sync::Mutex::new(BufferPool::new()) })
        }

        /// Smart streaming dispatch with dynamic buffer limit validation and chunking.
        pub fn compute(&self, pairs: &[(&str, &str)]) -> Result<Vec<u32>> {
            // Serialize GPU dispatch across threads (gfx-rs/wgpu#10085): held
            // for the whole call, released on return.
            let _dispatch = self.engine.dispatch_lock();
            let n = pairs.len();
            if n == 0 { return Ok(vec![]); }

            let mut results = vec![0u32; n];
            let mut short_indices: Vec<usize> = Vec::with_capacity(n);
            let mut long_indices: Vec<usize> = Vec::new();
            let mut cpu_indices: Vec<usize> = Vec::new();

            for (i, (a, b)) in pairs.iter().enumerate() {
                if a.is_empty() || b.is_empty() {
                    let a_count = if a.is_ascii() { a.len() } else { a.chars().count() };
                    let b_count = if b.is_ascii() { b.len() } else { b.chars().count() };
                    results[i] = (a_count.max(b_count)) as u32;
                } else if *a == *b {
                    results[i] = 0;
                } else {
                    let a_len = a.chars().count();
                    let b_len = b.chars().count();
                    if a_len > GPU_MAX_STRING_LEN || b_len > GPU_MAX_STRING_LEN {
                        cpu_indices.push(i);
                    } else if a_len <= SHORT_MAX_LEN && b_len <= SHORT_MAX_LEN {
                        short_indices.push(i);
                    } else {
                        long_indices.push(i);
                    }
                }
            }

            if !cpu_indices.is_empty() {
                let cpu_results: Vec<u32> = cpu_indices.par_iter()
                    .map(|&i| levenshtein_distance_raw(pairs[i].0, pairs[i].1))
                    .collect();
                for (idx, &orig_i) in cpu_indices.iter().enumerate() {
                    results[orig_i] = cpu_results[idx];
                }
            }

            let total_gpu = short_indices.len() + long_indices.len();
            let gpu_threshold = self.engine.effective_gpu_threshold();
            if total_gpu < gpu_threshold {
                // Below the (auto or user-set) threshold the whole batch is
                // cheaper on CPU. Record the routing so callers can confirm
                // GPU mode did not silently dispatch.
                crate::gpu::GpuEngine::record_routing(0, n);
                let mut all_gpu: Vec<usize> = short_indices.clone();
                all_gpu.extend_from_slice(&long_indices);
                let cpu_results: Vec<u32> = all_gpu.par_iter()
                    .map(|&i| levenshtein_distance_raw(pairs[i].0, pairs[i].1))
                    .collect();
                for (idx, &orig_i) in all_gpu.iter().enumerate() {
                    results[orig_i] = cpu_results[idx];
                }
                return Ok(results);
            }
            crate::gpu::GpuEngine::record_routing(total_gpu, cpu_indices.len());

            // Long-string pairs (65..=256 chars): general kernel, chunked by
            // device limits.
            if !long_indices.is_empty() {
                let max_allowed_binding = self.engine.max_storage_buffer_binding_size as usize;
                let bytes_per_pair = (GPU_MAX_STRING_LEN * 4 * 2 + 8).max(128);
                let dynamic_chunk_size = (max_allowed_binding / bytes_per_pair)
                    .min(MAX_DESIRED_CHUNK_PAIRS)
                    .max(512);

                for stream_chunk in long_indices.chunks(dynamic_chunk_size) {
                    let chunk_results = self.compute_gpu_subset(pairs, stream_chunk)?;
                    for (idx, &orig_i) in stream_chunk.iter().enumerate() {
                        if chunk_results[idx] == 0xFFFFFFFF {
                            results[orig_i] = levenshtein_distance_raw(pairs[orig_i].0, pairs[orig_i].1);
                        } else {
                            results[orig_i] = chunk_results[idx];
                        }
                    }
                }
            }

            // Short-string pairs (<= 64 chars): workgroup-shared-row kernel,
            // chunked by its larger per-pair buffer footprint.
            if !short_indices.is_empty() {
                let max_allowed_binding = self.engine.max_storage_buffer_binding_size as usize;
                let bytes_per_pair = ((SHORT_MAX_LEN + 1) * 4 * 2 + 8).max(128);
                let short_chunk_size = (max_allowed_binding / bytes_per_pair)
                    .min(MAX_DESIRED_CHUNK_PAIRS)
                    .max(512);

                for stream_chunk in short_indices.chunks(short_chunk_size) {
                    let chunk_results = self.compute_gpu_short(pairs, stream_chunk)?;
                    for (idx, &orig_i) in stream_chunk.iter().enumerate() {
                        results[orig_i] = chunk_results[idx];
                    }
                }
            }

            Ok(results)
        }

        fn compute_gpu_subset(&self, pairs: &[(&str, &str)], indices: &[usize]) -> Result<Vec<u32>> {
            let batch_size = indices.len() as u32;

            let mut offsets_a: Vec<u32> = Vec::with_capacity(indices.len() + 1);
            let mut chars_a: Vec<u32> = Vec::new();
            let mut offsets_b: Vec<u32> = Vec::with_capacity(indices.len() + 1);
            let mut chars_b: Vec<u32> = Vec::new();
            offsets_a.push(0);
            offsets_b.push(0);

            let mut max_len = 0u32;
            for &i in indices {
                let (a, b) = pairs[i];
                chars_a.extend(a.chars().map(|c| c as u32));
                offsets_a.push(chars_a.len() as u32);
                chars_b.extend(b.chars().map(|c| c as u32));
                offsets_b.push(chars_b.len() as u32);
                let a_count = a.chars().count();
                let b_count = b.chars().count();
                max_len = max_len.max(a_count.max(b_count) as u32);
            }

            if chars_a.is_empty() { chars_a.push(0); }
            if chars_b.is_empty() { chars_b.push(0); }

            // Validate total buffer allocations against hardware max binding size
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

            // Persistent buffers: ensure capacity for this batch, upload the
            // packed data once, and reuse the allocations for the whole
            // dispatch (and future calls). The guard is held across dispatch so
            // the readback staging is not reused until unmapped below.
            let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
            let offsets_bytes = ((offsets_a.len() * 4) as u64).max(results_size);
            pool.ensure(&self.engine.device, SLOT_OFFSETS_A, offsets_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, "oa");
            pool.ensure(&self.engine.device, SLOT_OFFSETS_B, offsets_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, "ob");
            pool.ensure(&self.engine.device, SLOT_CHARS_A, chars_a_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, "ca");
            pool.ensure(&self.engine.device, SLOT_CHARS_B, chars_b_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, "cb");
            pool.ensure(&self.engine.device, SLOT_RESULTS, results_size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, "res");
            pool.ensure(&self.engine.device, SLOT_STAGING, results_size, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, "stg");
            pool.ensure(&self.engine.device, SLOT_PARAMS, std::mem::size_of::<Params>() as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, "p");

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

            // Each chunk gets its OWN submit + readback. The params buffer is
            // written through the queue, so with one shared submit all
            // dispatches would see the LAST chunk's offset (wrong results for
            // batches > MAX_DISPATCH); per-chunk submit sequences each write
            // before its dispatch. The common case (<= 65535 pairs) is a
            // single chunk, so this costs nothing there.
            let mut gpu_results: Vec<u32> = Vec::with_capacity(batch_size as usize);
            let mut remaining = batch_size;
            let mut offset = 0u32;

            while remaining > 0 {
                let chunk = remaining.min(MAX_DISPATCH);
                let params = Params { batch_size, max_len, offset };
                pool.write(&self.engine.queue, SLOT_PARAMS, bytemuck::bytes_of(&params));

                let bg = self.engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: buf_offsets_a.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: buf_chars_a.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: buf_offsets_b.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: buf_chars_b.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: buf_results.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 5, resource: buf_params.as_entire_binding() },
                    ],
                });

                let mut encoder = self.engine.device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor { label: Some("levenshtein encoder") },
                );
                let workgroups = chunk.div_ceil(64);
                {
                    let mut pass = encoder.begin_compute_pass(
                        &wgpu::ComputePassDescriptor { label: None, timestamp_writes: None },
                    );
                    pass.set_pipeline(&self.pipeline);
                    pass.set_bind_group(0, &bg, &[]);
                    pass.dispatch_workgroups(workgroups, 1, 1);
                }

                let chunk_bytes = (chunk as u64) * 4;
                encoder.copy_buffer_to_buffer(buf_results, 0, buf_staging, 0, chunk_bytes);
                let bytes = self.engine.readback(encoder, &pool, chunk_bytes)?;
                let flat: &[u32] = bytemuck::cast_slice(&bytes);
                gpu_results.extend_from_slice(&flat[..chunk as usize]);

                remaining -= chunk;
                offset += chunk;
            }
            Ok(gpu_results)
        }

        /// Short-string path (both strings <= 64 chars): workgroup-shared DP
        /// rows + coalesced transposed char loads (levenshtein_short.wgsl).
        ///
        /// The transposed layout is chars_T[i*B + t] = i-th char of pair t
        /// (i in 1..=len; the padded region is never read because the shader's
        /// loops are bounded by the exact per-pair lengths). Per-pair lengths
        /// go in the offsets slots.
        fn compute_gpu_short(&self, pairs: &[(&str, &str)], indices: &[usize]) -> Result<Vec<u32>> {
            let batch_size = indices.len() as u32;
            if batch_size == 0 {
                return Ok(vec![]);
            }

            // Myers fast path: when every pair shares the same non-empty ASCII
            // query (<= 64 chars), one Peq table is shared per workgroup and
            // each thread runs the bit-vector in registers — ~15 ops per text
            // char, no per-thread DP row (see shaders/levenshtein_myers.wgsl).
            if let Some(query) = shared_ascii_query(pairs, indices) {
                return self.compute_gpu_myers(query, pairs, indices);
            }

            // Pass 1: per-pair lengths + max length (sizes the transpose).
            let mut len_a: Vec<u32> = Vec::with_capacity(indices.len());
            let mut len_b: Vec<u32> = Vec::with_capacity(indices.len());
            let mut max_len: u32 = 0;
            for &i in indices {
                let (a, b) = pairs[i];
                let al = a.chars().count() as u32;
                let bl = b.chars().count() as u32;
                len_a.push(al);
                len_b.push(bl);
                max_len = max_len.max(al.max(bl));
            }
            let rows = max_len + 1; // 0 never read; rows 1..=max_len valid

            // Pass 2: transposed char matrices.
            let mut chars_a_t = vec![0u32; (rows as usize) * indices.len()];
            let mut chars_b_t = vec![0u32; (rows as usize) * indices.len()];
            for (t, &i) in indices.iter().enumerate() {
                let (a, b) = pairs[i];
                for (k, ch) in a.chars().enumerate() {
                    chars_a_t[(k + 1) * indices.len() + t] = ch as u32;
                }
                for (k, ch) in b.chars().enumerate() {
                    chars_b_t[(k + 1) * indices.len() + t] = ch as u32;
                }
            }

            let chars_bytes = (chars_a_t.len() * 4) as u64;
            let lens_bytes = (batch_size as u64) * 4;
            let results_size = (batch_size as u64) * 4;
            let limit = self.engine.max_buffer_size_effective();
            if chars_bytes > limit || results_size > limit {
                return Err(FuzzGpuError::BufferError(
                    "Short batch buffer size exceeds device max_buffer_size".into(),
                ));
            }

            let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
            // Slot reuse: chars slots hold the transposed matrices, offsets
            // slots hold the per-pair lengths.
            pool.ensure(&self.engine.device, SLOT_CHARS_A, chars_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, "sca");
            pool.ensure(&self.engine.device, SLOT_CHARS_B, chars_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, "scb");
            pool.ensure(&self.engine.device, SLOT_OFFSETS_A, lens_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, "sla");
            pool.ensure(&self.engine.device, SLOT_OFFSETS_B, lens_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, "slb");
            pool.ensure(&self.engine.device, SLOT_RESULTS, results_size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, "sres");
            pool.ensure(&self.engine.device, SLOT_STAGING, results_size, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, "sstg");
            pool.ensure(&self.engine.device, SLOT_PARAMS, std::mem::size_of::<Params>() as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, "sp");

            pool.write(&self.engine.queue, SLOT_CHARS_A, bytemuck::cast_slice(&chars_a_t));
            pool.write(&self.engine.queue, SLOT_CHARS_B, bytemuck::cast_slice(&chars_b_t));
            pool.write(&self.engine.queue, SLOT_OFFSETS_A, bytemuck::cast_slice(&len_a));
            pool.write(&self.engine.queue, SLOT_OFFSETS_B, bytemuck::cast_slice(&len_b));

            let buf_chars_a = pool.get(SLOT_CHARS_A);
            let buf_chars_b = pool.get(SLOT_CHARS_B);
            let buf_len_a = pool.get(SLOT_OFFSETS_A);
            let buf_len_b = pool.get(SLOT_OFFSETS_B);
            let buf_results = pool.get(SLOT_RESULTS);
            let buf_params = pool.get(SLOT_PARAMS);

            let params = Params { batch_size, max_len, offset: 0 };
            pool.write(&self.engine.queue, SLOT_PARAMS, bytemuck::bytes_of(&params));

            let bg = self.engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: buf_chars_a.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: buf_chars_b.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: buf_len_a.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: buf_len_b.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: buf_results.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: buf_params.as_entire_binding() },
                ],
            });

            let mut encoder = self.engine.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("levenshtein short encoder") },
            );
            let workgroups = batch_size.div_ceil(64);
            {
                let mut pass = encoder.begin_compute_pass(
                    &wgpu::ComputePassDescriptor { label: None, timestamp_writes: None },
                );
                pass.set_pipeline(&self.short_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }

            encoder.copy_buffer_to_buffer(buf_results, 0, pool.get(SLOT_STAGING), 0, results_size);
            let bytes = self.engine.readback(encoder, &pool, results_size)?;
            let flat: &[u32] = bytemuck::cast_slice(&bytes);
            Ok(flat.to_vec())
        }

        /// Myers (1999) bit-vector path: one shared ASCII query vs N texts.
        ///
        /// Buffer slots: `SLOT_CHARS_A` = pattern bytes (linear), `SLOT_CHARS_B`
        /// = transposed text bytes (`chars_text[i*B + t]`), `SLOT_OFFSETS_A/B` =
        /// per-text byte lengths. See `levenshtein_myers.wgsl` for the kernel.
        fn compute_gpu_myers(
            &self,
            query: &str,
            pairs: &[(&str, &str)],
            indices: &[usize],
        ) -> Result<Vec<u32>> {
            let batch_size = indices.len() as u32;
            let m = query.len() as u32; // ASCII gate => len == char count

            // Per-text byte lengths + max (sizes the transpose).
            let mut lens: Vec<u32> = Vec::with_capacity(indices.len());
            let mut max_len: u32 = 0;
            for &i in indices {
                let bl = pairs[i].1.len() as u32;
                lens.push(bl);
                max_len = max_len.max(bl);
            }
            let rows = max_len + 1; // 0 never read

            // Transposed text bytes: chars_t[(k + 1) * B + t] = k-th byte of text t.
            let mut chars_t = vec![0u32; (rows as usize) * indices.len()];
            for (t, &i) in indices.iter().enumerate() {
                for (k, &ch) in pairs[i].1.as_bytes().iter().enumerate() {
                    chars_t[(k + 1) * indices.len() + t] = ch as u32;
                }
            }
            let pattern_chars: Vec<u32> = query.as_bytes().iter().map(|&b| b as u32).collect();

            let chars_bytes = (chars_t.len() * 4) as u64;
            let lens_bytes = (batch_size as u64) * 4;
            let results_size = (batch_size as u64) * 4;
            let limit = self.engine.max_buffer_size_effective();
            if chars_bytes > limit || results_size > limit {
                return Err(FuzzGpuError::BufferError(
                    "Myers batch buffer size exceeds device max_buffer_size".into(),
                ));
            }

            let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
            pool.ensure(&self.engine.device, SLOT_CHARS_A, (pattern_chars.len() * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, "mpa");
            pool.ensure(&self.engine.device, SLOT_CHARS_B, chars_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, "mtx");
            pool.ensure(&self.engine.device, SLOT_OFFSETS_A, lens_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, "mla");
            pool.ensure(&self.engine.device, SLOT_OFFSETS_B, lens_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, "mlb");
            pool.ensure(&self.engine.device, SLOT_RESULTS, results_size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, "mres");
            pool.ensure(&self.engine.device, SLOT_STAGING, results_size, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, "mstg");
            pool.ensure(&self.engine.device, SLOT_PARAMS, std::mem::size_of::<Params>() as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, "mprm");

            pool.write(&self.engine.queue, SLOT_CHARS_A, bytemuck::cast_slice(&pattern_chars));
            pool.write(&self.engine.queue, SLOT_CHARS_B, bytemuck::cast_slice(&chars_t));
            pool.write(&self.engine.queue, SLOT_OFFSETS_A, bytemuck::cast_slice(&lens));
            pool.write(&self.engine.queue, SLOT_OFFSETS_B, bytemuck::cast_slice(&lens));

            let buf_pattern = pool.get(SLOT_CHARS_A);
            let buf_texts = pool.get(SLOT_CHARS_B);
            let buf_lens = pool.get(SLOT_OFFSETS_A);
            let buf_lens_dup = pool.get(SLOT_OFFSETS_B);
            let buf_results = pool.get(SLOT_RESULTS);
            let buf_params = pool.get(SLOT_PARAMS);

            // Params.max_len carries the pattern length m (the bit-vector width).
            let params = Params { batch_size, max_len: m, offset: 0 };
            pool.write(&self.engine.queue, SLOT_PARAMS, bytemuck::bytes_of(&params));

            let bg = self.engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: buf_pattern.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: buf_texts.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: buf_lens.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: buf_lens_dup.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: buf_results.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: buf_params.as_entire_binding() },
                ],
            });

            let mut encoder = self.engine.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("levenshtein myers encoder") },
            );
            let workgroups = batch_size.div_ceil(64);
            {
                let mut pass = encoder.begin_compute_pass(
                    &wgpu::ComputePassDescriptor { label: None, timestamp_writes: None },
                );
                pass.set_pipeline(&self.myers_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }

            encoder.copy_buffer_to_buffer(buf_results, 0, pool.get(SLOT_STAGING), 0, results_size);
            let bytes = self.engine.readback(encoder, &pool, results_size)?;
            let flat: &[u32] = bytemuck::cast_slice(&bytes);
            Ok(flat.to_vec())
        }

        /// Dedicated 2D Matrix GPU Compute: O(N + M) data upload instead of O(N * M)
        /// Validates string lengths and memory limits *before* any GPU dispatch.
        pub fn compute_matrix(&self, list_a: &[&str], list_b: &[&str]) -> Result<Vec<Vec<u32>>> {
            // Serialize GPU dispatch across threads (gfx-rs/wgpu#10085).
            let _dispatch = self.engine.dispatch_lock();
            let rows = list_a.len();
            let cols = list_b.len();
            if rows == 0 || cols == 0 { return Ok(vec![]); }

            let total_pairs = rows * cols;
            // Small matrices: compute on CPU directly with Rayon (zero PCIe
            // transfer overhead) — threshold auto-selected from the adapter
            // (or user-set via GpuEngine::set_gpu_threshold).
            if total_pairs < self.engine.effective_gpu_threshold() {
                crate::gpu::GpuEngine::record_routing(0, total_pairs);
                return Ok(levenshtein_cdist_cpu(list_a, list_b));
            }
            crate::gpu::GpuEngine::record_routing(total_pairs, 0);

            // Row-wise Myers fast path: replaces the O(m·n) DP matrix kernel
            // with the bit-vector when every row query is non-empty ASCII
            // <= 64 bytes (the pattern) and every text is ASCII (any length).
            // Empty queries can't use it (distance = text length), so those
            // rows fall through to the general kernel below.
            if list_a.iter().all(|s| !s.is_empty() && s.is_ascii() && s.len() <= SHORT_MAX_LEN)
                && list_b.iter().all(|s| s.is_ascii())
            {
                return self.compute_matrix_myers(list_a, list_b);
            }

            // CRITICAL FIX: Pre-filter oversized strings BEFORE GPU buffer allocation or dispatch
            let has_oversized = list_a.iter().any(|s| s.chars().count() > GPU_MAX_STRING_LEN)
                || list_b.iter().any(|s| s.chars().count() > GPU_MAX_STRING_LEN);

            let matrix_size = (total_pairs as u64) * 4;

            // If strings exceed GPU shader capacity (256 chars) or matrix exceeds GPU buffer limit, fall back to CPU immediately
            if has_oversized || matrix_size > self.engine.max_buffer_size_effective() {
                return Ok(levenshtein_cdist_cpu(list_a, list_b));
            }

            // Pack List A (O(N) data)
            let mut offsets_a: Vec<u32> = Vec::with_capacity(rows + 1);
            let mut chars_a: Vec<u32> = Vec::new();
            offsets_a.push(0);
            for a in list_a {
                chars_a.extend(a.chars().map(|c| c as u32));
                offsets_a.push(chars_a.len() as u32);
            }

            // Pack List B (O(M) data)
            let mut offsets_b: Vec<u32> = Vec::with_capacity(cols + 1);
            let mut chars_b: Vec<u32> = Vec::new();
            offsets_b.push(0);
            for b in list_b {
                chars_b.extend(b.chars().map(|c| c as u32));
                offsets_b.push(chars_b.len() as u32);
            }

            if chars_a.is_empty() { chars_a.push(0); }
            if chars_b.is_empty() { chars_b.push(0); }

            // Persistent buffers (see gpu::BufferPool) — same arena as the batch
            // path; matrix/staging slots grow to the larger matrix size.
            let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
            let offsets_bytes = ((offsets_a.len() * 4) as u64).max(matrix_size);
            pool.ensure(&self.engine.device, SLOT_OFFSETS_A, offsets_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, "moa");
            pool.ensure(&self.engine.device, SLOT_OFFSETS_B, offsets_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, "mob");
            pool.ensure(&self.engine.device, SLOT_CHARS_A, (chars_a.len() * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, "mca");
            pool.ensure(&self.engine.device, SLOT_CHARS_B, (chars_b.len() * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, "mcb");
            pool.ensure(&self.engine.device, SLOT_RESULTS, matrix_size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, "mres");
            pool.ensure(&self.engine.device, SLOT_STAGING, matrix_size, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, "mstg");
            pool.ensure(&self.engine.device, SLOT_PARAMS, std::mem::size_of::<MatrixParams>() as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, "mp");

            pool.write(&self.engine.queue, SLOT_OFFSETS_A, bytemuck::cast_slice(&offsets_a));
            pool.write(&self.engine.queue, SLOT_CHARS_A, bytemuck::cast_slice(&chars_a));
            pool.write(&self.engine.queue, SLOT_OFFSETS_B, bytemuck::cast_slice(&offsets_b));
            pool.write(&self.engine.queue, SLOT_CHARS_B, bytemuck::cast_slice(&chars_b));

            let params = MatrixParams { rows: rows as u32, cols: cols as u32 };
            pool.write(&self.engine.queue, SLOT_PARAMS, bytemuck::bytes_of(&params));

            let buf_offsets_a = pool.get(SLOT_OFFSETS_A);
            let buf_chars_a = pool.get(SLOT_CHARS_A);
            let buf_offsets_b = pool.get(SLOT_OFFSETS_B);
            let buf_chars_b = pool.get(SLOT_CHARS_B);
            let buf_matrix = pool.get(SLOT_RESULTS);
            let buf_staging = pool.get(SLOT_STAGING);
            let buf_params = pool.get(SLOT_PARAMS);

            let bg = self.engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bind_group_layout,
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
                &wgpu::CommandEncoderDescriptor { label: Some("levenshtein matrix encoder") },
            );

            // 2D dispatch: workgroups across cols (x) and rows (y)
            let workgroups_x = (cols as u32).div_ceil(16);
            let workgroups_y = (rows as u32).div_ceil(16);

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
                .map_err(|_| FuzzGpuError::Timeout("GPU matrix readback timed out after 10s".into()))?
                .map_err(|e| FuzzGpuError::BufferError(format!("GPU matrix buffer map failed: {}", e)))?;

            let data = slice
                .get_mapped_range()
                .map_err(|e| FuzzGpuError::BufferError(format!("GPU buffer map range failed: {e}")))?;
            let flat: &[u32] = bytemuck::cast_slice(&data);

            let mut matrix: Vec<Vec<u32>> = Vec::with_capacity(rows);
            for i in 0..rows {
                let start = i * cols;
                let end = start + cols;
                matrix.push(flat[start..end].to_vec());
            }

            drop(data);
            buf_staging.unmap();
            Ok(matrix)
        }

        /// Row-wise Myers matrix: one workgroup per row, shared Peq per row in
        /// SLM, bit-vector per cell instead of DP. Preconditions (checked by
        /// the caller before dispatch): every `list_a` string is non-empty
        /// ASCII <= 64 bytes (the pattern); every `list_b` string is ASCII
        /// (any length).
        fn compute_matrix_myers(&self, list_a: &[&str], list_b: &[&str]) -> Result<Vec<Vec<u32>>> {
            let rows = list_a.len();
            let cols = list_b.len();
            if rows == 0 || cols == 0 {
                return Ok(vec![]);
            }

            let matrix_size = ((rows * cols) as u64) * 4;
            if matrix_size > self.engine.max_buffer_size_effective() {
                return Ok(levenshtein_cdist_cpu(list_a, list_b));
            }

            // Pack list_a (patterns) with offsets; pack list_b (texts) with
            // offsets. ASCII bytes become u32 values 0..=127.
            let mut offsets_a: Vec<u32> = Vec::with_capacity(rows + 1);
            let mut chars_a: Vec<u32> = Vec::new();
            offsets_a.push(0);
            for a in list_a {
                chars_a.extend(a.as_bytes().iter().map(|&b| b as u32));
                offsets_a.push(chars_a.len() as u32);
            }
            let mut offsets_b: Vec<u32> = Vec::with_capacity(cols + 1);
            let mut chars_b: Vec<u32> = Vec::new();
            offsets_b.push(0);
            for b in list_b {
                chars_b.extend(b.as_bytes().iter().map(|&b| b as u32));
                offsets_b.push(chars_b.len() as u32);
            }
            if chars_a.is_empty() {
                chars_a.push(0);
            }
            if chars_b.is_empty() {
                chars_b.push(0);
            }

            let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
            let offsets_bytes = (((rows + 1).max(cols + 1)) * 4) as u64;
            pool.ensure(&self.engine.device, SLOT_OFFSETS_A, offsets_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, "cmoa");
            pool.ensure(&self.engine.device, SLOT_OFFSETS_B, offsets_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, "cmob");
            pool.ensure(&self.engine.device, SLOT_CHARS_A, (chars_a.len() * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, "cmca");
            pool.ensure(&self.engine.device, SLOT_CHARS_B, (chars_b.len() * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, "cmcb");
            pool.ensure(&self.engine.device, SLOT_RESULTS, matrix_size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, "cmres");
            pool.ensure(&self.engine.device, SLOT_STAGING, matrix_size, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, "cmstg");
            pool.ensure(&self.engine.device, SLOT_PARAMS, std::mem::size_of::<MatrixParams>() as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, "cmprm");

            pool.write(&self.engine.queue, SLOT_OFFSETS_A, bytemuck::cast_slice(&offsets_a));
            pool.write(&self.engine.queue, SLOT_CHARS_A, bytemuck::cast_slice(&chars_a));
            pool.write(&self.engine.queue, SLOT_OFFSETS_B, bytemuck::cast_slice(&offsets_b));
            pool.write(&self.engine.queue, SLOT_CHARS_B, bytemuck::cast_slice(&chars_b));

            let buf_offsets_a = pool.get(SLOT_OFFSETS_A);
            let buf_chars_a = pool.get(SLOT_CHARS_A);
            let buf_offsets_b = pool.get(SLOT_OFFSETS_B);
            let buf_chars_b = pool.get(SLOT_CHARS_B);
            let buf_matrix = pool.get(SLOT_RESULTS);
            let buf_staging = pool.get(SLOT_STAGING);
            let buf_params = pool.get(SLOT_PARAMS);

            let bg = self.engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: buf_chars_a.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: buf_offsets_a.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: buf_chars_b.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: buf_offsets_b.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: buf_matrix.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: buf_params.as_entire_binding() },
                ],
            });

            let mut flat = vec![0u32; rows * cols];
            let params = MatrixParams { rows: rows as u32, cols: cols as u32 };
            pool.write(&self.engine.queue, SLOT_PARAMS, bytemuck::bytes_of(&params));

            // Chunk over rows so workgroup counts stay <= 65535 (dispatch
            // limit); each chunk dispatches + reads back before the next, so
            // the params buffer (written once per chunk, same values) can't
            // race with a later dispatch.
            const ROWS_PER_CHUNK: u32 = 60000;
            let mut dispatched = 0usize;
            while dispatched < rows {
                let chunk_rows = ((rows - dispatched) as u32).min(ROWS_PER_CHUNK);
                let chunk_params = MatrixParams { rows: chunk_rows, cols: cols as u32 };
                pool.write(&self.engine.queue, SLOT_PARAMS, bytemuck::bytes_of(&chunk_params));

                let mut encoder = self.engine.device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor { label: Some("levenshtein myers cdist encoder") },
                );
                {
                    let mut pass = encoder.begin_compute_pass(
                        &wgpu::ComputePassDescriptor { label: None, timestamp_writes: None },
                    );
                    pass.set_pipeline(&self.myers_cdist_pipeline);
                    pass.set_bind_group(0, &bg, &[]);
                    pass.dispatch_workgroups(chunk_rows, 1, 1);
                }

                // Copy only this chunk's rows back.
                let chunk_bytes = (chunk_rows as usize * cols * 4) as u64;
                encoder.copy_buffer_to_buffer(
                    buf_matrix,
                    dispatched as u64 * cols as u64 * 4,
                    buf_staging,
                    0,
                    chunk_bytes,
                );
                let bytes = self.engine.readback(encoder, &pool, chunk_bytes)?;
                let chunk_flat: &[u32] = bytemuck::cast_slice(&bytes);
                let chunk_len = chunk_rows as usize * cols;
                flat[dispatched * cols..dispatched * cols + chunk_len].copy_from_slice(&chunk_flat[..chunk_len]);
                dispatched += chunk_rows as usize;
            }

            let mut matrix: Vec<Vec<u32>> = Vec::with_capacity(rows);
            for i in 0..rows {
                let start = i * cols;
                matrix.push(flat[start..start + cols].to_vec());
            }
            Ok(matrix)
        }

        /// Create a batched dispatch: enqueue several pair-lists, then
        /// [`GpuLevenshteinBatch::execute`] once. All GPU-eligible pairs across
        /// every enqueued op are packed into a single command encoder and read
        /// back with one sync, amortizing the per-call round-trip that
        /// dominates small dispatches (measured: ~1 ms/call on iGPUs).
        pub fn batch(&self) -> GpuLevenshteinBatch<'_> {
            GpuLevenshteinBatch { kernel: self, ops: Vec::new() }
        }
    }

    /// A queued set of Levenshtein batch operations executed with a single GPU
    /// dispatch + readback. Each enqueued op returns its own `Vec<u32>` of
    /// distances, with the same semantics as [`GpuLevenshteinKernel::compute`]
    /// (empty/identical short-circuits, >256-char pairs routed to CPU, sentinel
    /// recompute) applied per op.
    pub struct GpuLevenshteinBatch<'k> {
        kernel: &'k GpuLevenshteinKernel,
        ops: Vec<Vec<(&'k str, &'k str)>>,
    }

    impl<'k> GpuLevenshteinBatch<'k> {
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
        pub fn execute(self) -> Result<Vec<Vec<u32>>> {
            // Serialize GPU dispatch across threads (gfx-rs/wgpu#10085).
            let _dispatch = self.kernel.engine.dispatch_lock();
            let n_ops = self.ops.len();
            if n_ops == 0 {
                return Ok(vec![]);
            }

            // Classify every pair across ops: empty/identical short-circuits,
            // >256-char pairs to CPU, the rest to one shared GPU batch. The
            // offsets/chars are packed during classification so we never walk
            // the pairs twice.
            let mut out: Vec<Vec<u32>> = Vec::with_capacity(n_ops);
            let mut gpu_ranges: Vec<(u32, u32)> = Vec::with_capacity(n_ops); // (start, count) in global GPU index space (long path)
            let mut op_gpu_to_pair: Vec<Vec<usize>> = Vec::with_capacity(n_ops);
            let mut short_pairs: Vec<(usize, usize)> = Vec::new(); // (op, pair in op) for the <=64-char kernel
            let mut cpu_oversized: Vec<(usize, usize)> = Vec::new(); // (op, pair in op)
            let mut offsets_a: Vec<u32> = vec![0];
            let mut chars_a: Vec<u32> = Vec::new();
            let mut offsets_b: Vec<u32> = vec![0];
            let mut chars_b: Vec<u32> = Vec::new();
            let mut gpu_global: u32 = 0;
            let mut max_len: u32 = 0;

            for (op_i, pairs) in self.ops.iter().enumerate() {
                let mut op_results = vec![0u32; pairs.len()];
                let mut op_gpu: Vec<usize> = Vec::new();
                for (j, (a, b)) in pairs.iter().enumerate() {
                    if a.is_empty() || b.is_empty() {
                        let a_count = if a.is_ascii() { a.len() } else { a.chars().count() };
                        let b_count = if b.is_ascii() { b.len() } else { b.chars().count() };
                        op_results[j] = (a_count.max(b_count)) as u32;
                    } else if *a == *b {
                        op_results[j] = 0;
                    } else {
                        let a_len = a.chars().count();
                        let b_len = b.chars().count();
                        if a_len > GPU_MAX_STRING_LEN || b_len > GPU_MAX_STRING_LEN {
                            cpu_oversized.push((op_i, j));
                        } else if a_len <= SHORT_MAX_LEN && b_len <= SHORT_MAX_LEN {
                            short_pairs.push((op_i, j));
                        } else {
                            chars_a.extend(a.chars().map(|c| c as u32));
                            offsets_a.push(chars_a.len() as u32);
                            chars_b.extend(b.chars().map(|c| c as u32));
                            offsets_b.push(chars_b.len() as u32);
                            op_gpu.push(j);
                            max_len = max_len.max(a_len.max(b_len) as u32);
                        }
                    }
                }
                let start = gpu_global;
                gpu_global += op_gpu.len() as u32;
                gpu_ranges.push((start, op_gpu.len() as u32));
                op_gpu_to_pair.push(op_gpu);
                out.push(op_results);
            }

            let long_count = gpu_global as usize;
            let total_gpu = long_count + short_pairs.len();

            // Oversized pairs are always computed on CPU (Rayon, parallel).
            if !cpu_oversized.is_empty() {
                let cpu_res: Vec<u32> = cpu_oversized
                    .par_iter()
                    .map(|&(op, j)| levenshtein_distance_raw(self.ops[op][j].0, self.ops[op][j].1))
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
                gpu_op_pair.extend_from_slice(&short_pairs);
                let cpu_res: Vec<u32> = gpu_op_pair
                    .par_iter()
                    .map(|&(op, j)| levenshtein_distance_raw(self.ops[op][j].0, self.ops[op][j].1))
                    .collect();
                for (idx, &(op, j)) in gpu_op_pair.iter().enumerate() {
                    out[op][j] = cpu_res[idx];
                }
                return Ok(out);
            }
            crate::gpu::GpuEngine::record_routing(total_gpu, 0);

            // Long-path pairs (65..=256 chars): one encoder, all chunks, one
            // readback. Skipped entirely when the batch has no long pairs.
            if long_count > 0 {
            if chars_a.is_empty() { chars_a.push(0); }
            if chars_b.is_empty() { chars_b.push(0); }

            // Validate the long-path allocations against the device limit.
            let chars_a_bytes = (chars_a.len() * 4) as u64;
            let chars_b_bytes = (chars_b.len() * 4) as u64;
            let results_size = (long_count as u64) * 4;
            let limit = self.kernel.engine.max_buffer_size_effective();
            if chars_a_bytes > limit || chars_b_bytes > limit || results_size > limit {
                return Err(FuzzGpuError::BufferError(
                    "Batch buffer size exceeds device max_buffer_size".into(),
                ));
            }

            let mut pool = self.kernel.pool.lock().unwrap_or_else(|e| e.into_inner());
            let offsets_bytes = ((offsets_a.len() * 4) as u64).max(results_size);
            pool.ensure(&self.kernel.engine.device, SLOT_OFFSETS_A, offsets_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, "boa");
            pool.ensure(&self.kernel.engine.device, SLOT_OFFSETS_B, offsets_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, "bob");
            pool.ensure(&self.kernel.engine.device, SLOT_CHARS_A, chars_a_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, "bca");
            pool.ensure(&self.kernel.engine.device, SLOT_CHARS_B, chars_b_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, "bcb");
            pool.ensure(&self.kernel.engine.device, SLOT_RESULTS, results_size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, "bres");
            pool.ensure(&self.kernel.engine.device, SLOT_STAGING, results_size, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, "bstg");
            pool.ensure(&self.kernel.engine.device, SLOT_PARAMS, std::mem::size_of::<Params>() as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, "bp");

            pool.write(&self.kernel.engine.queue, SLOT_OFFSETS_A, bytemuck::cast_slice(&offsets_a));
            pool.write(&self.kernel.engine.queue, SLOT_CHARS_A, bytemuck::cast_slice(&chars_a));
            pool.write(&self.kernel.engine.queue, SLOT_OFFSETS_B, bytemuck::cast_slice(&offsets_b));
            pool.write(&self.kernel.engine.queue, SLOT_CHARS_B, bytemuck::cast_slice(&chars_b));

            let buf_offsets_a = pool.get(SLOT_OFFSETS_A);
            let buf_chars_a = pool.get(SLOT_CHARS_A);
            let buf_offsets_b = pool.get(SLOT_OFFSETS_B);
            let buf_chars_b = pool.get(SLOT_CHARS_B);
            let buf_results = pool.get(SLOT_RESULTS);
            let buf_params = pool.get(SLOT_PARAMS);

            let mut remaining = long_count as u32;
            let mut offset = 0u32;
            let mut flat: Vec<u32> = Vec::with_capacity(long_count);
            while remaining > 0 {
                let chunk = remaining.min(MAX_DISPATCH);
                let params = Params { batch_size: long_count as u32, max_len, offset };
                pool.write(&self.kernel.engine.queue, SLOT_PARAMS, bytemuck::bytes_of(&params));

                let bg = self.kernel.engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.kernel.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: buf_offsets_a.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: buf_chars_a.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: buf_offsets_b.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: buf_chars_b.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: buf_results.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 5, resource: buf_params.as_entire_binding() },
                    ],
                });

                let mut encoder = self.kernel.engine.device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor { label: Some("levenshtein batch encoder") },
                );
                let workgroups = chunk.div_ceil(64);
                {
                    let mut pass = encoder.begin_compute_pass(
                        &wgpu::ComputePassDescriptor { label: None, timestamp_writes: None },
                    );
                    pass.set_pipeline(&self.kernel.pipeline);
                    pass.set_bind_group(0, &bg, &[]);
                    pass.dispatch_workgroups(workgroups, 1, 1);
                }
                let chunk_bytes = (chunk as u64) * 4;
                encoder.copy_buffer_to_buffer(buf_results, 0, pool.get(SLOT_STAGING), 0, chunk_bytes);
                let bytes = self.kernel.engine.readback(encoder, &pool, chunk_bytes)?;
                flat.extend_from_slice(bytemuck::cast_slice(&bytes));

                remaining -= chunk;
                offset += chunk;
            }

            // Split the flat result range back into per-op vectors.
            for (op, &(start, count)) in gpu_ranges.iter().enumerate() {
                for k in 0..count as usize {
                    let j = op_gpu_to_pair[op][k];
                    let v = flat[(start as usize) + k];
                    if v == 0xFFFFFFFF {
                        out[op][j] = levenshtein_distance_raw(self.ops[op][j].0, self.ops[op][j].1);
                    } else {
                        out[op][j] = v;
                    }
                }
            }
            } // end long_count > 0

            // Short-path pairs (<= 64 chars): workgroup-shared-row kernel.
            if !short_pairs.is_empty() {
                let flat_short: Vec<(&str, &str)> = short_pairs
                    .iter()
                    .map(|&(op, j)| self.ops[op][j])
                    .collect();
                let indices: Vec<usize> = (0..flat_short.len()).collect();
                let short_res = self.kernel.compute_gpu_short(&flat_short, &indices)?;
                for (k, &(op, j)) in short_pairs.iter().enumerate() {
                    out[op][j] = short_res[k];
                }
            }
            Ok(out)
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

        fn gpu_kernel_or_skip() -> Option<&'static GpuLevenshteinKernel> {
            match GpuLevenshteinKernel::get() {
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

        /// Exercises shader compilation, buffer sizing/allocation, dispatch,
        /// readback, and the 0xFFFFFFFF sentinel path end-to-end against CPU results.
        #[test]
        fn test_gpu_compute_matches_cpu() {
            let _gpu_guard = crate::gpu::gpu_test_lock();
            let Some(kernel) = gpu_kernel_or_skip() else { return; };

            let a = gen_strings(1000, 0xDEADBEEF);
            let b = gen_strings(1000, 0xCAFEBABE);
            let mut pairs: Vec<(&str, &str)> =
                a.iter().zip(b.iter()).map(|(x, y)| (x.as_str(), y.as_str())).collect();
            // Include edge cases that must be resolved on the CPU path.
            pairs[0] = ("", "");
            pairs[1] = ("", "xyz");
            pairs[2] = ("hello", "hello");

            let gpu = kernel.compute(&pairs).expect("GPU compute should succeed");
            let cpu: Vec<u32> = pairs.iter()
                .map(|(x, y)| levenshtein_distance_raw(x, y))
                .collect();
            assert_eq!(gpu, cpu, "GPU Levenshtein results must match CPU");
            // The 0xFFFFFFFF sentinel must never leak into results.
            assert!(!gpu.contains(&0xFFFFFFFF), "0xFFFFFFFF sentinel leaked into results");
        }

        /// Validates the 2D matrix pipeline (shader, O(N+M) packing, readback).
        #[test]
        fn test_gpu_compute_matrix_matches_cpu() {
            let _gpu_guard = crate::gpu::gpu_test_lock();
            let Some(kernel) = gpu_kernel_or_skip() else { return; };

            let a = gen_strings(30, 0x1234ABCD);
            let b = gen_strings(30, 0x5678EF01);
            let refs_a: Vec<&str> = a.iter().map(|s| s.as_str()).collect();
            let refs_b: Vec<&str> = b.iter().map(|s| s.as_str()).collect();

            let gpu = kernel.compute_matrix(&refs_a, &refs_b).expect("GPU matrix should succeed");
            let cpu = levenshtein_cdist_cpu(&refs_a, &refs_b);
            assert_eq!(gpu, cpu, "GPU Levenshtein matrix must match CPU");
            for row in &gpu {
                assert!(!row.contains(&0xFFFFFFFF), "0xFFFFFFFF sentinel leaked into matrix");
            }
        }

        /// Arms the readback-timeout fault and verifies the batch path returns
        /// `FuzzGpuError::Timeout` deterministically (fast, and never Ok).
        #[test]
        fn test_batch_readback_timeout_returns_timeout_error() {
            let _gpu_guard = crate::gpu::gpu_test_lock();
            let Some(kernel) = gpu_kernel_or_skip() else { return; };
            // Force GPU dispatch: below the auto threshold the fault is never
            // exercised because the batch routes to CPU before reaching the GPU
            // readback path.
            GpuEngine::set_gpu_threshold(Some(1));

            let a = gen_strings(1000, 0xABCDEF01);
            let b = gen_strings(1000, 0x23456789);
            let pairs: Vec<(&str, &str)> =
                a.iter().zip(b.iter()).map(|(x, y)| (x.as_str(), y.as_str())).collect();

            crate::gpu::arm_readback_timeout_fault();
            let result = kernel.compute(&pairs);
            crate::gpu::disarm_readback_timeout_fault();
            GpuEngine::set_gpu_threshold(None);

            match result {
                Err(FuzzGpuError::Timeout(_)) => {} // expected
                Err(e) => panic!("expected FuzzGpuError::Timeout, got: {}", e),
                Ok(_) => panic!("expected FuzzGpuError::Timeout, got Ok results"),
            }
        }

        /// Same fault-injection check for the 2D matrix readback path.
        #[test]
        fn test_matrix_readback_timeout_returns_timeout_error() {
            let _gpu_guard = crate::gpu::gpu_test_lock();
            let Some(kernel) = gpu_kernel_or_skip() else { return; };
            GpuEngine::set_gpu_threshold(Some(1));

            let a = gen_strings(30, 0x11111111);
            let b = gen_strings(30, 0x22222222);
            let refs_a: Vec<&str> = a.iter().map(|s| s.as_str()).collect();
            let refs_b: Vec<&str> = b.iter().map(|s| s.as_str()).collect();

            crate::gpu::arm_readback_timeout_fault();
            let result = kernel.compute_matrix(&refs_a, &refs_b);
            crate::gpu::disarm_readback_timeout_fault();
            GpuEngine::set_gpu_threshold(None);

            match result {
                Err(FuzzGpuError::Timeout(_)) => {} // expected
                Err(e) => panic!("expected FuzzGpuError::Timeout, got: {}", e),
                Ok(_) => panic!("expected FuzzGpuError::Timeout, got Ok results"),
            }
        }

        /// Arms the small-buffer fault and verifies the batch path returns
        /// `FuzzGpuError::BufferError` when inputs exceed the (faulted) limit.
        #[test]
        fn test_buffer_size_validation_returns_buffer_error() {
            let _gpu_guard = crate::gpu::gpu_test_lock();
            let Some(kernel) = gpu_kernel_or_skip() else { return; };
            GpuEngine::set_gpu_threshold(Some(1));

            let a = gen_strings(1000, 0x0BADF00D);
            let b = gen_strings(1000, 0xF00DBABE);
            let pairs: Vec<(&str, &str)> =
                a.iter().zip(b.iter()).map(|(x, y)| (x.as_str(), y.as_str())).collect();

            crate::gpu::arm_small_buffer_fault();
            let result = kernel.compute(&pairs);
            crate::gpu::disarm_small_buffer_fault();
            GpuEngine::set_gpu_threshold(None);

            match result {
                Err(FuzzGpuError::BufferError(_)) => {} // expected
                Err(e) => panic!("expected FuzzGpuError::BufferError, got: {}", e),
                Ok(_) => panic!("expected FuzzGpuError::BufferError, got Ok results"),
            }
        }

        /// With the small-buffer fault armed, the matrix path must gracefully
        /// fall back to CPU results instead of erroring.
        #[test]
        fn test_matrix_oversize_falls_back_to_cpu() {
            let _gpu_guard = crate::gpu::gpu_test_lock();
            let Some(kernel) = gpu_kernel_or_skip() else { return; };

            let a = gen_strings(32, 0x5EEDC0DE);
            let b = gen_strings(32, 0x0DDBA11);
            let refs_a: Vec<&str> = a.iter().map(|s| s.as_str()).collect();
            let refs_b: Vec<&str> = b.iter().map(|s| s.as_str()).collect();

            crate::gpu::arm_small_buffer_fault();
            let result = kernel.compute_matrix(&refs_a, &refs_b);
            crate::gpu::disarm_small_buffer_fault();

            let cpu = levenshtein_cdist_cpu(&refs_a, &refs_b);
            match result {
                Ok(matrix) => {
                    assert_eq!(matrix, cpu, "matrix fallback must equal CPU results");
                }
                Err(e) => panic!("expected graceful CPU fallback, got error: {}", e),
            }
        }

        /// Arms the shader-error fault and verifies `new_inner` returns
        /// `FuzzGpuError::ShaderError` instead of panicking on invalid WGSL.
        #[test]
        fn test_shader_validation_error_returns_shader_error() {
            let _gpu_guard = crate::gpu::gpu_test_lock();
            let engine = match GpuEngine::get() {
                Ok(e) => e,
                Err(e) => {
                    if crate::gpu::require_gpu() {
                        panic!("FUZZGPU_REQUIRE_GPU is set but no usable GPU device: {}", e);
                    }
                    eprintln!("skipping GPU test (no usable device): {}", e);
                    return;
                }
            };

            crate::gpu::arm_shader_error_fault();
            let result = GpuLevenshteinKernel::new_inner(engine);
            crate::gpu::disarm_shader_error_fault();

            match result {
                Err(FuzzGpuError::ShaderError(_)) => {} // expected
                Err(e) => panic!("expected FuzzGpuError::ShaderError, got: {}", e),
                Ok(_) => panic!("expected FuzzGpuError::ShaderError, got Ok kernel"),
            }
        }

        /// The batched API must return exactly what per-op `compute` returns,
        /// including the CPU-routed edge cases and Unicode char counting.
        #[test]
        fn test_gpu_batch_matches_compute() {
            let _gpu_guard = crate::gpu::gpu_test_lock();
            let Some(kernel) = gpu_kernel_or_skip() else { return; };

            let a1 = gen_strings(600, 0x1A2B3C4D);
            let b1 = gen_strings(600, 0x5E6F7081);
            let mut op1: Vec<(&str, &str)> =
                a1.iter().zip(b1.iter()).map(|(x, y)| (x.as_str(), y.as_str())).collect();
            // CPU-routed edge cases inside a GPU-sized op.
            op1[0] = ("", "");
            op1[1] = ("", "xyz");
            op1[2] = ("hello", "hello");

            // Unicode pair (char-counted, not byte-counted) + an oversized pair.
            let long = "a".repeat(300);
            let op2: Vec<(&str, &str)> = vec![
                ("日本語のテキスト", "日本語のテスト"),
                (&long, "short"),
                ("emoji 😀🎉 test", "emoji 😀 test"),
            ];

            let op3: Vec<(&str, &str)> = vec![("kitten", "sitting"), ("", "x"), ("same", "same")];

            let expected = vec![
                kernel.compute(&op1).expect("op1 compute"),
                kernel.compute(&op2).expect("op2 compute"),
                kernel.compute(&op3).expect("op3 compute"),
            ];

            let mut batch = kernel.batch();
            batch.add(&op1);
            batch.add(&op2);
            batch.add(&op3);
            assert_eq!(batch.len(), 3);
            let got = batch.execute().expect("batch execute");

            assert_eq!(got, expected, "batch results must equal per-op compute results");
            for op in &got {
                assert!(!op.contains(&0xFFFFFFFF), "0xFFFFFFFF sentinel leaked into batch results");
            }
        }

        /// Batch results must agree with the raw CPU distance for every pair.
        #[test]
        fn test_gpu_batch_matches_cpu() {
            let _gpu_guard = crate::gpu::gpu_test_lock();
            let Some(kernel) = gpu_kernel_or_skip() else { return; };

            let a = gen_strings(700, 0xDEADBEEF);
            let b = gen_strings(700, 0xCAFEBABE);
            let op1: Vec<(&str, &str)> =
                a.iter().zip(b.iter()).map(|(x, y)| (x.as_str(), y.as_str())).collect();
            let op2: Vec<(&str, &str)> = vec![("", ""), ("a", "b"), ("test", "taste")];

            let mut batch = kernel.batch();
            batch.add(&op1);
            batch.add(&op2);
            let got = batch.execute().expect("batch execute");

            for (op, pairs) in got.iter().zip([&op1, &op2]) {
                for (i, &v) in op.iter().enumerate() {
                    let expected = levenshtein_distance_raw(pairs[i].0, pairs[i].1);
                    assert_eq!(v, expected, "batch result {} must equal CPU distance", i);
                }
            }
        }

        /// An empty batch is a no-op that returns an empty result set.
        #[test]
        fn test_gpu_batch_empty_returns_empty() {
            let _gpu_guard = crate::gpu::gpu_test_lock();
            let Some(kernel) = gpu_kernel_or_skip() else { return; };

            let batch = kernel.batch();
            assert!(batch.is_empty());
            let got = batch.execute().expect("empty batch execute");
            assert!(got.is_empty());
        }

        /// The 65..=256-char range must take the general kernel (not the short
        /// kernel, not CPU) and still match CPU exactly, including the 64/65
        /// short/long routing boundary.
        #[test]
        fn test_gpu_long_string_path_matches_cpu() {
            let _gpu_guard = crate::gpu::gpu_test_lock();
            let Some(kernel) = gpu_kernel_or_skip() else { return; };

            fn gen_long(count: usize, seed: u64) -> Vec<String> {
                let mut state = seed;
                let mut out = Vec::with_capacity(count);
                for _ in 0..count {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let len = 70 + ((state >> 33) as usize % 21); // 70..=90 chars
                    let mut s = String::with_capacity(len);
                    for _ in 0..len {
                        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        s.push((b'a' + ((state >> 33) as u8 % 26)) as char);
                    }
                    out.push(s);
                }
                out
            }

            let a = gen_long(600, 0xABCD);
            let b = gen_long(600, 0x1234);
            let mut pairs: Vec<(&str, &str)> =
                a.iter().zip(b.iter()).map(|(x, y)| (x.as_str(), y.as_str())).collect();
            // Boundary: 64 chars still short-kernel, 65 chars long-kernel.
            let x64 = "x".repeat(64);
            let y64 = "y".repeat(64);
            let x65 = "x".repeat(65);
            let y65 = "y".repeat(65);
            pairs.push((&x64, &y64));
            pairs.push((&x65, &y65));

            let gpu = kernel.compute(&pairs).expect("GPU compute should succeed");
            let cpu: Vec<u32> = pairs.iter()
                .map(|(x, y)| levenshtein_distance_raw(x, y))
                .collect();
            assert_eq!(gpu, cpu, "GPU long-string results must match CPU");
        }

        /// The Myers (1999) bit-vector kernel must agree with the CPU for a
        /// shared-ASCII-query batch — the exact shape that triggers it (all
        /// pairs share one query <= 64 chars, all ASCII). Forced via the
        /// private dispatch fn so the test does not depend on the routing
        /// heuristic.
        #[test]
        fn test_gpu_myers_matches_cpu() {
            let _gpu_guard = crate::gpu::gpu_test_lock();
            let Some(kernel) = gpu_kernel_or_skip() else { return; };

            let query: &str = "kitten-sitting-fuzzy-matching";
            let mut state: u64 = 0xFEED_CAFE;
            let mut texts: Vec<String> = Vec::with_capacity(600);
            for _ in 0..600 {
                let len = 1 + ((state >> 33) as usize % 24);
                let mut s = String::with_capacity(len);
                for _ in 0..len {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    s.push((b'a' + ((state >> 33) as u8 % 26)) as char);
                }
                texts.push(s);
            }
            // Edge cases: identical text (score 0), empty text (score = query len).
            texts[0] = query.to_string();
            texts[1] = String::new();

            let pairs: Vec<(&str, &str)> =
                texts.iter().map(|t| (query, t.as_str())).collect();
            let indices: Vec<usize> = (0..pairs.len()).collect();

            let gpu = kernel
                .compute_gpu_myers(query, &pairs, &indices)
                .expect("Myers GPU dispatch should succeed");
            let cpu: Vec<u32> = pairs.iter()
                .map(|(a, b)| crate::simd::levenshtein_myers(a.as_bytes(), b.as_bytes()))
                .collect();
            assert_eq!(gpu, cpu, "Myers GPU results must match CPU Myers");
        }

        /// Mask edge: a 64-char query exercises the `m == 64` all-ones-mask
        /// branch, and a 65-char query must NOT be routed to Myers (falls back
        /// to the DP short kernel, still correct).
        #[test]
        fn test_gpu_myers_64char_boundary() {
            let _gpu_guard = crate::gpu::gpu_test_lock();
            let Some(kernel) = gpu_kernel_or_skip() else { return; };

            let q64: String = (0..64).map(|i| (b'a' + (i % 26) as u8) as char).collect();
            let mut state: u64 = 0xBA5E_BA11;
            let mut texts: Vec<String> = Vec::with_capacity(600);
            for _ in 0..600 {
                let len = 1 + ((state >> 33) as usize % 24);
                let mut s = String::with_capacity(len);
                for _ in 0..len {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    s.push((b'a' + ((state >> 33) as u8 % 26)) as char);
                }
                texts.push(s);
            }
            texts[0] = q64.clone();

            let pairs: Vec<(&str, &str)> =
                texts.iter().map(|t| (q64.as_str(), t.as_str())).collect();
            let indices: Vec<usize> = (0..pairs.len()).collect();

            let gpu = kernel
                .compute_gpu_myers(&q64, &pairs, &indices)
                .expect("Myers 64-char dispatch should succeed");
            let cpu: Vec<u32> = pairs.iter()
                .map(|(a, b)| crate::simd::levenshtein_myers(a.as_bytes(), b.as_bytes()))
                .collect();
            assert_eq!(gpu, cpu, "Myers 64-char boundary results must match CPU Myers");

            // 65-char query: shared_ascii_query must reject it, so compute()
            // takes the DP short kernel and still matches CPU.
            let q65 = format!("x{}", q64);
            let pairs65: Vec<(&str, &str)> =
                texts.iter().map(|t| (q65.as_str(), t.as_str())).collect();
            let gpu65 = kernel.compute(&pairs65).expect("65-char compute should succeed");
            let cpu65: Vec<u32> = pairs65.iter()
                .map(|(a, b)| levenshtein_distance_raw(a, b))
                .collect();
            assert_eq!(gpu65, cpu65, "65-char query must route to DP and match CPU");
        }

        /// Non-ASCII inputs must never take the Myers path (Peq is indexed by
        /// byte value, valid for ASCII only) — the DP short kernel handles
        /// them and must still match CPU.
        #[test]
        fn test_gpu_myers_unicode_falls_back_to_dp() {
            let _gpu_guard = crate::gpu::gpu_test_lock();
            let Some(kernel) = gpu_kernel_or_skip() else { return; };

            let query: &str = "héllo wörld";
            let mut state: u64 = 0x0DD_BA11;
            let mut texts: Vec<String> = Vec::with_capacity(600);
            for _ in 0..600 {
                let len = 1 + ((state >> 33) as usize % 16);
                let mut s = String::with_capacity(len * 2);
                for _ in 0..len {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    s.push(if state % 3 == 0 { 'é' } else { (b'a' + ((state >> 33) as u8 % 26)) as char });
                }
                texts.push(s);
            }

            let pairs: Vec<(&str, &str)> =
                texts.iter().map(|t| (query, t.as_str())).collect();
            let gpu = kernel.compute(&pairs).expect("Unicode compute should succeed");
            let cpu: Vec<u32> = pairs.iter()
                .map(|(a, b)| levenshtein_distance_raw(a, b))
                .collect();
            assert_eq!(gpu, cpu, "Unicode pairs must fall back to DP and match CPU");
        }

        /// `shared_ascii_query` must accept only batches where every pair
        /// shares one non-empty ASCII query <= 64 bytes.
        #[test]
        fn test_shared_ascii_query_detection() {
            let a = "query";
            let b = "other";
            let pairs: Vec<(&str, &str)> =
                vec![(a, "x"), (a, "y"), (a, "z")];
            let indices: Vec<usize> = (0..3).collect();
            assert_eq!(shared_ascii_query(&pairs, &indices), Some(a));

            // Mixed queries: reject.
            let mixed: Vec<(&str, &str)> = vec![(a, "x"), (b, "y")];
            assert_eq!(shared_ascii_query(&mixed, &[0, 1]), None);
            // Non-ASCII query: reject.
            let uni: Vec<(&str, &str)> = vec![("héllo", "x")];
            assert_eq!(shared_ascii_query(&uni, &[0]), None);
            // Non-ASCII text: reject.
            let uni_t: Vec<(&str, &str)> = vec![(a, "héllo")];
            assert_eq!(shared_ascii_query(&uni_t, &[0]), None);
            // > 64-byte query: reject.
            let long_q = "q".repeat(65);
            let long: Vec<(&str, &str)> = vec![(long_q.as_str(), "x")];
            assert_eq!(shared_ascii_query(&long, &[0]), None);
        }

        /// Multi-chunk dispatch: batches > MAX_DISPATCH (65535) pairs split
        /// into multiple dispatches. Each chunk gets its own submit + readback
        /// so the per-chunk params offset is applied before its dispatch (a
        /// single shared submit would run every chunk with the last offset and
        /// produce wrong results — the regression this test pins).
        #[test]
        fn test_gpu_multi_chunk_batch_matches_cpu() {
            let _gpu_guard = crate::gpu::gpu_test_lock();
            let Some(kernel) = gpu_kernel_or_skip() else { return; };

            // 70,000 pairs -> two 65535-pair dispatches plus remainder.
            let mut state: u64 = 0x0C0FFEE;
            let mut pairs: Vec<(String, String)> = Vec::with_capacity(70_000);
            for _ in 0..70_000 {
                let mut a = String::with_capacity(12);
                let mut b = String::with_capacity(12);
                for _ in 0..12 {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    a.push((b'a' + ((state >> 33) as u8 % 26)) as char);
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    b.push((b'a' + ((state >> 33) as u8 % 26)) as char);
                }
                pairs.push((a, b));
            }
            let refs: Vec<(&str, &str)> =
                pairs.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();

            let gpu = kernel.compute(&refs).expect("GPU multi-chunk compute should succeed");
            assert_eq!(gpu.len(), 70_000);
            // Spot-check all four boundaries: last of chunk 1 (idx 65534),
            // first of chunk 2 (idx 65535), and the final element.
            for &i in &[0usize, 65534, 65535, 65536, 69_999] {
                let expected = levenshtein_distance_raw(&refs[i].0, &refs[i].1);
                assert_eq!(gpu[i], expected, "multi-chunk result {} must match CPU", i);
            }
        }

        /// The production dispatch lock must make concurrent GPU calls from
        /// many threads safe and correct. Upstream gfx-rs/wgpu#10085 crashes
        /// only under >=3 concurrent dispatchers on the shared device (Intel
        /// iGPUs); the engine-level lock serializes them. Each worker uses a
        /// GPU-sized batch (600 pairs) so a real dispatch happens every call.
        #[test]
        fn test_concurrent_dispatch_is_serialized_and_correct() {
            let _gpu_guard = crate::gpu::gpu_test_lock();
            let Some(kernel) = gpu_kernel_or_skip() else { return; };

            let workers: Vec<std::thread::JoinHandle<bool>> = (0..8)
                .map(|t| {
                    std::thread::spawn(move || {
                        let mut state = 0x1234_5678 + t as u64;
                        let mut all_ok = true;
                        for _ in 0..20 {
                            // Deterministic 16-char pairs (same LCG as the rest
                            // of the suite) so results are checkable on CPU.
                            let mut pairs: Vec<(String, String)> = Vec::with_capacity(600);
                            for _ in 0..600 {
                                let mut a = String::with_capacity(16);
                                let mut b = String::with_capacity(16);
                                for _ in 0..16 {
                                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                                    a.push((b'a' + ((state >> 33) as u8 % 26)) as char);
                                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                                    b.push((b'a' + ((state >> 33) as u8 % 26)) as char);
                                }
                                pairs.push((a, b));
                            }
                            let refs: Vec<(&str, &str)> =
                                pairs.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
                            match kernel.compute(&refs) {
                                Ok(res) => {
                                    for (i, ((a, b), got)) in refs.iter().zip(res).enumerate() {
                                        let expect = levenshtein_distance_raw(a, b);
                                        if got != expect {
                                            eprintln!("worker {t} iter pair {i}: got {got}, expect {expect}");
                                            all_ok = false;
                                        }
                                    }
                                }
                                Err(e) => {
                                    eprintln!("worker {t} compute failed: {e}");
                                    all_ok = false;
                                }
                            }
                        }
                        all_ok
                    })
                })
                .collect();

            for (t, worker) in workers.into_iter().enumerate() {
                assert!(worker.join().expect("worker panicked"), "worker {t}: concurrent GPU compute produced wrong results");
            }
        }
    }
}
