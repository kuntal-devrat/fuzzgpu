use rayon::prelude::*;

/// Jaro similarity between two strings.
///
/// Returns 1.0 for identical strings (including both empty), 0.0 if no matches.
/// Supports both ASCII fast-path and full Unicode characters.
pub fn jaro(a: &str, b: &str) -> f64 {
    if a.is_ascii() && b.is_ascii() {
        jaro_bytes(a.as_bytes(), b.as_bytes())
    } else {
        let a_chars: Vec<char> = a.chars().collect();
        let b_chars: Vec<char> = b.chars().collect();
        jaro_chars(&a_chars, &b_chars)
    }
}

fn jaro_bytes(a: &[u8], b: &[u8]) -> f64 {
    let (m, n) = (a.len(), b.len());
    if m == 0 && n == 0 { return 1.0; }
    if m == 0 || n == 0 { return 0.0; }
    if a == b { return 1.0; }

    let match_distance = (m.max(n) / 2).saturating_sub(1);

    // Bit-parallel fast path: u64 position masks replace the window scan.
    if m <= 64 && n <= 64 {
        return crate::simd::jaro_bitpar(a, b);
    }

    if m <= 128 && n <= 128 {
        let mut a_matches = [false; 128];
        let mut b_matches = [false; 128];
        jaro_inner_slice(a, b, &mut a_matches[..m], &mut b_matches[..n], match_distance)
    } else {
        let mut a_matches = vec![false; m];
        let mut b_matches = vec![false; n];
        jaro_inner_slice(a, b, &mut a_matches, &mut b_matches, match_distance)
    }
}

fn jaro_chars(a: &[char], b: &[char]) -> f64 {
    let (m, n) = (a.len(), b.len());
    if m == 0 && n == 0 { return 1.0; }
    if m == 0 || n == 0 { return 0.0; }
    if a == b { return 1.0; }

    let match_distance = (m.max(n) / 2).saturating_sub(1);

    if m <= 128 && n <= 128 {
        let mut a_matches = [false; 128];
        let mut b_matches = [false; 128];
        jaro_inner_slice(a, b, &mut a_matches[..m], &mut b_matches[..n], match_distance)
    } else {
        let mut a_matches = vec![false; m];
        let mut b_matches = vec![false; n];
        jaro_inner_slice(a, b, &mut a_matches, &mut b_matches, match_distance)
    }
}

#[inline]
fn jaro_inner_slice<T: PartialEq>(a: &[T], b: &[T], a_matches: &mut [bool], b_matches: &mut [bool], match_distance: usize) -> f64 {
    let (m, n) = (a.len(), b.len());
    let mut matches = 0u32;

    for i in 0..m {
        let lo = i.saturating_sub(match_distance);
        let hi = (i + match_distance + 1).min(n);
        let ai = &a[i];
        for j in lo..hi {
            if b_matches[j] || ai != &b[j] { continue; }
            a_matches[i] = true;
            b_matches[j] = true;
            matches += 1;
            break;
        }
    }

    if matches == 0 { return 0.0; }

    let mut transpositions = 0u32;
    let mut k = 0;
    for i in 0..m {
        if !a_matches[i] { continue; }
        while !b_matches[k] { k += 1; }
        if a[i] != b[k] { transpositions += 1; }
        k += 1;
    }

    (matches as f64 / m as f64
        + matches as f64 / n as f64
        + (matches as f64 - (transpositions / 2) as f64) / matches as f64) / 3.0
}

/// Jaro-Winkler similarity with prefix bonus.
/// `p` is the prefix weight (0.0–0.25, default 0.1).
/// Winkler prefix boost is applied when base Jaro similarity is >= 0.7 (Winkler 1990 standard).
pub fn jaro_winkler(a: &str, b: &str, p: f64) -> f64 {
    if a == b { return 1.0; }

    let jaro_score = jaro(a, b);
    if jaro_score < 0.7 {
        return jaro_score;
    }

    let p = p.clamp(0.0, 0.25);

    let prefix_len = if a.is_ascii() && b.is_ascii() {
        let a_bytes = a.as_bytes();
        let b_bytes = b.as_bytes();
        let max_prefix = a_bytes.len().min(b_bytes.len()).min(4);
        let mut len = 0usize;
        for i in 0..max_prefix {
            if a_bytes[i] == b_bytes[i] {
                len += 1;
            } else {
                break;
            }
        }
        len
    } else {
        let a_chars: Vec<char> = a.chars().take(4).collect();
        let b_chars: Vec<char> = b.chars().take(4).collect();
        let max_prefix = a_chars.len().min(b_chars.len());
        let mut len = 0usize;
        for i in 0..max_prefix {
            if a_chars[i] == b_chars[i] {
                len += 1;
            } else {
                break;
            }
        }
        len
    };

    let boost = (prefix_len as f64) * p * (1.0 - jaro_score);
    (jaro_score + boost).min(1.0)
}

/// Winkler prefix bonus: length of the shared ASCII prefix (≤ 4), or 0 when
/// the base Jaro is below the 0.7 threshold or the inputs are non-ASCII.
#[inline]
fn winkler_prefix_len(query: &str, cand: &str, jaro_score: f64) -> usize {
    if jaro_score < 0.7 {
        return 0;
    }
    if query.is_ascii() && cand.is_ascii() {
        let max = query.len().min(cand.len()).min(4);
        let qb = query.as_bytes();
        let cb = cand.as_bytes();
        let mut len = 0;
        while len < max && qb[len] == cb[len] {
            len += 1;
        }
        len
    } else {
        let qc: Vec<char> = query.chars().take(4).collect();
        let cc: Vec<char> = cand.chars().take(4).collect();
        let max = qc.len().min(cc.len());
        let mut len = 0;
        while len < max && qc[len] == cc[len] {
            len += 1;
        }
        len
    }
}

/// Batch Jaro-Winkler on CPU using Rayon, with a 4-way AVX2 fast path.
///
/// When the shared query and every candidate are non-empty ASCII ≤ 64 bytes
/// (the fuzzy-matching shape), the matching-window pass runs four candidates
/// per 256-bit vector ([`crate::simd::jaro_4way`]); the Winkler prefix bonus
/// is applied per candidate afterwards. Anything else falls back to per-pair
/// Rayon.
pub fn jaro_winkler_batch(query: &str, candidates: &[&str], p: f64) -> Vec<f64> {
    let p = p.clamp(0.0, 0.25);
    let gate = !query.is_empty()
        && query.is_ascii()
        && query.len() <= 64
        && candidates.iter().all(|c| !c.is_empty() && c.is_ascii() && c.len() <= 64);
    if gate {
        let qb = query.as_bytes();
        let w = crate::simd::jaro_simd_width();
        let mut out = vec![0.0; candidates.len()];
        const CHUNK: usize = 4096;
        out.par_chunks_mut(CHUNK).enumerate().for_each(|(ci, chunk_out)| {
            let start = ci * CHUNK;
            let mut k = 0;
            // Stack-allocated buffer avoids a heap Vec per SIMD group.
            let mut group_buf: [&[u8]; 8] = [b""; 8];
            while k + w <= chunk_out.len() {
                for t in 0..w {
                    group_buf[t] = candidates[start + k + t].as_bytes();
                }
                let res = crate::simd::jaro_width(qb, &group_buf[..w]);
                for (lane, score) in res.into_iter().enumerate() {
                    let cand = candidates[start + k + lane];
                    chunk_out[k + lane] = jaro_winkler_apply_boost(query, cand, score, p);
                }
                k += w;
            }
            while k < chunk_out.len() {
                chunk_out[k] = jaro_winkler(query, candidates[start + k], p);
                k += 1;
            }
        });
        return out;
    }
    candidates.par_iter().map(|c| jaro_winkler(query, c, p)).collect()
}

/// Jaro + Winkler prefix boost, sharing the prefix computation with
/// `jaro_winkler`'s own boost logic.
#[inline]
fn jaro_winkler_apply_boost(query: &str, cand: &str, jaro_score: f64, p: f64) -> f64 {
    if jaro_score >= 0.7 {
        let len = winkler_prefix_len(query, cand, jaro_score);
        let boost = len as f64 * p * (1.0 - jaro_score);
        (jaro_score + boost).min(1.0)
    } else {
        jaro_score
    }
}

/// Cross-product matrix for Jaro-Winkler on CPU using Rayon.
pub fn jaro_winkler_cdist_cpu(list_a: &[&str], list_b: &[&str], p: f64) -> Vec<Vec<f64>> {
    if list_a.is_empty() || list_b.is_empty() {
        return vec![];
    }
    list_a.par_iter().map(|a| {
        list_b.iter().map(|b| jaro_winkler(a, b, p)).collect()
    }).collect()
}

/// Optimized Jaro variant using the bit-parallel fast path for ≤ 64-byte
/// ASCII inputs and the stack-allocated window scan otherwise.
pub fn jaro_optimized(a: &[u8], b: &[u8]) -> f64 {
    jaro_bytes(a, b)
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

    const SHADER_SRC: &str = include_str!("shaders/jaro.wgsl");
    const MATRIX_SHADER_SRC: &str = include_str!("shaders/jaro_matrix.wgsl");

    const GPU_MAX_STRING_LEN: usize = 128;
    const MAX_DISPATCH: u32 = 65535;
    const MAX_DESIRED_CHUNK_PAIRS: usize = 500_000;

    /// Scales the discrete-GPU auto threshold (64) for the Jaro bitmap
    /// kernel: measured on Iris Xe, the window-scan kernel needs a much
    /// larger batch than the Myers bit-vector to amortize the sync round-trip
    /// and win (64 × 16 = 1024 pairs on dGPUs). On integrated GPUs the SIMD
    /// CPU path wins at every scale, so auto-routing never dispatches there
    /// (see `GpuEngine::metric_gpu_threshold`).
    const JARO_DISCRETE_FACTOR: usize = 16;

    #[repr(C)]
    #[derive(Copy, Clone, Pod, Zeroable)]
    struct JaroParams {
        batch_size: u32,
        max_len: u32,
        offset: u32,
    }

    #[repr(C)]
    #[derive(Copy, Clone, Pod, Zeroable)]
    struct JaroMatrixParams {
        rows: u32,
        cols: u32,
    }

    /// Sentinel written by the shaders when a pair must be recomputed on CPU
    /// (strings > 128 chars, empty inputs).
    const GPU_RECOMPUTE: u32 = u32::MAX;

    /// Assemble the f64 Jaro score from the GPU kernel's integer parts using
    /// rapidfuzz's exact formula (integer floor division on transpositions,
    /// f64 arithmetic throughout) — bit-identical to the CPU reference.
    #[inline]
    fn jaro_score_from_parts(matches: u32, transpositions: u32, a_len: u32, b_len: u32) -> f64 {
        if matches == 0 {
            return 0.0;
        }
        let t = (transpositions / 2) as f64;
        let mf = matches as f64;
        (mf / a_len as f64 + mf / b_len as f64 + (mf - t) / mf) / 3.0
    }

    /// Assemble Jaro-Winkler from GPU parts: f64 Jaro, then the Winkler prefix
    /// boost (gate at 0.7, clamp to 1.0) — identical to the CPU path.
    #[inline]
    fn jaro_winkler_from_parts(
        matches: u32,
        transpositions: u32,
        prefix_len: u32,
        a_len: u32,
        b_len: u32,
        p: f64,
    ) -> f64 {
        let jaro = jaro_score_from_parts(matches, transpositions, a_len, b_len);
        if jaro >= 0.7 {
            let boost = prefix_len as f64 * p * (1.0 - jaro);
            (jaro + boost).min(1.0)
        } else {
            jaro
        }
    }

    /// CPU fallback for GPU-eligible pairs. When the pairs share one non-empty
    /// ASCII query ≤ 64 bytes (the fuzzy-matching shape), this routes through
    /// the SIMD batch kernel — the same fast path a CPU-only caller gets from
    /// [`jaro_winkler_batch`] — so enabling the GPU on an iGPU (where Jaro is
    /// auto-routed to CPU) never regresses below the pure-CPU path. Any other
    /// shape falls back to per-pair Rayon.
    fn cpu_compute_jaro(pairs: &[(&str, &str)], indices: &[usize], p: f64) -> Vec<f64> {
        let Some(&first) = indices.first() else {
            return vec![];
        };
        let query = pairs[first].0;
        let simd_shape = !query.is_empty()
            && query.is_ascii()
            && query.len() <= 64
            && indices.iter().all(|&i| {
                let b = pairs[i].1;
                !b.is_empty() && b.is_ascii() && b.len() <= 64 && pairs[i].0 == query
            });
        if simd_shape {
            let cands: Vec<&str> = indices.iter().map(|&i| pairs[i].1).collect();
            return crate::jaro_winkler_batch(query, &cands, p);
        }
        indices.par_iter().map(|&i| crate::jaro_winkler(pairs[i].0, pairs[i].1, p)).collect()
    }

    pub struct GpuJaroKernel {
        engine: std::sync::Arc<GpuEngine>,
        pipeline: wgpu::ComputePipeline,
        matrix_pipeline: wgpu::ComputePipeline,
        bind_group_layout: wgpu::BindGroupLayout,
        // Persistent buffer arena (see gpu::BufferPool) — removes the per-call
        // `create_buffer` cost that dominated small-batch dispatches.
        pool: std::sync::Mutex<BufferPool>,
    }

    static GLOBAL_GPU_JARO_KERNEL: OnceLock<GpuJaroKernel> = OnceLock::new();

    impl GpuJaroKernel {
        pub fn get() -> Result<&'static Self> {
            if let Some(k) = GLOBAL_GPU_JARO_KERNEL.get() { return Ok(k); }
            let engine = GpuEngine::get()?;
            let kernel = Self::new_inner(engine)?;
            let _ = GLOBAL_GPU_JARO_KERNEL.set(kernel);
            GLOBAL_GPU_JARO_KERNEL.get().ok_or_else(|| FuzzGpuError::NoDevice(
                "Jaro kernel unexpectedly absent after init".into()
            ))
        }

        fn new_inner(engine: std::sync::Arc<GpuEngine>) -> Result<Self> {
            // Both kernels register through the public kernel-registration API
            // (`GpuEngine::build_compute_pipeline`), so shader/pipeline
            // validation failures surface as `FuzzGpuError::ShaderError` instead
            // of panicking. Test builds can fault-inject invalid WGSL via
            // `effective_shader_source`.
            let bind_group_layout = engine.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("jaro bgl"),
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
                "jaro pipeline",
                &crate::gpu::effective_shader_source(SHADER_SRC),
                &layout,
            )?;
            let matrix_pipeline = engine.build_compute_pipeline(
                "jaro matrix pipeline",
                &crate::gpu::effective_shader_source(MATRIX_SHADER_SRC),
                &layout,
            )?;

            Ok(Self { engine, pipeline, matrix_pipeline, bind_group_layout, pool: std::sync::Mutex::new(BufferPool::new()) })
        }

        /// Smart streaming GPU/CPU dispatch for batch Jaro-Winkler with dynamic chunk sizing.
        ///
        /// `p` (Winkler prefix weight) must be in `0.0..=0.25`; anything else —
        /// including NaN — is rejected with `FuzzGpuError::InvalidInput` before
        /// any dispatch. (The CPU `jaro_winkler` clamps instead; this is the
        /// strict boundary of the Result-returning GPU API.)
        pub fn compute_batch(&self, pairs: &[(&str, &str)], p: f64) -> Result<Vec<f64>> {
            // Serialize GPU dispatch across threads (gfx-rs/wgpu#10085).
            let _dispatch = self.engine.dispatch_lock();
            if !(0.0..=0.25).contains(&p) {
                return Err(FuzzGpuError::InvalidInput(format!(
                    "Jaro-Winkler prefix weight p must be within 0.0..=0.25, got {p}"
                )));
            }
            let n = pairs.len();
            if n == 0 { return Ok(vec![]); }

            let mut results = vec![0.0f64; n];
            let mut gpu_indices: Vec<usize> = Vec::with_capacity(n);
            let mut cpu_indices: Vec<usize> = Vec::new();

            for (i, (a, b)) in pairs.iter().enumerate() {
                if a.is_empty() && b.is_empty() {
                    results[i] = 1.0;
                } else if a.is_empty() || b.is_empty() {
                    results[i] = 0.0;
                } else if *a == *b {
                    results[i] = 1.0;
                } else {
                    let a_len = a.chars().count();
                    let b_len = b.chars().count();
                    if a_len > GPU_MAX_STRING_LEN || b_len > GPU_MAX_STRING_LEN {
                        cpu_indices.push(i);
                    } else {
                        gpu_indices.push(i);
                    }
                }
            }

            if !cpu_indices.is_empty() {
                let cpu_results = cpu_compute_jaro(pairs, &cpu_indices, p);
                for (idx, &orig_i) in cpu_indices.iter().enumerate() {
                    results[orig_i] = cpu_results[idx];
                }
            }

            if gpu_indices.len() < self.engine.metric_gpu_threshold(JARO_DISCRETE_FACTOR) {
                // Below the (auto or user-set) threshold: CPU is cheaper — and
                // on iGPUs (auto threshold = never) this branch always wins.
                crate::gpu::GpuEngine::record_routing(0, n);
                let cpu_results = cpu_compute_jaro(pairs, &gpu_indices, p);
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
                let gpu_results = self.compute_gpu_subset(pairs, stream_chunk, p)?;
                for (idx, &orig_i) in stream_chunk.iter().enumerate() {
                    if gpu_results[idx] < 0.0 {
                        results[orig_i] = crate::jaro_winkler(pairs[orig_i].0, pairs[orig_i].1, p);
                    } else {
                        results[orig_i] = gpu_results[idx];
                    }
                }
            }

            Ok(results)
        }

        fn compute_gpu_subset(&self, pairs: &[(&str, &str)], indices: &[usize], p: f64) -> Result<Vec<f64>> {
            let batch_size = indices.len() as u32;

            // Transposed, pair-major packing: chars_x[i * B + t] is the i-th
            // char of pair t (stride B = batch size). For a fixed position all
            // threads in a workgroup read consecutive addresses (coalesced) —
            // the layout lesson from the Levenshtein short kernel; the old
            // per-pair-contiguous layout made every (i, j) window-scan load a
            // scattered cache-line fetch (~200 cycles, kernel-bound). Positions
            // past a pair's length are never read (loops bounded by lens).
            let mut lens_a: Vec<u32> = Vec::with_capacity(indices.len());
            let mut lens_b: Vec<u32> = Vec::with_capacity(indices.len());
            let mut max_len = 0u32;
            for &i in indices {
                let (a, b) = pairs[i];
                let a_count = a.chars().count() as u32;
                let b_count = b.chars().count() as u32;
                lens_a.push(a_count);
                lens_b.push(b_count);
                max_len = max_len.max(a_count.max(b_count));
            }
            let rows = (max_len as usize).max(1);
            let mut chars_a = vec![0u32; rows * indices.len()];
            let mut chars_b = vec![0u32; rows * indices.len()];
            for (t, &i) in indices.iter().enumerate() {
                for (k, c) in pairs[i].0.chars().enumerate() {
                    chars_a[k * indices.len() + t] = c as u32;
                }
                for (k, c) in pairs[i].1.chars().enumerate() {
                    chars_b[k * indices.len() + t] = c as u32;
                }
            }

            let chars_a_bytes = (chars_a.len() * 4) as u64;
            let chars_b_bytes = (chars_b.len() * 4) as u64;
            // 4×u32 per pair: [matches, transpositions, prefix_len, 0].
            let results_size = (batch_size as u64) * 16;

            if chars_a_bytes > self.engine.max_buffer_size_effective()
                || chars_b_bytes > self.engine.max_buffer_size_effective()
                || results_size > self.engine.max_buffer_size_effective()
            {
                return Err(FuzzGpuError::BufferError(
                    "Buffer size exceeds device max_buffer_size".into(),
                ));
            }

            // Persistent buffers (see gpu::BufferPool) — same arena pattern as
            // the Levenshtein kernel: ensure capacity, upload once, reuse.
            let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
            let lens_bytes = ((lens_a.len() * 4) as u64).max(results_size);
            pool.ensure(&self.engine.device, SLOT_OFFSETS_A, lens_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, "joa");
            pool.ensure(&self.engine.device, SLOT_OFFSETS_B, lens_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, "job");
            pool.ensure(&self.engine.device, SLOT_CHARS_A, chars_a_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, "jca");
            pool.ensure(&self.engine.device, SLOT_CHARS_B, chars_b_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, "jcb");
            pool.ensure(&self.engine.device, SLOT_RESULTS, results_size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, "jres");
            pool.ensure(&self.engine.device, SLOT_STAGING, results_size, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, "jstg");
            pool.ensure(&self.engine.device, SLOT_PARAMS, std::mem::size_of::<JaroParams>() as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, "jp");

            pool.write(&self.engine.queue, SLOT_OFFSETS_A, bytemuck::cast_slice(&lens_a));
            pool.write(&self.engine.queue, SLOT_CHARS_A, bytemuck::cast_slice(&chars_a));
            pool.write(&self.engine.queue, SLOT_OFFSETS_B, bytemuck::cast_slice(&lens_b));
            pool.write(&self.engine.queue, SLOT_CHARS_B, bytemuck::cast_slice(&chars_b));

            let buf_offsets_a = pool.get(SLOT_OFFSETS_A);
            let buf_chars_a = pool.get(SLOT_CHARS_A);
            let buf_offsets_b = pool.get(SLOT_OFFSETS_B);
            let buf_chars_b = pool.get(SLOT_CHARS_B);
            let buf_results = pool.get(SLOT_RESULTS);
            let buf_staging = pool.get(SLOT_STAGING);
            let buf_params = pool.get(SLOT_PARAMS);

            // Per-chunk submit + readback (see the Levenshtein kernel: the
            // params buffer is written through the queue, so one shared submit
            // would give every dispatch the LAST chunk's offset).
            let mut gpu_results: Vec<f64> = Vec::with_capacity(batch_size as usize);
            let mut remaining = batch_size;
            let mut offset = 0u32;

            while remaining > 0 {
                let chunk = remaining.min(MAX_DISPATCH);
                let params = JaroParams { batch_size, max_len, offset };
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

                let mut encoder = self.engine.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("jaro encoder") });
                let workgroups = (chunk + 63) / 64;
                { let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None }); pass.set_pipeline(&self.pipeline); pass.set_bind_group(0, &bg, &[]); pass.dispatch_workgroups(workgroups, 1, 1); }
                let chunk_bytes = (chunk as u64) * 16;
                encoder.copy_buffer_to_buffer(&buf_results, 0, &buf_staging, 0, chunk_bytes);
                let bytes = self.engine.readback(encoder, &pool, chunk_bytes)?;
                let raw: &[u32] = bytemuck::cast_slice(&bytes);
                // Decode 4×u32 per pair and assemble the f64 score host-side
                // (bit-exact with the CPU reference). GPU_RECOMPUTE pairs are
                // returned as -1.0; compute_batch recomputes them on CPU.
                for (t, part) in raw.chunks_exact(4).enumerate() {
                    if part[0] == GPU_RECOMPUTE {
                        gpu_results.push(-1.0);
                    } else {
                        gpu_results.push(jaro_winkler_from_parts(
                            part[0], part[1], part[2], lens_a[t], lens_b[t], p,
                        ));
                    }
                }

                remaining -= chunk;
                offset += chunk;
            }
            Ok(gpu_results)
        }

        /// Dedicated 2D Jaro-Winkler Matrix GPU compute: O(N + M) upload instead of O(N * M).
        /// Pre-filters oversized strings and validates buffer limits before dispatch.
        ///
        /// `p` (Winkler prefix weight) must be in `0.0..=0.25`; anything else —
        /// including NaN — is rejected with `FuzzGpuError::InvalidInput` before
        /// any dispatch.
        pub fn compute_matrix(&self, list_a: &[&str], list_b: &[&str], p: f64) -> Result<Vec<Vec<f64>>> {
            // Serialize GPU dispatch across threads (gfx-rs/wgpu#10085).
            let _dispatch = self.engine.dispatch_lock();
            if !(0.0..=0.25).contains(&p) {
                return Err(FuzzGpuError::InvalidInput(format!(
                    "Jaro-Winkler prefix weight p must be within 0.0..=0.25, got {p}"
                )));
            }
            let rows = list_a.len();
            let cols = list_b.len();
            if rows == 0 || cols == 0 { return Ok(vec![]); }

            let total_pairs = rows * cols;
            if total_pairs < self.engine.metric_gpu_threshold(JARO_DISCRETE_FACTOR) {
                crate::gpu::GpuEngine::record_routing(0, total_pairs);
                return Ok(jaro_winkler_cdist_cpu(list_a, list_b, p));
            }
            crate::gpu::GpuEngine::record_routing(total_pairs, 0);

            // CRITICAL FIX: Pre-filter oversized strings BEFORE GPU buffer allocation or dispatch
            let has_oversized = list_a.iter().any(|s| s.chars().count() > GPU_MAX_STRING_LEN)
                || list_b.iter().any(|s| s.chars().count() > GPU_MAX_STRING_LEN);

            let matrix_size = (total_pairs as u64) * 16;

            if has_oversized || matrix_size > self.engine.max_buffer_size_effective() {
                return Ok(jaro_winkler_cdist_cpu(list_a, list_b, p));
            }

            // Transposed packing: chars_a[i * rows + row] (stride = rows),
            // chars_b[j * cols + col] (stride = cols) — coalesced reads for a
            // fixed position across a workgroup row / column.
            let mut lens_a: Vec<u32> = Vec::with_capacity(rows);
            let mut lens_b: Vec<u32> = Vec::with_capacity(cols);
            let mut max_len_a = 0u32;
            let mut max_len_b = 0u32;
            for a in list_a {
                let la = a.chars().count() as u32;
                lens_a.push(la);
                max_len_a = max_len_a.max(la);
            }
            for b in list_b {
                let lb = b.chars().count() as u32;
                lens_b.push(lb);
                max_len_b = max_len_b.max(lb);
            }
            let stride_a = (max_len_a as usize).max(1);
            let stride_b = (max_len_b as usize).max(1);
            let mut chars_a = vec![0u32; stride_a * rows];
            let mut chars_b = vec![0u32; stride_b * cols];
            for (row, a) in list_a.iter().enumerate() {
                for (k, c) in a.chars().enumerate() {
                    chars_a[k * rows + row] = c as u32;
                }
            }
            for (col, b) in list_b.iter().enumerate() {
                for (k, c) in b.chars().enumerate() {
                    chars_b[k * cols + col] = c as u32;
                }
            }

            // Persistent buffers (see gpu::BufferPool) — same arena as the batch
            // path; matrix/staging slots grow to the larger matrix size.
            let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
            let lens_bytes = ((lens_a.len() * 4) as u64).max(matrix_size);
            pool.ensure(&self.engine.device, SLOT_OFFSETS_A, lens_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, "jmoa");
            pool.ensure(&self.engine.device, SLOT_OFFSETS_B, lens_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, "jmob");
            pool.ensure(&self.engine.device, SLOT_CHARS_A, (chars_a.len() * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, "jmca");
            pool.ensure(&self.engine.device, SLOT_CHARS_B, (chars_b.len() * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, "jmcb");
            pool.ensure(&self.engine.device, SLOT_RESULTS, matrix_size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, "jmres");
            pool.ensure(&self.engine.device, SLOT_STAGING, matrix_size, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, "jmstg");
            pool.ensure(&self.engine.device, SLOT_PARAMS, std::mem::size_of::<JaroMatrixParams>() as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, "jmp");

            pool.write(&self.engine.queue, SLOT_OFFSETS_A, bytemuck::cast_slice(&lens_a));
            pool.write(&self.engine.queue, SLOT_CHARS_A, bytemuck::cast_slice(&chars_a));
            pool.write(&self.engine.queue, SLOT_OFFSETS_B, bytemuck::cast_slice(&lens_b));
            pool.write(&self.engine.queue, SLOT_CHARS_B, bytemuck::cast_slice(&chars_b));

            let params = JaroMatrixParams { rows: rows as u32, cols: cols as u32 };
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
                &wgpu::CommandEncoderDescriptor { label: Some("jaro matrix encoder") },
            );

            let workgroups_x = (cols as u32 + 15) / 16;
            let workgroups_y = (rows as u32 + 15) / 16;

            {
                let mut pass = encoder.begin_compute_pass(
                    &wgpu::ComputePassDescriptor { label: None, timestamp_writes: None },
                );
                pass.set_pipeline(&self.matrix_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
            }

            encoder.copy_buffer_to_buffer(&buf_matrix, 0, &buf_staging, 0, matrix_size);
            self.engine.submit(encoder);

            let slice = buf_staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            self.engine.map_readback(&slice, move |r| { let _ = tx.send(r); });
            self.engine.poll();

            rx.recv_timeout(GpuEngine::readback_timeout())
                .map_err(|_| FuzzGpuError::Timeout("GPU Jaro matrix readback timed out after 10s".into()))?
                .map_err(|e| FuzzGpuError::BufferError(format!("GPU buffer map failed: {}", e)))?;

            let data = slice
                .get_mapped_range()
                .map_err(|e| FuzzGpuError::BufferError(format!("GPU buffer map range failed: {e}")))?;
            let raw: &[u32] = bytemuck::cast_slice(&data);

            // Decode 4×u32 per cell and assemble the f64 score host-side
            // (bit-exact with the CPU reference). GPU_RECOMPUTE cells — empty
            // inputs and strings in (128, 256] chars that overflow the
            // shader's 128-bit match bitmap — are recomputed on CPU here.
            let mut matrix: Vec<Vec<f64>> = Vec::with_capacity(rows);
            for i in 0..rows {
                let mut row_vec: Vec<f64> = Vec::with_capacity(cols);
                for j in 0..cols {
                    let part = &raw[(i * cols + j) * 4..(i * cols + j) * 4 + 4];
                    if part[0] == GPU_RECOMPUTE {
                        row_vec.push(crate::jaro_winkler(list_a[i], list_b[j], p));
                    } else {
                        row_vec.push(jaro_winkler_from_parts(
                            part[0], part[1], part[2], lens_a[i], lens_b[j], p,
                        ));
                    }
                }
                matrix.push(row_vec);
            }

            drop(data);
            buf_staging.unmap();
            Ok(matrix)
        }

        /// Create a batched dispatch: enqueue several pair-lists, then
        /// [`GpuJaroBatch::execute`] once. All GPU-eligible pairs across every
        /// enqueued op share one command encoder and one readback, amortizing
        /// the per-call sync round-trip. `p` is validated here, matching
        /// [`Self::compute_batch`].
        pub fn batch(&self, p: f64) -> Result<GpuJaroBatch<'_>> {
            if !(0.0..=0.25).contains(&p) {
                return Err(FuzzGpuError::InvalidInput(format!(
                    "Jaro-Winkler prefix weight p must be within 0.0..=0.25, got {p}"
                )));
            }
            Ok(GpuJaroBatch { kernel: self, p, ops: Vec::new() })
        }
    }

    /// A queued set of Jaro-Winkler batch operations executed with a single GPU
    /// dispatch + readback. Each enqueued op returns its own `Vec<f64>` of
    /// scores, with the same semantics as [`GpuJaroKernel::compute_batch`]
    /// (empty/identical short-circuits, >256-char pairs routed to CPU, negative
    /// sentinel recompute) applied per op.
    pub struct GpuJaroBatch<'k> {
        kernel: &'k GpuJaroKernel,
        p: f64,
        ops: Vec<Vec<(&'k str, &'k str)>>,
    }

    impl<'k> GpuJaroBatch<'k> {
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
        pub fn execute(self) -> Result<Vec<Vec<f64>>> {
            // Serialize GPU dispatch across threads (gfx-rs/wgpu#10085).
            let _dispatch = self.kernel.engine.dispatch_lock();
            let n_ops = self.ops.len();
            if n_ops == 0 {
                return Ok(vec![]);
            }

            // Classify + pack every pair across ops (mirrors compute_batch).
            let mut out: Vec<Vec<f64>> = Vec::with_capacity(n_ops);
            let mut gpu_ranges: Vec<(u32, u32)> = Vec::with_capacity(n_ops);
            let mut op_gpu_to_pair: Vec<Vec<usize>> = Vec::with_capacity(n_ops);
            let mut cpu_pairs: Vec<(usize, usize)> = Vec::new();
            let mut gpu_pair_list: Vec<(usize, usize)> = Vec::new(); // (op, j) in dispatch order
            let mut lens_a: Vec<u32> = Vec::new();
            let mut lens_b: Vec<u32> = Vec::new();
            let mut gpu_global: u32 = 0;
            let mut max_len: u32 = 0;

            for (op_i, pairs) in self.ops.iter().enumerate() {
                let mut op_results = vec![0.0f64; pairs.len()];
                let mut op_gpu: Vec<usize> = Vec::new();
                for (j, (a, b)) in pairs.iter().enumerate() {
                    if a.is_empty() && b.is_empty() {
                        op_results[j] = 1.0;
                    } else if a.is_empty() || b.is_empty() {
                        op_results[j] = 0.0;
                    } else if *a == *b {
                        op_results[j] = 1.0;
                    } else {
                        let a_len = a.chars().count();
                        let b_len = b.chars().count();
                        if a_len > GPU_MAX_STRING_LEN || b_len > GPU_MAX_STRING_LEN {
                            cpu_pairs.push((op_i, j));
                        } else {
                            gpu_pair_list.push((op_i, j));
                            lens_a.push(a_len as u32);
                            lens_b.push(b_len as u32);
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

            let total_gpu = gpu_global as usize;

            // Oversized pairs are always computed on CPU (Rayon).
            if !cpu_pairs.is_empty() {
                let cpu_res: Vec<f64> = cpu_pairs
                    .par_iter()
                    .map(|&(op, j)| crate::jaro_winkler(self.ops[op][j].0, self.ops[op][j].1, self.p))
                    .collect();
                for (k, &(op, j)) in cpu_pairs.iter().enumerate() {
                    out[op][j] = cpu_res[k];
                }
            }

            // Below the (auto or user-set) GPU threshold the whole batch is
            // cheaper on CPU.
            if total_gpu < self.kernel.engine.metric_gpu_threshold(JARO_DISCRETE_FACTOR) {
                crate::gpu::GpuEngine::record_routing(0, total_gpu);
                let mut gpu_op_pair: Vec<(usize, usize)> = Vec::with_capacity(total_gpu);
                for (op, &(_, count)) in gpu_ranges.iter().enumerate() {
                    for k in 0..count as usize {
                        gpu_op_pair.push((op, op_gpu_to_pair[op][k]));
                    }
                }
                let cpu_res: Vec<f64> = gpu_op_pair
                    .par_iter()
                    .map(|&(op, j)| crate::jaro_winkler(self.ops[op][j].0, self.ops[op][j].1, self.p))
                    .collect();
                for (idx, &(op, j)) in gpu_op_pair.iter().enumerate() {
                    out[op][j] = cpu_res[idx];
                }
                return Ok(out);
            }
            crate::gpu::GpuEngine::record_routing(total_gpu, 0);

            // Transposed pair-major packing over the flat GPU pair list
            // (stride = total_gpu) — coalesced reads in the shader.
            let rows = (max_len as usize).max(1);
            let mut chars_a = vec![0u32; rows * total_gpu];
            let mut chars_b = vec![0u32; rows * total_gpu];
            for (t, &(op, j)) in gpu_pair_list.iter().enumerate() {
                let (a, b) = self.ops[op][j];
                for (k, c) in a.chars().enumerate() {
                    chars_a[k * total_gpu + t] = c as u32;
                }
                for (k, c) in b.chars().enumerate() {
                    chars_b[k * total_gpu + t] = c as u32;
                }
            }

            let chars_a_bytes = (chars_a.len() * 4) as u64;
            let chars_b_bytes = (chars_b.len() * 4) as u64;
            // 4×u32 per pair: [matches, transpositions, prefix_len, 0].
            let results_size = (total_gpu as u64) * 16;
            let limit = self.kernel.engine.max_buffer_size_effective();
            if chars_a_bytes > limit || chars_b_bytes > limit || results_size > limit {
                return Err(FuzzGpuError::BufferError(
                    "Batch buffer size exceeds device max_buffer_size".into(),
                ));
            }

            // Single submit: all chunks recorded into one encoder, read back once.
            let mut pool = self.kernel.pool.lock().unwrap_or_else(|e| e.into_inner());
            let lens_bytes = ((lens_a.len() * 4) as u64).max(results_size);
            pool.ensure(&self.kernel.engine.device, SLOT_OFFSETS_A, lens_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, "bjoa");
            pool.ensure(&self.kernel.engine.device, SLOT_OFFSETS_B, lens_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, "bjob");
            pool.ensure(&self.kernel.engine.device, SLOT_CHARS_A, chars_a_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, "bjca");
            pool.ensure(&self.kernel.engine.device, SLOT_CHARS_B, chars_b_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, "bjcb");
            pool.ensure(&self.kernel.engine.device, SLOT_RESULTS, results_size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, "bjres");
            pool.ensure(&self.kernel.engine.device, SLOT_STAGING, results_size, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, "bjstg");
            pool.ensure(&self.kernel.engine.device, SLOT_PARAMS, std::mem::size_of::<JaroParams>() as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, "bjp");

            pool.write(&self.kernel.engine.queue, SLOT_OFFSETS_A, bytemuck::cast_slice(&lens_a));
            pool.write(&self.kernel.engine.queue, SLOT_CHARS_A, bytemuck::cast_slice(&chars_a));
            pool.write(&self.kernel.engine.queue, SLOT_OFFSETS_B, bytemuck::cast_slice(&lens_b));
            pool.write(&self.kernel.engine.queue, SLOT_CHARS_B, bytemuck::cast_slice(&chars_b));

            let buf_offsets_a = pool.get(SLOT_OFFSETS_A);
            let buf_chars_a = pool.get(SLOT_CHARS_A);
            let buf_offsets_b = pool.get(SLOT_OFFSETS_B);
            let buf_chars_b = pool.get(SLOT_CHARS_B);
            let buf_results = pool.get(SLOT_RESULTS);
            let buf_params = pool.get(SLOT_PARAMS);

            // Per-chunk submit + readback (see the Levenshtein kernel: one
            // shared submit would give every dispatch the LAST chunk's offset).
            let mut raw: Vec<u32> = Vec::with_capacity(total_gpu);
            let mut remaining = total_gpu as u32;
            let mut offset = 0u32;
            while remaining > 0 {
                let chunk = remaining.min(MAX_DISPATCH);
                let params = JaroParams { batch_size: total_gpu as u32, max_len, offset };
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
                    &wgpu::CommandEncoderDescriptor { label: Some("jaro batch encoder") },
                );
                let workgroups = (chunk + 63) / 64;
                {
                    let mut pass = encoder.begin_compute_pass(
                        &wgpu::ComputePassDescriptor { label: None, timestamp_writes: None },
                    );
                    pass.set_pipeline(&self.kernel.pipeline);
                    pass.set_bind_group(0, &bg, &[]);
                    pass.dispatch_workgroups(workgroups, 1, 1);
                }
                let chunk_bytes = (chunk as u64) * 16;
                encoder.copy_buffer_to_buffer(&buf_results, 0, pool.get(SLOT_STAGING), 0, chunk_bytes);
                let bytes = self.kernel.engine.readback(encoder, &pool, chunk_bytes)?;
                raw.extend_from_slice(bytemuck::cast_slice(&bytes));

                remaining -= chunk;
                offset += chunk;
            }

            // Split the flat result range back into per-op vectors, decoding
            // the 4×u32 parts and assembling f64 scores host-side (bit-exact
            // with the CPU reference). GPU_RECOMPUTE pairs recompute on CPU.
            for (op, &(start, count)) in gpu_ranges.iter().enumerate() {
                for k in 0..count as usize {
                    let j = op_gpu_to_pair[op][k];
                    let part = &raw[(start as usize + k) * 4..(start as usize + k) * 4 + 4];
                    if part[0] == GPU_RECOMPUTE {
                        out[op][j] = crate::jaro_winkler(self.ops[op][j].0, self.ops[op][j].1, self.p);
                    } else {
                        out[op][j] = jaro_winkler_from_parts(
                            part[0],
                            part[1],
                            part[2],
                            lens_a[start as usize + k],
                            lens_b[start as usize + k],
                            self.p,
                        );
                    }
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

        fn gpu_kernel_or_skip() -> Option<&'static GpuJaroKernel> {
            match GpuJaroKernel::get() {
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

        fn assert_close(gpu: &[f64], cpu: &[f64]) {
            assert_eq!(gpu.len(), cpu.len());
            for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
                assert!(
                    (g - c).abs() < 1e-4,
                    "GPU Jaro result {} differs from CPU: {} vs {}",
                    i, g, c
                );
            }
        }

        /// Force GPU dispatch regardless of metric/auto routing (which sends
        /// Jaro to CPU on integrated GPUs where the SIMD path wins). The
        /// override is global, so the RAII guard restores `None` on drop;
        /// tests hold the GPU lock, so it cannot race with other tests.
        fn force_gpu() -> impl Drop {
            crate::gpu::force_gpu_threshold(1)
        }

        /// Exercises shader compilation, buffer sizing/allocation, dispatch,
        /// readback, and the negative sentinel fallback end-to-end against CPU.
        #[test]
        fn test_gpu_batch_matches_cpu() {
            let _gpu_guard = crate::gpu::gpu_test_lock();
            let Some(kernel) = gpu_kernel_or_skip() else { return; };
            let _force = force_gpu();

            let a = gen_strings(1000, 0xFEEDFACE);
            let b = gen_strings(1000, 0x0DDBA11);
            let mut pairs: Vec<(&str, &str)> =
                a.iter().zip(b.iter()).map(|(x, y)| (x.as_str(), y.as_str())).collect();
            pairs[0] = ("", "");
            pairs[1] = ("", "xyz");
            pairs[2] = ("hello", "hello");

            let gpu = kernel.compute_batch(&pairs, 0.1).expect("GPU batch should succeed");
            let cpu: Vec<f64> = pairs.iter()
                .map(|(x, y)| crate::jaro_winkler(x, y, 0.1))
                .collect();
            assert_close(&gpu, &cpu);
            // The negative sentinel (written for > 128-char strings) must never leak.
            assert!(gpu.iter().all(|&v| v >= 0.0), "negative sentinel leaked into results");
        }

        /// Validates the 2D Jaro-Winkler matrix pipeline.
        #[test]
        fn test_gpu_matrix_matches_cpu() {
            let _gpu_guard = crate::gpu::gpu_test_lock();
            let Some(kernel) = gpu_kernel_or_skip() else { return; };
            let _force = force_gpu();

            let a = gen_strings(30, 0x13579BDF);
            let b = gen_strings(30, 0x2468ACE0);
            let refs_a: Vec<&str> = a.iter().map(|s| s.as_str()).collect();
            let refs_b: Vec<&str> = b.iter().map(|s| s.as_str()).collect();

            let gpu = kernel.compute_matrix(&refs_a, &refs_b, 0.1).expect("GPU matrix should succeed");
            let cpu = jaro_winkler_cdist_cpu(&refs_a, &refs_b, 0.1);
            for (grow, crow) in gpu.iter().zip(cpu.iter()) {
                assert_close(grow, crow);
            }
        }

        /// Arms the readback-timeout fault and verifies the batch path returns
        /// `FuzzGpuError::Timeout` deterministically (fast, and never Ok).
        #[test]
        fn test_batch_readback_timeout_returns_timeout_error() {
            let _gpu_guard = crate::gpu::gpu_test_lock();
            let Some(kernel) = gpu_kernel_or_skip() else { return; };
            let _force = force_gpu();

            let a = gen_strings(1000, 0xFEED1234);
            let b = gen_strings(1000, 0x0DD5A11);
            let pairs: Vec<(&str, &str)> =
                a.iter().zip(b.iter()).map(|(x, y)| (x.as_str(), y.as_str())).collect();

            crate::gpu::arm_readback_timeout_fault();
            let result = kernel.compute_batch(&pairs, 0.1);
            crate::gpu::disarm_readback_timeout_fault();

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
            let _force = force_gpu();

            let a = gen_strings(30, 0x33333333);
            let b = gen_strings(30, 0x44444444);
            let refs_a: Vec<&str> = a.iter().map(|s| s.as_str()).collect();
            let refs_b: Vec<&str> = b.iter().map(|s| s.as_str()).collect();

            crate::gpu::arm_readback_timeout_fault();
            let result = kernel.compute_matrix(&refs_a, &refs_b, 0.1);
            crate::gpu::disarm_readback_timeout_fault();

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
            let _force = force_gpu();

            let a = gen_strings(1000, 0xC0FFEE01);
            let b = gen_strings(1000, 0x0FF1CE11);
            let pairs: Vec<(&str, &str)> =
                a.iter().zip(b.iter()).map(|(x, y)| (x.as_str(), y.as_str())).collect();

            crate::gpu::arm_small_buffer_fault();
            let result = kernel.compute_batch(&pairs, 0.1);
            crate::gpu::disarm_small_buffer_fault();

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
            let _force = force_gpu();

            let a = gen_strings(32, 0xA11CEB00);
            let b = gen_strings(32, 0xB00B1355);
            let refs_a: Vec<&str> = a.iter().map(|s| s.as_str()).collect();
            let refs_b: Vec<&str> = b.iter().map(|s| s.as_str()).collect();

            crate::gpu::arm_small_buffer_fault();
            let result = kernel.compute_matrix(&refs_a, &refs_b, 0.1);
            crate::gpu::disarm_small_buffer_fault();

            let cpu = jaro_winkler_cdist_cpu(&refs_a, &refs_b, 0.1);
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
            let result = GpuJaroKernel::new_inner(engine);
            crate::gpu::disarm_shader_error_fault();

            match result {
                Err(FuzzGpuError::ShaderError(_)) => {} // expected
                Err(e) => panic!("expected FuzzGpuError::ShaderError, got: {}", e),
                Ok(_) => panic!("expected FuzzGpuError::ShaderError, got Ok kernel"),
            }
        }

        /// Malformed `p` values (outside the Winkler 0.0–0.25 range, or NaN)
        /// must be rejected with `FuzzGpuError::InvalidInput` before any GPU
        /// work; the valid boundary values must still pass through.
        #[test]
        fn test_jaro_batch_invalid_p_returns_invalid_input() {
            let _gpu_guard = crate::gpu::gpu_test_lock();
            let Some(kernel) = gpu_kernel_or_skip() else { return; };
            let _force = force_gpu();

            let pairs: Vec<(&str, &str)> = vec![("MARTHA", "MARHTA"), ("hello", "world")];
            for bad in [0.26f64, -0.1, 0.5, f64::NAN] {
                match kernel.compute_batch(&pairs, bad) {
                    Err(FuzzGpuError::InvalidInput(_)) => {} // expected
                    other => panic!("expected FuzzGpuError::InvalidInput for p={bad}, got: {other:?}"),
                }
            }
            for good in [0.0f64, 0.1, 0.25] {
                assert!(kernel.compute_batch(&pairs, good).is_ok(), "p={good} should be accepted");
            }
        }

        /// Same InvalidInput rejection on the 2D matrix entry point.
        #[test]
        fn test_jaro_matrix_invalid_p_returns_invalid_input() {
            let _gpu_guard = crate::gpu::gpu_test_lock();
            let Some(kernel) = gpu_kernel_or_skip() else { return; };
            let _force = force_gpu();

            let list_a = ["MARTHA", "hello"];
            let list_b = ["MARHTA", "world"];
            for bad in [0.26f64, -0.1, 0.5, f64::NAN] {
                match kernel.compute_matrix(&list_a, &list_b, bad) {
                    Err(FuzzGpuError::InvalidInput(_)) => {} // expected
                    other => panic!("expected FuzzGpuError::InvalidInput for p={bad}, got: {other:?}"),
                }
            }
            for good in [0.0f64, 0.1, 0.25] {
                assert!(kernel.compute_matrix(&list_a, &list_b, good).is_ok(), "p={good} should be accepted");
            }
        }

        /// The batched API must return exactly what per-op `compute_batch`
        /// returns, including the CPU-routed edge cases.
        #[test]
        fn test_gpu_batch_matches_compute_batch() {
            let _gpu_guard = crate::gpu::gpu_test_lock();
            let Some(kernel) = gpu_kernel_or_skip() else { return; };
            let _force = force_gpu();

            let a1 = gen_strings(600, 0x1A2B3C4D);
            let b1 = gen_strings(600, 0x5E6F7081);
            let mut op1: Vec<(&str, &str)> =
                a1.iter().zip(b1.iter()).map(|(x, y)| (x.as_str(), y.as_str())).collect();
            op1[0] = ("", "");
            op1[1] = ("", "xyz");
            op1[2] = ("MARTHA", "MARHTA");

            let long = "a".repeat(300);
            let op2: Vec<(&str, &str)> = vec![
                ("日本", "日本語"),
                (&long, "short"),
                ("dwayne", "duane"),
            ];

            let p = 0.1;
            let expected = vec![
                kernel.compute_batch(&op1, p).expect("op1 compute_batch"),
                kernel.compute_batch(&op2, p).expect("op2 compute_batch"),
            ];

            let mut batch = kernel.batch(p).expect("batch creation");
            batch.add(&op1);
            batch.add(&op2);
            assert_eq!(batch.len(), 2);
            let got = batch.execute().expect("batch execute");

            for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
                assert_eq!(g.len(), e.len(), "op {i} length");
                for (j, (&gv, &ev)) in g.iter().zip(e).enumerate() {
                    assert!((gv - ev).abs() < 1e-4, "op {i} pair {j}: batch {gv} != compute {ev}");
                }
                assert!(!g.iter().any(|&v| v < 0.0), "op {i}: negative sentinel leaked");
            }
        }

        /// `batch()` must reject an out-of-range `p` the same way `compute_batch`
        /// does — at creation, before any dispatch.
        #[test]
        fn test_gpu_batch_invalid_p_returns_invalid_input() {
            let _gpu_guard = crate::gpu::gpu_test_lock();
            let Some(kernel) = gpu_kernel_or_skip() else { return; };
            let _force = force_gpu();

            for bad in [0.26f64, -0.1, 0.5, f64::NAN] {
                match kernel.batch(bad) {
                    Err(FuzzGpuError::InvalidInput(_)) => {} // expected
                    Err(e) => panic!("expected FuzzGpuError::InvalidInput for p={bad}, got: {e}"),
                    Ok(_) => panic!("expected FuzzGpuError::InvalidInput for p={bad}, got Ok batch"),
                }
            }
            assert!(kernel.batch(0.1).is_ok(), "p=0.1 should be accepted");
        }

        /// An empty batch is a no-op that returns an empty result set.
        #[test]
        fn test_gpu_batch_empty_returns_empty() {
            let _gpu_guard = crate::gpu::gpu_test_lock();
            let Some(kernel) = gpu_kernel_or_skip() else { return; };
            let _force = force_gpu();

            let batch = kernel.batch(0.1).expect("batch creation");
            assert!(batch.is_empty());
            let got = batch.execute().expect("empty batch execute");
            assert!(got.is_empty());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// The byte fast path and the char path must agree exactly on ASCII
        /// input. Lengths up to 200 bytes cross the 128-byte stack/heap split
        /// in both implementations.
        #[test]
        fn ascii_byte_and_char_paths_agree(
            a in prop::collection::vec(prop::char::range('a', 'z'), 0..=200usize),
            b in prop::collection::vec(prop::char::range('a', 'z'), 0..=200usize),
        ) {
            let a: String = a.into_iter().collect();
            let b: String = b.into_iter().collect();
            let a_chars: Vec<char> = a.chars().collect();
            let b_chars: Vec<char> = b.chars().collect();
            let by = jaro_bytes(a.as_bytes(), b.as_bytes());
            let ch = jaro_chars(&a_chars, &b_chars);
            prop_assert!(
                (by - ch).abs() < 1e-12,
                "jaro_bytes {} != jaro_chars {} for {:?} vs {:?}",
                by, ch, a, b
            );
        }
    }
}
