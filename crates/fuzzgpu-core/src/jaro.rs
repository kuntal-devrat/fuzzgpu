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
        + (matches as f64 - transpositions as f64 / 2.0) / matches as f64) / 3.0
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

/// Batch Jaro-Winkler on CPU using Rayon.
pub fn jaro_winkler_batch(query: &str, candidates: &[&str], p: f64) -> Vec<f64> {
    candidates.par_iter().map(|c| jaro_winkler(query, c, p)).collect()
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

/// Optimized Jaro variant using stack allocations and early termination.
pub fn jaro_optimized(a: &[u8], b: &[u8]) -> f64 {
    jaro_bytes(a, b)
}

#[cfg(feature = "gpu")]
pub mod gpu_ext {
    use super::*;
    use bytemuck::{Pod, Zeroable};
    use std::sync::OnceLock;
    use wgpu::util::DeviceExt;
    use crate::gpu::{FuzzGpuError, GpuEngine, Result};

    const SHADER_SRC: &str = include_str!("shaders/jaro.wgsl");
    const MATRIX_SHADER_SRC: &str = include_str!("shaders/jaro_matrix.wgsl");

    const GPU_THRESHOLD: usize = 500;
    const GPU_MAX_STRING_LEN: usize = 128;
    const MAX_DISPATCH: u32 = 65535;
    const MAX_DESIRED_CHUNK_PAIRS: usize = 500_000;

    #[repr(C)]
    #[derive(Copy, Clone, Pod, Zeroable)]
    struct JaroParams {
        batch_size: u32,
        max_len: u32,
        offset: u32,
        winkler_p_bits: u32,
    }

    #[repr(C)]
    #[derive(Copy, Clone, Pod, Zeroable)]
    struct JaroMatrixParams {
        rows: u32,
        cols: u32,
        winkler_p_bits: u32,
    }

    pub struct GpuJaroKernel {
        engine: std::sync::Arc<GpuEngine>,
        pipeline: wgpu::ComputePipeline,
        matrix_pipeline: wgpu::ComputePipeline,
        bind_group_layout: wgpu::BindGroupLayout,
    }

    static GLOBAL_GPU_JARO_KERNEL: OnceLock<GpuJaroKernel> = OnceLock::new();

    impl GpuJaroKernel {
        pub fn get() -> Result<&'static Self> {
            if let Some(k) = GLOBAL_GPU_JARO_KERNEL.get() { return Ok(k); }
            let engine = GpuEngine::get()?;
            let kernel = Self::new_inner(engine)?;
            let _ = GLOBAL_GPU_JARO_KERNEL.set(kernel);
            Ok(GLOBAL_GPU_JARO_KERNEL.get().unwrap())
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

            Ok(Self { engine, pipeline, matrix_pipeline, bind_group_layout })
        }

        /// Smart streaming GPU/CPU dispatch for batch Jaro-Winkler with dynamic chunk sizing.
        ///
        /// `p` (Winkler prefix weight) must be in `0.0..=0.25`; anything else —
        /// including NaN — is rejected with `FuzzGpuError::InvalidInput` before
        /// any dispatch. (The CPU `jaro_winkler` clamps instead; this is the
        /// strict boundary of the Result-returning GPU API.)
        pub fn compute_batch(&self, pairs: &[(&str, &str)], p: f64) -> Result<Vec<f64>> {
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
                let cpu_results: Vec<f64> = cpu_indices.par_iter()
                    .map(|&i| crate::jaro_winkler(pairs[i].0, pairs[i].1, p))
                    .collect();
                for (idx, &orig_i) in cpu_indices.iter().enumerate() {
                    results[orig_i] = cpu_results[idx];
                }
            }

            if gpu_indices.len() < GPU_THRESHOLD {
                let cpu_results: Vec<f64> = gpu_indices.par_iter()
                    .map(|&i| crate::jaro_winkler(pairs[i].0, pairs[i].1, p))
                    .collect();
                for (idx, &orig_i) in gpu_indices.iter().enumerate() {
                    results[orig_i] = cpu_results[idx];
                }
                return Ok(results);
            }

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

            let buf_offsets_a = self.engine.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("joa"), contents: bytemuck::cast_slice(&offsets_a), usage: wgpu::BufferUsages::STORAGE });
            let buf_chars_a = self.engine.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("jca"), contents: bytemuck::cast_slice(&chars_a), usage: wgpu::BufferUsages::STORAGE });
            let buf_offsets_b = self.engine.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("job"), contents: bytemuck::cast_slice(&offsets_b), usage: wgpu::BufferUsages::STORAGE });
            let buf_chars_b = self.engine.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("jcb"), contents: bytemuck::cast_slice(&chars_b), usage: wgpu::BufferUsages::STORAGE });

            let buf_results = self.engine.device.create_buffer(&wgpu::BufferDescriptor { label: Some("jres"), size: results_size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
            let buf_staging = self.engine.device.create_buffer(&wgpu::BufferDescriptor { label: Some("jstg"), size: results_size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

            let winkler_p_bits = (p as f32).to_bits();

            let mut encoder = self.engine.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("jaro encoder") });
            let mut remaining = batch_size;
            let mut offset = 0u32;

            while remaining > 0 {
                let chunk = remaining.min(MAX_DISPATCH);
                let params = JaroParams { batch_size, max_len, offset, winkler_p_bits };
                let buf_params = self.engine.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("jp"), contents: bytemuck::bytes_of(&params), usage: wgpu::BufferUsages::UNIFORM });

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

                let workgroups = (chunk + 63) / 64;
                { let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None }); pass.set_pipeline(&self.pipeline); pass.set_bind_group(0, &bg, &[]); pass.dispatch_workgroups(workgroups, 1, 1); }
                remaining -= chunk;
                offset += chunk;
            }

            encoder.copy_buffer_to_buffer(&buf_results, 0, &buf_staging, 0, results_size);
            self.engine.submit(encoder);

            let slice = buf_staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            self.engine.map_readback(&slice, move |r| { let _ = tx.send(r); });
            self.engine.poll();

            rx.recv_timeout(GpuEngine::readback_timeout())
                .map_err(|_| FuzzGpuError::Timeout("GPU Jaro readback timed out after 10s".into()))?
                .map_err(|e| FuzzGpuError::BufferError(format!("GPU buffer map failed: {}", e)))?;

            let data = slice
                .get_mapped_range()
                .map_err(|e| FuzzGpuError::BufferError(format!("GPU buffer map range failed: {e}")))?;
            let raw: &[u32] = bytemuck::cast_slice(&data);
            let gpu_results: Vec<f64> = raw.iter().map(|&bits| f32::from_bits(bits) as f64).collect();
            drop(data);
            buf_staging.unmap();
            Ok(gpu_results)
        }

        /// Dedicated 2D Jaro-Winkler Matrix GPU compute: O(N + M) upload instead of O(N * M).
        /// Pre-filters oversized strings and validates buffer limits before dispatch.
        ///
        /// `p` (Winkler prefix weight) must be in `0.0..=0.25`; anything else —
        /// including NaN — is rejected with `FuzzGpuError::InvalidInput` before
        /// any dispatch.
        pub fn compute_matrix(&self, list_a: &[&str], list_b: &[&str], p: f64) -> Result<Vec<Vec<f64>>> {
            if !(0.0..=0.25).contains(&p) {
                return Err(FuzzGpuError::InvalidInput(format!(
                    "Jaro-Winkler prefix weight p must be within 0.0..=0.25, got {p}"
                )));
            }
            let rows = list_a.len();
            let cols = list_b.len();
            if rows == 0 || cols == 0 { return Ok(vec![]); }

            let total_pairs = rows * cols;
            if total_pairs < GPU_THRESHOLD {
                return Ok(jaro_winkler_cdist_cpu(list_a, list_b, p));
            }

            // CRITICAL FIX: Pre-filter oversized strings BEFORE GPU buffer allocation or dispatch
            let has_oversized = list_a.iter().any(|s| s.chars().count() > GPU_MAX_STRING_LEN)
                || list_b.iter().any(|s| s.chars().count() > GPU_MAX_STRING_LEN);

            let matrix_size = (total_pairs as u64) * 4;

            if has_oversized || matrix_size > self.engine.max_buffer_size_effective() {
                return Ok(jaro_winkler_cdist_cpu(list_a, list_b, p));
            }

            // Pack List A
            let mut offsets_a: Vec<u32> = Vec::with_capacity(rows + 1);
            let mut chars_a: Vec<u32> = Vec::new();
            offsets_a.push(0);
            for a in list_a {
                chars_a.extend(a.chars().map(|c| c as u32));
                offsets_a.push(chars_a.len() as u32);
            }

            // Pack List B
            let mut offsets_b: Vec<u32> = Vec::with_capacity(cols + 1);
            let mut chars_b: Vec<u32> = Vec::new();
            offsets_b.push(0);
            for b in list_b {
                chars_b.extend(b.chars().map(|c| c as u32));
                offsets_b.push(chars_b.len() as u32);
            }

            if chars_a.is_empty() { chars_a.push(0); }
            if chars_b.is_empty() { chars_b.push(0); }

            let buf_offsets_a = self.engine.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("jmoa"), contents: bytemuck::cast_slice(&offsets_a), usage: wgpu::BufferUsages::STORAGE,
            });
            let buf_chars_a = self.engine.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("jmca"), contents: bytemuck::cast_slice(&chars_a), usage: wgpu::BufferUsages::STORAGE,
            });
            let buf_offsets_b = self.engine.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("jmob"), contents: bytemuck::cast_slice(&offsets_b), usage: wgpu::BufferUsages::STORAGE,
            });
            let buf_chars_b = self.engine.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("jmcb"), contents: bytemuck::cast_slice(&chars_b), usage: wgpu::BufferUsages::STORAGE,
            });

            let buf_matrix = self.engine.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("jmres"), size: matrix_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let buf_staging = self.engine.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("jmstg"), size: matrix_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let winkler_p_bits = (p as f32).to_bits();
            let params = JaroMatrixParams { rows: rows as u32, cols: cols as u32, winkler_p_bits };
            let buf_params = self.engine.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("jmp"), contents: bytemuck::bytes_of(&params), usage: wgpu::BufferUsages::UNIFORM,
            });

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

            let mut matrix: Vec<Vec<f64>> = Vec::with_capacity(rows);
            for i in 0..rows {
                let start = i * cols;
                let end = start + cols;
                matrix.push(raw[start..end].iter().map(|&bits| f32::from_bits(bits) as f64).collect());
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

        /// Exercises shader compilation, buffer sizing/allocation, dispatch,
        /// readback, and the negative sentinel fallback end-to-end against CPU.
        #[test]
        fn test_gpu_batch_matches_cpu() {
            let _gpu_guard = crate::gpu::gpu_test_lock();
            let Some(kernel) = gpu_kernel_or_skip() else { return; };

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
