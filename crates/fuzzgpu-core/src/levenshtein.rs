use rayon::prelude::*;

/// Standard single-row DP Levenshtein distance with diagonal optimization.
/// Supports both ASCII (byte-fast path) and Unicode characters (scalar value codepoint path).
pub fn levenshtein_distance_raw(a: &str, b: &str) -> u32 {
    if a.is_ascii() && b.is_ascii() {
        levenshtein_distance_slice(a.as_bytes(), b.as_bytes())
    } else {
        let a_chars: Vec<char> = a.chars().collect();
        let b_chars: Vec<char> = b.chars().collect();
        levenshtein_distance_slice(&a_chars, &b_chars)
    }
}

fn levenshtein_distance_slice<T: PartialEq>(a: &[T], b: &[T]) -> u32 {
    let (m, n) = (a.len(), b.len());
    if m == 0 { return n as u32; }
    if n == 0 { return m as u32; }
    if a == b { return 0; }

    // Single-row + diagonal optimization: halves memory vs two-row.
    let mut row = vec![0u32; n + 1];
    for (j, item) in row.iter_mut().enumerate() {
        *item = j as u32;
    }

    for i in 1..=m {
        let mut prev_diag = row[0];
        row[0] = i as u32;
        let ai = &a[i - 1];
        for j in 1..=n {
            let old = row[j];
            let cost = if ai == &b[j - 1] { 0 } else { 1 };
            row[j] = (prev_diag + cost).min(row[j] + 1).min(row[j - 1] + 1);
            prev_diag = old;
        }
    }
    row[n]
}

/// Cross-product matrix computed on CPU via Rayon.
pub fn levenshtein_cdist_cpu(list_a: &[&str], list_b: &[&str]) -> Vec<Vec<u32>> {
    if list_a.is_empty() || list_b.is_empty() {
        return vec![];
    }
    list_a.par_iter().map(|a| {
        list_b.iter().map(|b| levenshtein_distance_raw(a, b)).collect()
    }).collect()
}

/// CPU-parallel Levenshtein kernel using Rayon.
pub struct LevenshteinKernel;

impl LevenshteinKernel {
    pub fn compute(&self, pairs: &[(&str, &str)]) -> crate::Result<Vec<u32>> {
        Ok(pairs.par_iter().map(|(a, b)| levenshtein_distance_raw(a, b)).collect())
    }
}

#[cfg(feature = "gpu")]
pub mod gpu_ext {
    use super::*;
    use bytemuck::{Pod, Zeroable};
    use std::sync::OnceLock;
    use wgpu::util::DeviceExt;
    use crate::gpu::{FuzzGpuError, GpuEngine, Result};

    const SHADER_SRC: &str = include_str!("shaders/levenshtein.wgsl");
    const MATRIX_SHADER_SRC: &str = include_str!("shaders/levenshtein_matrix.wgsl");

    const GPU_THRESHOLD: usize = 500;
    const GPU_MAX_STRING_LEN: usize = 256;
    const MAX_DISPATCH: u32 = 65535;
    const MAX_DESIRED_CHUNK_PAIRS: usize = 500_000;

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
        matrix_pipeline: wgpu::ComputePipeline,
        bind_group_layout: wgpu::BindGroupLayout,
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

            Ok(Self { engine, pipeline, matrix_pipeline, bind_group_layout })
        }

        /// Smart streaming dispatch with dynamic buffer limit validation and chunking.
        pub fn compute(&self, pairs: &[(&str, &str)]) -> Result<Vec<u32>> {
            let n = pairs.len();
            if n == 0 { return Ok(vec![]); }

            let mut results = vec![0u32; n];
            let mut gpu_indices: Vec<usize> = Vec::with_capacity(n);
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
                    } else {
                        gpu_indices.push(i);
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

            if gpu_indices.len() < GPU_THRESHOLD {
                let cpu_results: Vec<u32> = gpu_indices.par_iter()
                    .map(|&i| levenshtein_distance_raw(pairs[i].0, pairs[i].1))
                    .collect();
                for (idx, &orig_i) in gpu_indices.iter().enumerate() {
                    results[orig_i] = cpu_results[idx];
                }
                return Ok(results);
            }

            // Calculate chunk size dynamically based on hardware limits
            let max_allowed_binding = self.engine.max_storage_buffer_binding_size as usize;
            let bytes_per_pair = (GPU_MAX_STRING_LEN * 4 * 2 + 8).max(128);
            let dynamic_chunk_size = (max_allowed_binding / bytes_per_pair)
                .min(MAX_DESIRED_CHUNK_PAIRS)
                .max(512);

            for stream_chunk in gpu_indices.chunks(dynamic_chunk_size) {
                let chunk_results = self.compute_gpu_subset(pairs, stream_chunk)?;
                for (idx, &orig_i) in stream_chunk.iter().enumerate() {
                    if chunk_results[idx] == 0xFFFFFFFF {
                        results[orig_i] = levenshtein_distance_raw(pairs[orig_i].0, pairs[orig_i].1);
                    } else {
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

            let buf_offsets_a = self.engine.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("oa"), contents: bytemuck::cast_slice(&offsets_a),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let buf_chars_a = self.engine.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ca"), contents: bytemuck::cast_slice(&chars_a),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let buf_offsets_b = self.engine.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ob"), contents: bytemuck::cast_slice(&offsets_b),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let buf_chars_b = self.engine.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cb"), contents: bytemuck::cast_slice(&chars_b),
                usage: wgpu::BufferUsages::STORAGE,
            });

            let buf_results = self.engine.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("res"), size: results_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let buf_staging = self.engine.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("stg"), size: results_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let mut encoder = self.engine.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("levenshtein encoder") },
            );

            let mut remaining = batch_size;
            let mut offset = 0u32;

            while remaining > 0 {
                let chunk = remaining.min(MAX_DISPATCH);
                let params = Params { batch_size, max_len, offset };
                let buf_params = self.engine.device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("p"),
                        contents: bytemuck::bytes_of(&params),
                        usage: wgpu::BufferUsages::UNIFORM,
                    },
                );

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

                let workgroups = (chunk + 63) / 64;
                {
                    let mut pass = encoder.begin_compute_pass(
                        &wgpu::ComputePassDescriptor { label: None, timestamp_writes: None },
                    );
                    pass.set_pipeline(&self.pipeline);
                    pass.set_bind_group(0, &bg, &[]);
                    pass.dispatch_workgroups(workgroups, 1, 1);
                }
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
                .map_err(|_| FuzzGpuError::Timeout("GPU buffer mapping timed out after 10s".into()))?
                .map_err(|e| FuzzGpuError::BufferError(format!("GPU buffer map failed: {}", e)))?;

            let data = slice
                .get_mapped_range()
                .map_err(|e| FuzzGpuError::BufferError(format!("GPU buffer map range failed: {e}")))?;
            let gpu_results: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            buf_staging.unmap();
            Ok(gpu_results)
        }

        /// Dedicated 2D Matrix GPU Compute: O(N + M) data upload instead of O(N * M)
        /// Validates string lengths and memory limits *before* any GPU dispatch.
        pub fn compute_matrix(&self, list_a: &[&str], list_b: &[&str]) -> Result<Vec<Vec<u32>>> {
            let rows = list_a.len();
            let cols = list_b.len();
            if rows == 0 || cols == 0 { return Ok(vec![]); }

            let total_pairs = rows * cols;
            // Small matrices (< 500 pairs): compute on CPU directly with Rayon (zero PCIe transfer overhead)
            if total_pairs < GPU_THRESHOLD {
                return Ok(levenshtein_cdist_cpu(list_a, list_b));
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

            let buf_offsets_a = self.engine.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("moa"), contents: bytemuck::cast_slice(&offsets_a),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let buf_chars_a = self.engine.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mca"), contents: bytemuck::cast_slice(&chars_a),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let buf_offsets_b = self.engine.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mob"), contents: bytemuck::cast_slice(&offsets_b),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let buf_chars_b = self.engine.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mcb"), contents: bytemuck::cast_slice(&chars_b),
                usage: wgpu::BufferUsages::STORAGE,
            });

            let buf_matrix = self.engine.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mres"), size: matrix_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let buf_staging = self.engine.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mstg"), size: matrix_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let params = MatrixParams { rows: rows as u32, cols: cols as u32 };
            let buf_params = self.engine.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mp"), contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

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

            let a = gen_strings(1000, 0xABCDEF01);
            let b = gen_strings(1000, 0x23456789);
            let pairs: Vec<(&str, &str)> =
                a.iter().zip(b.iter()).map(|(x, y)| (x.as_str(), y.as_str())).collect();

            crate::gpu::arm_readback_timeout_fault();
            let result = kernel.compute(&pairs);
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

            let a = gen_strings(30, 0x11111111);
            let b = gen_strings(30, 0x22222222);
            let refs_a: Vec<&str> = a.iter().map(|s| s.as_str()).collect();
            let refs_b: Vec<&str> = b.iter().map(|s| s.as_str()).collect();

            crate::gpu::arm_readback_timeout_fault();
            let result = kernel.compute_matrix(&refs_a, &refs_b);
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

            let a = gen_strings(1000, 0x0BADF00D);
            let b = gen_strings(1000, 0xF00DBABE);
            let pairs: Vec<(&str, &str)> =
                a.iter().zip(b.iter()).map(|(x, y)| (x.as_str(), y.as_str())).collect();

            crate::gpu::arm_small_buffer_fault();
            let result = kernel.compute(&pairs);
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
    }
}
