use rayon::prelude::*;

/// Standard single-row DP Levenshtein distance with diagonal optimization.
pub fn levenshtein_distance_raw(a: &str, b: &str) -> u32 {
    let a = a.as_bytes();
    let b = b.as_bytes();
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
        let ai = a[i - 1];
        for j in 1..=n {
            let old = row[j];
            let cost = if ai == b[j - 1] { 0 } else { 1 };
            row[j] = (prev_diag + cost).min(row[j] + 1).min(row[j - 1] + 1);
            prev_diag = old;
        }
    }
    row[n]
}

/// Cross-product matrix computed on CPU via Rayon.
pub fn levenshtein_cdist_cpu(list_a: &[&str], list_b: &[&str]) -> Vec<Vec<u32>> {
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
    const STREAMING_CHUNK_PAIRS: usize = 500_000;

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
            let module = engine.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("levenshtein shader"),
                source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
            });

            let matrix_module = engine.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("levenshtein matrix shader"),
                source: wgpu::ShaderSource::Wgsl(MATRIX_SHADER_SRC.into()),
            });

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
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

            let pipeline = engine.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("levenshtein pipeline"),
                layout: Some(&layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

            let matrix_pipeline = engine.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("levenshtein matrix pipeline"),
                layout: Some(&layout),
                module: &matrix_module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

            Ok(Self { engine, pipeline, matrix_pipeline, bind_group_layout })
        }

        /// Smart streaming dispatch with streaming chunking for arbitrary dataset sizes (>128MB).
        pub fn compute(&self, pairs: &[(&str, &str)]) -> Result<Vec<u32>> {
            let n = pairs.len();
            if n == 0 { return Ok(vec![]); }

            let mut results = vec![0u32; n];
            let mut gpu_indices: Vec<usize> = Vec::with_capacity(n);
            let mut cpu_indices: Vec<usize> = Vec::new();

            for (i, (a, b)) in pairs.iter().enumerate() {
                if a.is_empty() || b.is_empty() {
                    results[i] = (a.len().max(b.len())) as u32;
                } else if *a == *b {
                    results[i] = 0;
                } else if a.len() > GPU_MAX_STRING_LEN || b.len() > GPU_MAX_STRING_LEN {
                    cpu_indices.push(i);
                } else {
                    gpu_indices.push(i);
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

            for stream_chunk in gpu_indices.chunks(STREAMING_CHUNK_PAIRS) {
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
                chars_a.extend(a.bytes().map(|c| c as u32));
                offsets_a.push(chars_a.len() as u32);
                chars_b.extend(b.bytes().map(|c| c as u32));
                offsets_b.push(chars_b.len() as u32);
                max_len = max_len.max(a.len().max(b.len()) as u32);
            }

            if chars_a.is_empty() { chars_a.push(0); }
            if chars_b.is_empty() { chars_b.push(0); }

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

            let results_size = (batch_size as u64) * 4;
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
            self.engine.queue.submit(Some(encoder.finish()));

            let slice = buf_staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
            self.engine.device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().map_err(|e| FuzzGpuError::BufferError(e.to_string()))?;
            let data = slice.get_mapped_range();
            let gpu_results: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            buf_staging.unmap();
            Ok(gpu_results)
        }

        /// Dedicated 2D Matrix GPU Compute: O(N + M) data upload instead of O(N * M)
        pub fn compute_matrix(&self, list_a: &[&str], list_b: &[&str]) -> Result<Vec<Vec<u32>>> {
            let rows = list_a.len();
            let cols = list_b.len();
            if rows == 0 || cols == 0 { return Ok(vec![]); }

            let total_pairs = rows * cols;
            // Small matrices (< 500 pairs): compute on CPU directly with Rayon (zero PCIe transfer overhead)
            if total_pairs < GPU_THRESHOLD {
                return Ok(levenshtein_cdist_cpu(list_a, list_b));
            }

            // Pack List A (O(N) data)
            let mut offsets_a: Vec<u32> = Vec::with_capacity(rows + 1);
            let mut chars_a: Vec<u32> = Vec::new();
            offsets_a.push(0);
            for a in list_a {
                chars_a.extend(a.bytes().map(|c| c as u32));
                offsets_a.push(chars_a.len() as u32);
            }

            // Pack List B (O(M) data)
            let mut offsets_b: Vec<u32> = Vec::with_capacity(cols + 1);
            let mut chars_b: Vec<u32> = Vec::new();
            offsets_b.push(0);
            for b in list_b {
                chars_b.extend(b.bytes().map(|c| c as u32));
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

            let matrix_size = (total_pairs as u64) * 4;
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
            self.engine.queue.submit(Some(encoder.finish()));

            let slice = buf_staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
            self.engine.device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().map_err(|e| FuzzGpuError::BufferError(e.to_string()))?;
            let data = slice.get_mapped_range();
            let flat: &[u32] = bytemuck::cast_slice(&data);

            let has_oversized = list_a.iter().any(|s| s.len() > GPU_MAX_STRING_LEN)
                || list_b.iter().any(|s| s.len() > GPU_MAX_STRING_LEN);

            let mut matrix: Vec<Vec<u32>> = Vec::with_capacity(rows);
            if !has_oversized {
                for i in 0..rows {
                    let start = i * cols;
                    let end = start + cols;
                    matrix.push(flat[start..end].to_vec());
                }
            } else {
                for i in 0..rows {
                    let start = i * cols;
                    let end = start + cols;
                    let mut row = flat[start..end].to_vec();
                    for (j, val) in row.iter_mut().enumerate() {
                        if *val == 0xFFFFFFFF {
                            *val = levenshtein_distance_raw(list_a[i], list_b[j]);
                        }
                    }
                    matrix.push(row);
                }
            }

            drop(data);
            buf_staging.unmap();
            Ok(matrix)
        }
    }
}
