use rayon::prelude::*;

/// Jaro similarity between two strings.
///
/// Returns 1.0 for identical strings (including both empty), 0.0 if no matches.
pub fn jaro(a: &str, b: &str) -> f64 {
    let a = a.as_bytes();
    let b = b.as_bytes();
    let (m, n) = (a.len(), b.len());

    if m == 0 && n == 0 { return 1.0; }
    if m == 0 || n == 0 { return 0.0; }
    if a == b { return 1.0; }

    let match_distance = (m.max(n) / 2).saturating_sub(1);

    if m <= 128 && n <= 128 {
        let mut a_matches = [false; 128];
        let mut b_matches = [false; 128];
        jaro_inner(a, b, &mut a_matches[..m], &mut b_matches[..n], match_distance)
    } else {
        let mut a_matches = vec![false; m];
        let mut b_matches = vec![false; n];
        jaro_inner(a, b, &mut a_matches, &mut b_matches, match_distance)
    }
}

#[inline]
fn jaro_inner(a: &[u8], b: &[u8], a_matches: &mut [bool], b_matches: &mut [bool], match_distance: usize) -> f64 {
    let (m, n) = (a.len(), b.len());
    let mut matches = 0u32;

    for i in 0..m {
        let lo = i.saturating_sub(match_distance);
        let hi = (i + match_distance + 1).min(n);
        let ai = a[i];
        for j in lo..hi {
            if b_matches[j] || ai != b[j] { continue; }
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
pub fn jaro_winkler(a: &str, b: &str, p: f64) -> f64 {
    if a == b { return 1.0; }

    let jaro_score = jaro(a, b);
    let prefix_len = a.as_bytes().iter()
        .zip(b.as_bytes().iter())
        .take_while(|(x, y)| x == y)
        .count()
        .min(4);

    jaro_score + (prefix_len as f64 * p * (1.0 - jaro_score))
}

/// Batch Jaro-Winkler: one query vs many candidates, parallelized with Rayon.
pub fn jaro_winkler_batch(query: &str, candidates: &[&str], p: f64) -> Vec<f64> {
    candidates.par_iter().map(|c| jaro_winkler(query, c, p)).collect()
}

/// Cross-product similarity matrix on CPU via Rayon.
pub fn jaro_winkler_cdist_cpu(list_a: &[&str], list_b: &[&str], p: f64) -> Vec<Vec<f64>> {
    if list_a.is_empty() || list_b.is_empty() {
        return vec![];
    }
    list_a.par_iter().map(|a| {
        list_b.iter().map(|b| jaro_winkler(a, b, p)).collect()
    }).collect()
}

// ── GPU-accelerated Jaro-Winkler ────────────────────────────

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
    const STREAMING_CHUNK_PAIRS: usize = 500_000;

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
            let module = engine.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("jaro shader"),
                source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
            });

            let matrix_module = engine.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("jaro matrix shader"),
                source: wgpu::ShaderSource::Wgsl(MATRIX_SHADER_SRC.into()),
            });

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
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

            let pipeline = engine.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("jaro pipeline"),
                layout: Some(&layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

            let matrix_pipeline = engine.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("jaro matrix pipeline"),
                layout: Some(&layout),
                module: &matrix_module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

            Ok(Self { engine, pipeline, matrix_pipeline, bind_group_layout })
        }

        /// Smart streaming GPU/CPU dispatch for batch Jaro-Winkler with streaming chunking for >128MB.
        pub fn compute_batch(&self, pairs: &[(&str, &str)], p: f64) -> Result<Vec<f64>> {
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
                } else if a.len() > GPU_MAX_STRING_LEN || b.len() > GPU_MAX_STRING_LEN {
                    cpu_indices.push(i);
                } else {
                    gpu_indices.push(i);
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

            for stream_chunk in gpu_indices.chunks(STREAMING_CHUNK_PAIRS) {
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
                chars_a.extend(a.bytes().map(|c| c as u32));
                offsets_a.push(chars_a.len() as u32);
                chars_b.extend(b.bytes().map(|c| c as u32));
                offsets_b.push(chars_b.len() as u32);
                max_len = max_len.max(a.len().max(b.len()) as u32);
            }

            if chars_a.is_empty() { chars_a.push(0); }
            if chars_b.is_empty() { chars_b.push(0); }

            let buf_offsets_a = self.engine.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("joa"), contents: bytemuck::cast_slice(&offsets_a), usage: wgpu::BufferUsages::STORAGE });
            let buf_chars_a = self.engine.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("jca"), contents: bytemuck::cast_slice(&chars_a), usage: wgpu::BufferUsages::STORAGE });
            let buf_offsets_b = self.engine.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("job"), contents: bytemuck::cast_slice(&offsets_b), usage: wgpu::BufferUsages::STORAGE });
            let buf_chars_b = self.engine.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("jcb"), contents: bytemuck::cast_slice(&chars_b), usage: wgpu::BufferUsages::STORAGE });

            let results_size = (batch_size as u64) * 4;
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
            self.engine.queue.submit(Some(encoder.finish()));

            let slice = buf_staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
            self.engine.device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().map_err(|e| FuzzGpuError::BufferError(e.to_string()))?;
            let data = slice.get_mapped_range();
            let raw: &[u32] = bytemuck::cast_slice(&data);
            let gpu_results: Vec<f64> = raw.iter().map(|&bits| f32::from_bits(bits) as f64).collect();
            drop(data);
            buf_staging.unmap();
            Ok(gpu_results)
        }

        /// Dedicated 2D Jaro-Winkler Matrix GPU compute: O(N + M) upload instead of O(N * M)
        pub fn compute_matrix(&self, list_a: &[&str], list_b: &[&str], p: f64) -> Result<Vec<Vec<f64>>> {
            let rows = list_a.len();
            let cols = list_b.len();
            if rows == 0 || cols == 0 { return Ok(vec![]); }

            let total_pairs = rows * cols;
            if total_pairs < GPU_THRESHOLD {
                return Ok(jaro_winkler_cdist_cpu(list_a, list_b, p));
            }

            // Pack List A
            let mut offsets_a: Vec<u32> = Vec::with_capacity(rows + 1);
            let mut chars_a: Vec<u32> = Vec::new();
            offsets_a.push(0);
            for a in list_a {
                chars_a.extend(a.bytes().map(|c| c as u32));
                offsets_a.push(chars_a.len() as u32);
            }

            // Pack List B
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

            let matrix_size = (total_pairs as u64) * 4;
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

            let mut matrix: Vec<Vec<f64>> = Vec::with_capacity(rows);
            if !has_oversized {
                for i in 0..rows {
                    let start = i * cols;
                    let end = start + cols;
                    matrix.push(flat[start..end].iter().map(|&bits| f32::from_bits(bits) as f64).collect());
                }
            } else {
                for i in 0..rows {
                    let start = i * cols;
                    let end = start + cols;
                    let mut row: Vec<f64> = flat[start..end].iter().map(|&bits| f32::from_bits(bits) as f64).collect();
                    for (j, val) in row.iter_mut().enumerate() {
                        if *val < 0.0 {
                            *val = jaro_winkler(list_a[i], list_b[j], p);
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
