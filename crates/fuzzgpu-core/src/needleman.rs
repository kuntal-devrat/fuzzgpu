use crate::{sat_add, sat_mul};
use rayon::prelude::*;

/// Needleman-Wunsch global alignment score with linear gap penalty.
///
/// Uses single-row DP + scalar diagonal for minimal memory.
/// Supports both ASCII fast-path and full Unicode characters.
/// All scores use `i64` with saturating arithmetic to prevent integer overflow on long sequences.
pub fn needleman_wunsch(a: &str, b: &str, match_score: i64, mismatch_score: i64, gap_penalty: i64) -> i64 {
    if a.is_ascii() && b.is_ascii() {
        needleman_wunsch_bytes(a.as_bytes(), b.as_bytes(), match_score, mismatch_score, gap_penalty)
    } else {
        let a_chars: Vec<char> = a.chars().collect();
        let b_chars: Vec<char> = b.chars().collect();
        needleman_wunsch_chars(&a_chars, &b_chars, match_score, mismatch_score, gap_penalty)
    }
}

fn needleman_wunsch_bytes(a: &[u8], b: &[u8], match_score: i64, mismatch_score: i64, gap_penalty: i64) -> i64 {
    needleman_wunsch_slice(a, b, match_score, mismatch_score, gap_penalty)
}

fn needleman_wunsch_chars(a: &[char], b: &[char], match_score: i64, mismatch_score: i64, gap_penalty: i64) -> i64 {
    needleman_wunsch_slice(a, b, match_score, mismatch_score, gap_penalty)
}

fn needleman_wunsch_slice<T: PartialEq>(a: &[T], b: &[T], match_score: i64, mismatch_score: i64, gap_penalty: i64) -> i64 {
    let (m, n) = (a.len(), b.len());

    if m == 0 { return sat_mul(n as i64, gap_penalty); }
    if n == 0 { return sat_mul(m as i64, gap_penalty); }
    if a == b { return sat_mul(m as i64, match_score); }

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
            let score = if ai == &b[j - 1] { match_score } else { mismatch_score };
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
    query: &str, candidates: &[&str],
    match_score: i64, mismatch_score: i64, gap_penalty: i64,
) -> Vec<i64> {
    candidates.par_iter().map(|c| {
        needleman_wunsch(query, c, match_score, mismatch_score, gap_penalty)
    }).collect()
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
        needleman_wunsch_affine_slice(a.as_bytes(), b.as_bytes(), match_score, mismatch_score, gap_open, gap_extend)
    } else {
        let a_chars: Vec<char> = a.chars().collect();
        let b_chars: Vec<char> = b.chars().collect();
        needleman_wunsch_affine_slice(&a_chars, &b_chars, match_score, mismatch_score, gap_open, gap_extend)
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

    if m == 0 && n == 0 { return 0; }
    if m == 0 { return sat_add(gap_open, sat_mul(n as i64, gap_extend)); }
    if n == 0 { return sat_add(gap_open, sat_mul(m as i64, gap_extend)); }
    if a == b { return sat_mul(m as i64, match_score); }

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
            let sub_score = if ai == bj { match_score } else { mismatch_score };

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
    candidates.par_iter().map(|c| {
        needleman_wunsch_affine(query, c, match_score, mismatch_score, gap_open, gap_extend)
    }).collect()
}

#[cfg(feature = "gpu")]
pub mod gpu_ext {
    use super::*;
    use bytemuck::{Pod, Zeroable};
    use std::sync::OnceLock;
    use wgpu::util::DeviceExt;
    use crate::gpu::{FuzzGpuError, GpuEngine, Result};

    const SHADER_SRC: &str = include_str!("shaders/needleman_affine.wgsl");
    const GPU_THRESHOLD: usize = 500;
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
        bind_group_layout: wgpu::BindGroupLayout,
    }

    static GLOBAL_GPU_KERNEL: OnceLock<GpuNeedlemanAffineKernel> = OnceLock::new();

    impl GpuNeedlemanAffineKernel {
        pub fn get() -> Result<&'static Self> {
            if let Some(k) = GLOBAL_GPU_KERNEL.get() { return Ok(k); }
            let engine = GpuEngine::get()?;
            let kernel = Self::new_inner(engine)?;
            let _ = GLOBAL_GPU_KERNEL.set(kernel);
            Ok(GLOBAL_GPU_KERNEL.get().unwrap())
        }

        fn new_inner(engine: std::sync::Arc<GpuEngine>) -> Result<Self> {
            let bind_group_layout =
                engine.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            let layout = engine.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
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
            Ok(Self { engine, pipeline, bind_group_layout })
        }

        /// GPU/CPU batch for affine-gap Needleman-Wunsch (Gotoh).
        ///
        /// Strings up to `GPU_MAX_STRING_LEN` (128) chars run on the GPU
        /// shader; longer ones route to the CPU implementation. Scores are f32
        /// on the GPU (WGSL has no i64) and exact for the practical scoring
        /// ranges — see the shader note. Below `GPU_THRESHOLD` pairs the whole
        /// batch routes to CPU, mirroring the Levenshtein kernel.
        pub fn compute_batch(
            &self,
            pairs: &[(&str, &str)],
            match_score: i64,
            mismatch_score: i64,
            gap_open: i64,
            gap_extend: i64,
        ) -> Result<Vec<i64>> {
            let n = pairs.len();
            if n == 0 { return Ok(vec![]); }

            let mut results = vec![0i64; n];
            let mut gpu_indices: Vec<usize> = Vec::with_capacity(n);
            let mut cpu_indices: Vec<usize> = Vec::new();

            for (i, (a, b)) in pairs.iter().enumerate() {
                if a.chars().count() > GPU_MAX_STRING_LEN || b.chars().count() > GPU_MAX_STRING_LEN {
                    cpu_indices.push(i);
                } else {
                    gpu_indices.push(i);
                }
            }

            if !cpu_indices.is_empty() {
                let cpu_results: Vec<i64> = cpu_indices.par_iter()
                    .map(|&i| needleman_wunsch_affine(pairs[i].0, pairs[i].1, match_score, mismatch_score, gap_open, gap_extend))
                    .collect();
                for (idx, &orig_i) in cpu_indices.iter().enumerate() {
                    results[orig_i] = cpu_results[idx];
                }
            }

            if gpu_indices.is_empty() || gpu_indices.len() < GPU_THRESHOLD {
                let cpu_results: Vec<i64> = gpu_indices.par_iter()
                    .map(|&i| needleman_wunsch_affine(pairs[i].0, pairs[i].1, match_score, mismatch_score, gap_open, gap_extend))
                    .collect();
                for (idx, &orig_i) in gpu_indices.iter().enumerate() {
                    results[orig_i] = cpu_results[idx];
                }
                return Ok(results);
            }

            // Chunk sizing from device limits, mirroring the Levenshtein kernel.
            let max_allowed_binding = self.engine.max_storage_buffer_binding_size as usize;
            let bytes_per_pair = (GPU_MAX_STRING_LEN * 4 * 2 + 8).max(128);
            let dynamic_chunk_size = (max_allowed_binding / bytes_per_pair)
                .min(MAX_DESIRED_CHUNK_PAIRS)
                .max(512);

            for stream_chunk in gpu_indices.chunks(dynamic_chunk_size) {
                let chunk_results = self.compute_gpu_chunk(
                    pairs, stream_chunk, match_score, mismatch_score, gap_open, gap_extend,
                )?;
                for (idx, &orig_i) in stream_chunk.iter().enumerate() {
                    if chunk_results[idx] < SENTINEL_THRESHOLD {
                        results[orig_i] = needleman_wunsch_affine(
                            pairs[orig_i].0, pairs[orig_i].1, match_score, mismatch_score, gap_open, gap_extend,
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

            if chars_a.is_empty() { chars_a.push(0); }
            if chars_b.is_empty() { chars_b.push(0); }

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
                label: Some("res"),
                size: results_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let buf_staging = self.engine.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("stg"),
                size: results_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            let mut encoder = self.engine.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("affine enc") },
            );

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

                let workgroups = (chunk + 15) / 16;
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
            let gpu_results: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            buf_staging.unmap();
            Ok(gpu_results)
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

    #[test]
    fn test_affine_gap() {
        let s1 = "AGCT";
        let s2 = "AGCT";
        assert_eq!(needleman_wunsch_affine(s1, s2, 2, -1, -3, -1), 8);

        let score = needleman_wunsch_affine("ACGT", "AT", 2, -1, -3, -1);
        assert!(score < 8);
    }
}
