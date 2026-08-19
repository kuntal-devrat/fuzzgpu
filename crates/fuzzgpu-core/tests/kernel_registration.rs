//! End-to-end coverage of the `FuzzGpuError::ShaderError` path from the
//! public API.
//!
//! `GpuEngine::build_compute_pipeline` is the public kernel-registration
//! primitive (it's what the built-in Levenshtein/Jaro kernels use internally).
//! Feeding it a deliberately broken WGSL file must surface as a caught
//! `FuzzGpuError::ShaderError` — not a panic — and a valid shader must still
//! register successfully.

use fuzzgpu_core::gpu::{FuzzGpuError, GpuEngine};

fn engine_or_skip() -> Option<std::sync::Arc<GpuEngine>> {
    match GpuEngine::get() {
        Ok(e) => Some(e),
        Err(e) => {
            // Same env-var contract as the lib suite: CI with a software
            // Vulkan adapter must fail instead of skip.
            if std::env::var("FUZZGPU_REQUIRE_GPU")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false)
            {
                panic!("FUZZGPU_REQUIRE_GPU is set but no usable GPU device: {e}");
            }
            eprintln!("skipping GPU test (no usable device): {e}");
            None
        }
    }
}

/// The broken WGSL lives in a fixture file so the test literally feeds a file
/// through the API, matching how a user would register a shader.
const BROKEN: &str = include_str!("fixtures/broken.wgsl");
const VALID: &str = r#"
@compute @workgroup_size(1)
fn main() {}
"#;

#[test]
fn broken_wgsl_file_returns_shader_error() {
    let engine = match engine_or_skip() {
        Some(e) => e,
        None => return,
    };
    let layout = engine
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            // wgpu 30: bind group layouts are Option-wrapped; push constants are
            // now `immediate_size` (0 = none).
            bind_group_layouts: &[],
            immediate_size: 0,
        });

    match engine.build_compute_pipeline("broken kernel", BROKEN, &layout) {
        Err(FuzzGpuError::ShaderError(msg)) => {
            assert!(
                !msg.is_empty(),
                "shader error must carry the wgpu diagnostic"
            );
        }
        other => panic!("expected FuzzGpuError::ShaderError, got: {other:?}"),
    }
}

#[test]
fn valid_wgsl_registers_successfully() {
    let engine = match engine_or_skip() {
        Some(e) => e,
        None => return,
    };
    let layout = engine
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            // wgpu 30: bind group layouts are Option-wrapped; push constants are
            // now `immediate_size` (0 = none).
            bind_group_layouts: &[],
            immediate_size: 0,
        });

    let pipeline = engine
        .build_compute_pipeline("valid kernel", VALID, &layout)
        .expect("valid WGSL must register without error");
    drop(pipeline);
}
