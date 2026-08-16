use std::sync::{Arc, OnceLock};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum FuzzGpuError {
    #[error("GPU not available: {0}")]
    NoDevice(String),
    #[error("Shader compilation failed: {0}")]
    ShaderError(String),
    #[error("Buffer error: {0}")]
    BufferError(String),
    #[error("Pipeline error: {0}")]
    PipelineError(String),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

pub type Result<T> = std::result::Result<T, FuzzGpuError>;

pub struct GpuEngine {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: GpuInfo,
}

static GLOBAL_ENGINE: OnceLock<Arc<GpuEngine>> = OnceLock::new();

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub name: String,
    pub backend: String,
}

impl GpuEngine {
    pub fn get() -> Result<Arc<Self>> {
        if let Some(engine) = GLOBAL_ENGINE.get() {
            return Ok(Arc::clone(engine));
        }

        let engine = pollster::block_on(Self::new_inner())?;
        let _ = GLOBAL_ENGINE.set(Arc::clone(&engine));
        Ok(engine)
    }

    async fn new_inner() -> Result<Arc<Self>> {
        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| FuzzGpuError::NoDevice("No GPU adapter found".into()))?;

        let info = GpuInfo {
            name: adapter.get_info().name.clone(),
            backend: format!("{:?}", adapter.get_info().backend),
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("fuzzgpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    max_storage_buffer_binding_size: 128 * 1024 * 1024,
                    max_buffer_size: 128 * 1024 * 1024,
                    max_compute_workgroup_storage_size: 16 * 1024,
                    max_compute_invocations_per_workgroup: 256,
                    ..Default::default()
                },
                memory_hints: wgpu::MemoryHints::Performance,
            }, None)
            .await
            .map_err(|e| FuzzGpuError::NoDevice(e.to_string()))?;

        log::info!("fuzzgpu: Using {} ({})", info.name, info.backend);

        Ok(Arc::new(Self {
            device,
            queue,
            info,
        }))
    }
}
