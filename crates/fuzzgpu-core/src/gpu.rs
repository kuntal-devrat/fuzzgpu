use std::sync::atomic::{AtomicBool, Ordering};
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
    #[error("GPU operation timed out: {0}")]
    Timeout(String),
    #[error("Invalid input or parameters: {0}")]
    InvalidInput(String),
}

pub type Result<T> = std::result::Result<T, FuzzGpuError>;

pub struct GpuEngine {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: GpuInfo,
    pub max_buffer_size: u64,
    pub max_storage_buffer_binding_size: u32,
}

static GLOBAL_ENGINE: OnceLock<Arc<GpuEngine>> = OnceLock::new();
static CPU_ONLY_FLAG: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub name: String,
    pub backend: String,
}

impl GpuEngine {
    /// Enable or disable CPU-only fallback globally.
    pub fn set_cpu_only(cpu_only: bool) {
        CPU_ONLY_FLAG.store(cpu_only, Ordering::SeqCst);
    }

    /// Check whether CPU-only mode is active.
    pub fn is_cpu_only() -> bool {
        if CPU_ONLY_FLAG.load(Ordering::Relaxed) {
            return true;
        }
        if let Ok(v) = std::env::var("FUZZGPU_USE_CPU") {
            if v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes") {
                return true;
            }
        }
        false
    }

    /// Returns whether a GPU device is available and ready for compute.
    pub fn is_available() -> bool {
        if Self::is_cpu_only() {
            return false;
        }
        Self::get().is_ok()
    }

    /// Get or lazily initialize the singleton GPU engine.
    pub fn get() -> Result<Arc<Self>> {
        if Self::is_cpu_only() {
            return Err(FuzzGpuError::NoDevice("CPU-only mode is forced (FUZZGPU_USE_CPU)".into()));
        }

        if let Some(engine) = GLOBAL_ENGINE.get() {
            return Ok(Arc::clone(engine));
        }

        let engine = pollster::block_on(Self::new_inner())?;
        let _ = GLOBAL_ENGINE.set(Arc::clone(&engine));
        Ok(engine)
    }

    async fn new_inner() -> Result<Arc<Self>> {
        let instance = wgpu::Instance::default();

        let adapter = match instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
        {
            Some(a) => a,
            None => instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::None,
                    compatible_surface: None,
                    force_fallback_adapter: true,
                })
                .await
                .ok_or_else(|| FuzzGpuError::NoDevice("No GPU adapter found".into()))?,
        };

        let adapter_info = adapter.get_info();
        let info = GpuInfo {
            name: adapter_info.name.clone(),
            backend: format!("{:?}", adapter_info.backend),
        };

        // Query adapter limits dynamically rather than blindly hardcoding 128MB
        let adapter_limits = adapter.limits();
        let target_storage_size = (128 * 1024 * 1024).min(adapter_limits.max_storage_buffer_binding_size);
        let target_buffer_size = (128 * 1024 * 1024).min(adapter_limits.max_buffer_size);

        let required_limits = wgpu::Limits {
            max_storage_buffer_binding_size: target_storage_size,
            max_buffer_size: target_buffer_size,
            max_compute_workgroup_storage_size: 16 * 1024.min(adapter_limits.max_compute_workgroup_storage_size),
            max_compute_invocations_per_workgroup: 256.min(adapter_limits.max_compute_invocations_per_workgroup),
            ..Default::default()
        };

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("fuzzgpu"),
                    required_features: wgpu::Features::empty(),
                    required_limits,
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| FuzzGpuError::NoDevice(format!("Failed to create GPU device: {}", e)))?;

        if std::env::var("FUZZGPU_DEBUG").is_ok() {
            eprintln!(
                "fuzzgpu: Initialized {} ({}) with max buffer size {} MB",
                info.name,
                info.backend,
                target_buffer_size / (1024 * 1024)
            );
        }

        Ok(Arc::new(Self {
            device,
            queue,
            info,
            max_buffer_size: target_buffer_size,
            max_storage_buffer_binding_size: target_storage_size,
        }))
    }
}
