use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use thiserror::Error;

#[cfg(test)]
use std::cell::Cell;

// Test-only fault injection state, thread-local so only the arming test
// thread is affected (tests run in parallel and share the global engine).
// See docs/GPU_TESTING.md for how to use these hooks and the conventions
// every GPU test must follow.
#[cfg(test)]
thread_local! {
    static TEST_INJECT_READBACK_TIMEOUT: Cell<bool> = const { Cell::new(false) };
    static TEST_INJECT_SMALL_BUFFER: Cell<bool> = const { Cell::new(false) };
    static TEST_INJECT_SHADER_ERROR: Cell<bool> = const { Cell::new(false) };
}

/// Test gate: when `FUZZGPU_REQUIRE_GPU` is set, GPU tests must not skip — a
/// missing device is a hard failure instead. CI sets this on runners that
/// install a (software) Vulkan adapter so the GPU paths are genuinely
/// exercised on every push rather than silently skipped.
#[cfg(test)]
pub(crate) fn require_gpu() -> bool {
    std::env::var("FUZZGPU_REQUIRE_GPU")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Serialize GPU access across tests. This is a workaround for a wgpu/driver
/// crash on Intel Iris Xe under >=3 concurrent dispatchers on the shared
/// device (heap corruption on DX12, segfault on Vulkan) — see
/// `repro/wgpu-parallel-crash` and upstream gfx-rs/wgpu#10085.
///
/// Setting `FUZZGPU_SKIP_DISPATCH_LOCK=1` bypasses the workaround so the
/// underlying crash can be reproduced / bisected in CI or locally.
#[cfg(test)]
pub(crate) fn gpu_test_lock() -> Option<std::sync::MutexGuard<'static, ()>> {
    if std::env::var("FUZZGPU_SKIP_DISPATCH_LOCK")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        return None;
    }
    Some(GPU_TEST_DISPATCH_LOCK.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Arm the readback-timeout fault on the current thread: `map_async`
/// registrations are skipped (so no poll on *any* thread can complete the
/// mapping) and the readback deterministically fails with
/// `FuzzGpuError::Timeout` via `recv_timeout`.
#[cfg(test)]
pub(crate) fn arm_readback_timeout_fault() {
    TEST_INJECT_READBACK_TIMEOUT.with(|c| c.set(true));
}

/// Disarm the readback-timeout fault on the current thread.
#[cfg(test)]
pub(crate) fn disarm_readback_timeout_fault() {
    TEST_INJECT_READBACK_TIMEOUT.with(|c| c.set(false));
}

/// Arm the small-buffer fault on the current thread: the effective max buffer
/// size shrinks so any real input trips the `FuzzGpuError::BufferError` branch.
#[cfg(test)]
pub(crate) fn arm_small_buffer_fault() {
    TEST_INJECT_SMALL_BUFFER.with(|c| c.set(true));
}

/// Disarm the small-buffer fault on the current thread.
#[cfg(test)]
pub(crate) fn disarm_small_buffer_fault() {
    TEST_INJECT_SMALL_BUFFER.with(|c| c.set(false));
}

/// Arm the shader-error fault on the current thread: kernel initialization
/// compiles deliberately invalid WGSL so shader validation fails and surfaces
/// as `FuzzGpuError::ShaderError` instead of a panic.
#[cfg(test)]
pub(crate) fn arm_shader_error_fault() {
    TEST_INJECT_SHADER_ERROR.with(|c| c.set(true));
}

/// Disarm the shader-error fault on the current thread.
#[cfg(test)]
pub(crate) fn disarm_shader_error_fault() {
    TEST_INJECT_SHADER_ERROR.with(|c| c.set(false));
}

/// The WGSL source to compile for a kernel shader. In test builds the
/// shader-error fault substitutes deliberately invalid WGSL to exercise shader
/// validation errors; in production this is a pass-through.
pub(crate) fn effective_shader_source(real: &'static str) -> std::borrow::Cow<'static, str> {
    #[cfg(test)]
    if TEST_INJECT_SHADER_ERROR.with(|c| c.get()) {
        return std::borrow::Cow::Owned("this is deliberately invalid wgsl !!!".into());
    }
    std::borrow::Cow::Borrowed(real)
}

// Serializes GPU dispatch across tests. The GPU is one shared hardware
// resource, and hammering it with 3+ concurrent dispatches across many rapid
// process runs trips a heap corruption / segfault in the wgpu stack on some
// hardware (observed on Intel Iris Xe with both DX12 and Vulkan backends;
// reproduces identically with tests that contain no fault-injection code).
// Serializing access — standard practice for shared-device GPU test suites —
// keeps parallel test execution stable. Production code is unaffected.
#[cfg(test)]
pub(crate) static GPU_TEST_DISPATCH_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

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
static ENV_CPU_CHECK: OnceLock<bool> = OnceLock::new();

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

    /// Check whether CPU-only mode is active (cached, zero syscall overhead in loops).
    pub fn is_cpu_only() -> bool {
        if CPU_ONLY_FLAG.load(Ordering::Relaxed) {
            return true;
        }
        *ENV_CPU_CHECK.get_or_init(|| {
            if let Ok(v) = std::env::var("FUZZGPU_USE_CPU") {
                v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes")
            } else {
                false
            }
        })
    }

    /// Returns whether a GPU device is available and ready for compute.
    pub fn is_available() -> bool {
        if Self::is_cpu_only() {
            return false;
        }
        Self::get().is_ok()
    }

    /// Submit a finished command encoder to the GPU queue.
    ///
    /// Runs unconditionally — including in fault-injected tests — so the
    /// device's normal maintenance paths stay healthy. (The fault only skips
    /// the readback `map_async` registration in [`Self::map_readback`].)
    pub fn submit(&self, encoder: wgpu::CommandEncoder) {
        self.queue.submit(Some(encoder.finish()));
    }

    /// Block until the device is idle, driving pending `map_async` callbacks.
    pub fn poll(&self) {
        // wgpu 26+ renamed `Maintain::Wait` to `PollType::wait_indefinitely`
        // and made `poll` fallible; a plain wait-for-idle can only fail on an
        // invalid submission index, which we never supply, so ignore the error.
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }

    /// Compile a WGSL compute shader and build a compute pipeline — the public
    /// kernel-registration primitive for custom kernels.
    ///
    /// Both shader-module parsing and pipeline linking run inside a wgpu
    /// validation error scope, so a broken shader surfaces as
    /// [`FuzzGpuError::ShaderError`] (with the full wgpu diagnostic) instead of
    /// a panic. `entry_point` is assumed to be `"main"`.
    pub fn build_compute_pipeline(
        &self,
        label: &str,
        source: &str,
        layout: &wgpu::PipelineLayout,
    ) -> Result<wgpu::ComputePipeline> {
        // wgpu 26+ error scopes are RAII guards: push returns a guard whose
        // `pop()` resolves to the captured error.
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);

        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        if let Some(err) = pollster::block_on(scope.pop()) {
            return Err(FuzzGpuError::ShaderError(format!(
                "GPU shader validation failed: {err}"
            )));
        }
        Ok(pipeline)
    }

    /// Register an asynchronous readback mapping.
    ///
    /// Test-only fault injection: when armed on the current thread (see
    /// [`arm_readback_timeout_fault`]), the mapping is *never registered*.
    /// This is what makes the timeout deterministic — wgpu fires the callback
    /// from whichever thread first polls the (globally shared) device, so
    /// merely skipping the submission would still let a parallel test's `poll`
    /// complete our map. Skipping registration means no thread can complete
    /// it, and `recv_timeout` deterministically yields `FuzzGpuError::Timeout`.
    pub fn map_readback(
        &self,
        slice: &wgpu::BufferSlice<'_>,
        callback: impl FnOnce(std::result::Result<(), wgpu::BufferAsyncError>) + Send + 'static,
    ) {
        #[cfg(test)]
        if TEST_INJECT_READBACK_TIMEOUT.with(|c| c.get()) {
            return;
        }
        slice.map_async(wgpu::MapMode::Read, callback);
    }

    /// How long a GPU readback may take before `FuzzGpuError::Timeout` is
    /// returned.
    pub fn readback_timeout() -> Duration {
        Duration::from_secs(10)
    }

    /// The maximum buffer size to allow for GPU allocations.
    ///
    /// Test-only fault injection: when the small-buffer fault is armed on the
    /// current thread, this shrinks to a tiny value so any real input trips the
    /// `FuzzGpuError::BufferError` branch deterministically. Production returns
    /// the real device limit.
    pub fn max_buffer_size_effective(&self) -> u64 {
        #[cfg(test)]
        if TEST_INJECT_SMALL_BUFFER.with(|c| c.get()) {
            return 1024;
        }
        self.max_buffer_size
    }

    /// Get or lazily initialize the singleton GPU engine.
    pub fn get() -> Result<Arc<Self>> {
        if Self::is_cpu_only() {
            return Err(FuzzGpuError::NoDevice("CPU-only mode is active (FUZZGPU_USE_CPU)".into()));
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
                apply_limit_buckets: false,
            })
            .await
        {
            Ok(a) => a,
            Err(_) => instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::None,
                    compatible_surface: None,
                    force_fallback_adapter: true,
                    apply_limit_buckets: false,
                })
                .await
                .map_err(|e| FuzzGpuError::NoDevice(format!("No GPU adapter found: {e}")))?,
        };

        let adapter_info = adapter.get_info();
        let info = GpuInfo {
            name: adapter_info.name.clone(),
            backend: format!("{:?}", adapter_info.backend),
        };

        // Query adapter limits dynamically rather than hardcoding static limits
        let adapter_limits = adapter.limits();
        let target_storage_size = (128 * 1024 * 1024).min(adapter_limits.max_storage_buffer_binding_size);
        let target_buffer_size = (128 * 1024 * 1024).min(adapter_limits.max_buffer_size);

        let required_limits = wgpu::Limits {
            max_storage_buffer_binding_size: target_storage_size,
            max_buffer_size: target_buffer_size,
            max_compute_workgroup_storage_size: 16384.min(adapter_limits.max_compute_workgroup_storage_size),
            max_compute_invocations_per_workgroup: 256.min(adapter_limits.max_compute_invocations_per_workgroup),
            ..Default::default()
        };

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("fuzzgpu device"),
                    required_features: wgpu::Features::empty(),
                    required_limits,
                    experimental_features: wgpu::ExperimentalFeatures::disabled(),
                    // Explicit rather than `Default::default()`: `Performance` is the
                    // default variant, but stating it makes the intent (prefer
                    // throughput over memory footprint) unambiguous.
                    memory_hints: wgpu::MemoryHints::Performance,
                    trace: wgpu::Trace::Off,
                },
            )
            .await
            .map_err(|e| FuzzGpuError::NoDevice(format!("Failed to create GPU device: {}", e)))?;

        let engine = Arc::new(Self {
            device,
            queue,
            info,
            max_buffer_size: target_buffer_size as u64,
            // wgpu 30 widened this limit to u64; the 128 MiB cap fits u32.
            max_storage_buffer_binding_size: target_storage_size as u32,
        });

        Ok(engine)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Validates adapter discovery, device creation, and the queried hardware limits.
    #[test]
    fn test_engine_initialization() {
        let _gpu_guard = gpu_test_lock();
        if GpuEngine::is_cpu_only() {
            if require_gpu() {
                panic!("FUZZGPU_REQUIRE_GPU is set but CPU-only mode is active (FUZZGPU_USE_CPU)");
            }
            eprintln!("skipping GPU test (CPU-only mode active)");
            return;
        }
        match GpuEngine::get() {
            Ok(engine) => {
                assert!(!engine.info.name.is_empty(), "GPU name must be populated");
                assert!(!engine.info.backend.is_empty(), "GPU backend must be populated");
                assert!(engine.max_buffer_size > 0, "max_buffer_size must be > 0");
                assert!(engine.max_storage_buffer_binding_size > 0, "max_storage_buffer_binding_size must be > 0");
                assert!(engine.max_storage_buffer_binding_size as u64 <= engine.max_buffer_size);
            }
            Err(e) => {
                if require_gpu() {
                    panic!("FUZZGPU_REQUIRE_GPU is set but no usable GPU device: {}", e);
                }
                eprintln!("skipping GPU test (no usable device): {}", e);
            }
        }
    }
}
