use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
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

/// Whether the dispatch-serialization workaround is bypassed. Shared by the
/// test-only lock below and the production [`GpuEngine::dispatch_lock`] so
/// `FUZZGPU_SKIP_DISPATCH_LOCK=1` disables BOTH — the repro harness needs the
/// bypass to reproduce upstream gfx-rs/wgpu#10085 under real concurrency.
pub(crate) fn dispatch_lock_bypass() -> bool {
    std::env::var("FUZZGPU_SKIP_DISPATCH_LOCK")
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
    if dispatch_lock_bypass() {
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

/// Slot ids for the shared [`BufferPool`] — the same layout every kernel uses
/// (offsets/chars for the two input lists, results, readback staging, and the
/// per-dispatch uniform).
pub(crate) const SLOT_OFFSETS_A: usize = 0;
pub(crate) const SLOT_CHARS_A: usize = 1;
pub(crate) const SLOT_OFFSETS_B: usize = 2;
pub(crate) const SLOT_CHARS_B: usize = 3;
pub(crate) const SLOT_RESULTS: usize = 4;
pub(crate) const SLOT_STAGING: usize = 5;
pub(crate) const SLOT_PARAMS: usize = 6;

/// Reusable GPU buffer arena shared by all kernels.
///
/// Every fuzzgpu dispatch is fully synchronous (submit → map → poll → read →
/// unmap) before it returns, so a buffer that was just read back is idle and
/// safe to reuse for the next dispatch. Keeping buffers alive across calls
/// removes the per-call `create_buffer` cost — the dominant fixed overhead for
/// small/medium batches (each allocation round-trips through the driver and
/// gpu-allocator). Growth is geometric to amortize reallocation.
pub(crate) struct BufferPool {
    slots: Vec<Option<(wgpu::Buffer, u64)>>,
}

impl BufferPool {
    pub(crate) fn new() -> Self {
        BufferPool { slots: Vec::new() }
    }

    /// Ensure slot `id` holds at least `needed` bytes, creating or growing it
    /// geometrically. Call before [`Self::get`] / [`Self::write`].
    pub(crate) fn ensure(
        &mut self,
        device: &wgpu::Device,
        id: usize,
        needed: u64,
        usage: wgpu::BufferUsages,
        label: &str,
    ) {
        if self.slots.len() <= id {
            self.slots.resize(id + 1, None);
        }
        let current = self.slots[id].as_ref().map(|(_, cap)| *cap).unwrap_or(0);
        if current >= needed {
            return;
        }
        // Geometric growth: at least 2x the previous capacity, floor 1 KiB.
        let new_cap = needed.max(1024).max(current * 2);
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: new_cap,
            usage,
            mapped_at_creation: false,
        });
        self.slots[id] = Some((buf, new_cap));
    }

    /// Borrow the buffer for slot `id` (must have been `ensure`d).
    pub(crate) fn get(&self, id: usize) -> &wgpu::Buffer {
        &self.slots[id]
            .as_ref()
            .expect("BufferPool::get called before ensure")
            .0
    }

    /// Upload `data` into slot `id` (the buffer must have `COPY_DST` usage).
    pub(crate) fn write(&self, queue: &wgpu::Queue, id: usize, data: &[u8]) {
        queue.write_buffer(self.get(id), 0, data);
    }
}

pub struct GpuEngine {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: GpuInfo,
    pub max_buffer_size: u64,
    pub max_storage_buffer_binding_size: u32,
    /// Serializes GPU dispatch across threads. The upstream wgpu/driver crash
    /// (gfx-rs/wgpu#10085) triggers under >=3 concurrent dispatchers on the
    /// shared device (Intel iGPUs, DX12/Vulkan) — e.g. two Python threads
    /// calling the GIL-releasing GPU bindings simultaneously. Every public GPU
    /// entry point holds this lock for the duration of its dispatch + readback,
    /// so at most one submission is ever in flight. `FUZZGPU_SKIP_DISPATCH_LOCK`
    /// bypasses it (repro harness only).
    dispatch_lock: std::sync::Mutex<()>,
}

static GLOBAL_ENGINE: OnceLock<Arc<GpuEngine>> = OnceLock::new();
static CPU_ONLY_FLAG: AtomicBool = AtomicBool::new(false);
static ENV_CPU_CHECK: OnceLock<bool> = OnceLock::new();

/// User override for the GPU dispatch threshold (pairs below which a batch is
/// routed to CPU). `None` = auto-select from the adapter (see
/// [`GpuEngine::auto_gpu_threshold`]). Set via [`GpuEngine::set_gpu_threshold`]
/// or the Python/JS `set_gpu_threshold(None)` API. A `Mutex` (not `OnceLock`)
/// so the value can be changed at runtime.
static GPU_THRESHOLD_OVERRIDE: std::sync::Mutex<Option<usize>> = std::sync::Mutex::new(None);

/// Diagnostics: how many pairs were routed to GPU vs CPU by the most recent
/// GPU-eligible call (best-effort, for `hardware_info`/debugging — not a
/// synchronization contract).
static LAST_ROUTING_GPU: AtomicUsize = AtomicUsize::new(0);
static LAST_ROUTING_CPU: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub name: String,
    pub backend: String,
    /// Adapter device type (DiscreteGpu / IntegratedGpu / VirtualGpu / Cpu / Other).
    /// Drives the auto-routing threshold: discrete GPUs get a low threshold
    /// (GPU wins early), integrated and software (virtual) GPUs get
    /// conservative thresholds where Rayon is usually faster.
    pub device_type: String,
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

    /// Override the GPU dispatch threshold (pairs below which a batch is routed
    /// to CPU). `None` restores automatic selection from the adapter.
    ///
    /// The auto value (see [`Self::auto_gpu_threshold`]) is a good default, but
    /// callers on a known machine — e.g. a workstation with a discrete GPU and
    /// a weak CPU — may want to force GPU routing below the auto threshold, or
    /// force CPU routing on a software renderer.
    pub fn set_gpu_threshold(threshold: Option<usize>) {
        *GPU_THRESHOLD_OVERRIDE.lock().unwrap_or_else(|e| e.into_inner()) = threshold;
    }

    /// The user-supplied threshold override, if any.
    pub fn gpu_threshold_override() -> Option<usize> {
        *GPU_THRESHOLD_OVERRIDE.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The auto-selected GPU dispatch threshold for a *hypothetical* adapter
    /// class — used before any engine exists and by the Python diagnostics.
    ///
    /// - Discrete GPUs: low threshold (64) — fast PCIe/NVLink sync and high
    ///   shader throughput make the GPU win even for modest batches.
    /// - Integrated GPUs: 500 (the historical default) — the per-dispatch
    ///   round-trip (~1 ms) dominates below this, and Rayon usually beats the
    ///   iGPU's DP kernels.
    /// - Virtual/software GPUs (lavapipe, llvmpipe, SwiftShader): 2000 — a
    ///   software rasterizer almost never beats 8 CPU cores.
    /// - CPU adapters: `usize::MAX` (never route to GPU).
    pub fn auto_gpu_threshold(device_type: &str) -> usize {
        match device_type {
            "DiscreteGpu" => 64,
            "IntegratedGpu" => 500,
            "VirtualGpu" => 2000,
            "Cpu" => usize::MAX,
            _ => 500,
        }
    }

    /// The effective GPU dispatch threshold for this engine: the user override
    /// if set, else auto-selected from the adapter's device type.
    pub fn effective_gpu_threshold(&self) -> usize {
        match Self::gpu_threshold_override() {
            Some(t) => t,
            None => Self::auto_gpu_threshold(&self.info.device_type),
        }
    }

    /// Effective GPU threshold for a *specific metric*. The auto thresholds
    /// are tuned for the Myers bit-vector Levenshtein kernel, which wins on
    /// iGPUs at scale; the O(m·w) Jaro bitmap kernel and the Lowrance-Wagner
    /// Damerau SLM kernel lose to the SIMD CPU path on integrated GPUs at
    /// every measured scale (Iris Xe: Jaro ~2×, Damerau ~6× slower at 50k
    /// pairs), so auto-routing never dispatches them there. On discrete GPUs
    /// the per-pair compute advantage flips, but the ~0.5 ms sync round-trip
    /// still demands a much larger batch than Myers — `discrete_factor` scales
    /// the (already low) discrete auto threshold accordingly. An explicit
    /// [`Self::set_gpu_threshold`] override always wins.
    pub fn metric_gpu_threshold(&self, discrete_factor: usize) -> usize {
        match Self::gpu_threshold_override() {
            Some(t) => t,
            None => match self.info.device_type.as_str() {
                // dGPU: auto threshold (64) × factor, e.g. 16 → 1024 pairs for
                // Jaro, 32 → 2048 for Damerau.
                "DiscreteGpu" => Self::auto_gpu_threshold("DiscreteGpu")
                    .saturating_mul(discrete_factor),
                // Integrated, virtual/software, CPU, unknown: never auto-route
                // these kernels (CPU SIMD wins at every measured scale).
                _ => usize::MAX,
            },
        }
    }

    /// The effective GPU dispatch threshold for the *current* engine (or the
    /// auto default if no engine exists yet). Convenience for diagnostics.
    pub fn current_gpu_threshold() -> usize {
        if let Some(engine) = GLOBAL_ENGINE.get() {
            engine.effective_gpu_threshold()
        } else if let Ok(engine) = Self::get() {
            engine.effective_gpu_threshold()
        } else {
            Self::auto_gpu_threshold("IntegratedGpu")
        }
    }

    /// Record a routing decision for the diagnostics API: `gpu_pairs` pairs were
    /// dispatched to the GPU and `cpu_pairs` were computed on CPU (oversized or
    /// below-threshold) by the most recent GPU-eligible call.
    pub fn record_routing(gpu_pairs: usize, cpu_pairs: usize) {
        LAST_ROUTING_GPU.store(gpu_pairs, Ordering::Relaxed);
        LAST_ROUTING_CPU.store(cpu_pairs, Ordering::Relaxed);
    }

    /// Last routing decision: `(gpu_pairs, cpu_pairs)` of the most recent
    /// GPU-eligible call. Lets callers confirm a "GPU mode" call actually
    /// dispatched to the GPU (it is *not* silently CPU-routed).
    pub fn last_routing() -> (usize, usize) {
        (LAST_ROUTING_GPU.load(Ordering::Relaxed), LAST_ROUTING_CPU.load(Ordering::Relaxed))
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

    /// Submit an encoder and synchronously read back `size` bytes from the
    /// pool's staging slot (the encoder must have copied results into it).
    ///
    /// This is the single-sync primitive the batched APIs use: N queued
    /// dispatches are recorded into one encoder, submitted once, and read back
    /// with one map/poll round-trip — amortizing the per-call sync cost that
    /// dominates small dispatches.
    pub(crate) fn readback(
        &self,
        encoder: wgpu::CommandEncoder,
        pool: &BufferPool,
        size: u64,
    ) -> Result<Vec<u8>> {
        self.submit(encoder);
        let staging = pool.get(SLOT_STAGING);
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        self.map_readback(&slice, move |r| { let _ = tx.send(r); });
        self.poll();
        rx.recv_timeout(Self::readback_timeout())
            .map_err(|_| FuzzGpuError::Timeout("GPU readback timed out after 10s".into()))?
            .map_err(|e| FuzzGpuError::BufferError(format!("GPU buffer map failed: {e}")))?;
        let data = slice
            .get_mapped_range()
            .map_err(|e| FuzzGpuError::BufferError(format!("GPU buffer map range failed: {e}")))?;
        let bytes = data[..size as usize].to_vec();
        drop(data);
        staging.unmap();
        Ok(bytes)
    }

    /// Take the production dispatch lock (see the field doc). Returns `None`
    /// when `FUZZGPU_SKIP_DISPATCH_LOCK` is set so the repro harness can
    /// reproduce upstream #10085 under real concurrency.
    pub(crate) fn dispatch_lock(&self) -> Option<std::sync::MutexGuard<'_, ()>> {
        if dispatch_lock_bypass() {
            return None;
        }
        Some(self.dispatch_lock.lock().unwrap_or_else(|e| e.into_inner()))
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
            device_type: format!("{:?}", adapter_info.device_type),
        };

        // Query adapter limits dynamically rather than hardcoding static limits
        let adapter_limits = adapter.limits();
        let target_storage_size = (128 * 1024 * 1024).min(adapter_limits.max_storage_buffer_binding_size);
        let target_buffer_size = (128 * 1024 * 1024).min(adapter_limits.max_buffer_size);

        let required_limits = wgpu::Limits {
            max_storage_buffer_binding_size: target_storage_size,
            max_buffer_size: target_buffer_size,
            // 32 KiB target (up from 16 KiB): the Damerau kernel keeps a full
            // Lowrance-Wagner matrix per pair in workgroup shared memory
            // (~22.6 KiB at workgroup size 4 with a 32-char cap). Requesting
            // min(adapter, 32 KiB) never exceeds the adapter, so device
            // creation cannot fail from this change; devices below the
            // Damerau budget simply route that kernel to CPU.
            max_compute_workgroup_storage_size: 32768.min(adapter_limits.max_compute_workgroup_storage_size),
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
            dispatch_lock: std::sync::Mutex::new(()),
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

    /// The auto-routing threshold must reflect the adapter class: discrete
    /// GPUs route to GPU early (low threshold), integrated and software GPUs
    /// are conservative, CPU adapters never route.
    #[test]
    fn test_auto_threshold_by_device_type() {
        assert_eq!(GpuEngine::auto_gpu_threshold("DiscreteGpu"), 64);
        assert_eq!(GpuEngine::auto_gpu_threshold("IntegratedGpu"), 500);
        assert_eq!(GpuEngine::auto_gpu_threshold("VirtualGpu"), 2000);
        assert_eq!(GpuEngine::auto_gpu_threshold("Cpu"), usize::MAX);
        assert_eq!(GpuEngine::auto_gpu_threshold("Other"), 500);
    }

    /// `set_gpu_threshold` must override the auto value and be resettable to
    /// auto (`None`) at runtime (a `Mutex`, not a one-shot `OnceLock`). The
    /// override is global, so this test holds the GPU test lock — serializing
    /// it against every other GPU test, which would otherwise see the
    /// temporary override (e.g. a 1000-pair fault-injection test routed to
    /// CPU instead of dispatching) — and restores `None` before returning.
    #[test]
    fn test_set_gpu_threshold_override_and_reset() {
        let _gpu_guard = gpu_test_lock();
        GpuEngine::set_gpu_threshold(Some(1234));
        assert_eq!(GpuEngine::gpu_threshold_override(), Some(1234));
        if let Ok(engine) = GpuEngine::get() {
            assert_eq!(engine.effective_gpu_threshold(), 1234);
        }
        // Resettable: back to auto (no override).
        GpuEngine::set_gpu_threshold(None);
        assert_eq!(GpuEngine::gpu_threshold_override(), None);
        if let Ok(engine) = GpuEngine::get() {
            assert_eq!(
                engine.effective_gpu_threshold(),
                GpuEngine::auto_gpu_threshold(&engine.info.device_type),
                "effective threshold must follow the adapter's auto value after reset"
            );
        }
    }

    /// `record_routing` / `last_routing` must surface the most recent routing
    /// decision so callers can confirm GPU mode actually dispatched.
    #[test]
    fn test_last_routing_diagnostics() {
        GpuEngine::record_routing(0, 0);
        assert_eq!(GpuEngine::last_routing(), (0, 0));
        GpuEngine::record_routing(750, 250);
        assert_eq!(GpuEngine::last_routing(), (750, 250));
    }
}
