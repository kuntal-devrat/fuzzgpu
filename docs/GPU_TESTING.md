# GPU Test Fault Injection & Conventions

How to deterministically trigger every failure branch of `FuzzGpuError` in GPU
tests, and the conventions every GPU test must follow. All fault-injection
state and helpers live in `crates/fuzzgpu-core/src/gpu.rs`.

## Quick reference

| `FuzzGpuError` branch | Hook to trigger it | Mechanism |
| --- | --- | --- |
| `Timeout` | `arm_readback_timeout_fault()` / `disarm_readback_timeout_fault()` | readback `map_async` is never registered |
| `BufferError` | `arm_small_buffer_fault()` / `disarm_small_buffer_fault()` | effective max buffer size shrinks to 1 KiB |
| `ShaderError` | `arm_shader_error_fault()` / `disarm_shader_error_fault()` | kernel source replaced with invalid WGSL |
| `NoDevice` | none | CPU-only mode / no adapter — tests skip (or hard-fail with `FUZZGPU_REQUIRE_GPU=1`) |
| `InvalidInput` | none — pass the bad value | out-of-range `p` to `compute_batch` / `compute_matrix` |

The three fault hooks share two properties you can rely on:

1. **`#[cfg(test)]`-only.** They are stripped from production builds, where the
   helpers they feed become trivial pass-throughs (`map_readback` maps,
   `max_buffer_size_effective` returns the real limit,
   `effective_shader_source` returns the real source). Fault hooks must never
   be called from production code.
2. **Thread-local state.** Arming affects only the calling test thread, so
   parallel tests in other threads are untouched and there is no cross-test
   contamination. The state dies with the test thread, but always
   `arm` → exercise → `disarm` around the call being tested regardless.

## Timeout fault (`FuzzGpuError::Timeout`)

```rust
crate::gpu::arm_readback_timeout_fault();
let result = kernel.compute(&pairs); // or compute_matrix / compute_batch
crate::gpu::disarm_readback_timeout_fault();
```

`GpuEngine::map_readback` skips `slice.map_async(...)` entirely when armed.
This is what makes the timeout **deterministic**: wgpu fires the map callback
from whichever thread first polls the (globally shared) device, so merely
skipping the submission would still let a parallel test's `poll` complete our
mapping. Because the mapping is never registered, no thread can complete it.
The callback closure — which owns the channel sender — is dropped when
`map_readback` returns, closing the channel, so the readback's
`rx.recv_timeout(GpuEngine::readback_timeout())` fails *instantly* with
`RecvTimeoutError::Disconnected`, mapped to `FuzzGpuError::Timeout`.

Example tests: `test_batch_readback_timeout_returns_timeout_error` and
`test_matrix_readback_timeout_returns_timeout_error` in both
`levenshtein::gpu_ext::tests` and `jaro::gpu_ext::tests`.

## Small-buffer fault (`FuzzGpuError::BufferError`)

```rust
crate::gpu::arm_small_buffer_fault();
let result = kernel.compute(&pairs); // batch path → Err(BufferError)
crate::gpu::disarm_small_buffer_fault();
```

`GpuEngine::max_buffer_size_effective` returns 1024 when armed instead of the
real device limit. Any real packed input exceeds 1 KiB, so:

- the **batch** path returns `Err(FuzzGpuError::BufferError)` before any
  allocation, and
- the **matrix** path gracefully falls back to CPU results (`Ok`) instead of
  erroring — a distinct branch worth asserting.

Example tests: `test_buffer_size_validation_returns_buffer_error` (batch) and
`test_matrix_oversize_falls_back_to_cpu` (matrix), both kernels.

## Shader-error fault (`FuzzGpuError::ShaderError`)

```rust
crate::gpu::arm_shader_error_fault();
let result = GpuLevenshteinKernel::new_inner(engine); // direct, bypasses the cache
crate::gpu::disarm_shader_error_fault();
```

`effective_shader_source(real)` substitutes deliberately invalid WGSL
(`"this is deliberately invalid wgsl !!!"`) when armed. Kernel construction —
`new_inner`, which now routes through the public
`GpuEngine::build_compute_pipeline` — compiles the source inside a wgpu
validation error scope, so the naga parse failure surfaces as
`Err(FuzzGpuError::ShaderError)` with the full diagnostic instead of a panic.

Two gotchas:

- **Call `new_inner` directly, not `get()`.** Kernels are cached in a global
  `OnceLock`; `get()` returns the already-compiled kernel and would never
  recompile, so the fault would have no effect (and would poison the cache
  for later tests).
- The same `ShaderError` path is also covered end-to-end from the public API
  in `crates/fuzzgpu-core/tests/kernel_registration.rs`, which feeds a real
  broken WGSL file through `GpuEngine::build_compute_pipeline`.

Example tests: `test_shader_validation_error_returns_shader_error` in both
kernel test modules.

## Conventions every GPU test must follow

1. **Hold the dispatch lock first.** GPU tests share one device; 3+ concurrent
   dispatchers across rapid process runs crash the wgpu/driver stack on some
   hardware (heap corruption on DX12, segfault on Vulkan — observed on Intel
   Iris Xe; see `repro/wgpu-parallel-crash` and upstream gfx-rs/wgpu#10085).
   Every GPU test starts with
   `let _gpu_guard = crate::gpu::gpu_test_lock();` so tests still run on
   parallel threads but dispatch one at a time. (`FUZZGPU_SKIP_DISPATCH_LOCK=1`
   bypasses this — reproduction/bisection only, never set it in CI.)

   **Production is protected too.** The test lock is `#[cfg(test)]`-only, but
   the same crash class applies to any multi-threaded caller of the GPU
   bindings (the Python bindings release the GIL around kernel calls, so two
   Python threads can dispatch concurrently). `GpuEngine` therefore carries a
   production `dispatch_lock` that every public GPU entry point
   (`compute` / `compute_matrix` / `compute_batch` / `batch().execute()`, all
   kernels) holds for the duration of its dispatch + readback — at most one
   submission is ever in flight. `test_concurrent_dispatch_is_serialized_and_correct`
   stress-tests this with 8 threads × 20 GPU dispatches. The same
   `FUZZGPU_SKIP_DISPATCH_LOCK` env var disables both locks (repro only).
2. **Skip cleanly without a device.** Acquire the kernel through the module's
   `gpu_kernel_or_skip()` helper; on failure it logs and returns `None`, and
   the test returns early. Under `FUZZGPU_REQUIRE_GPU=1` (set by CI's
   lavapipe job) the same helper panics instead, so a missing device fails the
   job rather than silently passing.
3. **Assert the exact error variant.** `match` on the result and panic with
   the actual error for anything unexpected; never just `.is_err()`.
4. **Arm → exercise → disarm** in the same test, and disarm before any
   assertion that could panic (the fault is thread-local, but leaving it armed
   obscures later failures in the same test).

## Why the faults exist

These branches are unreachable through normal inputs in a healthy environment
(a working GPU never times out, buffers never exceed the limit, shipped
shaders always compile). The hooks make them deterministic and testable so the
error handling — the 10s timeout path, the buffer-limit validation, and the
error-scope shader compilation — is continuously verified instead of being
dead code.
