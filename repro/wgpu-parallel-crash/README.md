# wgpu parallel-dispatch crash — investigation findings

Crash under concurrent compute dispatch on a shared `wgpu::Device`
(gfx-rs/wgpu#10085).

- Upstream issue: https://github.com/gfx-rs/wgpu/issues/10085
- Related upstream issue: https://github.com/gfx-rs/wgpu/issues/5270
- Vulkan loader bug: https://github.com/KhronosGroup/Vulkan-Loader/issues/1863
- Workaround PR: https://github.com/gfx-rs/wgpu/pull/10121

---

## Root cause: Vulkan loader use-after-free in `vkSetDebugUtilsObjectNameEXT`

Native minidump analysis (`!analyze -v` + disassembly + register dump) pinpoints
the crash to a **use-after-free in the Vulkan loader** (vulkan-1.dll), NOT in
the Intel driver, NOT in wgpu-core, and NOT in `Queue::drop`.

### Crash signature

```
Exception: 0xC0000005 (ACCESS_VIOLATION)
Faulting address: 0xFEEEFEEEFEEEFEEE  (freed heap fill pattern)
Module: vulkan-1.dll  (Vulkan loader v1.4.313.0, offset +0x41fc4)
```

### Call stack (at fault)

```
GpuEngine::readback
  → wgpu::Queue::submit
    → Queue::submit
      → open_pass
        → begin_encoding
          → DeviceShared::set_object_name
            → ash::set_debug_utils_object_name
              → vkSetDebugUtilsObjectNameEXT
                → loader_get_icd_and_device  ← CRASH HERE
```

### What happens

The Vulkan loader's handle-ownership walk in `vkSetDebugUtilsObjectNameEXT`
dereferences a freed node (fill pattern `0xFEEEFEEE`). The loader's internal
handle table has a use-after-free: it holds a pointer to a device object that
has already been freed.

### Evidence

1. **6 native minidumps** captured early in the session, all with identical
   signature (ACCESS_VIOLATION at freed-heap address, same module offset)
2. **No wgpu-core frames** at the fault — `Queue::drop` never runs
   (static `OnceLock<Arc<GpuEngine>>` is never dropped)
3. **Same crash pattern** as gfx-rs/wgpu#5270 (NVIDIA, MoltenVK, AMD — Feb 2024)
4. **Same crash** as KhronosGroup/Vulkan-Loader#1863 (AMD + llvmpipe, Feb 2026)
5. **`InstanceFlags::DISCARD_HAL_LABELS`** prevents the crash by skipping the
   `set_object_name` call entirely (confirmed in #5270, implemented in PR #10121)

---

## Crash rate

Intermittent/environmental, NOT a fixed rate:

- 6 native crashes early in the session (~0.26% at 8 threads, ~10% at 32 threads)
- **0 crashes** in ~260 subsequent runs (driver state cycling)
- NVIDIA T4: 300/300 clean per #5270 — consistent with this being the same
  loader bug that affects specific driver/loader combos under concurrent use

---

## Workaround

PR #10121 adds `InstanceFlags::DISCARD_HAL_LABELS` support to wgpu-hal.
Setting this flag skips the `set_object_name` call entirely, avoiding the
loader's buggy code path. Requires a code change (not env-var controllable
with `Instance::default()`).

---

## What fuzzgpu's failures actually were (test-isolation bugs, fixed)

The 70-84% failure rates under the old dispatch-lock bypass were **fuzzgpu's
own test-isolation bugs**, NOT wgpu crashes:

1. **Threshold-override race**: concurrent tests stealing `gpu_threshold_override`
   mid-dispatch, silently routing fault-injection tests to CPU
2. **Fault-injection tests**: asserting "expected Timeout/BufferError" when the
   concurrent override theft routed to CPU, returning Ok instead

All backtraces were fuzzgpu frames; none contained wgpu-core. Fixed in v0.2.0
with `GPU_THRESHOLD_TEST_LOCK` and `force_gpu_threshold()` RAII guards.

---

## Investigation timeline

| Date | Finding |
|---|---|
| 2024-01 | First observed on wgpu 24.0.5, Intel Iris Xe, Windows 11 |
| 2026-08-16 | Migrated to wgpu 30.0.0 — crash persists |
| 2026-08-18 | WSL/lavapipe test — 70% exit-101 (fuzzgpu test bugs, not wgpu) |
| 2026-08-18 | Dispatch-lock workaround removed (default concurrent) |
| 2026-08-20 | Native minidump analysis: Vulkan loader use-after-free identified |
| 2026-08-20 | Matches gfx-rs/wgpu#5270 + KhronosGroup/Vulkan-Loader#1863 |
| 2026-08-20 | PR #10121 opened (DISCARD_HAL_LABELS workaround) |

---

## Environment

- Intel(R) Iris(R) Xe Graphics (Tiger Lake iGPU, 11th gen Core i7)
- Windows 11, Intel driver 32.0.101.7088 (June 2026)
- wgpu 30.0.0, Vulkan backend (wgpu::Instance::default() ignores WGPU_BACKEND)
