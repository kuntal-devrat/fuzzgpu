# wgpu parallel-dispatch crash — root cause found and fixed

Originally reported as heap corruption (`STATUS_HEAP_CORRUPTION` 0xC0000374)
and access violations (`STATUS_ACCESS_VIOLATION` 0xC0000005) when 3+ threads
concurrently dispatch compute on a shared `wgpu::Device`.

- Upstream issue: https://github.com/gfx-rs/wgpu/issues/10085
- **Fix PR:** `kuntal-devrat/wgpu` branch `fix/queue-drop-drain-loop` (PR open against `gfx-rs/wgpu:trunk`)
- Status: **root cause identified and fixed** — 0/200 hard crashes on Windows after the patch

---

## Root cause

`Queue::drop` in `wgpu-core/src/device/queue.rs` contained an unconditional
`assert!(queue_empty)` that panicked when a concurrent thread called
`track_submission()` between `wait_for_idle()` returning and `maintain()`
acquiring the life-tracker lock:

```
Queue::drop                          concurrent submitter
─────────────────────────────        ──────────────────────────────────
wait_for_idle()  ← GPU fully idle
                                     lock life_tracker
                                     track_submission(idx)
                                     unlock life_tracker
maintain()
  lock life_tracker
  triage_submissions()  ← sees idx
  queue_empty() → false
unlock life_tracker
assert!(queue_empty)    ← PANIC
```

On Windows the panic propagates through the wgpu callback boundary and is
caught by the OS as `STATUS_ACCESS_VIOLATION` (0xC0000005) or
`STATUS_HEAP_CORRUPTION` (0xC0000374) depending on timing. On Linux it exits
with code 101 (Rust panic). The bug is entirely in `wgpu-core` above the HAL
— it reproduces on lavapipe (pure CPU software Vulkan) with no GPU hardware.

This is a sibling of gfx-rs/wgpu#9958 which fixed the same race class on the
`Device::maintain` path.

---

## Fix

Replace the single `maintain + assert` with a drain loop that re-reads
`last_successful_submission_index` and calls `maintain` until the tracker is
empty. The loop terminates in at most two iterations — see the inline comment
in `queue.rs` for the termination proof.

---

## Evidence

| Test | Before fix | After fix |
|---|---|---|
| Linux lavapipe (CPU-only Vulkan, WSL2) | 14/20 panics (70%) | **0/30 (0%)** |
| Windows / Intel Iris Xe / DX12+Vulkan | ~1 crash per 10–30 runs | **0/200 runs** |
| Windows / Intel Iris Xe / DX12+Vulkan / June 2026 driver | 168/173 crashes | **0/200 runs** |

The lavapipe result is the decisive discriminator: no Intel hardware, no DX12,
no Windows, no driver involved — the crash is in pure `wgpu-core` Rust.

---

## Investigation timeline

| Date | Finding |
|---|---|
| 2024-01 | First observed on wgpu 24.0.5, Intel Iris Xe, Windows 11 |
| 2026-08-16 | Migrated to wgpu 30.0.0 — crash persists, same signature |
| 2026-08-18 | Intel driver update (Jan 2024 → Jun 2026) — same crash rate |
| 2026-08-18 | WSL/lavapipe test — reproduces at 70% on CPU-only renderer |
| 2026-08-18 | Root cause identified: `assert!(queue_empty)` race in `Queue::drop` |
| 2026-08-18 | Fix applied — 0/200 hard crashes on Windows |

---

## Workaround (still active until wgpu fix ships)

fuzzgpu serializes GPU dispatch across test threads via `gpu_test_lock()`.
Remove it when the wgpu fix lands in a crates.io release:

1. Bump `wgpu = "X.Y"` in `Cargo.toml` to the version containing the fix
2. Remove `.cargo/config.toml` patch section
3. Delete `gpu_test_lock()`, `GPU_TEST_DISPATCH_LOCK`, `dispatch_lock_bypass()`
4. Delete `GpuEngine::dispatch_lock` field and `dispatch_lock()` method
5. Remove `FUZZGPU_SKIP_DISPATCH_LOCK` env var handling

---

## The minimal reproducer in this crate

`cargo run --release -- --process-loop 300 --threads 6`

**This crate never reproduced the crash** (~1,500 process runs across variants).
The crash required the full fuzzgpu test harness — specifically the interaction
of fault-injection tests, matrix kernels, and 8 concurrent test threads. The
minimal case was never isolated; root cause analysis via WSL/lavapipe made
further minimization unnecessary.

---

## Environment

- Intel(R) Iris(R) Xe Graphics (Tiger Lake iGPU, 11th gen Core i7)
- Windows 11, Intel driver 32.0.101.7088 (June 2026)
- wgpu 30.0.0, both DX12 and Vulkan backends confirmed
