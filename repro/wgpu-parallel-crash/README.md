# wgpu parallel-dispatch crash on Intel Iris Xe

A wgpu-24.0.5 crash observed in the fuzzgpu GPU test suite: **3+ threads
concurrently dispatching compute on a shared device corrupt the process heap**.
On DX12 this surfaces as `STATUS_HEAP_CORRUPTION` (0xC0000374); on Vulkan as a
segfault (0xC0000005). Serial execution and 2 concurrent dispatchers are
stable.

- Upstream issue: https://github.com/gfx-rs/wgpu/issues/10085
- Affected: wgpu 24.0.5 **and 30.0.0** (migrated and re-tested 2026-08-16),
  Intel Iris Xe (Tiger Lake iGPU), Windows 11
- Status: **reproduced on both wgpu 24.0.5 and wgpu 30.0.0** (lockless test
  suite — `0xC0000374` on both backends under 30), **not yet isolated to a
  minimal case** (this crate is the attempted minimal reproducer and does not
  crash on its own so far)

> **wgpu 30.0.0 re-test (2026-08-16):** the fuzzgpu suite was migrated to
> wgpu 30.0.0 and the lockless loop re-run. The crash persists with the
> identical signature: `0xC0000374` on Vulkan (run 78/120) and DX12
> (runs 42, 72, 116/120); the serialized (locked) suite stays stable. So the
> workaround remains load-bearing and #10085 is **not** fixed by the v30
> release (the v30 locking relaxation, #9475, did not address it).

## Symptoms

| Configuration | Result |
| --- | --- |
| 1-2 concurrent dispatch threads | stable (300/300 process runs) |
| 3+ concurrent dispatch threads | crashes ~1 in 10-300 process runs |
| DX12 | `0xC0000374` heap corruption |
| Vulkan | `0xC0000005` / `0xC0000374` |
| Crash timing | whole-process abort, no Rust panic output |

The crash kills the entire process — no error is recoverable in Rust. It was
first found by looping the fuzzgpu GPU test suite
(`cargo test -p fuzzgpu-core --lib gpu -- --test-threads=8`) and observing
hard process exits at iterations 39, 77, 146, 232 (and, on re-testing,
iterations 9/21/54/55).

## Reproducing with the fuzzgpu test suite

The suite normally serializes GPU access with a test-only mutex
(`gpu_test_lock()` in `crates/fuzzgpu-core/src/gpu.rs`) as a workaround.
Bypass it and loop the suite:

```bash
cargo test -p fuzzgpu-core --lib --no-run
BIN=$(ls -t target/debug/deps/fuzzgpu_core-*.exe | head -1)
for i in $(seq 1 300); do
  FUZZGPU_SKIP_DISPATCH_LOCK=1 "$BIN" gpu --test-threads=8 || { echo "CRASH at iter $i"; break; }
done
```

Expect a crash (exit 127 / segfault / heap corruption) within ~10-300
iterations on Intel Iris Xe. With `WGPU_BACKEND=dx12` the crash tends to come
faster. With `--test-threads=2` it should never crash (300/300 observed).

## The minimal reproducer in this crate

`cargo run --release -- --process-loop 300 --threads 6` mirrors the workload:
N threads share one device; each iterates
`create_buffer_init` inputs → dispatch the real Levenshtein kernel →
submit → poll → `map_async` → `recv_timeout` → read → drop.

**It does not crash** (~1,500 process runs across variants: trivial shaders,
large buffers, mapped-at-creation inputs, the real kernel WGSL, both
backends). The crash is therefore not yet reduced to a minimal case; the
difference from the crashing test suite is still unknown (test-harness thread
mix, matrix kernels, the fault-injection paths, or something subtler).

## Changelog context (why upstream should look)

wgpu has a documented history of exactly this crash class:

- 27.0.3: "Fix STATUS_HEAP_CORRUPTION crash when concurrently calling
  create_sampler" (gfx-rs/wgpu#8043)
- v30: "Relaxed locking within wgpu-core to enable queue submission
  processing on one thread to proceed while another thread is blocked in a
  device poll" (gfx-rs/wgpu#9475)
- trunk (Unreleased): "Fix a spurious assertion failure in Device::maintain
  when multiple threads race polling the same device" (gfx-rs/wgpu#9958)

Whether any of these fix this crash is untested so far; the reproducer in
this crate is the vehicle for that check (bump `wgpu` in Cargo.toml).

## Environment

- Intel(R) Iris(R) Xe Graphics (Tiger Lake iGPU, 11th gen Core i7)
- Windows 11, wgpu 24.0.5 / naga 24.0.0
- Backends: DX12 (default) and Vulkan (Intel ICD) both crash
