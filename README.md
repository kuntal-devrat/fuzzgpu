<div align="center">

<img src="https://raw.githubusercontent.com/kuntal-devrat/fuzzgpu/main/assets/logo.png" alt="fuzzgpu logo" width="140" height="140" />

# fuzzgpu

**Hardware-Accelerated Fuzzy String Matching & Sequence Alignment**

*Cross-platform GPU compute via WebGPU (`wgpu`) & Multi-Core CPU parallelism with Rayon. Zero CUDA dependencies.*

[![PyPI Version](https://img.shields.io/badge/pypi-v0.1.6-blue.svg?style=flat-square)](https://pypi.org/project/fuzzgpu/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.87+-orange.svg?style=flat-square)](https://www.rust-lang.org)
[![Cross Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux%20%7C%20WASM-lightgrey.svg?style=flat-square)](https://github.com/kuntal-devrat/fuzzgpu)

</div>

---

## Overview

`fuzzgpu` is a high-throughput string distance and sequence alignment engine written in **Rust** with native **Python** and **WebAssembly** bindings. It leverages GPU compute shaders (`wgpu` / WGSL) and Rayon multi-threading to accelerate large-scale batch queries and distance matrix computations across:

- **Apple Silicon (Metal)**
- **Linux (Vulkan)**
- **Windows (DirectX 12 / Vulkan)**
- **Integrated GPUs (Intel Iris Xe, AMD Radeon)**
- **WebAssembly (In-browser execution)**

No NVIDIA CUDA drivers or complex toolkits required.

---

## What's New in v0.1.6

### Drop-in rapidfuzz parity (Python)
The full Python layer is now byte-identical to rapidfuzz 3.14.5 over a 169,744-pair differential harness across `ratio`, `partial_ratio`, `token_sort_ratio`, `token_set_ratio`, `token_ratio`, `WRatio`, `QRatio`, `partial_token_*`, `jaro`, `jaro_winkler`, `levenshtein`, `indel`, `hamming`, `osa` — **0 mismatches**.

### Bug fixes (Rust core, float parity)
- **`ratio` / `partial_ratio` cutoff imprecision** — port of rapidfuzz's load-bearing `NormSim_to_NormDist = min(1, 1 - cutoff/100 + 1e-5)` term. Without it, the branch-and-bound would silently reject ties at exact cutoffs (e.g. `partial_ratio("park", "ba", score_cutoff=50.0)` returned `0` instead of `50`).
- **`ratio` score formula** — switched from `((len-dist)/len)*100` to `(1 - dist/len)*100` to match rapidfuzz C++'s exact ulp order (`indel_normalized_similarity * 100`).

### New features (Python distance layer)
- **`Editops` / `Opcodes` / `Editop` / `Opcode` / `MatchingBlock` / `ScoreAlignment`** classes (rapidfuzz-compatible list-/tuple-likes with `as_list`, `as_opcodes`, `as_editops`, `as_matching_blocks`, `apply`, `inverse`, `remove_subsequence`, `from_*`).
- **`Levenshtein.editops` / `.opcodes`** — exact Myers bit-parallel port with `common_affix` (suffix measured on post-prefix slice, matching rapidfuzz).
- **`LCSseq`** module — Myers LCS bit-parallel matrix + editops/opcodes (delete-checked-first traceback).
- **`Prefix`** and **`Postfix`** modules with the C++ `1 - dist/maximum` ulp order.
- **`Hamming.editops` / `.opcodes`** — replace-per-mismatch + padding delete/insert model.
- **`Indel.editops` / `.opcodes`** — delegates to `LCSseq` (matching rapidfuzz C++).
- **`fuzz.partial_ratio_alignment`** now returns `ScoreAlignment(score, src_start, src_end, dest_start, dest_end)` — drop-in compatible with rapidfuzz.
- **`process.extract` / `extractOne` / `cdist`** default to `WRatio` (matches rapidfuzz 3.14.5).
- **`token_ratio` / `partial_token_ratio`** exposed at top level (`fuzzgpu.token_ratio`, `fuzzgpu.partial_token_ratio`).
- All alignment types re-exported at the package root (`fuzzgpu.Editop`, `fuzzgpu.Editops`, etc.).

### Type stubs
- `__init__.pyi`, `fuzz.pyi`, `process.pyi`, `distance/__init__.pyi` updated for the new APIs.

---

<details>
<summary><b>Previous (v0.1.5</b</summary>

### Bug fixes
- **Damerau-Levenshtein safety gate** now fires in release builds (`assert!` not `debug_assert!`) — non-ASCII inputs no longer silently produce wrong distances in production wheels
- **Needleman-Wunsch GPU f32 precision guard** — scoring parameters that exceed the exact f32 integer range (2²⁴ = 16,777,216) now automatically route to CPU, preventing silent precision loss
- **Wavefront shader race condition** fixed — `diags[1]` seed initialization consolidated into a single thread with a proper `workgroupBarrier()`
- **`extract_one` early-exit** fixed — the `break` at score==100.0 now only fires after the `is_better` check
- **Distance module processor bug** fixed across all `distance/*.py` modules — `similarity`/`normalized_*` now apply the processor once, then compute `maximum` on the processed strings

### Optimizations
- **Zero-allocation SIMD hot paths** — `levenshtein_cdist`, `levenshtein_batch`, and `jaro_winkler_batch` now use stack-allocated `[&[u8]; 8]` instead of per-group heap `Vec`, eliminating millions of tiny allocations at 1M-cell matrix scale
- **`token_set_ratio`** uses `Cow<str>` to skip heap allocation when intersection/difference sets are empty
- **`process.cdist`** fast path routes through the Rayon/GPU `ratio_batch` when the default scorer is used, instead of one Python call per cell

### New features
- `partial_ratio_alignment(s1, s2)` → `(score, src_start, dest_start, length)` — rapidfuzz-compatible alignment result
- `partial_token_sort_ratio`, `partial_token_set_ratio`, `QRatio` — now exposed at the top level
- **Jaro-Winkler GPU routing in Python** — `jaro_winkler_batch` and `jaro_winkler_cdist` now use the GPU kernel on discrete GPUs
- **Needleman-Wunsch GPU routing in Python** — `needleman_wunsch_affine_batch` now uses `GpuNeedlemanAffineKernel`
- `editops` and `opcodes` re-exported at the top level (`fuzzgpu.editops`, `fuzzgpu.opcodes`)
- Complete type stubs (`__init__.pyi`, `fuzz.pyi`, `process.pyi`)

</details>

---

## Benchmark Results

*Hardware: Intel(R) Iris(R) Xe Graphics (Vulkan) + Intel Core i7 (Rayon uses all cores)*
*Versions: fuzzgpu 0.1.6 · rapidfuzz 3.14.5 · python-Levenshtein 0.27.4*
*Median of 7 runs after warmup. Reproduce: `python benchmarks/bench_compare.py`*

### Levenshtein Batch (1 query × N candidates, 10-char strings)
| Batch Size | `fuzzgpu` (GPU) | `fuzzgpu` (CPU) | `rapidfuzz` | vs RF (GPU) | vs RF (CPU) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **100** | 0.12 ms | 0.01 ms | 0.03 ms | 0.25× | 2.21× |
| **1,000** | 0.83 ms | 0.09 ms | 0.15 ms | 0.18× | 1.67× |
| **10,000** | 2.60 ms | 0.87 ms | 1.47 ms | 0.57× | 1.68× |
| **50,000** | 9.92 ms | 5.04 ms | 6.38 ms | 0.64× | 1.27× |

### Damerau-Levenshtein Batch (unrestricted Lowrance-Wagner)
| Batch Size | `fuzzgpu` (GPU) | `fuzzgpu` (CPU) | `rapidfuzz` | vs RF (GPU) | vs RF (CPU) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1,000** | 0.58 ms | 0.40 ms | 1.63 ms | 2.81× | 4.06× |
| **10,000** | 2.95 ms | 2.15 ms | 20.86 ms | 7.06× | 9.70× |
| **50,000** | 17.49 ms | 12.08 ms | 120.25 ms | 6.88× | 9.95× |

> **Note:** rapidfuzz's `DamerauLevenshtein` uses Optimal String Alignment (OSA). fuzzgpu implements the **unrestricted** Lowrance-Wagner (1975) algorithm which allows non-adjacent transpositions. For example: `damerau("ca", "abc") == 2` (fuzzgpu) vs `3` (rapidfuzz OSA). Use `fuzzgpu.distance.OSA` for OSA-compatible semantics.

---

## Installation

```bash
pip install fuzzgpu
```

```toml
# Rust
[dependencies]
fuzzgpu-core = "0.1.6"
```

---

## Quickstart

```python
import fuzzgpu

# ── Core distance metrics ─────────────────────────────────────────────────────
lev = fuzzgpu.levenshtein("kitten", "sitting")              # 3
dam = fuzzgpu.damerau("ab", "ba")                           # 1  (transposition)
jw  = fuzzgpu.jaro_winkler("MARTHA", "MARHTA")             # 0.9611...

# ── Batch (auto-routed GPU/CPU) ───────────────────────────────────────────────
candidates = ["hallo", "hullo", "jello", "yellow", "hello world"] * 10_000
distances  = fuzzgpu.levenshtein_batch("hello", candidates)
jw_scores  = fuzzgpu.jaro_winkler_batch("hello", candidates, p=0.1)
nw_scores  = fuzzgpu.needleman_wunsch_affine_batch(
    "AGTACGCA", candidates, match=2, mismatch=-1, gap_open=-3, gap_extend=-1
)

# ── Cross-product distance matrix ─────────────────────────────────────────────
matrix = fuzzgpu.levenshtein_cdist(["abc", "def", "xyz"], ["abd", "axy", "def"])

# ── Zero-allocation outputs (write into preallocated numpy arrays) ────────────
import numpy as np
out_u32 = np.empty(len(candidates), dtype=np.uint32)
out_f64 = np.empty(len(candidates), dtype=np.float64)
mat_u32 = np.empty((3, 3), dtype=np.uint32)
fuzzgpu.levenshtein_batch_into("hello", candidates, out_u32)
fuzzgpu.jaro_winkler_batch_into("hello", candidates, out_f64)
fuzzgpu.levenshtein_cdist_into(["abc", "def", "xyz"], ["abd", "axy", "def"], mat_u32)

# ── Global sequence alignment (Gotoh 1982 affine gap) ────────────────────────
score = fuzzgpu.needleman_wunsch_affine("AGTACGCA", "TATGC", 2, -1, -3, -1)

# ── Fuzzy ratios (rapidfuzz-compatible) ──────────────────────────────────────
from fuzzgpu.fuzz import (
    ratio, partial_ratio, partial_ratio_alignment,
    token_sort_ratio, token_set_ratio,
    partial_token_sort_ratio, partial_token_set_ratio,
    QRatio, WRatio,
)

ratio("fuzzy was a bear", "fuzzy was a bear")           # 100.0
partial_ratio("hello", "oh hello there")               # 100.0
score, src, dst, length = partial_ratio_alignment("hello", "oh hello there")
# (100.0, 0, 3, 5)  ← window starts at char 3 of the longer string

token_sort_ratio("new york mets", "mets new york")      # 100.0
token_set_ratio("fuzzy was a bear", "fuzzy bear")       # 100.0

# ── Alignment helpers (rapidfuzz-compatible) ──────────────────────────────────
from fuzzgpu.distance import Levenshtein
ops   = fuzzgpu.editops("kitten", "sitting")           # top-level alias
codes = fuzzgpu.opcodes("kitten", "sitting")

# ── Search ────────────────────────────────────────────────────────────────────
from fuzzgpu.fuzz import extract, extractOne
best  = extractOne("hellp", ["hello", "world", "help"], score_cutoff=50.0)
# ("help", 88.88888888888889, 2)
top_3 = extract("apple", ["apply", "ape", "banana", "applesauce"],
                score_cutoff=50.0, limit=3)

# ── rapidfuzz.process-compatible API ─────────────────────────────────────────
from fuzzgpu.process import extract, extractOne, cdist
matrix = cdist(["hello", "world"], ["hallo", "wurld"])  # uses GPU/Rayon

# ── distance submodule (rapidfuzz.distance-compatible) ───────────────────────
from fuzzgpu.distance import Levenshtein, DamerauLevenshtein, Jaro, JaroWinkler
from fuzzgpu.distance import Hamming, OSA, Indel

Levenshtein.distance("kitten", "sitting")               # 3
Levenshtein.normalized_similarity("kitten", "sitting")  # 0.571...
Levenshtein.similarity(" abc ", "abc", processor=str.strip)  # 3
DamerauLevenshtein.distance("ca", "abc")                # 2  (unrestricted)
OSA.distance("ca", "abc")                               # 3  (OSA / rapidfuzz-compatible)
JaroWinkler.similarity("MARTHA", "MARHTA", prefix_weight=0.1)  # 0.9611...

# ── Hardware diagnostics ──────────────────────────────────────────────────────
print(fuzzgpu.gpu_info())        # "Intel(R) Iris(R) Xe Graphics (Vulkan)"
print(fuzzgpu.hardware_info())   # adapter, threshold, last routing stats

fuzzgpu.set_gpu_threshold(100)   # force GPU for batches >= 100 pairs
fuzzgpu.set_gpu_threshold(None)  # restore auto-selection
fuzzgpu.set_cpu_only(True)       # force CPU-only mode
```

---

## Rust API

```toml
[dependencies]
fuzzgpu-core = "0.1.6"                                        # GPU + CPU fallback
# fuzzgpu-core = { version = "0.1.6", default-features = false } # CPU-only
```

```rust
use fuzzgpu_core::levenshtein::gpu_ext::GpuLevenshteinKernel;

fn main() -> fuzzgpu_core::Result<()> {
    let kernel = GpuLevenshteinKernel::get()?;

    // Batch
    let pairs = vec![("kitten", "sitting"), ("hello", "hullo")];
    let distances = kernel.compute(&pairs)?;   // [3, 1]

    // Cross-product matrix
    let matrix = kernel.compute_matrix(&["abc", "def"], &["abc", "xyz"])?;

    // Multi-op batch (one GPU dispatch + readback amortized across all ops)
    let mut batch = kernel.batch();
    batch.add(&pairs);
    batch.add(&[("foo", "bar"), ("test", "taste")]);
    let results = batch.execute()?;   // Vec<Vec<u32>>
    Ok(())
}
```

Available GPU kernels: `GpuLevenshteinKernel`, `GpuJaroKernel`, `GpuNeedlemanAffineKernel`, `GpuDamerauKernel`.

---

## WebAssembly

```bash
cd crates/fuzzgpu-wasm
wasm-pack build --target web --release
```

```js
import init, {
    levenshtein_distance, jaro_winkler, ratio, extract,
    needleman_wunsch, needleman_wunsch_affine,
} from './pkg/fuzzgpu_wasm.js';
await init();

levenshtein_distance('kitten', 'sitting');   // 3
jaro_winkler('MARTHA', 'MARHTA', 0.1);       // 0.9611...

// Needleman-Wunsch scores are i64 → JavaScript BigInt
needleman_wunsch('AGTACGCA', 'TATGC', 2n, -1n, -2n);        // 1n
needleman_wunsch_affine('AGTACGCA', 'TATGC', 2n, -1n, -3n, -1n); // -2n
```

---

## Technical Architecture

### Execution pipeline

```
                    ┌─────────────────────────┐
                    │     User Query / API    │
                    └────────────┬────────────┘
                                 │
                   Batch size / dataset assessment
                                 │
          ┌──────────────────────┴──────────────────────┐
          ▼                                             ▼
  Small batches (< threshold)              Large batches (≥ threshold)
          │                                             │
  ┌───────────────────┐                  ┌─────────────────────────────┐
  │  Rayon Parallel   │                  │   wgpu WebGPU Compute       │
  │  Myers bit-vector │                  │   WGSL shaders              │
  │  AVX512/AVX2/NEON │                  │   Metal / Vulkan / DX12     │
  └───────────────────┘                  └─────────────────────────────┘
```

### Key design points

- **Myers (1999) bit-vector** — O(n) Levenshtein for patterns ≤ 64 chars, zero inner DP loop. Vectorized with AVX512 (8 texts/vector), AVX2 (4), NEON (2), portable scalar fallback. ISA detected at runtime via cached CPUID; override with `FUZZGPU_SIMD=portable|neon|avx2|avx512`.
- **Unrestricted Damerau-Levenshtein** — Full Lowrance-Wagner (1975) with non-adjacent transpositions. GPU shader keeps the full DP matrix in workgroup shared memory (≤ 32 chars ASCII).
- **Gotoh (1982) affine gaps** — 3-state recurrence, O(n) memory. GPU shader computes in f32; automatically routes to CPU when scoring parameters exceed f32 exact range (2²⁴).
- **WGSL shaders require no adapter features** — bit-vectors implemented as u32×2 pairs (no `SHADER_INT64`), works on every WebGPU backend including browsers and integrated GPUs.
- **Metric-aware routing** — iGPUs auto-route Jaro/Damerau to CPU (where AVX2 SIMD wins); discrete GPUs dispatch above a scaled threshold. `hardware_info()` shows every routing decision.
- **Dispatch lock** — serializes GPU calls across threads to work around `gfx-rs/wgpu#10085` (heap corruption under ≥3 concurrent dispatchers on Intel iGPUs).
- **Zero-copy Python bindings** — `Bound<PyString>` pointers, no `Vec<String>` copies; `*_into` APIs write directly into caller-supplied numpy arrays.

### GPU kernels

| Kernel | Algorithm | Max length | Notes |
|--------|-----------|------------|-------|
| `levenshtein.wgsl` | Standard DP | 256 chars | General path |
| `levenshtein_short.wgsl` | SLM row DP | 64 chars | Transposed layout, no register spill |
| `levenshtein_myers.wgsl` | Myers bit-vector | 64 chars | Shared Peq per workgroup, 2×u32 bitmask |
| `levenshtein_cdist_myers.wgsl` | Row-wise Myers | 64 chars | One workgroup per matrix row |
| `levenshtein_matrix.wgsl` | 2D DP grid | 256 chars | O(N+M) data upload |
| `jaro.wgsl` | Bitmap matcher | 128 chars | 128-bit registers, transposed layout |
| `jaro_matrix.wgsl` | 2D Jaro grid | 128 chars | O(N+M) data upload |
| `damerau.wgsl` | Lowrance-Wagner | 32 chars ASCII | Full matrix in SLM, non-adjacent transpositions |
| `damerau_matrix.wgsl` | 2D Damerau grid | 32 chars ASCII | Same |
| `needleman_affine.wgsl` | Gotoh serial | 128 chars | f32 scores, one thread per pair |
| `needleman_wavefront.wgsl` | Gotoh wavefront | 128 chars | Anti-diagonal parallel, O(m+n) steps |

---

## Project Structure

```
fuzzgpu/
├── crates/
│   ├── fuzzgpu-core/        # Core Rust engine + GPU shaders
│   ├── fuzzgpu-python/      # PyO3 Python extension
│   └── fuzzgpu-wasm/        # wasm-bindgen WebAssembly module
├── python/fuzzgpu/          # Python package wrapper + type stubs
│   ├── distance/            # rapidfuzz.distance-compatible modules
│   ├── fuzz.py              # rapidfuzz.fuzz-compatible scorers
│   └── process.py           # rapidfuzz.process-compatible helpers
├── fuzz/                    # libFuzzer targets + stable self-harness
├── benchmarks/              # Comparative benchmark scripts
├── tests/                   # Python pytest suite (174 tests)
└── docs/GPU_TESTING.md      # Fault injection & GPU test conventions
```

---

## Building from Source

```bash
# Prerequisites: Rust 1.87+, Python 3.10+, maturin
git clone https://github.com/kuntal-devrat/fuzzgpu.git
cd fuzzgpu
maturin develop --release
pytest tests/ -v
cargo test --workspace
```

---

## Environment Variables

| Variable | Effect |
|----------|--------|
| `FUZZGPU_USE_CPU` | Force CPU-only mode |
| `FUZZGPU_FORCE_GPU` | Error (not fallback) on GPU failure in Python |
| `FUZZGPU_DEBUG` | Log GPU→CPU fallback decisions |
| `FUZZGPU_SIMD` | Force ISA: `portable\|neon\|avx2\|avx512` |
| `FUZZGPU_READBACK_TIMEOUT_MS` | GPU readback timeout (default 10000 ms) |
| `FUZZGPU_SKIP_DISPATCH_LOCK` | Bypass serialization lock (repro only) |
| `FUZZGPU_REQUIRE_GPU` | In tests: fail instead of skip when no GPU |
| `WGPU_BACKEND` | Force wgpu backend: `vulkan\|metal\|dx12` |
| `PROPTEST_CASES` | Override proptest case count |

---

## License

[MIT](LICENSE)
