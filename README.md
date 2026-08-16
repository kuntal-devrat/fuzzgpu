<div align="center">

<img src="https://raw.githubusercontent.com/kuntal-devrat/fuzzgpu/main/assets/logo.png" alt="fuzzgpu logo" width="140" height="140" />

# fuzzgpu

**Hardware-Accelerated Fuzzy String Matching & Sequence Alignment**

*Cross-platform GPU compute via WebGPU (`wgpu`) & Multi-Core CPU parallelism with Rayon. Zero CUDA dependencies.*

[![PyPI Version](https://img.shields.io/badge/pypi-v0.1.4-blue.svg?style=flat-square)](https://pypi.org/project/fuzzgpu/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.87+-orange.svg?style=flat-square)](https://www.rust-lang.org)
[![Cross Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux%20%7C%20WASM-lightgrey.svg?style=flat-square)](https://github.com/kuntal-devrat/fuzzgpu)
[![Release wasm package](https://github.com/kuntal-devrat/fuzzgpu/actions/workflows/wasm-release.yml/badge.svg)](https://github.com/kuntal-devrat/fuzzgpu/actions/workflows/wasm-release.yml)

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

## Benchmark Results

*Hardware: Intel(R) Iris(R) Xe Graphics (Vulkan) + Intel Core i7 (Rayon uses all cores)*
*Versions: fuzzgpu 0.1.4 (release) · rapidfuzz 3.14.5 · python-Levenshtein 0.27.4*
*Method: median of 7 runs after a warmup call, one full library call per measurement.
Reproduce with `python benchmarks/bench_compare.py`.*

> **How to read these tables.** `fuzzgpu`'s CPU path is multi-threaded (Rayon)
> and uses the **Myers (1999) bit-vector** for ASCII pairs with a ≤ 64-char
> pattern (only the *pattern* must be short — the text can be any length),
> amplified by width-aware SIMD kernels — **AVX512** (8 texts/vector),
> **AVX2** (4), **NEON** (2), portable fallback — for Levenshtein *and*
> Jaro (bit-parallel matching-window pass). This is why the Levenshtein and
> Jaro-Winkler CPU numbers below *beat* rapidfuzz's C++/SIMD at scale, and
> Damerau (unrestricted Lowrance-Wagner) crushes it. The Python bindings are
> **zero-copy** (`abi3-py310` + pyo3 `Bound<str>` views — no `Vec<String>`
> copies per call).
>
> **GPU kernels exist for every metric.** Levenshtein uses the Myers bit-vector
> shader (shared Peq per workgroup, two-u32 bit-vector so no `SHADER_INT64`
> is needed) plus a **row-wise Myers cdist kernel** (~10× faster than the old
> DP matrix shader). Jaro-Winkler has a **bitmap-matching shader** (128-bit
> bitmaps in registers, transposed pair-major char layout for coalesced
> loads). Damerau has a **Lowrance-Wagner shader** that keeps each pair's full
> DP matrix in workgroup shared memory (bit-exact with the CPU reference,
> including non-adjacent transpositions like `ca`/`abc` = 2).
>
> **Routing is metric-aware and backend-aware.** The GPU carries a per-dispatch
> sync round-trip (~1 ms on an iGPU), and the Jaro/Damerau kernels are
> heavier per pair than Myers — measured on Iris Xe they lose to the SIMD CPU
> path at *every* scale (Jaro ~2×, Damerau ~6× at 50k pairs). Auto-routing
> therefore never sends them to an integrated GPU (the "GPU" columns below
> are the auto-routed result, i.e. the fast CPU path); on discrete GPUs they
> dispatch above a scaled threshold (Jaro ≥ 1,024 pairs, Damerau ≥ 2,048).
> Levenshtein's Myers kernel is cheap enough per pair to win on iGPUs at
> scale and is routed normally. `hardware_info()` shows the adapter class,
> auto threshold, and how many pairs actually went to the GPU;
> `set_gpu_threshold(n)` overrides any of it. These tables replace the
> v0.1.0 numbers, which were not reproducible: they compared against
> pure-Python loops and timed single un-warmed calls.

### 1. Levenshtein Batch (1 query × N candidates, 10-char strings)
| Batch Size | `fuzzgpu` (GPU) | `fuzzgpu` (CPU) | `rapidfuzz` | vs RF (GPU) | vs RF (CPU) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **100** | 0.12 ms | 0.01 ms | 0.03 ms | 0.25× | 2.21× |
| **1,000** | 0.83 ms | 0.09 ms | 0.15 ms | 0.18× | 1.67× |
| **10,000** | 2.60 ms | 0.87 ms | 1.47 ms | 0.57× | 1.68× |
| **50,000** | 9.92 ms | 5.04 ms | 6.38 ms | 0.64× | 1.27× |

### 2. Damerau-Levenshtein Batch
*Unrestricted Lowrance-Wagner (non-adjacent transpositions included, unlike
rapidfuzz's optimal-string-alignment). The GPU kernel exists and is
bit-exact, but on this iGPU auto-routing sends it to the CPU path (the GPU
column is the auto-routed result).*
| Batch Size | `fuzzgpu` (GPU) | `fuzzgpu` (CPU) | `rapidfuzz` | vs RF (GPU) | vs RF (CPU) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **100** | 0.10 ms | 0.06 ms | 0.14 ms | 1.43× | 2.18× |
| **1,000** | 0.58 ms | 0.40 ms | 1.63 ms | 2.81× | 4.06× |
| **10,000** | 2.95 ms | 2.15 ms | 20.86 ms | 7.06× | 9.70× |
| **50,000** | 17.49 ms | 12.08 ms | 120.25 ms | 6.88× | 9.95× |

### 3. Jaro-Winkler Batch (p = 0.1)
*GPU bitmap-matching kernel exists; on this iGPU auto-routing sends it to the
SIMD CPU path (the GPU column is the auto-routed result).*
| Batch Size | `fuzzgpu` (GPU) | `fuzzgpu` (CPU) | `rapidfuzz` | vs RF (GPU) | vs RF (CPU) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **100** | 0.01 ms | 0.01 ms | 0.01 ms | 1.06× | 1.13× |
| **1,000** | 0.11 ms | 0.10 ms | 0.10 ms | 0.95× | 0.99× |
| **10,000** | 1.50 ms | 1.68 ms | 1.72 ms | 1.15× | 1.02× |
| **50,000** | 10.96 ms | 14.34 ms | 15.62 ms | 1.43× | 1.09× |

### 4. Needleman-Wunsch Batch (affine, match=1, mismatch=-1, gap_open=-2, gap_extend=-1)
*rapidfuzz has no affine-gap Needleman-Wunsch scorer, so no comparison column.*
| Batch Size | `fuzzgpu` (GPU) | `fuzzgpu` (CPU) |
| :--- | :---: | :---: |
| **100** | 0.30 ms | 0.21 ms |
| **1,000** | 1.02 ms | 1.24 ms |
| **10,000** | 7.16 ms | 4.39 ms |
| **50,000** | 20.82 ms | 19.09 ms |

### 5. Levenshtein Cross-Product Matrix (`cdist`)
| Matrix Size | Total Pairs | `fuzzgpu` (GPU) | `fuzzgpu` (CPU) | `rapidfuzz` | python-Levenshtein | vs RF (GPU) | vs RF (CPU) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10 × 10** | 100 | 0.06 ms | 0.03 ms | 0.01 ms | 0.05 ms | 0.12× | 0.22× |
| **50 × 50** | 2,500 | 0.61 ms | 0.11 ms | 0.04 ms | 1.20 ms | 0.07× | 0.38× |
| **100 × 100** | 10,000 | 0.81 ms | 0.28 ms | 0.25 ms | 6.69 ms | 0.31× | 0.90× |
| **200 × 200** | 40,000 | 1.56 ms | 1.56 ms | 1.13 ms | 41.35 ms | 0.73× | 0.73× |

---

## Installation

### Python
```bash
pip install fuzzgpu
```

### Rust (Cargo.toml)
```toml
[dependencies]
fuzzgpu-core = "0.1.4"
```

---

## Quickstart

```python
import fuzzgpu
from fuzzgpu.fuzz import ratio, partial_ratio, token_sort_ratio, token_set_ratio, extract, extractOne

# 1. Classical Distance Metrics
lev = fuzzgpu.levenshtein_distance("kitten", "sitting")         # 3
dam = fuzzgpu.damerau_levenshtein_distance("ab", "ba")          # 1 (transposition-aware)
jw  = fuzzgpu.jaro_winkler_similarity("MARTHA", "MARHTA", 0.1)  # 0.9611

# 2. High-Throughput Batch Processing (Auto-dispatched to GPU/CPU)
candidates = ["hallo", "hullo", "jello", "yellow", "hello world"] * 10_000
distances  = fuzzgpu.levenshtein_batch("hello", candidates)
jw_scores  = fuzzgpu.jaro_winkler_batch("hello", candidates, prefix_weight=0.1)

# 3. 2D Cross-Product Distance Matrix (Dedicated 2D Grid Shader)
matrix = fuzzgpu.levenshtein_cdist(["abc", "def", "xyz"], ["abd", "axy", "def"])

# 3b. Zero-Allocation Outputs (write into preallocated numpy buffers — the
#     rapidfuzz binding model: no per-call Python int boxing, no GC churn)
import numpy as np
out_u32  = np.empty(len(candidates), dtype=np.uint32)
out_f64  = np.empty(len(candidates), dtype=np.float64)
mat_u32  = np.empty((3, 3), dtype=np.uint32)
fuzzgpu.levenshtein_batch_into("hello", candidates, out_u32)        # fills in place
fuzzgpu.jaro_winkler_batch_into("hello", candidates, out_f64)      # jaro/winkler: float64
fuzzgpu.levenshtein_cdist_into(["abc", "def", "xyz"], ["abd", "axy", "def"], mat_u32)
# Supported: levenshtein/damerau/jaro batch + cdist. `out` must be a numpy
# array of the exact shape/dtype (uint32 for distances, float64 for Jaro);
# it is validated (length, dtype, writable, contiguous) before any compute,
# and left untouched if validation fails.

# 4. Global Sequence Alignment (Gotoh 1982 Linear & Affine Gap Penalties)
score_linear = fuzzgpu.needleman_wunsch_score("AGTACGCA", "TATGC", match=2, mismatch=-1, gap=-2)
score_affine = fuzzgpu.needleman_wunsch_affine("AGTACGCA", "TATGC", match=2, mismatch=-1, gap_open=-3, gap_extend=-1)

# 5. RapidFuzz-Compatible Scorer & Search API
score = ratio("fuzzy was a bear", "fuzzy was a bear")          # 100.0
part  = partial_ratio("hello", "oh hello there")              # 100.0
tsr   = token_sort_ratio("new york mets", "mets new york")     # 100.0
tset  = token_set_ratio("fuzzy was a bear", "fuzzy bear")      # 100.0

# 6. Top-K Best Match Search
best  = extractOne("hellp", ["hello", "world", "help"], score_cutoff=50.0)
# Output: ("hello", 80.0, 0)

top_3 = extract("apple", ["apply", "ape", "banana", "applesauce"], score_cutoff=50.0, limit=3)

# 7. Hardware Diagnostics
print(fuzzgpu.gpu_info())
# Output: Intel(R) Iris(R) Xe Graphics (Vulkan) / Apple M2 (Metal)

# Full routing diagnostics: adapter class, auto threshold, last routing.
print(fuzzgpu.hardware_info())
# Output: GPU: Intel(R) Iris(R) Xe Graphics (Vulkan, IntegratedGpu) |
#         auto threshold: 500 | override: auto | last routing: 50000 GPU / 0 CPU pairs | ...

# GPU/CPU routing is backend-aware by default (discrete GPUs route earlier;
# integrated and software GPUs are conservative). Override it explicitly:
fuzzgpu.set_gpu_threshold(100)   # force GPU dispatch for batches >= 100 pairs
fuzzgpu.set_gpu_threshold(None)  # restore auto-selection from the adapter
```

---

## Rust API (fuzzgpu-core)

Add the dependency — the `gpu` feature (WebGPU via `wgpu`) is on by default with an automatic Rayon CPU fallback; set `default-features = false` for a pure-CPU build:

```toml
[dependencies]
fuzzgpu-core = "0.1.4"                                            # GPU + CPU fallback
# fuzzgpu-core = { version = "0.1.4", default-features = false } # CPU-only
```

### Batch compute & cross-product matrix (GPU, with CPU fallback)

The GPU kernel lazily initializes the wgpu device and auto-routes every workload: empty/identical pairs short-circuit, strings over 256 chars and batches under 500 pairs run on Rayon CPU, and everything else dispatches to the compute shader in chunks sized to the adapter's buffer limits:

```rust
use fuzzgpu_core::levenshtein::gpu_ext::GpuLevenshteinKernel;

fn main() -> fuzzgpu_core::Result<()> {
    let kernel = GpuLevenshteinKernel::get()?; // lazy wgpu device + pipeline setup

    // Batch: one query vs N candidates.
    let pairs: Vec<(&str, &str)> = vec![
        ("kitten", "sitting"),
        ("kitten", "kittens"),
        ("hello", "hullo"),
    ];
    let distances = kernel.compute(&pairs)?;
    assert_eq!(distances, vec![3, 1, 1]);

    // Cross-product N×M matrix via the dedicated 2D-grid shader.
    let list_a = ["kitten", "sitting"];
    let list_b = ["kitten", "mittens", "sitting"];
    let matrix = kernel.compute_matrix(&list_a, &list_b)?;
    assert_eq!(matrix, vec![vec![0, 2, 3], vec![3, 3, 0]]);
    Ok(())
}
```

Jaro-Winkler (`fuzzgpu_core::jaro::gpu_ext::GpuJaroKernel`) and affine Needleman-Wunsch (`fuzzgpu_core::needleman::gpu_ext::GpuNeedlemanAffineKernel`) kernels follow the same `get()` / batch / matrix pattern.

### CPU-only build (`default-features = false`)

```rust
use fuzzgpu_core::LevenshteinKernel;
use fuzzgpu_core::levenshtein::levenshtein_cdist_cpu;

let kernel = LevenshteinKernel;
let distances = kernel.compute(&pairs)?;              // Rayon-parallel batch
let matrix = levenshtein_cdist_cpu(&list_a, &list_b); // Rayon-parallel matrix
```

### Single-value API

The crate root also exposes the scalar functions behind the Python and wasm APIs:

```rust
use fuzzgpu_core::{
    damerau_levenshtein_distance, extract, jaro_winkler, levenshtein_distance_raw,
    needleman_wunsch, needleman_wunsch_affine, partial_ratio, ratio,
};

assert_eq!(levenshtein_distance_raw("kitten", "sitting"), 3);
assert_eq!(damerau_levenshtein_distance("ab", "ba"), 1);
assert_eq!(jaro_winkler("MARTHA", "MARHTA", 0.1), 0.9611111111111111);
assert_eq!(ratio("fuzzy was a bear", "fuzzy was a bear"), 100.0);
assert_eq!(needleman_wunsch("AGTACGCA", "TATGC", 2, -1, -2), 1);
```

> With the `gpu` feature, fallible APIs return `fuzzgpu_core::Result<T>` (`FuzzGpuError`); without it, the same alias is `Result<T, String>`.

---

## WebAssembly (JavaScript API)

Build the wasm module for your target and import it like any ES module:

```bash
cd crates/fuzzgpu-wasm
wasm-pack build --target web --release     # browser (ESM)
wasm-pack build --target nodejs --release  # Node.js (CommonJS)
```

```js
// Browser (ESM) — `init` is generated for --target web
import init, { levenshtein_distance, jaro_winkler, needleman_wunsch, ratio } from './pkg/fuzzgpu_wasm.js';
await init();

// Node.js (CommonJS): const fg = require('./pkg/fuzzgpu_wasm.js'); // no init needed

// Classic distance & similarity metrics return plain JS numbers
levenshtein_distance('kitten', 'sitting');   // 3
jaro_winkler('MARTHA', 'MARHTA', 0.1);       // 0.9611...
ratio('fuzzy was a bear', 'fuzzy was a bear'); // 100.0

// Batch & search helpers (returns match objects)
extract('apple', ['apply', 'ape', 'banana'], 50.0, 3);
```

### Generated package layout

`bash build-wasm.sh` (or `wasm-pack build --target web --release` — the script just wraps it with a release profile and `--out-dir ../../pkg`) produces the browser package at the repo root `pkg/`:

| File | Purpose |
| :--- | :--- |
| `fuzzgpu_wasm.js` | **ES module entry** — exports the whole API plus a default `init()`. This is the file you import. |
| `fuzzgpu_wasm_bg.wasm` | The compiled **WebAssembly module** (~140 KB). It is *not* inlined — `init()` fetches it at runtime, so it must ship alongside the glue. |
| `fuzzgpu_wasm.d.ts` | **TypeScript declarations** for every export, including the `bigint` (i64) Needleman-Wunsch signatures. |
| `fuzzgpu_wasm_bg.wasm.d.ts` | wasm-level type declaration picked up by some TS tooling. |
| `package.json` | npm manifest: `"type": "module"`, `"main": "fuzzgpu_wasm.js"`, `"types": "fuzzgpu_wasm.d.ts"`, plus a `files` list that controls exactly what ships when you publish the package. |

> The `.js` glue and the `.wasm` are **two separate files that must be served together** — the browser fetches `fuzzgpu_wasm_bg.wasm` by URL after the glue module loads. Bundlers that tree-shake or inline assets need the explicit import wiring below.

### Wiring into a bundler (Vite / webpack)

The generated `init()` resolves the `.wasm` with `new URL('fuzzgpu_wasm_bg.wasm', import.meta.url)`, which **Vite and webpack 5 both understand natively** — the default pattern needs zero bundler config:

```js
// main.js — default pattern (Vite and webpack 5)
import init, { levenshtein_distance, ratio, needleman_wunsch } from './pkg/fuzzgpu_wasm.js';

// init() fetches + instantiates fuzzgpu_wasm_bg.wasm (resolved relative to
// this module's URL) and returns a promise. All exports throw until it resolves.
await init();

console.log(levenshtein_distance('kitten', 'sitting')); // 3
```

- **Await `init()` exactly once at startup.** Top-level `await` works in both bundlers; otherwise wrap it in your app's async bootstrap. Calling any export before `init()` resolves throws.
- **TypeScript:** with `"moduleResolution": "bundler"` the adjacent `fuzzgpu_wasm.d.ts` is picked up automatically — BigInt score parameters are typed `bigint`.

If the `.wasm` lives somewhere non-default (CDN, custom base path, or a bundler that doesn't follow `new URL`), pass the URL explicitly — `init()` accepts a string / `URL` / `Response`, or an object `{ module_or_path }`:

```js
// Vite — explicit asset URL via the ?url suffix
import wasmUrl from './pkg/fuzzgpu_wasm_bg.wasm?url';
await init(wasmUrl);

// webpack 5 — asset/resource emits the .wasm as a URL string
// (module.rules: { test: /\.wasm$/, type: 'asset/resource' })
import wasmUrl from './pkg/fuzzgpu_wasm_bg.wasm';
await init(wasmUrl);
```

To consume the package from npm instead of a local path, publish the `pkg/` directory and import by name — `fuzzgpu_wasm.js` is `"main"` and the `files` array keeps the `.wasm` + `.d.ts` in the published tarball:

```js
import init, { levenshtein_distance } from 'fuzzgpu-wasm';
await init();
```

### BigInt Needleman-Wunsch scores

Needleman-Wunsch alignment scores are 64-bit integers. wasm-bindgen maps `i64` to JavaScript **`BigInt`** — not `Number` — so scores beyond the 32-bit range are never truncated:

```js
// Linear gap penalty — pass BigInt arguments, receive a BigInt back
const s = needleman_wunsch('AGTACGCA', 'TATGC', 2n, -1n, -2n);
// s === 1n

// Affine (Gotoh) gap penalties
const a = needleman_wunsch_affine('AGTACGCA', 'TATGC', 2n, -1n, -3n, -1n);
// a === -2n

// Scores far beyond i32::MAX (~2.1e9) survive exactly:
const long = 'A'.repeat(100);
needleman_wunsch(long, long, 30_000_000n, -1n, -2n);
// 3000000000n — exact BigInt, not wrapped/truncated
```

> **Note:** The score parameters are `i64`, so they must be passed as `BigInt` literals (`2n`) — passing a plain `Number` throws a `TypeError` (verified by the test suite). `BigInt` requires a modern runtime: all current browsers, Node ≥ 10.4.

---

## Technical Architecture

`fuzzgpu` combines a tiered execution pipeline to balance low-latency single queries and high-throughput batch workloads:

```
                          ┌──────────────────────────┐
                          │     User Query / API     │
                          └─────────────┬────────────┘
                                        │
                         Batch Size / Dataset Assessment
                                        │
                ┌───────────────────────┴───────────────────────┐
                ▼                                               ▼
     Small Workloads (< 500)                         Large Batches (≥ 500)
                │                                               │
   ┌───────────────────────────┐                 ┌───────────────────────────┐
   │    Rayon Multi-Threaded   │                 │     wgpu WebGPU Compute   │
   │      CPU Parallelism      │                 │  Shaders (Metal/Vulkan)   │
   │  - Myers 1999 Bit-Vector  │                 │  - 2D Workgroup Grids     │
   │  - Zero PCIe Latency      │                 │  - Streaming Chunking     │
   └───────────────────────────┘                 └───────────────────────────┘
```

### Key Architectural Optimizations

1. **GPU kernels for every metric**: Levenshtein runs the Myers bit-vector
   shader (`levenshtein_myers.wgsl`, two-u32 bit-vector, no `SHADER_INT64`)
   plus a row-wise Myers cdist kernel (`levenshtein_cdist_myers.wgsl`);
   Jaro-Winkler runs a bitmap-matching shader (`jaro.wgsl` / `jaro_matrix.wgsl`
   — 128-bit register bitmaps, transposed pair-major char layout for
   coalesced loads, no per-thread arrays); Damerau runs a Lowrance-Wagner
   shader (`damerau.wgsl` / `damerau_matrix.wgsl`) that keeps each pair's full
   DP matrix in workgroup shared memory, bit-exact with the CPU reference
   including non-adjacent transpositions (`ca`/`abc` = 2). Matrix shaders
   upload List A and List B once ($O(N + M)$ bandwidth) instead of
   duplicating pairs across PCIe.
2. **Myers (1999) Bit-Parallel CPU Engine**:
   For strings $\le 64$ characters, computes Levenshtein edit distance using bit-vector operations with zero inner dynamic programming loops ($O(N)$ execution).
3. **Lowrance & Wagner (1975) Unrestricted Damerau-Levenshtein**:
   Full support for character insertions, deletions, substitutions, and arbitrary transpositions.
4. **Gotoh (1982) Affine Gap Sequence Alignment**:
   Memory-efficient 3-state recurrence ($O(N)$ auxiliary space) for bioinformatics and long-sequence alignment.
5. **Streaming Chunk Partitioner**:
   Datasets exceeding GPU buffer limits (>128MB or >500,000 pairs) are automatically streamed in chunks to prevent VRAM overflow.
6. **Metric-Aware Backend Routing**:
   The Myers kernel wins on iGPUs at scale and is routed at the auto threshold; the heavier Jaro/Damerau kernels are auto-routed to CPU on integrated GPUs (measured 2–6× slower there at every scale) and to GPU on discrete GPUs above a scaled threshold. `hardware_info()` reports every routing decision; `set_gpu_threshold(n)` overrides.
7. **ISA-Aware SIMD Kernels (Levenshtein Myers & Jaro)**:
   The bit-parallel kernels dispatch at runtime to the widest available instruction set — **AVX512** (8 texts per 512-bit vector), **AVX2** (4 texts per 256-bit vector), **NEON** on aarch64 (2 texts per 128-bit vector), or a portable scalar fallback. Every kernel is differentially tested against the portable reference (AVX512/AVX2 on x86 CI, NEON on a native arm64 CI runner). To pin a specific ISA (e.g. to work around a 512-bit downclocking part, or for benchmarking) set `FUZZGPU_SIMD=portable|neon|avx2|avx512`. The GPU shaders are backend-agnostic WGSL (no `u64`, no adapter features) and run on Vulkan, Metal, DX12, and WebGPU.

---

## Project Structure

```
fuzzgpu/
├── assets/
│   └── logo.svg               # Vector brand asset
├── crates/
│   ├── fuzzgpu-core/          # Core Rust engine & compute shaders
│   │   ├── src/
│   │   │   ├── gpu.rs         # wgpu instance and device singleton
│   │   │   ├── levenshtein.rs # Levenshtein kernel & 2D matrix dispatch
│   │   │   ├── damerau.rs     # Lowrance-Wagner Damerau-Levenshtein
│   │   │   ├── needleman.rs   # Needleman-Wunsch (Linear & Affine)
│   │   │   ├── jaro.rs        # Jaro / Jaro-Winkler GPU & CPU kernels
│   │   │   ├── fuzz.rs        # Fuzzy ratio, token sort/set, extract
│   │   │   ├── simd.rs        # Myers bit-vector algorithms
│   │   │   └── shaders/       # WGSL compute shaders (1D & 2D)
│   ├── fuzzgpu-python/        # PyO3 CPython C-extension module
│   └── fuzzgpu-wasm/          # wasm-bindgen WebAssembly module
├── python/
│   └── fuzzgpu/               # Python package wrapper & typing
├── tests/
│   └── test_basic.py          # Comprehensive test suite (50 tests)
└── benchmarks/
    └── bench_compare.py       # Comparative benchmarking harness
```

---

## Building from Source

### Prerequisites
- [Rust Toolchain (1.87+)](https://rustup.rs/) (wgpu 30 MSRV)
- Python 3.10+ & `pip install maturin`

### Build Python Extension
```bash
# Clone the repository
git clone https://github.com/Flaxmbot/fuzzgpu.git
cd fuzzgpu

# Build and install into current virtual environment
maturin develop --release
```

### Run Tests & Benchmarks
```bash
# Run pytest verification suite
pytest tests/ -v

# Run comparative benchmark harness
python benchmarks/bench_compare.py
```

### Fuzz Testing

The `fuzz/` crate holds libFuzzer targets (nightly + `cargo fuzz run <target>`)
for Levenshtein, Jaro, Needleman-Wunsch, and the fuzzy ratios, each asserting
its fast path against a naive oracle. The same drivers run on **stable** via a
self-harness — `cargo test --manifest-path fuzz/Cargo.toml --lib --release` —
which is wired into CI, so the differential fuzz checks execute on every push
without a nightly toolchain.

GPU test writers: see [docs/GPU_TESTING.md](docs/GPU_TESTING.md) for the fault-injection hooks (timeout / buffer / shader-error branches) and the dispatch-lock, skip, and CI conventions every GPU test must follow.

### Build WebAssembly (Browser Target)
```bash
cd crates/fuzzgpu-wasm
wasm-pack build --target web --release
```

See [WebAssembly (JavaScript API)](#webassembly-javascript-api) for the JS usage patterns and the BigInt scoring API.

---

## Releasing

### WebAssembly (`wasm-v*`)

The wasm package is cut with the **Release wasm package** workflow — manually triggered from the Actions tab with the new semver version (e.g. `0.2.0`):

1. **Run the workflow**: *Actions → Release wasm package → Run workflow*, enter the new version. It validates the semver and refuses to re-release the current version.
2. **Bump**: rewrites `version` in `crates/fuzzgpu-wasm/Cargo.toml` (the wasm package's version — it is its own workspace, independent of the core/Python versions).
3. **Gate**: runs the `#[wasm_bindgen_test]` suite (BigInt i64 signatures, >2⁵³ scores) before anything ships.
4. **Build & verify**: builds with the exact user-facing `bash build-wasm.sh` and checks `pkg/package.json` carries the new version plus the `.wasm` magic bytes.
5. **Release**: commits the bump, creates the `wasm-v<version>` tag and a GitHub release attaching `fuzzgpu_wasm_bg.wasm`, the JS glue, `.d.ts`, `package.json`, and a zip of the full `pkg/`.

### Python / PyPI (`v*`)

The Python wheels take a different path — the **Release & Publish to PyPI** workflow, triggered by pushing a `v*` tag (e.g. `v0.2.0`): it builds Linux/macOS/Windows wheels with maturin, publishes them to PyPI, and attaches the wheels to the tag's GitHub release.

**Why two flows?** The wasm package is a browser artifact (ESM glue + `.wasm` + TypeScript types) whose version lives in `crates/fuzzgpu-wasm/Cargo.toml`; the Python package is a native extension published to PyPI and versioned with the core crate. Keeping them on separate tag namespaces (`wasm-v*` vs `v*`) lets each cut independently without colliding, and each workflow is self-contained (bump → test → build → attach) so nothing ships untested.

**Version parity is enforced.** fuzzgpu releases core/python/wasm in lockstep: the wasm workflow refuses to cut a version that doesn't equal the core crate's current version (`crates/fuzzgpu-core/Cargo.toml`), so the browser artifact can never drift from the Python package. To release a new version, bump core first (e.g. via the normal `v*` release process), then cut the wasm release at the same version — the parity check passes, and the wasm bump (`wasm-v<version>`) is exactly the core version.

---

## License

This project is licensed under the [MIT License](LICENSE).
