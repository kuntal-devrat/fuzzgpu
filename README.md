<div align="center">

<img src="https://raw.githubusercontent.com/kuntal-devrat/fuzzgpu/main/assets/logo.png" alt="fuzzgpu logo" width="140" height="140" />

# fuzzgpu

**Hardware-Accelerated Fuzzy String Matching & Sequence Alignment**

*Cross-platform GPU compute via WebGPU (`wgpu`) & Multi-Core CPU parallelism with Rayon. Zero CUDA dependencies.*

[![PyPI Version](https://img.shields.io/badge/pypi-v0.1.4-blue.svg?style=flat-square)](https://pypi.org/project/fuzzgpu/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.75+-orange.svg?style=flat-square)](https://www.rust-lang.org)
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

*Hardware: Intel(R) Iris(R) Xe Graphics (Vulkan) / Intel Core i7 CPU*

### 1. Damerau-Levenshtein Batch (1 Query × N Candidates)
| Batch Size | `fuzzgpu` | `rapidfuzz` | `python-Levenshtein` | Speedup vs RapidFuzz |
| :--- | :---: | :---: | :---: | :---: |
| **100** | **0.21 ms** | 0.36 ms | N/A | **1.74×** |
| **1,000** | **0.96 ms** | 3.72 ms | N/A | **3.88×** |
| **5,000** | **3.67 ms** | 18.44 ms | N/A | **5.02×** |
| **10,000** | **5.95 ms** | 37.46 ms | N/A | **6.30×** |
| **50,000** | **31.74 ms** | 85.87 ms | N/A | **2.71×** |

### 2. Levenshtein Cross-Product Matrix (`cdist` $N \times M$)
*Utilizing dedicated 2D Grid Workgroup Shaders with $O(N + M)$ memory bandwidth:*
| Matrix Size | Total Pairs | `fuzzgpu` | `rapidfuzz` | `python-Levenshtein` | Speedup vs RF | Speedup vs py-Lev |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **10 × 10** | 100 | **0.04 ms** | 0.04 ms | 0.05 ms | 1.00× | **1.31×** |
| **50 × 50** | 2,500 | **2.22 ms** | 0.71 ms | 0.99 ms | 0.32× | 0.45× |
| **100 × 100** | 10,000 | **5.29 ms** | 3.51 ms | 4.03 ms | 0.66× | 0.76× |
| **200 × 200** | 40,000 | **15.05 ms** | 33.58 ms | 46.82 ms | **2.23×** | **3.11×** |

### 3. Jaro-Winkler Similarity Batch
| Batch Size | `fuzzgpu` | `rapidfuzz` | `python-Levenshtein` | Speedup vs RapidFuzz |
| :--- | :---: | :---: | :---: | :---: |
| **1,000** | **3.10 ms** | 0.81 ms | 1.13 ms | 0.26× |
| **5,000** | **6.56 ms** | 5.12 ms | 5.78 ms | 0.78× |
| **10,000** | **8.56 ms** | 7.80 ms | 9.17 ms | 0.91× |
| **50,000** | **33.24 ms** | 47.11 ms | 24.86 ms | **1.42×** |

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

1. **2D Grid Matrix Shaders (`levenshtein_matrix.wgsl` & `jaro_matrix.wgsl`)**:
   Instead of duplicating string pairs across PCIe ($O(N \cdot M)$ transfers), List A and List B are uploaded once ($O(N + M)$ memory bandwidth). Workgroups execute on a 2D grid (`@workgroup_size(16, 16)`).
2. **Myers (1999) Bit-Parallel CPU Engine**:
   For strings $\le 64$ characters, computes Levenshtein edit distance using bit-vector operations with zero inner dynamic programming loops ($O(N)$ execution).
3. **Lowrance & Wagner (1975) Unrestricted Damerau-Levenshtein**:
   Full support for character insertions, deletions, substitutions, and arbitrary transpositions.
4. **Gotoh (1982) Affine Gap Sequence Alignment**:
   Memory-efficient 3-state recurrence ($O(N)$ auxiliary space) for bioinformatics and long-sequence alignment.
5. **Streaming Chunk Partitioner**:
   Datasets exceeding GPU buffer limits (>128MB or >500,000 pairs) are automatically streamed in chunks to prevent VRAM overflow.

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
- [Rust Toolchain (1.75+)](https://rustup.rs/)
- Python 3.8+ & `pip install maturin`

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
