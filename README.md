<div align="center">

<img src="https://raw.githubusercontent.com/kuntal-devrat/fuzzgpu/main/assets/logo.png" alt="fuzzgpu logo" width="140" height="140" />

# fuzzgpu

**Hardware-Accelerated Fuzzy String Matching & Sequence Alignment**

*Cross-platform GPU compute via WebGPU (`wgpu`) & Multi-Core CPU parallelism with Rayon. Zero CUDA dependencies.*

[![PyPI Version](https://img.shields.io/badge/pypi-v0.1.0-blue.svg?style=flat-square)](https://pypi.org/project/fuzzgpu/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.75+-orange.svg?style=flat-square)](https://www.rust-lang.org)
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
fuzzgpu-core = "0.1.0"
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

### Build WebAssembly (Browser Target)
```bash
cd crates/fuzzgpu-wasm
wasm-pack build --target web --release
```

---

## License

This project is licensed under the [MIT License](LICENSE).
