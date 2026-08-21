# Reddit & LinkedIn Post Drafts

---

## Reddit Post (r/rust, r/Python, r/programming)

**Title:** fuzzgpu — GPU-accelerated fuzzy string matching that's a drop-in rapidfuzz replacement (2-7x faster, no CUDA)

**Body:**

Hey everyone! I built `fuzzgpu` — a fuzzy string matching library in Rust with Python bindings that uses WebGPU for GPU acceleration.

**Why?** I needed to do fuzzy matching on millions of strings for a data deduplication pipeline. rapidfuzz is great but CPU-only, and at scale the bottleneck is raw throughput. So I wrote GPU compute shaders (WGSL) that run on any GPU via wgpu — no CUDA required.

**Key numbers (50K pairs, integrated Intel Iris Xe):**

| Algorithm | fuzzgpu | rapidfuzz | Speedup |
|---|---|---|---|
| Damerau-Levenshtein | 9.91 ms | 66.57 ms | **6.7x** |
| Levenshtein (Myers SIMD) | 3.36 ms | 4.46 ms | 1.3x |
| Jaro-Winkler | 3.25 ms | 6.69 ms | **2.1x** |
| NW Affine Gap (GPU) | 26.95 ms | no API | — |

**What makes it different:**

1. **Drop-in replacement** — Same API as rapidfuzz. Verified over 169,744 pairs, 0 mismatches.
2. **No CUDA** — Runs on Intel Iris Xe, AMD, Apple Metal, Vulkan, DX12 via WebGPU.
3. **Unrestricted Damerau-Levenshtein** — rapidfuzz only does OSA (restricted transpositions). fuzzgpu does the real algorithm.
4. **Myers bit-vector SIMD** — 8 distances computed simultaneously in one AVX512 vector. O(n) with zero inner DP loop.
5. **13 GPU shaders** — Levenshtein, Jaro, Damerau, Needleman-Wunsch, all with matrix variants.
6. **Zero-allocation `*_into()` APIs** — Write directly into numpy buffers, no Python object allocation.
7. **Needleman-Wunsch Affine Gap GPU batch** — rapidfuzz has no API for this at all.

**Install:** `pip install fuzzgpu`

**Architecture:** Rust core → PyO3 bindings → wgpu (WGSL shaders) + Rayon (CPU parallel). Works on Windows, macOS, Linux, and WebAssembly.

**Tech deep-dive:** During development I traced a crash through the Vulkan loader's `vkSetDebugUtilsObjectNameEXT` to a use-after-free in the loader's handle table — turns out it's a known issue in Intel's Vulkan ICD (KhronosGroup/Vulkan-Loader#1863). PR open upstream.

GitHub: https://github.com/kuntal-devrat/fuzzgpu

---

## LinkedIn Post

**fuzzgpu: GPU-accelerated fuzzy matching for Python — drop-in rapidfuzz replacement**

I built a fuzzy string matching library that runs on any GPU — no CUDA required.

Using WebGPU compute shaders (WGSL) via Rust + PyO3, fuzzgpu achieves 2-7x speedups over rapidfuzz on batch fuzzy matching:

- 50,000 Damerau-Levenshtein pairs in 9.91ms (vs 66.57ms = 6.7x)
- 1,000,000 distance evaluations (1000x1000 matrix) in under 15s
- Myers bit-vector SIMD: 8 distances in one AVX512 vector

**The hard parts:**

1. Vulkan loader use-after-free under concurrent dispatch (traced through minidumps to freed heap memory in vkSetDebugUtilsObjectNameEXT)
2. Smart metric-aware GPU routing: discrete GPU → threshold 64 pairs, integrated GPU → threshold 500, virtual GPU → threshold 2000
3. Cross-platform GPU compute that actually works on integrated GPUs (Intel Iris Xe, AMD Radeon)

**What makes it production-ready:**
- Drop-in rapidfuzz API compatibility (169,744 pairs tested, 0 mismatches)
- Unrestricted Damerau-Levenshtein (rapidfuzz only does OSA)
- Needleman-Wunsch Affine Gap GPU batch (rapidfuzz has no API for this)
- Zero-allocation numpy output buffers
- WebAssembly support for browser deployment
- pyo3 0.29 + numpy 0.29 (just upgraded from 0.23)

`pip install fuzzgpu` | github.com/kuntal-devrat/fuzzgpu

---

## Hashtags

#rust #python #gpu #webgpu #fuzzymatching #rapidfuzz #stringmatching #performance #simd #wgpu #pyo3 #opensource
