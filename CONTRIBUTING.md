# Contributing to FuzzGPU

Thank you for your interest in contributing to **FuzzGPU**! We welcome contributions from the community, whether reporting bugs, suggesting new features, improving documentation, or submitting pull requests.

## Code of Conduct

All contributors and maintainers are expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

---

## Architecture Overview

FuzzGPU is structured as a multi-crate Rust workspace with Python and WebAssembly bindings:

- **`crates/fuzzgpu-core/`**: The core computational engine. Implements:
  - Myers (1999) bit-vector Levenshtein and AVX-512 / AVX2 / NEON vectorization (`src/simd.rs`).
  - Lowrance-Wagner unrestricted Damerau-Levenshtein (`src/damerau.rs`).
  - Needleman-Wunsch global alignment and Gotoh (1982) affine gaps (`src/needleman.rs`).
  - Jaro and Jaro-Winkler distance and similarity (`src/jaro.rs`).
  - RapidFuzz-compatible fuzzy ratios and search extractors (`src/fuzz.rs`).
  - WebGPU / WGSL hardware acceleration abstraction via `wgpu` (`src/gpu.rs`, `src/shaders/`).
- **`crates/fuzzgpu-python/`**: PyO3 bindings compiling the C-extension module `fuzzgpu.fuzzgpu`.
- **`python/fuzzgpu/`**: High-level Python package providing full `rapidfuzz` drop-in compatibility:
  - `fuzz.py`: Fuzzy scorers (`ratio`, `partial_ratio`, `token_sort_ratio`, `WRatio`, `cdist`, etc.).
  - `process.py`: Search extractors (`extract`, `extractOne`, `cdist`).
  - `distance/`: RapidFuzz distance modules (`Levenshtein`, `DamerauLevenshtein`, `Jaro`, `JaroWinkler`, `Hamming`, `Indel`, `OSA`, `LCSseq`, `Prefix`, `Postfix`).
- **`crates/fuzzgpu-wasm/`**: WebAssembly bindings using `wasm-bindgen` for browser and Node.js environments.
- **`fuzz/`**: LibFuzzer targets and deterministic differential self-harnesses.
- **`tests/`**: Pytest test suite covering invariants, concurrency, edge cases, and differential tests against RapidFuzz.

---

## Development Setup

### Prerequisites

- **Rust**: 1.87+ (recommended: latest stable toolchain via `rustup`).
- **Python**: 3.10+ (CPython 3.10–3.13 or 3.14t free-threaded).
- **Maturin**: `pip install maturin`.
- **Optional for WASM**: `wasm-pack` and Node.js 18+.

### Setting Up Local Environment

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kuntal-devrat/fuzzgpu.git
   cd fuzzgpu
   ```

2. **Set up Python virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install maturin pytest pytest-timeout numpy rapidfuzz
   ```

3. **Build the extension in development mode**:
   ```bash
   maturin develop --release
   ```

---

## Testing & Quality Checks

Before submitting a PR, ensure all tests pass and code is cleanly formatted.

### Running Rust Tests

```bash
# Test workspace with default features (GPU + CPU)
cargo test --workspace

# Test with no default features (CPU-only build)
cargo test --workspace --no-default-features --no-run

# Run linter checks
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo clippy --workspace --all-targets --no-default-features -- -D warnings
```

### Running Python Tests

```bash
# Run pytest test suite
pytest tests -v

# Run RapidFuzz differential property tests
pytest tests/test_rapidfuzz_differential.py -v
```

### Running Fuzz Self-Harness

```bash
cargo test --manifest-path fuzz/Cargo.toml --lib --release
```

### Running WebAssembly Tests

```bash
cd crates/fuzzgpu-wasm
wasm-pack test --node
```

---

## Pull Request Guidelines

1. **Create a topic branch**: `git checkout -b feature/your-feature-name`.
2. **Adhere to Code Style**:
   - Run `cargo fmt --all` on Rust code.
   - Maintain full type stubs (`.pyi`) for any new public Python APIs.
3. **Add Tests**: Include unit tests and, where applicable, differential property tests.
4. **Documentation**: Update `README.md` or doc comments if you alter public behavior or add configuration options.
5. **PR Description**: Detail the motivation, approach, and how you verified your changes.
