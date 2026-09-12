## Description

Please include a summary of the changes and the related issue. Also include relevant motivation and context.

Fixes # (issue)

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Performance optimization (GPU/CPU/SIMD improvement without breaking behavior)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] CI / Infrastructure improvement

## Areas Affected

- [ ] `crates/fuzzgpu-core` (Rust algorithms & WGSL shaders)
- [ ] `crates/fuzzgpu-python` (PyO3 bindings)
- [ ] `crates/fuzzgpu-wasm` (Wasm bindings)
- [ ] `python/fuzzgpu` (Python package & type stubs)
- [ ] Documentation / Benchmarks / CI

## Benchmarks / Performance Impact

If this PR touches performance-critical code (GPU shaders, SIMD Myers, SIMD Jaro-Winkler, Python dispatch), please include benchmark results (before vs after):

```text
# Example benchmark comparison or notes
```

## Checklist

- [ ] My code follows the style guidelines of this project.
- [ ] I have performed a self-review of my own code.
- [ ] I have commented my code, particularly in hard-to-understand areas (e.g., WGSL shaders, SIMD intrinsics).
- [ ] I have made corresponding changes to the documentation (README, docstrings, type stubs `.pyi`).
- [ ] My changes pass all existing tests:
  - `cargo test --workspace`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo clippy --workspace --all-targets --no-default-features -- -D warnings`
  - `pytest tests`
- [ ] I have added tests that prove my fix is effective or that my feature works.
- [ ] Parity with RapidFuzz and mathematical distance invariants are preserved.
