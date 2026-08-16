use pyo3::prelude::*;
use pyo3::types::{PyString, PyStringMethods};
use std::sync::OnceLock;

#[cfg(feature = "gpu")]
use fuzzgpu_core::gpu::GpuEngine;

// ── Cached Environment Configuration ─────────────────────────

static FORCE_GPU_CACHE: OnceLock<bool> = OnceLock::new();
static DEBUG_CACHE: OnceLock<bool> = OnceLock::new();

#[inline]
fn is_force_gpu() -> bool {
    *FORCE_GPU_CACHE.get_or_init(|| {
        if let Ok(v) = std::env::var("FUZZGPU_FORCE_GPU") {
            v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes")
        } else {
            false
        }
    })
}

#[inline]
fn is_debug_mode() -> bool {
    *DEBUG_CACHE.get_or_init(|| {
        if let Ok(v) = std::env::var("FUZZGPU_DEBUG") {
            v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes")
        } else {
            false
        }
    })
}

/// Zero-copy extraction of a sequence of strings.
///
/// Each element is extracted as a `Bound<'py, PyString>` — a pointer to the
/// Python object, never a copy of its bytes (the previous signatures allocated
/// a `Vec<String>` per call). `&str` views are materialized on demand with
/// [`str_refs`], which borrows each string's UTF-8 representation via
/// `to_str()` (available because the bindings build with `abi3-py310`; the
/// old `abi3-py39` floor forced a transcode per string).
fn borrow_strs<'py>(seq: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyString>>> {
    let mut out = Vec::with_capacity(seq.len().unwrap_or(0));
    for item in seq.try_iter()? {
        out.push(item?.extract()?);
    }
    Ok(out)
}

/// Borrows the UTF-8 bytes of each held Python `str` as `&str` slices that are
/// valid as long as `bounds` (and the underlying Python objects) stay alive.
/// No string bytes are copied; Python's UTF-8 representation cache is reused.
fn str_refs<'a>(bounds: &'a [Bound<'_, PyString>]) -> PyResult<Vec<&'a str>> {
    bounds.iter().map(|b| b.to_str()).collect()
}

// ── Zero-allocation output buffers ───────────────────────────
//
// The `*_into` API writes results into a caller-supplied preallocated numpy
// array — no per-element Python objects are ever created, matching
// rapidfuzz's binding model for large batch calls. The plain
// `*_batch`/`*_cdist` functions box every result as a Python int/float,
// which dominates the return cost at 10k+ elements; `*_into` reduces that to
// one memcpy into preallocated memory.
//
// Implemented with rust-numpy's typed readwrite views, which validate the
// dtype exactly (`np.uint32` for distances, `np.float64` for ratios), the
// shape, and C-contiguity — and work under the package's `abi3-py310` floor
// (pyo3's own `buffer` module needs abi3-py311+ for the C buffer API).

/// `try_readwrite` fails (not panics) on read-only or concurrently borrowed
/// arrays; surface that as a clear `BufferError`.
fn readwrite_or_err<'py, T, D>(arr: &Bound<'py, numpy::PyArray<T, D>>) -> PyResult<numpy::PyReadwriteArray<'py, T, D>>
where
    T: numpy::Element,
    D: numpy::ndarray::Dimension,
{
    use numpy::PyArrayMethods;
    arr.try_readwrite().map_err(|_| {
        pyo3::exceptions::PyBufferError::new_err(
            "out must be a writable array not concurrently borrowed",
        )
    })
}

/// Validate `out` as a writable numpy `uint32` array of exactly `expected`
/// elements (1-D). Fails fast — before any compute — on wrong length,
/// wrong dtype, or a read-only array.
fn checked_u32_array1<'py>(out: &Bound<'py, PyAny>, expected: usize) -> PyResult<numpy::PyReadwriteArray1<'py, u32>> {
    use numpy::{PyArray1, PyUntypedArrayMethods};
    let arr = out.downcast::<PyArray1<u32>>()?;
    let rw = readwrite_or_err(arr)?;
    if rw.len() != expected {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "out must hold exactly {} uint32 elements, got {}",
            expected,
            rw.len()
        )));
    }
    Ok(rw)
}

/// Validate `out` as a writable numpy `float64` array of exactly `expected`
/// elements (1-D).
fn checked_f64_array1<'py>(out: &Bound<'py, PyAny>, expected: usize) -> PyResult<numpy::PyReadwriteArray1<'py, f64>> {
    use numpy::{PyArray1, PyUntypedArrayMethods};
    let arr = out.downcast::<PyArray1<f64>>()?;
    let rw = readwrite_or_err(arr)?;
    if rw.len() != expected {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "out must hold exactly {} float64 elements, got {}",
            expected,
            rw.len()
        )));
    }
    Ok(rw)
}

/// Validate `out` as a writable numpy `uint32` array of shape `(rows, cols)`
/// (2-D, row-major / C-contiguous) for cross-product matrices.
fn checked_u32_array2<'py>(out: &Bound<'py, PyAny>, rows: usize, cols: usize) -> PyResult<numpy::PyReadwriteArray2<'py, u32>> {
    use numpy::{PyArray2, PyUntypedArrayMethods};
    let arr = out.downcast::<PyArray2<u32>>()?;
    let rw = readwrite_or_err(arr)?;
    if rw.shape() != [rows, cols] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "out must have shape ({rows}, {cols}), got {:?}",
            rw.shape()
        )));
    }
    Ok(rw)
}

/// Validate `out` as a writable numpy `float64` array of shape `(rows, cols)`
/// (2-D, row-major / C-contiguous) for cross-product matrices.
fn checked_f64_array2<'py>(out: &Bound<'py, PyAny>, rows: usize, cols: usize) -> PyResult<numpy::PyReadwriteArray2<'py, f64>> {
    use numpy::{PyArray2, PyUntypedArrayMethods};
    let arr = out.downcast::<PyArray2<f64>>()?;
    let rw = readwrite_or_err(arr)?;
    if rw.shape() != [rows, cols] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "out must have shape ({rows}, {cols}), got {:?}",
            rw.shape()
        )));
    }
    Ok(rw)
}

/// Copy `values` into a validated 1-D `u32` view (GIL held; the view keeps
/// the numpy array alive and pinned).
fn write_u32_array1(mut rw: numpy::PyReadwriteArray1<'_, u32>, values: impl IntoIterator<Item = u32>) -> PyResult<()> {
    let slice = rw
        .as_slice_mut()
        .map_err(|_| pyo3::exceptions::PyBufferError::new_err("out must be C-contiguous"))?;
    for (dst, src) in slice.iter_mut().zip(values) {
        *dst = src;
    }
    Ok(())
}

/// Copy `values` (row-major) into a validated 2-D `u32` view.
fn write_u32_array2(mut rw: numpy::PyReadwriteArray2<'_, u32>, values: impl IntoIterator<Item = u32>) -> PyResult<()> {
    let slice = rw
        .as_slice_mut()
        .map_err(|_| pyo3::exceptions::PyBufferError::new_err("out must be C-contiguous"))?;
    for (dst, src) in slice.iter_mut().zip(values) {
        *dst = src;
    }
    Ok(())
}

/// Copy `values` into a validated 1-D `f64` view.
fn write_f64_array1(mut rw: numpy::PyReadwriteArray1<'_, f64>, values: impl IntoIterator<Item = f64>) -> PyResult<()> {
    let slice = rw
        .as_slice_mut()
        .map_err(|_| pyo3::exceptions::PyBufferError::new_err("out must be C-contiguous"))?;
    for (dst, src) in slice.iter_mut().zip(values) {
        *dst = src;
    }
    Ok(())
}

/// Copy `values` (row-major) into a validated 2-D `f64` view.
fn write_f64_array2(mut rw: numpy::PyReadwriteArray2<'_, f64>, values: impl IntoIterator<Item = f64>) -> PyResult<()> {
    let slice = rw
        .as_slice_mut()
        .map_err(|_| pyo3::exceptions::PyBufferError::new_err("out must be C-contiguous"))?;
    for (dst, src) in slice.iter_mut().zip(values) {
        *dst = src;
    }
    Ok(())
}

// ── Levenshtein ──────────────────────────────────────────────

#[pyfunction]
#[pyo3(text_signature = "(a, b, /)")]
fn levenshtein_distance(py: Python, a: &str, b: &str) -> PyResult<u32> {
    // A single pair never amortizes GPU setup/upload/readback. Batch APIs
    // retain GPU routing once that fixed cost can be amortized.
    py.allow_threads(|| Ok(fuzzgpu_core::levenshtein_distance_raw(a, b)))
}

/// Shared compute for `levenshtein_batch` / `levenshtein_batch_into`: Myers
/// GPU kernel (auto-routed, CPU fallback).
fn levenshtein_batch_core(py: Python<'_>, pairs: &[(&str, &str)]) -> PyResult<Vec<u32>> {
    #[cfg(feature = "gpu")]
    {
        if !GpuEngine::is_cpu_only() {
            if let Ok(kernel) = fuzzgpu_core::levenshtein::gpu_ext::GpuLevenshteinKernel::get() {
                match py.allow_threads(|| kernel.compute(pairs)) {
                    Ok(res) => return Ok(res),
                    Err(e) => {
                        if is_force_gpu() {
                            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                                "GPU Levenshtein batch failed: {}",
                                e
                            )));
                        }
                        if is_debug_mode() {
                            log::warn!("fuzzgpu [fallback]: GPU kernel failed ({}), switching to Rayon CPU", e);
                        }
                    }
                }
            }
        }
    }
    let kernel = fuzzgpu_core::LevenshteinKernel;
    py.allow_threads(|| {
        kernel
            .compute(pairs)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })
}

#[pyfunction]
#[pyo3(text_signature = "(query, candidates, /)")]
fn levenshtein_batch(py: Python, query: &str, candidates: &Bound<'_, PyAny>) -> PyResult<Vec<u32>> {
    // Borrowed candidates (zero-copy; see `borrow_strs`).
    let cands = borrow_strs(candidates)?;
    let cands = str_refs(&cands)?;
    let pairs: Vec<(&str, &str)> = cands.iter().map(|c| (query, *c)).collect();
    levenshtein_batch_core(py, &pairs)
}

/// Shared compute for `levenshtein_cdist` / `levenshtein_cdist_into`.
fn levenshtein_cdist_core(py: Python<'_>, refs_a: &[&str], refs_b: &[&str]) -> PyResult<Vec<Vec<u32>>> {
    #[cfg(feature = "gpu")]
    {
        if !GpuEngine::is_cpu_only() {
            if let Ok(kernel) = fuzzgpu_core::levenshtein::gpu_ext::GpuLevenshteinKernel::get() {
                match py.allow_threads(|| kernel.compute_matrix(refs_a, refs_b)) {
                    Ok(res) => return Ok(res),
                    Err(e) => {
                        if is_force_gpu() {
                            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                                "GPU Levenshtein matrix failed: {}",
                                e
                            )));
                        }
                        if is_debug_mode() {
                            log::warn!("fuzzgpu [fallback]: GPU matrix failed ({}), switching to Rayon CPU", e);
                        }
                    }
                }
            }
        }
    }
    py.allow_threads(|| Ok(fuzzgpu_core::levenshtein::levenshtein_cdist_cpu(refs_a, refs_b)))
}

#[pyfunction]
#[pyo3(text_signature = "(list_a, list_b, /)")]
fn levenshtein_cdist(py: Python, list_a: &Bound<'_, PyAny>, list_b: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<u32>>> {
    let refs_a = borrow_strs(list_a)?;
    let refs_b = borrow_strs(list_b)?;
    let refs_a = str_refs(&refs_a)?;
    let refs_b = str_refs(&refs_b)?;
    levenshtein_cdist_core(py, &refs_a, &refs_b)
}

/// Zero-allocation batch: results are written into a caller-supplied writable
/// buffer (numpy `uint32`, `array.array('I')`, `memoryview`, …) instead of
/// boxing each distance as a Python int. `out` must be exactly
/// `len(candidates)` elements and is validated before any compute.
#[pyfunction]
#[pyo3(text_signature = "(query, candidates, out, /)")]
fn levenshtein_batch_into(py: Python<'_>, query: &str, candidates: &Bound<'_, PyAny>, out: &Bound<'_, PyAny>) -> PyResult<()> {
    let cands = borrow_strs(candidates)?;
    let cands = str_refs(&cands)?;
    let pairs: Vec<(&str, &str)> = cands.iter().map(|c| (query, *c)).collect();
    let rw = checked_u32_array1(out, pairs.len())?;
    let results = levenshtein_batch_core(py, &pairs)?;
    write_u32_array1(rw, results)
}

/// Zero-allocation cross-product matrix: `out` must be a writable, contiguous
/// `(len(list_a), len(list_b))` buffer of `uint32` (row-major).
#[pyfunction]
#[pyo3(text_signature = "(list_a, list_b, out, /)")]
fn levenshtein_cdist_into(py: Python<'_>, list_a: &Bound<'_, PyAny>, list_b: &Bound<'_, PyAny>, out: &Bound<'_, PyAny>) -> PyResult<()> {
    let refs_a = borrow_strs(list_a)?;
    let refs_b = borrow_strs(list_b)?;
    let refs_a = str_refs(&refs_a)?;
    let refs_b = str_refs(&refs_b)?;
    let rw = checked_u32_array2(out, refs_a.len(), refs_b.len())?;
    let results = levenshtein_cdist_core(py, &refs_a, &refs_b)?;
    write_u32_array2(rw, results.into_iter().flatten())
}

// ── Damerau-Levenshtein ──────────────────────────────────────

#[pyfunction]
#[pyo3(text_signature = "(a, b, /)")]
fn damerau_levenshtein_distance(py: Python, a: &str, b: &str) -> PyResult<u32> {
    py.allow_threads(|| Ok(fuzzgpu_core::damerau_levenshtein_distance(a, b)))
}

/// Shared compute for `damerau_levenshtein_batch` / `..._into`: GPU
/// Lowrance-Wagner kernel (auto-routed, CPU fallback).
fn damerau_batch_core(py: Python<'_>, query: &str, cands: &[&str]) -> PyResult<Vec<u32>> {
    let pairs: Vec<(&str, &str)> = cands.iter().map(|c| (query, *c)).collect();
    #[cfg(feature = "gpu")]
    {
        if !GpuEngine::is_cpu_only() {
            if let Ok(kernel) = fuzzgpu_core::damerau::gpu_ext::GpuDamerauKernel::get() {
                match py.allow_threads(|| kernel.compute_batch(&pairs)) {
                    Ok(res) => return Ok(res),
                    Err(e) => {
                        if is_force_gpu() {
                            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                                "GPU Damerau-Levenshtein batch failed: {}",
                                e
                            )));
                        }
                        if is_debug_mode() {
                            log::warn!("fuzzgpu [fallback]: GPU kernel failed ({}), switching to Rayon CPU", e);
                        }
                    }
                }
            }
        }
    }
    py.allow_threads(|| Ok(fuzzgpu_core::damerau_levenshtein_batch(query, cands)))
}

#[pyfunction]
#[pyo3(text_signature = "(query, candidates, /)")]
fn damerau_levenshtein_batch(py: Python, query: &str, candidates: &Bound<'_, PyAny>) -> PyResult<Vec<u32>> {
    let refs = borrow_strs(candidates)?;
    let refs = str_refs(&refs)?;
    damerau_batch_core(py, query, &refs)
}

/// Shared compute for `damerau_levenshtein_cdist` / `..._into`.
fn damerau_cdist_core(py: Python<'_>, refs_a: &[&str], refs_b: &[&str]) -> PyResult<Vec<Vec<u32>>> {
    #[cfg(feature = "gpu")]
    {
        if !GpuEngine::is_cpu_only() {
            if let Ok(kernel) = fuzzgpu_core::damerau::gpu_ext::GpuDamerauKernel::get() {
                match py.allow_threads(|| kernel.compute_matrix(refs_a, refs_b)) {
                    Ok(res) => return Ok(res),
                    Err(e) => {
                        if is_force_gpu() {
                            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                                "GPU Damerau-Levenshtein matrix failed: {}",
                                e
                            )));
                        }
                        if is_debug_mode() {
                            log::warn!("fuzzgpu [fallback]: GPU matrix failed ({}), switching to Rayon CPU", e);
                        }
                    }
                }
            }
        }
    }
    py.allow_threads(|| Ok(fuzzgpu_core::damerau_levenshtein_cdist(refs_a, refs_b)))
}

#[pyfunction]
#[pyo3(text_signature = "(list_a, list_b, /)")]
fn damerau_levenshtein_cdist(py: Python, list_a: &Bound<'_, PyAny>, list_b: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<u32>>> {
    let refs_a = borrow_strs(list_a)?;
    let refs_b = borrow_strs(list_b)?;
    let refs_a = str_refs(&refs_a)?;
    let refs_b = str_refs(&refs_b)?;
    damerau_cdist_core(py, &refs_a, &refs_b)
}

/// Zero-allocation Damerau batch into a writable `uint32` buffer (see
/// [`levenshtein_batch_into`]).
#[pyfunction]
#[pyo3(text_signature = "(query, candidates, out, /)")]
fn damerau_levenshtein_batch_into(py: Python<'_>, query: &str, candidates: &Bound<'_, PyAny>, out: &Bound<'_, PyAny>) -> PyResult<()> {
    let refs = borrow_strs(candidates)?;
    let refs = str_refs(&refs)?;
    let rw = checked_u32_array1(out, refs.len())?;
    let results = damerau_batch_core(py, query, &refs)?;
    write_u32_array1(rw, results)
}

/// Zero-allocation Damerau cross-product matrix into a writable, contiguous
/// `(len(list_a), len(list_b))` buffer of `uint32` (row-major).
#[pyfunction]
#[pyo3(text_signature = "(list_a, list_b, out, /)")]
fn damerau_levenshtein_cdist_into(py: Python<'_>, list_a: &Bound<'_, PyAny>, list_b: &Bound<'_, PyAny>, out: &Bound<'_, PyAny>) -> PyResult<()> {
    let refs_a = borrow_strs(list_a)?;
    let refs_b = borrow_strs(list_b)?;
    let refs_a = str_refs(&refs_a)?;
    let refs_b = str_refs(&refs_b)?;
    let rw = checked_u32_array2(out, refs_a.len(), refs_b.len())?;
    let results = damerau_cdist_core(py, &refs_a, &refs_b)?;
    write_u32_array2(rw, results.into_iter().flatten())
}

#[pyfunction]
#[pyo3(text_signature = "(a, b, /)")]
fn damerau_ratio(py: Python, a: &str, b: &str) -> PyResult<f64> {
    py.allow_threads(|| Ok(fuzzgpu_core::damerau_ratio(a, b)))
}

// ── Needleman-Wunsch (i64 Score Support) ─────────────────────

#[pyfunction]
#[pyo3(text_signature = "(a, b, match_score, mismatch_score, gap_penalty, /)")]
fn needleman_wunsch_score(py: Python, a: &str, b: &str, match_score: i64, mismatch_score: i64, gap_penalty: i64) -> PyResult<i64> {
    py.allow_threads(|| Ok(fuzzgpu_core::needleman_wunsch(a, b, match_score, mismatch_score, gap_penalty)))
}

#[pyfunction]
#[pyo3(text_signature = "(query, candidates, match_score, mismatch_score, gap_penalty, /)")]
fn needleman_wunsch_batch_fn(py: Python, query: &str, candidates: &Bound<'_, PyAny>, match_score: i64, mismatch_score: i64, gap_penalty: i64) -> PyResult<Vec<i64>> {
    let refs = borrow_strs(candidates)?;
    let refs = str_refs(&refs)?;
    py.allow_threads(|| Ok(fuzzgpu_core::needleman_wunsch_batch(query, &refs, match_score, mismatch_score, gap_penalty)))
}

#[pyfunction]
#[pyo3(text_signature = "(a, b, match_score, mismatch_score, gap_open, gap_extend, /)")]
fn needleman_wunsch_affine(py: Python, a: &str, b: &str, match_score: i64, mismatch_score: i64, gap_open: i64, gap_extend: i64) -> PyResult<i64> {
    py.allow_threads(|| Ok(fuzzgpu_core::needleman_wunsch_affine(a, b, match_score, mismatch_score, gap_open, gap_extend)))
}

#[pyfunction]
#[pyo3(text_signature = "(query, candidates, match_score, mismatch_score, gap_open, gap_extend, /)")]
fn needleman_wunsch_affine_batch(py: Python, query: &str, candidates: &Bound<'_, PyAny>, match_score: i64, mismatch_score: i64, gap_open: i64, gap_extend: i64) -> PyResult<Vec<i64>> {
    let refs = borrow_strs(candidates)?;
    let refs = str_refs(&refs)?;
    py.allow_threads(|| Ok(fuzzgpu_core::needleman_wunsch_affine_batch(query, &refs, match_score, mismatch_score, gap_open, gap_extend)))
}

// ── Jaro-Winkler ────────────────────────────────────────────

#[pyfunction]
#[pyo3(text_signature = "(a, b, /)")]
fn jaro_similarity(py: Python, a: &str, b: &str) -> PyResult<f64> {
    py.allow_threads(|| Ok(fuzzgpu_core::jaro(a, b)))
}

#[pyfunction]
#[pyo3(signature = (a, b, p = 0.1), text_signature = "(a, b, p=0.1, /)")]
fn jaro_winkler_similarity(py: Python, a: &str, b: &str, p: f64) -> PyResult<f64> {
    if !(0.0..=0.25).contains(&p) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Prefix scaling parameter 'p' must be between 0.0 and 0.25",
        ));
    }
    py.allow_threads(|| Ok(fuzzgpu_core::jaro_winkler(a, b, p)))
}

/// Shared compute for `jaro_winkler_batch` / `..._into`.
fn jaro_batch_core(py: Python<'_>, query: &str, cands: &[&str], p: f64) -> PyResult<Vec<f64>> {
    // WGSL currently exposes portable f32 arithmetic only. Preserve the public
    // f64 contract (and exact CPU/GPU parity) until a portable f64 shader path
    // is available; the CPU implementation remains SIMD/Rayon parallel.
    py.allow_threads(|| Ok(fuzzgpu_core::jaro_winkler_batch(query, cands, p)))
}

#[pyfunction]
#[pyo3(signature = (query, candidates, p = 0.1), text_signature = "(query, candidates, p=0.1, /)")]
fn jaro_winkler_batch_fn(py: Python, query: &str, candidates: &Bound<'_, PyAny>, p: f64) -> PyResult<Vec<f64>> {
    if !(0.0..=0.25).contains(&p) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Prefix scaling parameter 'p' must be between 0.0 and 0.25",
        ));
    }
    let cands = borrow_strs(candidates)?;
    let cands = str_refs(&cands)?;
    jaro_batch_core(py, query, &cands, p)
}

/// Shared compute for `jaro_winkler_cdist` / `..._into`.
fn jaro_cdist_core(py: Python<'_>, refs_a: &[&str], refs_b: &[&str], p: f64) -> PyResult<Vec<Vec<f64>>> {
    // See `jaro_batch_core`: favor exact f64 results over the f32 GPU shader.
    py.allow_threads(|| Ok(fuzzgpu_core::jaro::jaro_winkler_cdist_cpu(refs_a, refs_b, p)))
}

#[pyfunction]
#[pyo3(signature = (list_a, list_b, p = 0.1), text_signature = "(list_a, list_b, p=0.1, /)")]
fn jaro_winkler_cdist(py: Python, list_a: &Bound<'_, PyAny>, list_b: &Bound<'_, PyAny>, p: f64) -> PyResult<Vec<Vec<f64>>> {
    if !(0.0..=0.25).contains(&p) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Prefix scaling parameter 'p' must be between 0.0 and 0.25",
        ));
    }
    let refs_a = borrow_strs(list_a)?;
    let refs_b = borrow_strs(list_b)?;
    let refs_a = str_refs(&refs_a)?;
    let refs_b = str_refs(&refs_b)?;
    jaro_cdist_core(py, &refs_a, &refs_b, p)
}

/// Zero-allocation Jaro-Winkler batch into a writable `float64` buffer (see
/// [`levenshtein_batch_into`]).
#[pyfunction]
#[pyo3(signature = (query, candidates, out, p = 0.1), text_signature = "(query, candidates, out, p=0.1, /)")]
fn jaro_winkler_batch_into(py: Python<'_>, query: &str, candidates: &Bound<'_, PyAny>, out: &Bound<'_, PyAny>, p: f64) -> PyResult<()> {
    if !(0.0..=0.25).contains(&p) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Prefix scaling parameter 'p' must be between 0.0 and 0.25",
        ));
    }
    let cands = borrow_strs(candidates)?;
    let cands = str_refs(&cands)?;
    let rw = checked_f64_array1(out, cands.len())?;
    let results = jaro_batch_core(py, query, &cands, p)?;
    write_f64_array1(rw, results)
}

/// Zero-allocation Jaro-Winkler cross-product matrix into a writable,
/// contiguous `(len(list_a), len(list_b))` buffer of `float64` (row-major).
#[pyfunction]
#[pyo3(signature = (list_a, list_b, out, p = 0.1), text_signature = "(list_a, list_b, out, p=0.1, /)")]
fn jaro_winkler_cdist_into(py: Python<'_>, list_a: &Bound<'_, PyAny>, list_b: &Bound<'_, PyAny>, out: &Bound<'_, PyAny>, p: f64) -> PyResult<()> {
    if !(0.0..=0.25).contains(&p) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Prefix scaling parameter 'p' must be between 0.0 and 0.25",
        ));
    }
    let refs_a = borrow_strs(list_a)?;
    let refs_b = borrow_strs(list_b)?;
    let refs_a = str_refs(&refs_a)?;
    let refs_b = str_refs(&refs_b)?;
    let rw = checked_f64_array2(out, refs_a.len(), refs_b.len())?;
    let results = jaro_cdist_core(py, &refs_a, &refs_b, p)?;
    write_f64_array2(rw, results.into_iter().flatten())
}

// ── Fuzzy matching ──────────────────────────────────────────

#[pyfunction]
#[pyo3(text_signature = "(a, b, /)")]
fn fuzz_ratio(py: Python, a: &str, b: &str) -> PyResult<f64> {
    py.allow_threads(|| Ok(fuzzgpu_core::ratio(a, b)))
}

#[pyfunction]
#[pyo3(text_signature = "(a, b, /)")]
fn fuzz_partial_ratio(py: Python, a: &str, b: &str) -> PyResult<f64> {
    py.allow_threads(|| Ok(fuzzgpu_core::partial_ratio(a, b)))
}

#[pyfunction]
#[pyo3(text_signature = "(a, b, /)")]
fn fuzz_token_sort_ratio(py: Python, a: &str, b: &str) -> PyResult<f64> {
    py.allow_threads(|| Ok(fuzzgpu_core::token_sort_ratio(a, b)))
}

#[pyfunction]
#[pyo3(text_signature = "(a, b, /)")]
fn fuzz_token_set_ratio(py: Python, a: &str, b: &str) -> PyResult<f64> {
    py.allow_threads(|| Ok(fuzzgpu_core::token_set_ratio(a, b)))
}

#[pyfunction]
#[pyo3(text_signature = "(a, b, /)")]
fn fuzz_wratio(py: Python, a: &str, b: &str) -> PyResult<f64> {
    py.allow_threads(|| Ok(fuzzgpu_core::wratio(a, b)))
}

#[pyfunction]
#[pyo3(text_signature = "(query, candidates, /)")]
fn fuzz_ratio_batch(py: Python, query: &str, candidates: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    let refs = borrow_strs(candidates)?;
    let refs = str_refs(&refs)?;
    py.allow_threads(|| Ok(fuzzgpu_core::ratio_batch(query, &refs)))
}

#[pyfunction]
#[pyo3(signature = (query, choices, score_cutoff = 0.0, limit = 5), text_signature = "(query, choices, score_cutoff=0.0, limit=5, /)")]
fn fuzz_extract(py: Python, query: &str, choices: &Bound<'_, PyAny>, score_cutoff: f64, limit: usize) -> PyResult<Vec<(String, f64, usize)>> {
    let refs = borrow_strs(choices)?;
    let refs = str_refs(&refs)?;
    py.allow_threads(|| Ok(fuzzgpu_core::extract(query, &refs, score_cutoff, limit)))
}

#[pyfunction]
#[pyo3(signature = (query, choices, score_cutoff = 0.0), text_signature = "(query, choices, score_cutoff=0.0, /)")]
fn fuzz_extract_one(py: Python, query: &str, choices: &Bound<'_, PyAny>, score_cutoff: f64) -> PyResult<Option<(String, f64, usize)>> {
    let refs = borrow_strs(choices)?;
    let refs = str_refs(&refs)?;
    py.allow_threads(|| Ok(fuzzgpu_core::extract_one(query, &refs, score_cutoff)))
}

// ── Optimized algorithm variants ────────────────────────────

#[pyfunction]
#[pyo3(text_signature = "(a, b, /)")]
fn levenshtein_myers(py: Python, a: &str, b: &str) -> PyResult<u32> {
    py.allow_threads(|| {
        if a.is_ascii() && b.is_ascii() {
            Ok(fuzzgpu_core::levenshtein_myers(a.as_bytes(), b.as_bytes()))
        } else {
            Ok(fuzzgpu_core::levenshtein_distance_raw(a, b))
        }
    })
}

#[pyfunction]
#[pyo3(text_signature = "(a, b, match_score, mismatch_score, gap_penalty, /)")]
fn needleman_wunsch_striped(py: Python, a: &str, b: &str, match_score: i64, mismatch_score: i64, gap_penalty: i64) -> PyResult<i64> {
    py.allow_threads(|| {
        if a.is_ascii() && b.is_ascii() {
            Ok(fuzzgpu_core::needleman_wunsch_striped(a.as_bytes(), b.as_bytes(), match_score, mismatch_score, gap_penalty))
        } else {
            Ok(fuzzgpu_core::needleman_wunsch(a, b, match_score, mismatch_score, gap_penalty))
        }
    })
}

#[pyfunction]
#[pyo3(text_signature = "(a, b, /)")]
fn jaro_optimized(py: Python, a: &str, b: &str) -> PyResult<f64> {
    py.allow_threads(|| {
        if a.is_ascii() && b.is_ascii() {
            Ok(fuzzgpu_core::jaro_optimized(a.as_bytes(), b.as_bytes()))
        } else {
            Ok(fuzzgpu_core::jaro(a, b))
        }
    })
}

// ── Control & Utilities ─────────────────────────────────────

#[pyfunction]
#[pyo3(text_signature = "(cpu_only, /)")]
fn set_cpu_only(cpu_only: bool) {
    #[cfg(feature = "gpu")]
    {
        GpuEngine::set_cpu_only(cpu_only);
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = cpu_only;
    }
}

#[pyfunction]
#[pyo3(text_signature = "()")]
fn is_gpu_available() -> bool {
    #[cfg(feature = "gpu")]
    {
        GpuEngine::is_available()
    }
    #[cfg(not(feature = "gpu"))]
    {
        false
    }
}

#[pyfunction]
#[pyo3(text_signature = "()")]
fn warmup() -> (bool, String) {
    #[cfg(feature = "gpu")]
    {
        if GpuEngine::is_cpu_only() {
            return (false, "CPU-only mode is active (set via set_cpu_only or FUZZGPU_USE_CPU)".into());
        }
        match GpuEngine::get() {
            Ok(engine) => (true, format!("GPU initialized: {} ({})", engine.info.name, engine.info.backend)),
            Err(e) => (false, format!("GPU initialization failed ({}), using Rayon CPU fallback", e)),
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        (false, "Built without GPU support (cpu-only)".into())
    }
}

#[pyfunction]
#[pyo3(text_signature = "()")]
fn gpu_info() -> PyResult<String> {
    #[cfg(feature = "gpu")]
    {
        if GpuEngine::is_cpu_only() {
            return Ok("CPU-only fallback mode (forced by user/env)".into());
        }
        match GpuEngine::get() {
            Ok(engine) => Ok(format!("{} ({})", engine.info.name, engine.info.backend)),
            Err(e) => Ok(format!("CPU-only fallback mode (GPU unavailable: {})", e)),
        }
    }
    #[cfg(not(feature = "gpu"))]
    { Ok("CPU-only mode (built without gpu feature)".into()) }
}

#[pyfunction]
#[pyo3(text_signature = "()")]
fn hardware_info() -> String {
    #[cfg(feature = "gpu")]
    {
        if GpuEngine::is_cpu_only() {
            return format!(
                "CPU-only fallback mode (GPU dispatch threshold: N/A, last routing: CPU)"
            );
        }
        match GpuEngine::get() {
            Ok(engine) => {
                let (gpu_pairs, cpu_pairs) = GpuEngine::last_routing();
                format!(
                    "GPU: {} ({}, {}) | auto threshold: {} | override: {} | last routing: {} GPU / {} CPU pairs | gpu_threshold: {}",
                    engine.info.name,
                    engine.info.backend,
                    engine.info.device_type,
                    GpuEngine::auto_gpu_threshold(&engine.info.device_type),
                    GpuEngine::gpu_threshold_override()
                        .map(|t| t.to_string())
                        .unwrap_or_else(|| "auto".into()),
                    gpu_pairs,
                    cpu_pairs,
                    engine.effective_gpu_threshold(),
                )
            }
            Err(e) => format!("CPU-only fallback mode (GPU unavailable: {})", e),
        }
    }
    #[cfg(not(feature = "gpu"))]
    { "CPU-only mode (built without gpu feature)".into() }
}

#[pyfunction]
#[pyo3(signature = (threshold = None))]
fn set_gpu_threshold(threshold: Option<usize>) {
    #[cfg(feature = "gpu")]
    {
        GpuEngine::set_gpu_threshold(threshold);
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = threshold;
    }
}

// ── Module ──────────────────────────────────────────────────

#[pymodule]
fn fuzzgpu(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Levenshtein
    m.add_function(wrap_pyfunction!(levenshtein_distance, m)?)?;
    m.add_function(wrap_pyfunction!(levenshtein_batch, m)?)?;
    m.add_function(wrap_pyfunction!(levenshtein_batch_into, m)?)?;
    m.add_function(wrap_pyfunction!(levenshtein_cdist, m)?)?;
    m.add_function(wrap_pyfunction!(levenshtein_cdist_into, m)?)?;
    // Damerau-Levenshtein
    m.add_function(wrap_pyfunction!(damerau_levenshtein_distance, m)?)?;
    m.add_function(wrap_pyfunction!(damerau_levenshtein_batch, m)?)?;
    m.add_function(wrap_pyfunction!(damerau_levenshtein_batch_into, m)?)?;
    m.add_function(wrap_pyfunction!(damerau_levenshtein_cdist, m)?)?;
    m.add_function(wrap_pyfunction!(damerau_levenshtein_cdist_into, m)?)?;
    m.add_function(wrap_pyfunction!(damerau_ratio, m)?)?;
    // Needleman-Wunsch
    m.add_function(wrap_pyfunction!(needleman_wunsch_score, m)?)?;
    m.add_function(wrap_pyfunction!(needleman_wunsch_batch_fn, m)?)?;
    m.add_function(wrap_pyfunction!(needleman_wunsch_affine, m)?)?;
    m.add_function(wrap_pyfunction!(needleman_wunsch_affine_batch, m)?)?;
    // Jaro-Winkler
    m.add_function(wrap_pyfunction!(jaro_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(jaro_winkler_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(jaro_winkler_batch_fn, m)?)?;
    m.add_function(wrap_pyfunction!(jaro_winkler_batch_into, m)?)?;
    m.add_function(wrap_pyfunction!(jaro_winkler_cdist, m)?)?;
    m.add_function(wrap_pyfunction!(jaro_winkler_cdist_into, m)?)?;
    // Fuzzy matching
    m.add_function(wrap_pyfunction!(fuzz_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(fuzz_partial_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(fuzz_token_sort_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(fuzz_token_set_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(fuzz_wratio, m)?)?;
    m.add_function(wrap_pyfunction!(fuzz_ratio_batch, m)?)?;
    m.add_function(wrap_pyfunction!(fuzz_extract, m)?)?;
    m.add_function(wrap_pyfunction!(fuzz_extract_one, m)?)?;
    // Optimized variants
    m.add_function(wrap_pyfunction!(levenshtein_myers, m)?)?;
    m.add_function(wrap_pyfunction!(needleman_wunsch_striped, m)?)?;
    m.add_function(wrap_pyfunction!(jaro_optimized, m)?)?;
    // Control & Utilities
    m.add_function(wrap_pyfunction!(set_cpu_only, m)?)?;
    m.add_function(wrap_pyfunction!(set_gpu_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(is_gpu_available, m)?)?;
    m.add_function(wrap_pyfunction!(warmup, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_info, m)?)?;
    m.add_function(wrap_pyfunction!(hardware_info, m)?)?;
    // Dynamic version directly from Cargo package metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
