use pyo3::prelude::*;
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

// ── Levenshtein ──────────────────────────────────────────────

#[pyfunction]
#[pyo3(text_signature = "(a, b, /)")]
fn levenshtein_distance(py: Python, a: &str, b: &str) -> PyResult<u32> {
    #[cfg(feature = "gpu")]
    {
        if !GpuEngine::is_cpu_only() {
            if let Ok(kernel) = fuzzgpu_core::levenshtein::gpu_ext::GpuLevenshteinKernel::get() {
                match py.allow_threads(|| kernel.compute(&[(a, b)])) {
                    Ok(res) => return Ok(res[0]),
                    Err(e) => {
                        if is_force_gpu() {
                            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                                "GPU Levenshtein compute failed: {}",
                                e
                            )));
                        }
                    }
                }
            }
        }
    }
    py.allow_threads(|| Ok(fuzzgpu_core::levenshtein_distance_raw(a, b)))
}

#[pyfunction]
#[pyo3(text_signature = "(query, candidates, /)")]
fn levenshtein_batch(py: Python, query: String, candidates: Vec<String>) -> PyResult<Vec<u32>> {
    let pairs: Vec<(&str, &str)> = candidates.iter().map(|c| (query.as_str(), c.as_str())).collect();
    #[cfg(feature = "gpu")]
    {
        if !GpuEngine::is_cpu_only() {
            if let Ok(kernel) = fuzzgpu_core::levenshtein::gpu_ext::GpuLevenshteinKernel::get() {
                match py.allow_threads(|| kernel.compute(&pairs)) {
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
            .compute(&pairs)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })
}

#[pyfunction]
#[pyo3(text_signature = "(list_a, list_b, /)")]
fn levenshtein_cdist(py: Python, list_a: Vec<String>, list_b: Vec<String>) -> PyResult<Vec<Vec<u32>>> {
    let refs_a: Vec<&str> = list_a.iter().map(|s| s.as_str()).collect();
    let refs_b: Vec<&str> = list_b.iter().map(|s| s.as_str()).collect();
    #[cfg(feature = "gpu")]
    {
        if !GpuEngine::is_cpu_only() {
            if let Ok(kernel) = fuzzgpu_core::levenshtein::gpu_ext::GpuLevenshteinKernel::get() {
                match py.allow_threads(|| kernel.compute_matrix(&refs_a, &refs_b)) {
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
    py.allow_threads(|| Ok(fuzzgpu_core::levenshtein::levenshtein_cdist_cpu(&refs_a, &refs_b)))
}

// ── Damerau-Levenshtein ──────────────────────────────────────

#[pyfunction]
#[pyo3(text_signature = "(a, b, /)")]
fn damerau_levenshtein_distance(py: Python, a: &str, b: &str) -> PyResult<u32> {
    py.allow_threads(|| Ok(fuzzgpu_core::damerau_levenshtein_distance(a, b)))
}

#[pyfunction]
#[pyo3(text_signature = "(query, candidates, /)")]
fn damerau_levenshtein_batch(py: Python, query: String, candidates: Vec<String>) -> PyResult<Vec<u32>> {
    let refs: Vec<&str> = candidates.iter().map(|s| s.as_str()).collect();
    py.allow_threads(|| Ok(fuzzgpu_core::damerau_levenshtein_batch(&query, &refs)))
}

#[pyfunction]
#[pyo3(text_signature = "(list_a, list_b, /)")]
fn damerau_levenshtein_cdist(py: Python, list_a: Vec<String>, list_b: Vec<String>) -> PyResult<Vec<Vec<u32>>> {
    let refs_a: Vec<&str> = list_a.iter().map(|s| s.as_str()).collect();
    let refs_b: Vec<&str> = list_b.iter().map(|s| s.as_str()).collect();
    py.allow_threads(|| Ok(fuzzgpu_core::damerau_levenshtein_cdist(&refs_a, &refs_b)))
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
fn needleman_wunsch_batch_fn(py: Python, query: String, candidates: Vec<String>, match_score: i64, mismatch_score: i64, gap_penalty: i64) -> PyResult<Vec<i64>> {
    let refs: Vec<&str> = candidates.iter().map(|s| s.as_str()).collect();
    py.allow_threads(|| Ok(fuzzgpu_core::needleman_wunsch_batch(&query, &refs, match_score, mismatch_score, gap_penalty)))
}

#[pyfunction]
#[pyo3(text_signature = "(a, b, match_score, mismatch_score, gap_open, gap_extend, /)")]
fn needleman_wunsch_affine(py: Python, a: &str, b: &str, match_score: i64, mismatch_score: i64, gap_open: i64, gap_extend: i64) -> PyResult<i64> {
    py.allow_threads(|| Ok(fuzzgpu_core::needleman_wunsch_affine(a, b, match_score, mismatch_score, gap_open, gap_extend)))
}

#[pyfunction]
#[pyo3(text_signature = "(query, candidates, match_score, mismatch_score, gap_open, gap_extend, /)")]
fn needleman_wunsch_affine_batch(py: Python, query: String, candidates: Vec<String>, match_score: i64, mismatch_score: i64, gap_open: i64, gap_extend: i64) -> PyResult<Vec<i64>> {
    let refs: Vec<&str> = candidates.iter().map(|s| s.as_str()).collect();
    py.allow_threads(|| Ok(fuzzgpu_core::needleman_wunsch_affine_batch(&query, &refs, match_score, mismatch_score, gap_open, gap_extend)))
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

#[pyfunction]
#[pyo3(signature = (query, candidates, p = 0.1), text_signature = "(query, candidates, p=0.1, /)")]
fn jaro_winkler_batch_fn(py: Python, query: String, candidates: Vec<String>, p: f64) -> PyResult<Vec<f64>> {
    if !(0.0..=0.25).contains(&p) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Prefix scaling parameter 'p' must be between 0.0 and 0.25",
        ));
    }
    let pairs: Vec<(&str, &str)> = candidates.iter().map(|c| (query.as_str(), c.as_str())).collect();
    #[cfg(feature = "gpu")]
    {
        if !GpuEngine::is_cpu_only() {
            if let Ok(kernel) = fuzzgpu_core::jaro::gpu_ext::GpuJaroKernel::get() {
                match py.allow_threads(|| kernel.compute_batch(&pairs, p)) {
                    Ok(res) => return Ok(res),
                    Err(e) => {
                        if is_force_gpu() {
                            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                                "GPU Jaro-Winkler batch failed: {}",
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
    let refs: Vec<&str> = candidates.iter().map(|s| s.as_str()).collect();
    py.allow_threads(|| Ok(fuzzgpu_core::jaro_winkler_batch(&query, &refs, p)))
}

#[pyfunction]
#[pyo3(signature = (list_a, list_b, p = 0.1), text_signature = "(list_a, list_b, p=0.1, /)")]
fn jaro_winkler_cdist(py: Python, list_a: Vec<String>, list_b: Vec<String>, p: f64) -> PyResult<Vec<Vec<f64>>> {
    if !(0.0..=0.25).contains(&p) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Prefix scaling parameter 'p' must be between 0.0 and 0.25",
        ));
    }
    let refs_a: Vec<&str> = list_a.iter().map(|s| s.as_str()).collect();
    let refs_b: Vec<&str> = list_b.iter().map(|s| s.as_str()).collect();
    #[cfg(feature = "gpu")]
    {
        if !GpuEngine::is_cpu_only() {
            if let Ok(kernel) = fuzzgpu_core::jaro::gpu_ext::GpuJaroKernel::get() {
                match py.allow_threads(|| kernel.compute_matrix(&refs_a, &refs_b, p)) {
                    Ok(res) => return Ok(res),
                    Err(e) => {
                        if is_force_gpu() {
                            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                                "GPU Jaro-Winkler matrix failed: {}",
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
    py.allow_threads(|| Ok(fuzzgpu_core::jaro::jaro_winkler_cdist_cpu(&refs_a, &refs_b, p)))
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
fn fuzz_ratio_batch(py: Python, query: String, candidates: Vec<String>) -> PyResult<Vec<f64>> {
    let refs: Vec<&str> = candidates.iter().map(|s| s.as_str()).collect();
    py.allow_threads(|| Ok(fuzzgpu_core::ratio_batch(&query, &refs)))
}

#[pyfunction]
#[pyo3(signature = (query, choices, score_cutoff = 0.0, limit = 5), text_signature = "(query, choices, score_cutoff=0.0, limit=5, /)")]
fn fuzz_extract(py: Python, query: String, choices: Vec<String>, score_cutoff: f64, limit: usize) -> PyResult<Vec<(String, f64, usize)>> {
    let refs: Vec<&str> = choices.iter().map(|s| s.as_str()).collect();
    py.allow_threads(|| Ok(fuzzgpu_core::extract(&query, &refs, score_cutoff, limit)))
}

#[pyfunction]
#[pyo3(signature = (query, choices, score_cutoff = 0.0), text_signature = "(query, choices, score_cutoff=0.0, /)")]
fn fuzz_extract_one(py: Python, query: String, choices: Vec<String>, score_cutoff: f64) -> PyResult<Option<(String, f64, usize)>> {
    let refs: Vec<&str> = choices.iter().map(|s| s.as_str()).collect();
    py.allow_threads(|| Ok(fuzzgpu_core::extract_one(&query, &refs, score_cutoff)))
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

// ── Module ──────────────────────────────────────────────────

#[pymodule]
fn fuzzgpu(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Levenshtein
    m.add_function(wrap_pyfunction!(levenshtein_distance, m)?)?;
    m.add_function(wrap_pyfunction!(levenshtein_batch, m)?)?;
    m.add_function(wrap_pyfunction!(levenshtein_cdist, m)?)?;
    // Damerau-Levenshtein
    m.add_function(wrap_pyfunction!(damerau_levenshtein_distance, m)?)?;
    m.add_function(wrap_pyfunction!(damerau_levenshtein_batch, m)?)?;
    m.add_function(wrap_pyfunction!(damerau_levenshtein_cdist, m)?)?;
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
    m.add_function(wrap_pyfunction!(jaro_winkler_cdist, m)?)?;
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
    m.add_function(wrap_pyfunction!(is_gpu_available, m)?)?;
    m.add_function(wrap_pyfunction!(warmup, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_info, m)?)?;
    // Dynamic version directly from Cargo package metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
