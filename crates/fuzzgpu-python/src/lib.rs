use pyo3::prelude::*;

#[cfg(feature = "gpu")]
use fuzzgpu_core::gpu::GpuEngine;

// ── Levenshtein ──────────────────────────────────────────────

#[pyfunction]
fn levenshtein_distance(py: Python, a: &str, b: &str) -> PyResult<u32> {
    #[cfg(feature = "gpu")]
    {
        if let Ok(kernel) = fuzzgpu_core::levenshtein::gpu_ext::GpuLevenshteinKernel::get() {
            if let Ok(res) = py.allow_threads(|| kernel.compute(&[(a, b)])) {
                return Ok(res[0]);
            }
        }
        let _ = py;
        Ok(fuzzgpu_core::levenshtein_distance_raw(a, b))
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = py;
        Ok(fuzzgpu_core::levenshtein_distance_raw(a, b))
    }
}

#[pyfunction]
fn levenshtein_batch(py: Python, query: String, candidates: Vec<String>) -> PyResult<Vec<u32>> {
    let pairs: Vec<(&str, &str)> = candidates.iter().map(|c| (query.as_str(), c.as_str())).collect();
    #[cfg(feature = "gpu")]
    {
        if let Ok(kernel) = fuzzgpu_core::levenshtein::gpu_ext::GpuLevenshteinKernel::get() {
            if let Ok(res) = py.allow_threads(|| kernel.compute(&pairs)) {
                return Ok(res);
            }
        }
        let kernel = fuzzgpu_core::LevenshteinKernel;
        kernel.compute(&pairs).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
    #[cfg(not(feature = "gpu"))]
    {
        Ok(py.allow_threads(|| pairs.iter().map(|(a, b)| fuzzgpu_core::levenshtein_distance_raw(a, b)).collect()))
    }
}

#[pyfunction]
fn levenshtein_cdist(py: Python, list_a: Vec<String>, list_b: Vec<String>) -> PyResult<Vec<Vec<u32>>> {
    let refs_a: Vec<&str> = list_a.iter().map(|s| s.as_str()).collect();
    let refs_b: Vec<&str> = list_b.iter().map(|s| s.as_str()).collect();
    #[cfg(feature = "gpu")]
    {
        if let Ok(kernel) = fuzzgpu_core::levenshtein::gpu_ext::GpuLevenshteinKernel::get() {
            if let Ok(res) = py.allow_threads(|| kernel.compute_matrix(&refs_a, &refs_b)) {
                return Ok(res);
            }
        }
        py.allow_threads(|| Ok(fuzzgpu_core::levenshtein::levenshtein_cdist_cpu(&refs_a, &refs_b)))
    }
    #[cfg(not(feature = "gpu"))]
    {
        py.allow_threads(|| Ok(fuzzgpu_core::levenshtein::levenshtein_cdist_cpu(&refs_a, &refs_b)))
    }
}

// ── Damerau-Levenshtein ──────────────────────────────────────

#[pyfunction]
fn damerau_levenshtein_distance(a: &str, b: &str) -> PyResult<u32> {
    Ok(fuzzgpu_core::damerau_levenshtein_distance(a, b))
}

#[pyfunction]
fn damerau_levenshtein_batch(py: Python, query: String, candidates: Vec<String>) -> PyResult<Vec<u32>> {
    let refs: Vec<&str> = candidates.iter().map(|s| s.as_str()).collect();
    py.allow_threads(|| Ok(fuzzgpu_core::damerau_levenshtein_batch(&query, &refs)))
}

#[pyfunction]
fn damerau_levenshtein_cdist(py: Python, list_a: Vec<String>, list_b: Vec<String>) -> PyResult<Vec<Vec<u32>>> {
    let refs_a: Vec<&str> = list_a.iter().map(|s| s.as_str()).collect();
    let refs_b: Vec<&str> = list_b.iter().map(|s| s.as_str()).collect();
    py.allow_threads(|| Ok(fuzzgpu_core::damerau_levenshtein_cdist(&refs_a, &refs_b)))
}

#[pyfunction]
fn damerau_ratio(a: &str, b: &str) -> PyResult<f64> {
    Ok(fuzzgpu_core::damerau_ratio(a, b))
}

// ── Needleman-Wunsch ────────────────────────────────────────

#[pyfunction]
fn needleman_wunsch_score(py: Python, a: &str, b: &str, match_score: i32, mismatch_score: i32, gap_penalty: i32) -> PyResult<i32> {
    py.allow_threads(|| Ok(fuzzgpu_core::needleman_wunsch(a, b, match_score, mismatch_score, gap_penalty)))
}

#[pyfunction]
fn needleman_wunsch_batch_fn(py: Python, query: String, candidates: Vec<String>, match_score: i32, mismatch_score: i32, gap_penalty: i32) -> PyResult<Vec<i32>> {
    let refs: Vec<&str> = candidates.iter().map(|s| s.as_str()).collect();
    py.allow_threads(|| Ok(fuzzgpu_core::needleman_wunsch_batch(&query, &refs, match_score, mismatch_score, gap_penalty)))
}

#[pyfunction]
fn needleman_wunsch_affine(py: Python, a: &str, b: &str, match_score: i32, mismatch_score: i32, gap_open: i32, gap_extend: i32) -> PyResult<i32> {
    py.allow_threads(|| Ok(fuzzgpu_core::needleman_wunsch_affine(a, b, match_score, mismatch_score, gap_open, gap_extend)))
}

#[pyfunction]
fn needleman_wunsch_affine_batch(py: Python, query: String, candidates: Vec<String>, match_score: i32, mismatch_score: i32, gap_open: i32, gap_extend: i32) -> PyResult<Vec<i32>> {
    let refs: Vec<&str> = candidates.iter().map(|s| s.as_str()).collect();
    py.allow_threads(|| Ok(fuzzgpu_core::needleman_wunsch_affine_batch(&query, &refs, match_score, mismatch_score, gap_open, gap_extend)))
}

// ── Jaro-Winkler ────────────────────────────────────────────

#[pyfunction]
fn jaro_similarity(a: &str, b: &str) -> PyResult<f64> {
    Ok(fuzzgpu_core::jaro(a, b))
}

#[pyfunction]
fn jaro_winkler_similarity(a: &str, b: &str, p: f64) -> PyResult<f64> {
    Ok(fuzzgpu_core::jaro_winkler(a, b, p))
}

#[pyfunction]
fn jaro_winkler_batch_fn(py: Python, query: String, candidates: Vec<String>, p: f64) -> PyResult<Vec<f64>> {
    let pairs: Vec<(&str, &str)> = candidates.iter().map(|c| (query.as_str(), c.as_str())).collect();
    #[cfg(feature = "gpu")]
    {
        if let Ok(kernel) = fuzzgpu_core::jaro::gpu_ext::GpuJaroKernel::get() {
            if let Ok(res) = py.allow_threads(|| kernel.compute_batch(&pairs, p)) {
                return Ok(res);
            }
        }
        let refs: Vec<&str> = candidates.iter().map(|s| s.as_str()).collect();
        py.allow_threads(|| Ok(fuzzgpu_core::jaro_winkler_batch(&query, &refs, p)))
    }
    #[cfg(not(feature = "gpu"))]
    {
        let refs: Vec<&str> = candidates.iter().map(|s| s.as_str()).collect();
        py.allow_threads(|| Ok(fuzzgpu_core::jaro_winkler_batch(&query, &refs, p)))
    }
}

#[pyfunction]
fn jaro_winkler_cdist(py: Python, list_a: Vec<String>, list_b: Vec<String>, p: f64) -> PyResult<Vec<Vec<f64>>> {
    let refs_a: Vec<&str> = list_a.iter().map(|s| s.as_str()).collect();
    let refs_b: Vec<&str> = list_b.iter().map(|s| s.as_str()).collect();
    #[cfg(feature = "gpu")]
    {
        if let Ok(kernel) = fuzzgpu_core::jaro::gpu_ext::GpuJaroKernel::get() {
            if let Ok(res) = py.allow_threads(|| kernel.compute_matrix(&refs_a, &refs_b, p)) {
                return Ok(res);
            }
        }
        py.allow_threads(|| Ok(fuzzgpu_core::jaro::jaro_winkler_cdist_cpu(&refs_a, &refs_b, p)))
    }
    #[cfg(not(feature = "gpu"))]
    {
        py.allow_threads(|| Ok(fuzzgpu_core::jaro::jaro_winkler_cdist_cpu(&refs_a, &refs_b, p)))
    }
}

// ── Fuzzy matching ──────────────────────────────────────────

#[pyfunction]
fn fuzz_ratio(a: &str, b: &str) -> PyResult<f64> { Ok(fuzzgpu_core::ratio(a, b)) }

#[pyfunction]
fn fuzz_partial_ratio(a: &str, b: &str) -> PyResult<f64> { Ok(fuzzgpu_core::partial_ratio(a, b)) }

#[pyfunction]
fn fuzz_token_sort_ratio(a: &str, b: &str) -> PyResult<f64> { Ok(fuzzgpu_core::token_sort_ratio(a, b)) }

#[pyfunction]
fn fuzz_token_set_ratio(a: &str, b: &str) -> PyResult<f64> { Ok(fuzzgpu_core::token_set_ratio(a, b)) }

#[pyfunction]
fn fuzz_wratio(a: &str, b: &str) -> PyResult<f64> { Ok(fuzzgpu_core::wratio(a, b)) }

#[pyfunction]
fn fuzz_ratio_batch(py: Python, query: String, candidates: Vec<String>) -> PyResult<Vec<f64>> {
    let refs: Vec<&str> = candidates.iter().map(|s| s.as_str()).collect();
    py.allow_threads(|| Ok(fuzzgpu_core::ratio_batch(&query, &refs)))
}

#[pyfunction]
fn fuzz_extract(py: Python, query: String, choices: Vec<String>, score_cutoff: f64, limit: usize) -> PyResult<Vec<(String, f64, usize)>> {
    let refs: Vec<&str> = choices.iter().map(|s| s.as_str()).collect();
    py.allow_threads(|| Ok(fuzzgpu_core::extract(&query, &refs, score_cutoff, limit)))
}

/// Returns the single best match above the cutoff, or None.
#[pyfunction]
fn fuzz_extract_one(py: Python, query: String, choices: Vec<String>, score_cutoff: f64) -> PyResult<Option<(String, f64, usize)>> {
    let refs: Vec<&str> = choices.iter().map(|s| s.as_str()).collect();
    py.allow_threads(|| Ok(fuzzgpu_core::extract_one(&query, &refs, score_cutoff)))
}

// ── Optimized algorithm variants ────────────────────────────

#[pyfunction]
fn levenshtein_myers(a: &str, b: &str) -> PyResult<u32> {
    Ok(fuzzgpu_core::levenshtein_myers(a.as_bytes(), b.as_bytes()))
}

#[pyfunction]
fn needleman_wunsch_striped(a: &str, b: &str, match_score: i32, mismatch_score: i32, gap_penalty: i32) -> PyResult<i32> {
    Ok(fuzzgpu_core::needleman_wunsch_striped(a.as_bytes(), b.as_bytes(), match_score, mismatch_score, gap_penalty))
}

#[pyfunction]
fn jaro_optimized(a: &str, b: &str) -> PyResult<f64> {
    Ok(fuzzgpu_core::jaro_optimized(a.as_bytes(), b.as_bytes()))
}

// ── GPU info ────────────────────────────────────────────────

#[pyfunction]
fn gpu_info() -> PyResult<String> {
    #[cfg(feature = "gpu")]
    {
        match GpuEngine::get() {
            Ok(engine) => Ok(format!("{} ({})", engine.info.name, engine.info.backend)),
            Err(_) => Ok("CPU-only fallback mode (no GPU device detected)".into()),
        }
    }
    #[cfg(not(feature = "gpu"))]
    { Ok("CPU-only mode (no GPU)".into()) }
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
    // GPU info & version
    m.add_function(wrap_pyfunction!(gpu_info, m)?)?;
    m.add("__version__", "0.1.0")?;
    Ok(())
}
