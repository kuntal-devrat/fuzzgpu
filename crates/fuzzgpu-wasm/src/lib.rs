use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn levenshtein_distance(a: &str, b: &str) -> u32 {
    fuzzgpu_core::levenshtein_distance_raw(a, b)
}

#[wasm_bindgen]
pub fn levenshtein_batch(query: &str, candidates: JsValue) -> Vec<u32> {
    let candidates: Vec<String> = serde_wasm_bindgen::from_value(candidates).unwrap_or_default();
    candidates.iter().map(|c| fuzzgpu_core::levenshtein_distance_raw(query, c)).collect()
}

#[wasm_bindgen]
pub fn damerau_levenshtein_distance(a: &str, b: &str) -> u32 {
    fuzzgpu_core::damerau_levenshtein_distance(a, b)
}

#[wasm_bindgen]
pub fn damerau_ratio(a: &str, b: &str) -> f64 {
    fuzzgpu_core::damerau_ratio(a, b)
}

#[wasm_bindgen]
pub fn needleman_wunsch(a: &str, b: &str, match_score: i32, mismatch_score: i32, gap_penalty: i32) -> i32 {
    fuzzgpu_core::needleman_wunsch(a, b, match_score, mismatch_score, gap_penalty)
}

#[wasm_bindgen]
pub fn needleman_wunsch_affine(a: &str, b: &str, match_score: i32, mismatch_score: i32, gap_open: i32, gap_extend: i32) -> i32 {
    fuzzgpu_core::needleman_wunsch_affine(a, b, match_score, mismatch_score, gap_open, gap_extend)
}

#[wasm_bindgen]
pub fn jaro(a: &str, b: &str) -> f64 {
    fuzzgpu_core::jaro(a, b)
}

#[wasm_bindgen]
pub fn jaro_winkler(a: &str, b: &str, p: f64) -> f64 {
    fuzzgpu_core::jaro_winkler(a, b, p)
}

#[wasm_bindgen]
pub fn ratio(a: &str, b: &str) -> f64 {
    fuzzgpu_core::ratio(a, b)
}

#[wasm_bindgen]
pub fn partial_ratio(a: &str, b: &str) -> f64 {
    fuzzgpu_core::partial_ratio(a, b)
}

#[wasm_bindgen]
pub fn token_sort_ratio(a: &str, b: &str) -> f64 {
    fuzzgpu_core::token_sort_ratio(a, b)
}

#[wasm_bindgen]
pub fn token_set_ratio(a: &str, b: &str) -> f64 {
    fuzzgpu_core::token_set_ratio(a, b)
}

#[wasm_bindgen]
pub fn wratio(a: &str, b: &str) -> f64 {
    fuzzgpu_core::wratio(a, b)
}

#[wasm_bindgen]
pub fn extract(query: &str, choices: JsValue, score_cutoff: f64, limit: usize) -> JsValue {
    let choices: Vec<String> = serde_wasm_bindgen::from_value(choices).unwrap_or_default();
    let refs: Vec<&str> = choices.iter().map(|s| s.as_str()).collect();
    let results = fuzzgpu_core::extract(query, &refs, score_cutoff, limit);
    serde_wasm_bindgen::to_value(&results).unwrap_or(JsValue::NULL)
}

/// Returns the single best match above cutoff, or null.
#[wasm_bindgen]
pub fn extract_one(query: &str, choices: JsValue, score_cutoff: f64) -> JsValue {
    let choices: Vec<String> = serde_wasm_bindgen::from_value(choices).unwrap_or_default();
    let refs: Vec<&str> = choices.iter().map(|s| s.as_str()).collect();
    match fuzzgpu_core::extract_one(query, &refs, score_cutoff) {
        Some(result) => serde_wasm_bindgen::to_value(&result).unwrap_or(JsValue::NULL),
        None => JsValue::NULL,
    }
}

/// Myers bit-vector Levenshtein (fast CPU path for short strings).
#[wasm_bindgen]
pub fn levenshtein_myers(a: &str, b: &str) -> u32 {
    fuzzgpu_core::levenshtein_myers(a.as_bytes(), b.as_bytes())
}
