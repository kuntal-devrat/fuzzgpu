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

/// Needleman-Wunsch global alignment score (linear gap penalty).
///
/// Scores are `i64` and are exposed to JavaScript as `BigInt` (wasm-bindgen
/// maps `i64` to `BigInt`), so scores beyond the i32 range are not truncated.
/// Pass BigInt arguments, e.g. `needleman_wunsch("AGTACGCA", "TATGC", 2n, -1n, -2n)`.
#[wasm_bindgen]
pub fn needleman_wunsch(a: &str, b: &str, match_score: i64, mismatch_score: i64, gap_penalty: i64) -> i64 {
    fuzzgpu_core::needleman_wunsch(a, b, match_score, mismatch_score, gap_penalty)
}

/// Needleman-Wunsch global alignment score with affine gap penalties (Gotoh).
///
/// Scores are `i64` and are exposed to JavaScript as `BigInt`, so scores
/// beyond the i32 range are not truncated.
#[wasm_bindgen]
pub fn needleman_wunsch_affine(a: &str, b: &str, match_score: i64, mismatch_score: i64, gap_open: i64, gap_extend: i64) -> i64 {
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

/// Myers bit-vector Levenshtein (fast CPU path for short ASCII strings).
/// Falls back to the Unicode-aware distance for non-ASCII input.
#[wasm_bindgen]
pub fn levenshtein_myers(a: &str, b: &str) -> u32 {
    if a.is_ascii() && b.is_ascii() {
        fuzzgpu_core::levenshtein_myers(a.as_bytes(), b.as_bytes())
    } else {
        fuzzgpu_core::levenshtein_distance_raw(a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::wasm_bindgen_test;

    // The i64 signatures below are pinned at compile time: if the API ever
    // regressed to i32, these tests would fail to compile. The value assertions
    // cover the whole point of the BigInt API — scores beyond i32::MAX and
    // beyond 2^53 (the largest f64 that can represent every integer) must
    // survive the wasm call chain intact. (These are in-module calls, so the
    // BigInt conversion itself is verified from JS in tests/js_api.test.cjs.)

    #[wasm_bindgen_test]
    fn needleman_wunsch_large_score_not_truncated() {
        let a = "A".repeat(100);
        let b = "A".repeat(100);
        // 100 matches * 1e14 = 1e16: exceeds i32::MAX (~2.1e9) and 2^53 (~9.0e15).
        assert_eq!(
            needleman_wunsch(&a, &b, 100_000_000_000_000, -1, -2),
            10_000_000_000_000_000
        );
    }

    #[wasm_bindgen_test]
    fn needleman_wunsch_affine_large_score_not_truncated() {
        let a = "A".repeat(100);
        let b = "A".repeat(100);
        assert_eq!(
            needleman_wunsch_affine(&a, &b, 100_000_000_000_000, -1, -10, -2),
            10_000_000_000_000_000
        );
    }

    #[wasm_bindgen_test]
    fn needleman_wunsch_known_values() {
        // Cross-checked against the Python bindings and the Node smoke test.
        assert_eq!(needleman_wunsch("AGTACGCA", "TATGC", 2, -1, -2), 1);
        assert_eq!(needleman_wunsch("hello", "hello", 2, -1, -2), 10);
        assert_eq!(needleman_wunsch("", "", 2, -1, -2), 0);
    }

    #[wasm_bindgen_test]
    fn needleman_wunsch_negative_scores() {
        // 4 mismatches at -1 each.
        assert_eq!(needleman_wunsch("AAAA", "TTTT", 2, -1, -2), -4);
        assert_eq!(needleman_wunsch_affine("AAAA", "TTTT", 2, -1, -10, -2), -4);
    }
}
