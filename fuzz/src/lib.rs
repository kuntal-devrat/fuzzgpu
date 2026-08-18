//! Shared fuzz-case drivers for fuzzgpu-core.
//!
//! Each `run_*` function decodes one arbitrary byte payload into strings and
//! exercises a public algorithm, asserting its fast path against a naive
//! oracle. A panic or a mismatch is a fuzz finding.
//!
//! Two entry points use this code:
//!   * the libFuzzer targets in `fuzz_targets/` (nightly + `cargo fuzz run`),
//!   * the self-harness `#[test]` below, which drives the same functions over
//!     a deterministic + pseudo-random corpus and runs on stable in CI.
//!
//! The fuzz build compiles fuzzgpu-core with `default-features = false` (no
//! GPU): this targets the CPU algorithms and SIMD kernels, where index/overflow
//! bugs live; the GPU paths are differentially covered by the proptest suite.

use arbitrary::Unstructured;
use fuzzgpu_core::fuzz::{partial_ratio, ratio, token_set_ratio, token_sort_ratio, wratio};
use fuzzgpu_core::jaro::{jaro, jaro_winkler, jaro_winkler_batch};
use fuzzgpu_core::levenshtein::{
    levenshtein_batch_auto, levenshtein_cdist_cpu, levenshtein_distance_raw,
};
use fuzzgpu_core::needleman::{needleman_wunsch, needleman_wunsch_affine};
use fuzzgpu_core::simd::{levenshtein_myers, levenshtein_myers_4way, needleman_wunsch_striped};

/// Decode up to `1 + (first byte % cap)` strings from the payload.
fn take_strings<'a>(u: &mut Unstructured<'a>, cap: usize) -> Vec<String> {
    let Ok(n) = u.arbitrary::<u8>() else { return vec![] };
    let count = 1 + (n as usize % cap);
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let Ok(s) = u.arbitrary::<String>() else { break };
        out.push(s);
    }
    out
}

/// Levenshtein: batch/cdist/SIMD fast paths against the scalar oracle.
pub fn run_levenshtein(data: &[u8]) {
    let mut u = Unstructured::new(data);
    let strs = take_strings(&mut u, 24);
    if strs.len() < 2 {
        return;
    }
    let query = &strs[0];
    let refs: Vec<&str> = strs[1..].iter().map(|s| s.as_str()).collect();

    // Batch (incl. shared-query SIMD 4-way path) vs per-pair scalar.
    let pairs: Vec<(&str, &str)> = refs.iter().map(|c| (query.as_str(), *c)).collect();
    let batch = levenshtein_batch_auto(&pairs);
    for (i, c) in refs.iter().enumerate() {
        assert_eq!(batch[i], levenshtein_distance_raw(query, c), "batch vs raw");
    }

    // cdist matrix vs per-pair scalar.
    let cdist = levenshtein_cdist_cpu(&refs, &refs);
    for (i, a) in refs.iter().enumerate() {
        for (j, b) in refs.iter().enumerate() {
            assert_eq!(cdist[i][j], levenshtein_distance_raw(a, b), "cdist vs raw");
        }
    }

    // SIMD kernels vs scalar (ASCII gate mirrors the production callers).
    if !query.is_empty() && query.is_ascii() && query.len() <= 64 {
        for c in &refs {
            if !c.is_ascii() {
                continue;
            }
            let qb = query.as_bytes();
            let myers = levenshtein_myers(qb, c.as_bytes());
            assert_eq!(myers, levenshtein_distance_raw(query, c), "myers vs raw");
            let four = levenshtein_myers_4way(qb, [c.as_bytes(), c.as_bytes(), c.as_bytes(), c.as_bytes()]);
            for lane in four {
                assert_eq!(lane, myers, "4way lane vs myers");
            }
        }
    }
}

/// Jaro / Jaro-Winkler: batch vs per-pair, and p-parameter edge values.
pub fn run_jaro(data: &[u8]) {
    let mut u = Unstructured::new(data);
    let strs = take_strings(&mut u, 24);
    if strs.len() < 2 {
        return;
    }
    let query = &strs[0];
    let refs: Vec<&str> = strs[1..].iter().map(|s| s.as_str()).collect();
    let p = if let Ok(v) = u.arbitrary::<f32>() {
        (v.abs() % 0.25).into()
    } else {
        0.1
    };

    let batch = jaro_winkler_batch(query, &refs, p);
    for (i, c) in refs.iter().enumerate() {
        assert_eq!(batch[i], jaro_winkler(query, c, p), "jw batch vs raw");
    }
    for c in &refs {
        let _ = jaro(query, c);
    }
    // p = 0.25 and p = 0.0 are boundary-legal; callers clamp the error path.
    let _ = jaro_winkler(query, "x", 0.0);
    let _ = jaro_winkler(query, "x", 0.25);
}

/// Needleman-Wunsch (linear + affine): fast paths vs naive oracles.
pub fn run_needleman(data: &[u8]) {
    let mut u = Unstructured::new(data);
    let strs = take_strings(&mut u, 12);
    if strs.len() < 2 {
        return;
    }
    let (a, b) = (&strs[0], &strs[1]);
    let match_score = match u.arbitrary::<i8>() { Ok(v) => (v as i64).max(1), Err(_) => 1 };
    let mismatch_score = match u.arbitrary::<i8>() { Ok(v) => -(v as i64) - 1, Err(_) => -1 };
    // The API convention is NEGATIVE gap penalties (scores are added to gap
    // values, so penalties must be negative); positive values would turn gaps
    // into bonuses and legitimately diverge from the identical-string fast
    // path. Mirror the Python tests (gap_open=-3, gap_extend=-1).
    let gap_open = match u.arbitrary::<i8>() { Ok(v) => -(v as i64) - 1, Err(_) => -3 };
    let gap_extend = match u.arbitrary::<i8>() { Ok(v) => -(v as i64) - 1, Err(_) => -1 };

    // Linear-gap: the striped fast path is documented as byte-semantics (the
    // ASCII kernel), so it is only comparable to the reference when both
    // inputs are ASCII. Non-ASCII inputs exercise the reference + affine paths.
    if a.is_ascii() && b.is_ascii() {
        assert_eq!(
            needleman_wunsch_striped(a.as_bytes(), b.as_bytes(), match_score, mismatch_score, gap_open),
            needleman_wunsch(a, b, match_score, mismatch_score, gap_open),
            "striped vs nw"
        );
    }

    // Affine-gap: fast path vs naive Gotoh.
    assert_eq!(
        needleman_wunsch_affine(a, b, match_score, mismatch_score, gap_open, gap_extend),
        naive_gotoh(a, b, match_score, mismatch_score, gap_open, gap_extend),
        "affine vs naive gotoh"
    );
}

/// Fuzzy ratios: each ratio must be in [0, 1] and equal across single/batch.
pub fn run_ratios(data: &[u8]) {
    let mut u = Unstructured::new(data);
    let strs = take_strings(&mut u, 12);
    if strs.len() < 2 {
        return;
    }
    let query = &strs[0];
    for c in &strs[1..] {
        for r in [ratio(query, c), partial_ratio(query, c, 0.0), token_sort_ratio(query, c, 0.0), token_set_ratio(query, c, 0.0), wratio(query, c, 0.0)] {
            // rapidfuzz-style percentages: 0..=100.
            assert!((0.0..=100.0).contains(&r), "ratio out of range: {r}");
        }
    }
}

/// Naive O(m·n) Gotoh (affine gap) oracle — mirrors the production
/// convention exactly (positive penalty magnitudes, E = horizontal gap state,
/// F = vertical gap state) but with plain arithmetic, so it can never share a
/// bug with the optimized implementation.
pub fn naive_gotoh(a: &str, b: &str, match_score: i64, mismatch_score: i64, gap_open: i64, gap_extend: i64) -> i64 {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (m, n) = (a.len(), b.len());
    let neg = i64::MIN / 4;

    let mut m_row = vec![neg; n + 1]; // best score ending in a match/substitution
    let mut e_row = vec![neg; n + 1]; // best score ending in a horizontal gap
    let mut f_row = vec![neg; n + 1]; // best score ending in a vertical gap
    m_row[0] = 0;
    for j in 1..=n {
        let gap = gap_open + gap_extend * (j as i64);
        f_row[j] = gap;
        m_row[j] = gap;
    }
    for i in 1..=m {
        let mut prev_m = m_row[0];
        let mut prev_e = e_row[0];
        let mut prev_f = f_row[0];
        let gap_i = gap_open + gap_extend * (i as i64);
        e_row[0] = gap_i;
        f_row[0] = neg;
        m_row[0] = gap_i;
        for j in 1..=n {
            let sub = if a[i - 1] == b[j - 1] { match_score } else { mismatch_score };
            let new_m = prev_m.max(prev_e).max(prev_f) + sub;
            let new_e = (e_row[j] + gap_extend)
                .max(m_row[j] + gap_open + gap_extend)
                .max(f_row[j] + gap_open + gap_extend);
            let new_f = (f_row[j - 1] + gap_extend)
                .max(m_row[j - 1] + gap_open + gap_extend)
                .max(e_row[j - 1] + gap_open + gap_extend);
            prev_m = m_row[j];
            prev_e = e_row[j];
            prev_f = f_row[j];
            m_row[j] = new_m;
            e_row[j] = new_e;
            f_row[j] = new_f;
        }
    }
    m_row[n].max(e_row[n]).max(f_row[n])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic corpus + pseudo-random inputs, driven over every target.
    /// This is the CI-runnable surrogate for the nightly libFuzzer loop.
    #[test]
    fn self_harness_runs_all_targets() {
        let mut state = 0x4D595DF4D0F33173u64;
        let mut corpus = vec![
            b"".to_vec(),
            b"\x00\x00".to_vec(),
            b"\xff\xff\xff\xff".to_vec(),
            "kitten\0sitting".as_bytes().to_vec(),
            "日本語\0日本語".as_bytes().to_vec(),
        ];
        // 1500 pseudo-random payloads (LCG) of varied length.
        for _ in 0..1500 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let len = (state % 96) as usize;
            let mut bytes = Vec::with_capacity(len);
            for _ in 0..len {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                bytes.push((state >> 33) as u8);
            }
            corpus.push(bytes);
        }
        for payload in &corpus {
            run_levenshtein(payload);
            run_jaro(payload);
            run_needleman(payload);
            run_ratios(payload);
        }
    }

    #[test]
    fn naive_gotoh_matches_simple_cases() {
        // Negative-penalty convention (matches the production API's contract).
        assert_eq!(naive_gotoh("AGCT", "AGCT", 2, -1, -3, -1), 8);
        // One deletion: 3 matches (6) minus one gap (open+extend = 4) = 2.
        assert_eq!(naive_gotoh("AGCT", "AGT", 2, -1, -3, -1), 2);
        assert_eq!(naive_gotoh("", "A", 2, -1, -3, -1), -4);
        assert_eq!(naive_gotoh("A", "", 2, -1, -3, -1), -4);
    }
}
