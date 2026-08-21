//! Property-based differential harness.
//!
//! Independent implementations of the same algorithm are compared against each
//! other (and against naive references) across randomly generated inputs:
//! ASCII and Unicode strings, plus edge lengths (0, 1, the 64/65 Myers
//! bit-vector boundary, the 128/129 Jaro stack/heap boundary, and strings that
//! exceed the GPU shader length limits so the CPU-fallback routing is
//! exercised too).
//!
//! The affine Needleman-Wunsch section additionally pins the Gotoh oracle's
//! boundary behavior directly: one-empty strings (closed form
//! `gap_open + n * gap_extend`), all-match (`len * match_score`), all-mismatch
//! over disjoint alphabets, and equal-length strings under positive-gap
//! regimes where `gap_extend > gap_open`.
//!
//! GPU tests skip cleanly when no device is available.

use fuzzgpu_core::{
    damerau_levenshtein_distance, jaro, jaro_optimized, jaro_winkler, levenshtein_distance_raw,
    levenshtein_myers, needleman_wunsch, needleman_wunsch_affine, needleman_wunsch_striped,
};
use proptest::prelude::*;

// ── Input strategies ─────────────────────────────────────────────────────────

/// ASCII strings over `[a-z]`, lengths `0..=max_len`.
fn ascii_string(max_len: usize) -> impl Strategy<Value = String> {
    prop::collection::vec(prop::char::range('a', 'z'), 0..=max_len)
        .prop_map(|v: Vec<char>| v.into_iter().collect())
}

/// Any valid Unicode string (including astral-plane and combining chars),
/// lengths `0..=max_len`.
fn unicode_string(max_len: usize) -> impl Strategy<Value = String> {
    prop::collection::vec(proptest::char::any(), 0..=max_len)
        .prop_map(|v: Vec<char>| v.into_iter().collect())
}

/// ASCII or Unicode, lengths `0..=max_len`.
fn any_string(max_len: usize) -> impl Strategy<Value = String> {
    prop_oneof![ascii_string(max_len), unicode_string(max_len)]
}

/// ASCII string, length `1..=max_len` (non-empty: the row-wise Myers cdist
/// kernel requires a non-empty pattern per row).
fn ascii_string_nz(max_len: usize) -> impl Strategy<Value = String> {
    (1usize..=max_len).prop_flat_map(move |len| {
        prop::collection::vec(prop::char::range('a', 'z'), len..=len)
            .prop_map(|chars| chars.into_iter().collect())
    })
}

fn any_pair(max_len: usize) -> impl Strategy<Value = (String, String)> {
    (any_string(max_len), any_string(max_len))
}

/// A pair of ASCII strings (the GPU Damerau kernel's gate is ASCII ≤ 32
/// chars, so ASCII pairs are what actually runs on the shader).
fn ascii_pair(max_len: usize) -> impl Strategy<Value = (String, String)> {
    (ascii_string(max_len), ascii_string(max_len))
}

/// A pair of ASCII strings with exactly equal length — needed for edge cases
/// that force every DP column into play with both sequences in the same shape.
fn equal_len_ascii_pair(max_len: usize) -> impl Strategy<Value = (String, String)> {
    (1usize..max_len).prop_flat_map(|len| {
        let chars = prop::collection::vec(prop::char::range('a', 'z'), len..=len);
        (chars.clone(), chars).prop_map(|(a, b): (Vec<char>, Vec<char>)| {
            (a.into_iter().collect(), b.into_iter().collect())
        })
    })
}

// ── Naive references (independent of the library implementations) ────────────

/// Naive two-row Levenshtein DP, used as an oracle for the optimized paths.
fn naive_levenshtein<T: PartialEq>(a: &[T], b: &[T]) -> u32 {
    let (m, n) = (a.len(), b.len());
    let mut prev: Vec<u32> = (0..=n as u32).collect();
    let mut cur = vec![0u32; n + 1];
    for i in 1..=m {
        cur[0] = i as u32;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j - 1] + cost).min(prev[j] + 1).min(cur[j - 1] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[n]
}

#[allow(clippy::needless_range_loop)]
/// Naive full-matrix Needleman-Wunsch (linear gap penalty), used as an oracle.
fn naive_needleman_wunsch<T: PartialEq>(
    a: &[T],
    b: &[T],
    match_score: i64,
    mismatch_score: i64,
    gap: i64,
) -> i64 {
    let (m, n) = (a.len(), b.len());
    let mut dp = vec![vec![0i64; n + 1]; m + 1];
    for i in 1..=m {
        dp[i][0] = (i as i64) * gap;
    }
    for j in 1..=n {
        dp[0][j] = (j as i64) * gap;
    }
    for i in 1..=m {
        for j in 1..=n {
            let score = if a[i - 1] == b[j - 1] {
                match_score
            } else {
                mismatch_score
            };
            dp[i][j] = (dp[i - 1][j - 1] + score)
                .max(dp[i - 1][j] + gap)
                .max(dp[i][j - 1] + gap);
        }
    }
    dp[m][n]
}

/// Naive full-matrix Gotoh (affine gap penalty), used as an oracle for
/// `needleman_wunsch_affine`. Independent implementation: three full matrices
/// (match/substitution, gap-in-a, gap-in-b) with the classic Gotoh 1982
/// recurrences, plain arithmetic (no saturating ops, no in-place row reuse).
/// A gap of length k costs `gap_open + k * gap_extend`.
fn naive_needleman_wunsch_affine<T: PartialEq>(
    a: &[T],
    b: &[T],
    match_score: i64,
    mismatch_score: i64,
    gap_open: i64,
    gap_extend: i64,
) -> i64 {
    const NEG_INF: i64 = -1_000_000_000_000_000_000;

    let (m, n) = (a.len(), b.len());
    if m == 0 && n == 0 {
        return 0;
    }
    if m == 0 {
        return gap_open + (n as i64) * gap_extend;
    }
    if n == 0 {
        return gap_open + (m as i64) * gap_extend;
    }
    if a == b {
        return (m as i64) * match_score;
    }

    let mut mtx = vec![vec![NEG_INF; n + 1]; m + 1];
    let mut ix = vec![vec![NEG_INF; n + 1]; m + 1];
    let mut iy = vec![vec![NEG_INF; n + 1]; m + 1];

    mtx[0][0] = 0;
    // Leading gaps in `b` (first row) and `a` (first column).
    for j in 1..=n {
        let cost = gap_open + (j as i64) * gap_extend;
        iy[0][j] = cost;
        mtx[0][j] = cost;
    }
    for i in 1..=m {
        let cost = gap_open + (i as i64) * gap_extend;
        ix[i][0] = cost;
        mtx[i][0] = cost;
    }

    for i in 1..=m {
        for j in 1..=n {
            let score = if a[i - 1] == b[j - 1] {
                match_score
            } else {
                mismatch_score
            };
            mtx[i][j] = mtx[i - 1][j - 1]
                .max(ix[i - 1][j - 1])
                .max(iy[i - 1][j - 1])
                + score;
            ix[i][j] = (ix[i - 1][j] + gap_extend)
                .max(mtx[i - 1][j] + gap_open + gap_extend)
                .max(iy[i - 1][j] + gap_open + gap_extend);
            iy[i][j] = (iy[i][j - 1] + gap_extend)
                .max(mtx[i][j - 1] + gap_open + gap_extend)
                .max(ix[i][j - 1] + gap_open + gap_extend);
        }
    }

    mtx[m][n].max(ix[m][n]).max(iy[m][n])
}

// ── Levenshtein ──────────────────────────────────────────────────────────────

proptest! {
    /// Myers bit-vector vs optimized row-DP vs naive, across the 64/65 boundary.
    #[test]
    fn levenshtein_ascii_all_impls_agree(a in ascii_string(200), b in ascii_string(200)) {
        let expected = naive_levenshtein(a.as_bytes(), b.as_bytes());
        prop_assert_eq!(levenshtein_myers(a.as_bytes(), b.as_bytes()), expected,
            "levenshtein_myers mismatch for {:?} vs {:?}", a, b);
        prop_assert_eq!(levenshtein_distance_raw(&a, &b), expected,
            "levenshtein_distance_raw mismatch for {:?} vs {:?}", a, b);
    }

    /// Pin the exact Myers 64-char bitmask boundary.
    #[test]
    fn levenshtein_myers_64_char_boundary(
        a in prop::collection::vec(prop::char::range('a', 'z'), 63usize..=66),
        b in prop::collection::vec(prop::char::range('a', 'z'), 63usize..=66),
    ) {
        let a: String = a.into_iter().collect();
        let b: String = b.into_iter().collect();
        prop_assert_eq!(
            levenshtein_myers(a.as_bytes(), b.as_bytes()),
            naive_levenshtein(a.as_bytes(), b.as_bytes()),
            "Myers 64-char boundary mismatch for {:?} vs {:?}", a, b
        );
    }

    /// Unicode goes through the char-based DP; validate against the oracle.
    #[test]
    fn levenshtein_unicode_matches_naive(a in unicode_string(60), b in unicode_string(60)) {
        let a_chars: Vec<char> = a.chars().collect();
        let b_chars: Vec<char> = b.chars().collect();
        prop_assert_eq!(
            levenshtein_distance_raw(&a, &b),
            naive_levenshtein(&a_chars, &b_chars),
            "Unicode Levenshtein mismatch for {:?} vs {:?}", a, b
        );
    }

    /// Levenshtein is a metric: triangle inequality must hold.
    #[test]
    fn levenshtein_triangle_inequality(a in ascii_string(40), b in ascii_string(40), c in ascii_string(40)) {
        let ab = levenshtein_distance_raw(&a, &b);
        let bc = levenshtein_distance_raw(&b, &c);
        let ac = levenshtein_distance_raw(&a, &c);
        prop_assert!(ac <= ab + bc, "d({:?},{:?})={} > d({:?},{:?})={} + d({:?},{:?})={}",
            a, c, ac, a, b, ab, b, c, bc);
    }
}

// ── Jaro ─────────────────────────────────────────────────────────────────────

proptest! {
    /// Stack/heap split at 128 bytes must not change results.
    #[test]
    fn jaro_ascii_optimized_matches_reference(a in ascii_string(300), b in ascii_string(300)) {
        let simd = jaro_optimized(a.as_bytes(), b.as_bytes());
        let reference = jaro(&a, &b);
        prop_assert!((simd - reference).abs() < 1e-12,
            "jaro_optimized {} != jaro {} for {:?} vs {:?}", simd, reference, a, b);
    }

    /// Pin the exact 128/129 stack-heap boundary.
    #[test]
    fn jaro_128_byte_boundary(
        a in prop::collection::vec(prop::char::range('a', 'z'), 127usize..=130),
        b in prop::collection::vec(prop::char::range('a', 'z'), 127usize..=130),
    ) {
        let a: String = a.into_iter().collect();
        let b: String = b.into_iter().collect();
        let simd = jaro_optimized(a.as_bytes(), b.as_bytes());
        let reference = jaro(&a, &b);
        prop_assert!((simd - reference).abs() < 1e-12,
            "128-byte boundary mismatch: {} vs {} for {:?} vs {:?}", simd, reference, a, b);
    }

    /// Unicode Jaro: `jaro_winkler(a, b, 0.0)` must reduce to `jaro(a, b)`.
    #[test]
    fn jaro_unicode_winkler_zero_equals_jaro(a in unicode_string(60), b in unicode_string(60)) {
        prop_assert!((jaro_winkler(&a, &b, 0.0) - jaro(&a, &b)).abs() < 1e-12,
            "jaro_winkler(.,.,0.0) {} != jaro {} for {:?} vs {:?}", jaro_winkler(&a, &b, 0.0), jaro(&a, &b), a, b);
    }

    #[test]
    fn jaro_identity_and_symmetry(a in any_string(60), b in any_string(60)) {
        prop_assert_eq!(jaro(&a, &a), 1.0);
        prop_assert!((jaro(&a, &b) - jaro(&b, &a)).abs() < 1e-12,
            "jaro not symmetric for {:?} vs {:?}", a, b);
    }
}

// ── Needleman-Wunsch ─────────────────────────────────────────────────────────

proptest! {
    /// Both linear-gap implementations must match the naive oracle.
    #[test]
    fn needleman_ascii_all_impls_agree(
        a in ascii_string(80), b in ascii_string(80),
        match_score in 1i64..5, mismatch_score in -5i64..-1, gap in -5i64..-1,
    ) {
        let expected = naive_needleman_wunsch(a.as_bytes(), b.as_bytes(), match_score, mismatch_score, gap);
        prop_assert_eq!(needleman_wunsch_striped(a.as_bytes(), b.as_bytes(), match_score, mismatch_score, gap), expected,
            "needleman_wunsch_striped mismatch for {:?} vs {:?}", a, b);
        prop_assert_eq!(needleman_wunsch(&a, &b, match_score, mismatch_score, gap), expected,
            "needleman_wunsch mismatch for {:?} vs {:?}", a, b);
    }

    #[test]
    fn needleman_unicode_matches_naive(
        a in unicode_string(40), b in unicode_string(40),
        match_score in 1i64..5, mismatch_score in -5i64..-1, gap in -5i64..-1,
    ) {
        let a_chars: Vec<char> = a.chars().collect();
        let b_chars: Vec<char> = b.chars().collect();
        prop_assert_eq!(
            needleman_wunsch(&a, &b, match_score, mismatch_score, gap),
            naive_needleman_wunsch(&a_chars, &b_chars, match_score, mismatch_score, gap),
            "Unicode Needleman-Wunsch mismatch for {:?} vs {:?}", a, b
        );
    }

    /// Affine-gap (Gotoh) ASCII path against the naive oracle.
    #[test]
    fn needleman_affine_ascii_matches_naive(
        a in ascii_string(60), b in ascii_string(60),
        match_score in 1i64..5, mismatch_score in -5i64..-1,
        gap_open in -10i64..-1, gap_extend in -5i64..-1,
    ) {
        prop_assert_eq!(
            needleman_wunsch_affine(&a, &b, match_score, mismatch_score, gap_open, gap_extend),
            naive_needleman_wunsch_affine(a.as_bytes(), b.as_bytes(), match_score, mismatch_score, gap_open, gap_extend),
            "affine Needleman-Wunsch mismatch for {:?} vs {:?}", a, b
        );
    }

    /// Affine-gap (Gotoh) Unicode path against the naive oracle.
    #[test]
    fn needleman_affine_unicode_matches_naive(
        a in unicode_string(40), b in unicode_string(40),
        match_score in 1i64..5, mismatch_score in -5i64..-1,
        gap_open in -10i64..-1, gap_extend in -5i64..-1,
    ) {
        let a_chars: Vec<char> = a.chars().collect();
        let b_chars: Vec<char> = b.chars().collect();
        prop_assert_eq!(
            needleman_wunsch_affine(&a, &b, match_score, mismatch_score, gap_open, gap_extend),
            naive_needleman_wunsch_affine(&a_chars, &b_chars, match_score, mismatch_score, gap_open, gap_extend),
            "Unicode affine Needleman-Wunsch mismatch for {:?} vs {:?}", a, b
        );
    }

    /// With `gap_open = 0`, affine cost `0 + k*gap_extend` degenerates to the
    /// linear model `k*gap_penalty`, so the affine and linear implementations
    /// must agree exactly — an independent cross-check between two library
    /// implementations.
    #[test]
    fn needleman_affine_open_zero_matches_linear(
        a in ascii_string(60), b in ascii_string(60),
        match_score in 1i64..5, mismatch_score in -5i64..-1, gap in -5i64..-1,
    ) {
        prop_assert_eq!(
            needleman_wunsch_affine(&a, &b, match_score, mismatch_score, 0, gap),
            needleman_wunsch(&a, &b, match_score, mismatch_score, gap),
            "affine(gap_open=0) != linear for {:?} vs {:?}", a, b
        );
    }

    // ── Affine (Gotoh) edge-case boundaries ────────────────────────────────

    /// One string empty: both implementations shortcut to a single gap of
    /// length |b|. Pin the closed form `gap_open + |b| * gap_extend` (and 0
    /// for both empty) directly — not just agreement between the two
    /// implementations, so a silent change to either shortcut fails loudly.
    #[test]
    fn needleman_affine_one_empty_boundary(
        b in prop::collection::vec(prop::char::range('a', 'z'), 1..=40),
        match_score in 1i64..5, mismatch_score in -5i64..-1,
        gap_open in -10i64..-1, gap_extend in -5i64..-1,
    ) {
        let b: String = b.into_iter().collect();
        let expected = gap_open + (b.len() as i64) * gap_extend;
        // ASCII byte path.
        prop_assert_eq!(
            needleman_wunsch_affine("", &b, match_score, mismatch_score, gap_open, gap_extend),
            expected,
            "empty vs {:?}", b
        );
        prop_assert_eq!(
            needleman_wunsch_affine(&b, "", match_score, mismatch_score, gap_open, gap_extend),
            expected,
            "{:?} vs empty", b
        );
        // Both empty — the degenerate alignment scores 0 regardless of gaps.
        prop_assert_eq!(
            needleman_wunsch_affine("", "", match_score, mismatch_score, gap_open, gap_extend),
            0
        );

        // Unicode char path: an empty ASCII string vs a non-ASCII string.
        let b_uni: String = "café\u{1F980}नमस्ते".chars().collect();
        let expected_uni = gap_open + (b_uni.chars().count() as i64) * gap_extend;
        prop_assert_eq!(
            needleman_wunsch_affine("", &b_uni, match_score, mismatch_score, gap_open, gap_extend),
            expected_uni,
            "empty vs {:?}", b_uni
        );
    }

    /// All-match: identical strings take the shortcut to `len * match_score`
    /// regardless of gap parameters — pin the exact value. Same-character
    /// strings of different lengths are also all-match where they overlap, but
    /// force the full DP through a trailing gap (no closed form — oracle
    /// agreement).
    #[test]
    fn needleman_affine_all_match_boundary(
        s in ascii_string(30), extra in 1usize..8,
        match_score in 1i64..5, mismatch_score in -5i64..-1,
        gap_open in -10i64..-1, gap_extend in -5i64..-1,
    ) {
        let expected = (s.len() as i64) * match_score;
        prop_assert_eq!(
            needleman_wunsch_affine(&s, &s, match_score, mismatch_score, gap_open, gap_extend),
            expected,
            "identical strings must score len*match for {:?}", s
        );

        // 'a'*m vs 'a'*(m+extra): every aligned pair matches; the DP decides
        // how to place `extra` gap characters.
        let short = "a".repeat(s.len());
        let long = "a".repeat(s.len() + extra);
        prop_assert_eq!(
            needleman_wunsch_affine(&short, &long, match_score, mismatch_score, gap_open, gap_extend),
            naive_needleman_wunsch_affine(short.as_bytes(), long.as_bytes(), match_score, mismatch_score, gap_open, gap_extend),
            "same-char unequal lengths: {}-vs-{}", short.len(), long.len()
        );
    }

    /// All-mismatch: strings over disjoint alphabets (every aligned pair is a
    /// substitution), equal and unequal lengths — exercises the DP's
    /// gap-vs-substitution tradeoff in full. No closed form, so pin by oracle
    /// agreement on both the byte and char paths.
    #[test]
    fn needleman_affine_all_mismatch_boundary(
        m in 1usize..40, n in 1usize..40,
        match_score in 1i64..5, mismatch_score in -5i64..-1,
        gap_open in -10i64..-1, gap_extend in -5i64..-1,
    ) {
        let a = "a".repeat(m);
        let b = "b".repeat(n);
        prop_assert_eq!(
            needleman_wunsch_affine(&a, &b, match_score, mismatch_score, gap_open, gap_extend),
            naive_needleman_wunsch_affine(a.as_bytes(), b.as_bytes(), match_score, mismatch_score, gap_open, gap_extend),
            "all-mismatch {}x{}", m, n
        );
        // Unicode path with disjoint astral-plane chars.
        let c: String = "\u{1F600}".repeat(m);
        let d: String = "\u{1F601}".repeat(n);
        let c_chars: Vec<char> = c.chars().collect();
        let d_chars: Vec<char> = d.chars().collect();
        prop_assert_eq!(
            needleman_wunsch_affine(&c, &d, match_score, mismatch_score, gap_open, gap_extend),
            naive_needleman_wunsch_affine(&c_chars, &d_chars, match_score, mismatch_score, gap_open, gap_extend),
            "all-mismatch (unicode) {}x{}", m, n
        );
    }

    /// Positive-gap regimes: `gap_extend > gap_open` with *both* gap costs
    /// positive flips the usual cost ordering (gaps can beat substitutions),
    /// and equal-length strings force every column through the DP. Both
    /// implementations must still agree.
    #[test]
    fn needleman_affine_positive_gap_regime_matches_naive(
        (a, b) in equal_len_ascii_pair(30),
        match_score in 1i64..5, mismatch_score in -5i64..-1,
        gap_open in 1i64..5, gap_extend in 5i64..9,
    ) {
        prop_assert_eq!(
            needleman_wunsch_affine(&a, &b, match_score, mismatch_score, gap_open, gap_extend),
            naive_needleman_wunsch_affine(a.as_bytes(), b.as_bytes(), match_score, mismatch_score, gap_open, gap_extend),
            "positive-gap (open={}, ext={}) {:?} vs {:?}", gap_open, gap_extend, a, b
        );
    }

    /// Mixed-sign regime with `gap_extend > gap_open`: opening a gap is
    /// cheaper than extending it (open negative, extend positive) — the
    /// opposite of the standard ordering, exercising a different DP corner.
    #[test]
    fn needleman_affine_positive_gap_mixed_sign_matches_naive(
        (a, b) in equal_len_ascii_pair(30),
        match_score in 1i64..5, mismatch_score in -5i64..-1,
        gap_open in -3i64..0, gap_extend in 2i64..6,
    ) {
        prop_assert_eq!(
            needleman_wunsch_affine(&a, &b, match_score, mismatch_score, gap_open, gap_extend),
            naive_needleman_wunsch_affine(a.as_bytes(), b.as_bytes(), match_score, mismatch_score, gap_open, gap_extend),
            "mixed-sign (open={}, ext={}) {:?} vs {:?}", gap_open, gap_extend, a, b
        );
    }
}

// ── Damerau-Levenshtein ──────────────────────────────────────────────────────

proptest! {
    /// No second full implementation exists publicly, so validate invariants
    /// that any correct Damerau-Levenshtein must satisfy.
    #[test]
    fn damerau_invariants(a in any_string(40), b in any_string(40)) {
        let d = damerau_levenshtein_distance(&a, &b);
        let l = levenshtein_distance_raw(&a, &b);

        // DL adds transpositions to the edit-operation set, so it can never
        // exceed plain Levenshtein.
        prop_assert!(d <= l, "damerau {} > levenshtein {} for {:?} vs {:?}", d, l, a, b);

        // Symmetry.
        prop_assert_eq!(d, damerau_levenshtein_distance(&b, &a),
            "damerau not symmetric for {:?} vs {:?}", a, b);

        // Every operation changes length by at most 1.
        let len_diff = (a.chars().count() as i64 - b.chars().count() as i64).unsigned_abs() as u32;
        prop_assert!(d >= len_diff, "damerau {} < |len diff| {} for {:?} vs {:?}", d, len_diff, a, b);

        // Identity.
        prop_assert_eq!(damerau_levenshtein_distance(&a, &a), 0);
    }
}

// ── GPU differential (skips when no device is available) ─────────────────────

#[cfg(feature = "gpu")]
mod gpu_differential {
    use super::*;
    use fuzzgpu_core::damerau::damerau_levenshtein_cdist;
    use fuzzgpu_core::damerau::gpu_ext::GpuDamerauKernel;
    use fuzzgpu_core::jaro::gpu_ext::GpuJaroKernel;
    use fuzzgpu_core::jaro::jaro_winkler_cdist_cpu;
    use fuzzgpu_core::levenshtein::gpu_ext::GpuLevenshteinKernel;
    use fuzzgpu_core::levenshtein::levenshtein_cdist_cpu;
    use fuzzgpu_core::needleman::gpu_ext::GpuNeedlemanAffineKernel;
    use std::sync::{Mutex, MutexGuard};

    /// Opt-in GPU serialization across the differential tests, mirroring the
    /// lib's contract: fully concurrent by default; when
    /// `FUZZGPU_SKIP_DISPATCH_LOCK=1` (opt-in safety valve for the rare
    /// gfx-rs/wgpu#10085 crash class) the lock serializes dispatch.
    static GPU_TEST_DISPATCH_LOCK: Mutex<()> = Mutex::new(());

    fn dispatch_serialize() -> bool {
        std::env::var("FUZZGPU_SKIP_DISPATCH_LOCK")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    }

    fn gpu_test_lock() -> Option<MutexGuard<'static, ()>> {
        if !dispatch_serialize() {
            return None;
        }
        Some(
            GPU_TEST_DISPATCH_LOCK
                .lock()
                .unwrap_or_else(|e| e.into_inner()),
        )
    }

    /// When `FUZZGPU_REQUIRE_GPU` is set (CI with a software Vulkan adapter),
    /// a missing device must fail the test instead of skipping — otherwise the
    /// GPU differentials would silently pass without exercising anything.
    /// (Integration tests can't see the lib's `#[cfg(test)]` helper, hence the
    /// local copy of the same env-var contract.)
    fn require_gpu() -> bool {
        std::env::var("FUZZGPU_REQUIRE_GPU")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    }

    fn gpu_levenshtein_or_skip() -> Option<&'static GpuLevenshteinKernel> {
        match GpuLevenshteinKernel::get() {
            Ok(k) => Some(k),
            Err(e) => {
                if require_gpu() {
                    panic!("FUZZGPU_REQUIRE_GPU is set but no usable GPU device: {}", e);
                }
                eprintln!("skipping GPU differential test (no usable device): {}", e);
                None
            }
        }
    }

    fn gpu_jaro_or_skip() -> Option<&'static GpuJaroKernel> {
        match GpuJaroKernel::get() {
            Ok(k) => Some(k),
            Err(e) => {
                if require_gpu() {
                    panic!("FUZZGPU_REQUIRE_GPU is set but no usable GPU device: {}", e);
                }
                eprintln!("skipping GPU differential test (no usable device): {}", e);
                None
            }
        }
    }

    fn gpu_needleman_affine_or_skip() -> Option<&'static GpuNeedlemanAffineKernel> {
        match GpuNeedlemanAffineKernel::get() {
            Ok(k) => Some(k),
            Err(e) => {
                if require_gpu() {
                    panic!("FUZZGPU_REQUIRE_GPU is set but no usable GPU device: {}", e);
                }
                eprintln!("skipping GPU differential test (no usable device): {}", e);
                None
            }
        }
    }

    fn gpu_damerau_or_skip() -> Option<&'static GpuDamerauKernel> {
        match GpuDamerauKernel::get() {
            Ok(k) => Some(k),
            Err(e) => {
                if require_gpu() {
                    panic!("FUZZGPU_REQUIRE_GPU is set but no usable GPU device: {}", e);
                }
                eprintln!("skipping GPU differential test (no usable device): {}", e);
                None
            }
        }
    }

    /// Force GPU dispatch regardless of metric/auto routing: Jaro and Damerau
    /// auto-route to CPU on integrated GPUs (where the SIMD path wins), so the
    /// differentials must set an explicit override to actually exercise the
    /// shaders. Uses `force_gpu_threshold` which holds the threshold test lock
    /// for its lifetime, preventing races with other tests.
    fn force_gpu() -> impl Drop {
        fuzzgpu_core::gpu::force_gpu_threshold(1)
    }

    /// GPU computes in f32, CPU in f64 — compare with epsilon.
    fn assert_close(a: &[f64], b: &[f64]) {
        assert_eq!(
            a.len(),
            b.len(),
            "length mismatch: {} vs {}",
            a.len(),
            b.len()
        );
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!(
                (x - y).abs() < 1e-4,
                "GPU Jaro result {} differs: {} vs {}",
                i,
                x,
                y
            );
        }
    }

    proptest! {
        // Default 10 cases for the GPU differentials (they dispatch on real
        // hardware), but respect the standard PROPTEST_CASES env var so CI or
        // local runs can scale coverage up — e.g. the `differential-deep` CI
        // job runs the whole suite at 64 cases to catch rare regressions.
        #![proptest_config(ProptestConfig::with_cases(
            std::env::var("PROPTEST_CASES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(10)
        ))]

        // Runtime note (measured): the GPU shaders run a serial O(n×m) DP per
        // pair, so long strings dominate the runtime quadratically — a
        // 1000-pair batch of ~150-char strings took ~1.8 s/dispatch on an
        // Intel iGPU (36 s for the Levenshtein batch test alone), versus
        // ~10-30 ms for ~30-char strings. Short strings keep every shader,
        // buffer, and packing code path covered while making the suite fast;
        // oversized pairs are appended to each case so the >256-char
        // CPU-fallback routing inside `compute` stays exercised (they never
        // reach the GPU shader, so they cost nothing there).

        /// Batch path: 1000 random ASCII/Unicode pairs per dispatch.
        #[test]
        fn levenshtein_gpu_batch_matches_cpu(pairs in prop::collection::vec(any_pair(32), 1000..=1000)) {
            let _gpu_guard = gpu_test_lock();
            let Some(kernel) = gpu_levenshtein_or_skip() else { return Ok(()); };
            let mut pairs = pairs;
            // Oversized (> 256 chars) pairs must route to CPU inside compute().
            pairs.push(("a".repeat(300), "b".repeat(300)));
            pairs.push(("x".repeat(280), "y".repeat(300)));
            let refs: Vec<(&str, &str)> =
                pairs.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
            let gpu = kernel.compute(&refs).expect("GPU compute should succeed");
            let cpu: Vec<u32> = refs.iter()
                .map(|(a, b)| levenshtein_distance_raw(a, b))
                .collect();
            prop_assert_eq!(gpu, cpu, "GPU Levenshtein batch must match CPU");
        }

        /// Matrix path: 32×32 strings, all within the 256-char shader limit so
        /// the GPU matrix shader is exercised on every case.
        #[test]
        fn levenshtein_gpu_matrix_matches_cpu(
            a in prop::collection::vec(any_string(32), 32..=32),
            b in prop::collection::vec(any_string(32), 32..=32),
        ) {
            let _gpu_guard = gpu_test_lock();
            let Some(kernel) = gpu_levenshtein_or_skip() else { return Ok(()); };
            let refs_a: Vec<&str> = a.iter().map(|s| s.as_str()).collect();
            let refs_b: Vec<&str> = b.iter().map(|s| s.as_str()).collect();
            let gpu = kernel.compute_matrix(&refs_a, &refs_b).expect("GPU matrix should succeed");
            let cpu = levenshtein_cdist_cpu(&refs_a, &refs_b);
            prop_assert_eq!(gpu, cpu, "GPU Levenshtein matrix must match CPU");
        }

        /// Row-wise Myers cdist: every query ASCII 1..=64 bytes (the Peq
        /// pattern) and every text ASCII of ANY length — including > 256
        /// chars, which the general DP matrix kernel rejects. Exercises the
        /// shared-Peq-per-row kernel + the long-text loop.
        #[test]
        fn levenshtein_gpu_matrix_myers_matches_cpu(
            a in prop::collection::vec(ascii_string_nz(64), 20..=20),
            b in prop::collection::vec(ascii_string(300), 40..=40),
        ) {
            let _gpu_guard = gpu_test_lock();
            let Some(kernel) = gpu_levenshtein_or_skip() else { return Ok(()); };
            let refs_a: Vec<&str> = a.iter().map(|s| s.as_str()).collect();
            let refs_b: Vec<&str> = b.iter().map(|s| s.as_str()).collect();
            // 20x40 = 800 cells, above the iGPU threshold (500) so the GPU
            // matrix path is taken, and every query is ASCII <= 64 so the
            // row-wise Myers route is selected.
            let gpu = kernel.compute_matrix(&refs_a, &refs_b).expect("GPU Myers matrix should succeed");
            let cpu = levenshtein_cdist_cpu(&refs_a, &refs_b);
            prop_assert_eq!(gpu, cpu, "GPU row-wise Myers matrix must match CPU");
        }

        /// Jaro-Winkler batch: 1000 random pairs, with appended > 128-char
        /// pairs so the CPU-fallback routing inside `compute_batch` stays
        /// exercised alongside the GPU kernel.
        #[test]
        fn jaro_gpu_batch_matches_cpu(pairs in prop::collection::vec(any_pair(32), 1000..=1000)) {
            let _gpu_guard = gpu_test_lock();
            let Some(kernel) = gpu_jaro_or_skip() else { return Ok(()); };
            let _force = force_gpu();
            let mut pairs = pairs;
            // Oversized (> 128 chars) pairs must route to CPU inside compute_batch().
            pairs.push(("a".repeat(140), "b".repeat(140)));
            pairs.push(("x".repeat(130), "y".repeat(140)));
            let refs: Vec<(&str, &str)> =
                pairs.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
            let gpu = kernel.compute_batch(&refs, 0.1).expect("GPU Jaro batch should succeed");
            let cpu: Vec<f64> = refs.iter()
                .map(|(a, b)| jaro_winkler(a, b, 0.1))
                .collect();
            assert_close(&gpu, &cpu);
        }

        /// Affine-gap Needleman-Wunsch GPU batch vs the CPU Gotoh oracle.
        /// Random ASCII/Unicode pairs (≤32 chars, so every pair runs on the GPU
        /// shader), plus appended oversized (>128 char) pairs that must route
        /// through the CPU fallback inside `compute_batch`. Scores are small,
        /// so the shader's f32 arithmetic is exact and the comparison is exact.
        #[test]
        fn needleman_affine_gpu_batch_matches_cpu(
            pairs in prop::collection::vec(any_pair(32), 1000..=1000),
            match_score in 1i64..5, mismatch_score in -5i64..-1,
            gap_open in -10i64..-1, gap_extend in -5i64..-1,
        ) {
            let _gpu_guard = gpu_test_lock();
            let Some(kernel) = gpu_needleman_affine_or_skip() else { return Ok(()); };
            let mut pairs = pairs;
            // Oversized (> 128 chars) pairs must route to CPU inside compute_batch.
            pairs.push(("a".repeat(140), "b".repeat(140)));
            pairs.push(("x".repeat(130), "y".repeat(150)));
            let refs: Vec<(&str, &str)> =
                pairs.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();

            let gpu = kernel
                .compute_batch(&refs, match_score, mismatch_score, gap_open, gap_extend)
                .expect("GPU affine batch should succeed");
            // Char-based oracle: correct for both ASCII and multi-byte Unicode.
            let cpu: Vec<i64> = refs.iter()
                .map(|(a, b)| {
                    let ca: Vec<char> = a.chars().collect();
                    let cb: Vec<char> = b.chars().collect();
                    naive_needleman_wunsch_affine(&ca, &cb, match_score, mismatch_score, gap_open, gap_extend)
                })
                .collect();
            prop_assert_eq!(gpu, cpu, "GPU affine batch must match the CPU Gotoh oracle");
        }

        /// Jaro-Winkler matrix: 32×32 strings, all within the 128-char shader limit.
        #[test]
        fn jaro_gpu_matrix_matches_cpu(
            a in prop::collection::vec(any_string(32), 32..=32),
            b in prop::collection::vec(any_string(32), 32..=32),
        ) {
            let _gpu_guard = gpu_test_lock();
            let Some(kernel) = gpu_jaro_or_skip() else { return Ok(()); };
            let _force = force_gpu();
            let refs_a: Vec<&str> = a.iter().map(|s| s.as_str()).collect();
            let refs_b: Vec<&str> = b.iter().map(|s| s.as_str()).collect();
            let gpu = kernel.compute_matrix(&refs_a, &refs_b, 0.1).expect("GPU Jaro matrix should succeed");
            let cpu = jaro_winkler_cdist_cpu(&refs_a, &refs_b, 0.1);
            for (grow, crow) in gpu.iter().zip(cpu.iter()) {
                assert_close(grow, crow);
            }
        }

        /// Damerau-Levenshtein batch: 1000 random ASCII pairs (all ≤ 32 chars,
        /// so every pair runs on the GPU's Lowrance-Wagner shader), plus
        /// appended > 32-char ASCII, Unicode, and non-ASCII pairs that must
        /// route through the CPU fallback inside `compute_batch`. The GPU
        /// result must be bit-exact with the CPU reference (both unrestricted
        /// Lowrance-Wagner), which is stronger than the Jaro epsilon compare.
        #[test]
        fn damerau_gpu_batch_matches_cpu(pairs in prop::collection::vec(ascii_pair(32), 1000..=1000)) {
            let _gpu_guard = gpu_test_lock();
            let Some(kernel) = gpu_damerau_or_skip() else { return Ok(()); };
            let _force = force_gpu();
            let mut pairs = pairs;
            // > 32 chars (CPU fallback) and Unicode (non-ASCII gate -> CPU).
            let long_a = "a".repeat(40);
            let long_b = "b".repeat(40);
            let uni_a = "héllo wörld".to_string();
            let uni_b = "héllo".to_string();
            pairs.push((long_a, long_b));
            pairs.push((uni_a, uni_b));
            let refs: Vec<(&str, &str)> =
                pairs.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
            let gpu = kernel.compute_batch(&refs).expect("GPU Damerau batch should succeed");
            let cpu: Vec<u32> = refs.iter()
                .map(|(a, b)| damerau_levenshtein_distance(a, b))
                .collect();
            prop_assert_eq!(&gpu, &cpu, "GPU Damerau batch must match CPU");
            prop_assert!(gpu.iter().all(|&d| d != u32::MAX), "u32::MAX sentinel leaked");
        }

        /// Damerau-Levenshtein matrix: 32×32 ASCII strings, all within the
        /// 32-char shader cap, so the 2D Lowrance-Wagner matrix kernel is
        /// exercised on every case.
        #[test]
        fn damerau_gpu_matrix_matches_cpu(
            a in prop::collection::vec(ascii_string(32), 32..=32),
            b in prop::collection::vec(ascii_string(32), 32..=32),
        ) {
            let _gpu_guard = gpu_test_lock();
            let Some(kernel) = gpu_damerau_or_skip() else { return Ok(()); };
            let _force = force_gpu();
            let refs_a: Vec<&str> = a.iter().map(|s| s.as_str()).collect();
            let refs_b: Vec<&str> = b.iter().map(|s| s.as_str()).collect();
            let gpu = kernel.compute_matrix(&refs_a, &refs_b).expect("GPU Damerau matrix should succeed");
            let cpu = damerau_levenshtein_cdist(&refs_a, &refs_b);
            prop_assert_eq!(gpu, cpu, "GPU Damerau matrix must match CPU");
        }
    }

    /// Wavefront path: the anti-diagonal wavefront kernel must agree exactly
    /// with the CPU Gotoh oracle. Called directly (it is an explicit API, not
    /// the default routing — measured ~140x slower than the serial shader on
    /// iGPU batches). Deterministic ASCII strings (70..=100 chars), an
    /// identical pair, an empty pair, a one-char pair, and a Unicode long
    /// pair (the wavefront compares u32 char codes). (Plain test, outside the
    /// `proptest!` block.)
    #[test]
    fn needleman_affine_gpu_wavefront_matches_cpu() {
        let _gpu_guard = gpu_test_lock();
        let Some(kernel) = gpu_needleman_affine_or_skip() else {
            return;
        };

        fn gen_long(count: usize, seed: u64) -> Vec<String> {
            let mut state = seed;
            let mut out = Vec::with_capacity(count);
            for _ in 0..count {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let len = 70 + ((state >> 33) as usize % 31); // 70..=100
                let mut s = String::with_capacity(len);
                for _ in 0..len {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    s.push((b'a' + ((state >> 33) as u8 % 26)) as char);
                }
                out.push(s);
            }
            out
        }

        let (match_score, mismatch_score, gap_open, gap_extend) = (2i64, -1i64, -3i64, -1i64);
        let a = gen_long(600, 0xACE0);
        let b = gen_long(600, 0xFACE);
        let mut pairs: Vec<(String, String)> = a.into_iter().zip(b).collect();
        // Edge cases: identical, empty, single-char, 64/65-char boundary,
        // Unicode (wavefront compares u32 char codes).
        pairs.push(("x".repeat(65), "x".repeat(65)));
        pairs.push(("".into(), "".into()));
        pairs.push(("a".into(), "b".into()));
        pairs.push(("x".repeat(64), "y".repeat(64)));
        pairs.push(("x".repeat(65), "y".repeat(65)));
        pairs.push(("é".repeat(80), "è".repeat(80)));
        let refs: Vec<(&str, &str)> = pairs
            .iter()
            .map(|(a, b)| (a.as_str(), b.as_str()))
            .collect();
        let indices: Vec<usize> = (0..refs.len()).collect();

        let gpu = kernel
            .compute_gpu_wavefront(
                &refs,
                &indices,
                match_score,
                mismatch_score,
                gap_open,
                gap_extend,
            )
            .expect("GPU wavefront dispatch should succeed");
        let cpu: Vec<i64> = refs
            .iter()
            .map(|(a, b)| {
                let ca: Vec<char> = a.chars().collect();
                let cb: Vec<char> = b.chars().collect();
                naive_needleman_wunsch_affine(
                    &ca,
                    &cb,
                    match_score,
                    mismatch_score,
                    gap_open,
                    gap_extend,
                )
            })
            .collect();
        let gpu_i64: Vec<i64> = gpu.iter().map(|&s| s as i64).collect();
        assert_eq!(
            gpu_i64, cpu,
            "GPU wavefront must match the CPU Gotoh oracle"
        );
    }

    /// Transposition-heavy edge cases must be bit-exact through the GPU
    /// Damerau kernel: adjacent transpositions, non-adjacent ("ca"/"abc" = 2,
    /// which optimal-string-alignment gets wrong), multi-transposition, and
    /// the classic "a cat"/"an act" (true Damerau = 2). (Plain test — no
    /// randomness, fast.)
    #[test]
    fn damerau_gpu_transposition_edges_match_cpu() {
        let _gpu_guard = gpu_test_lock();
        let Some(kernel) = gpu_damerau_or_skip() else {
            return;
        };
        let _force = force_gpu();

        let pairs: Vec<(&str, &str)> = vec![
            ("", ""),
            ("a", ""),
            ("", "b"),
            ("same", "same"),
            ("ab", "ba"),
            ("abc", "cba"),
            ("ca", "abc"),
            ("a cat", "an act"),
            ("sitting", "kitten"),
            ("dwayne", "duane"),
            ("abab", "baba"),
            ("xyz", "zyx"),
            ("abcdef", "abcfed"),
            ("abcdefgh", "abghefcd"),
        ];
        let gpu = kernel
            .compute_batch(&pairs)
            .expect("GPU Damerau batch should succeed");
        let cpu: Vec<u32> = pairs
            .iter()
            .map(|(a, b)| damerau_levenshtein_distance(a, b))
            .collect();
        assert_eq!(gpu, cpu, "GPU Damerau transposition edges must match CPU");
    }

    /// Jaro's classic transposition/shared-prefix cases through the GPU
    /// bitmap matcher (identical semantics to the CPU reference within 1e-4).
    #[test]
    fn jaro_gpu_classic_cases_match_cpu() {
        let _gpu_guard = gpu_test_lock();
        let Some(kernel) = gpu_jaro_or_skip() else {
            return;
        };
        let _force = force_gpu();

        let pairs: Vec<(&str, &str)> = vec![
            ("", ""),
            ("a", ""),
            ("martha", "martha"),
            ("martha", "marhta"),
            ("dwayne", "duane"),
            ("dixon", "dicksonx"),
            ("ab", "ba"),
            ("a cat", "an act"),
            ("crate", "trace"),
            ("", "xyz"),
            ("abcdef", "abcfed"),
        ];
        let gpu = kernel
            .compute_batch(&pairs, 0.1)
            .expect("GPU Jaro batch should succeed");
        let cpu: Vec<f64> = pairs.iter().map(|(a, b)| jaro_winkler(a, b, 0.1)).collect();
        assert_close(&gpu, &cpu);
    }

    /// The batched API (N ops queued, one dispatch + readback) must return
    /// exactly what N sequential per-op calls return, for all three kernels
    /// — including CPU-routed edge cases inside each op. (Lives outside the
    /// `proptest!` block so it runs as a plain deterministic test.)
    #[test]
    fn batch_api_matches_sequential_per_op() {
        let _gpu_guard = gpu_test_lock();
        // Force GPU dispatch: the Jaro/Damerau metric routing sends small and
        // mid-size batches to CPU on integrated GPUs, which would silently
        // drop the batched-API coverage.
        let _force = force_gpu();

        // Deterministic ASCII strings (same LCG as the lib tests).
        fn gen(count: usize, seed: u64) -> Vec<String> {
            let mut state = seed;
            let mut out = Vec::with_capacity(count);
            for _ in 0..count {
                let len = 4 + ((state >> 33) % 16) as usize;
                let mut s = String::with_capacity(len);
                for _ in 0..len {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    s.push((b'a' + ((state >> 33) as u8 % 26)) as char);
                }
                out.push(s);
            }
            out
        }

        let a = gen(600, 0x1111);
        let b = gen(600, 0x2222);
        let mut op1: Vec<(&str, &str)> = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x.as_str(), y.as_str()))
            .collect();
        op1[0] = ("", "");
        op1[1] = ("", "xyz");
        op1[2] = ("identical", "identical");

        let long = "a".repeat(300);
        let b100 = "b".repeat(100);
        let c100 = "c".repeat(100);
        let op2: Vec<(&str, &str)> = vec![
            ("kitten", "sitting"),
            (&long, "short"),
            ("日本語", "日本語のテスト"),
            // 100 chars: routes to the long GPU kernel (65..=256), not CPU.
            (&b100, &c100),
        ];
        let op3: Vec<(&str, &str)> = vec![("MARTHA", "MARHTA"), ("dwayne", "duane"), ("", "x")];

        // Levenshtein.
        if let Some(kernel) = gpu_levenshtein_or_skip() {
            let expected = vec![
                kernel.compute(&op1).expect("op1 compute"),
                kernel.compute(&op2).expect("op2 compute"),
                kernel.compute(&op3).expect("op3 compute"),
            ];
            let mut batch = kernel.batch();
            batch.add(&op1);
            batch.add(&op2);
            batch.add(&op3);
            assert_eq!(
                batch.execute().expect("levenshtein batch"),
                expected,
                "Levenshtein batch must equal sequential per-op compute"
            );
        }

        // Jaro-Winkler.
        if let Some(kernel) = gpu_jaro_or_skip() {
            let p = 0.1;
            let expected = vec![
                kernel.compute_batch(&op1, p).expect("op1 compute_batch"),
                kernel.compute_batch(&op2, p).expect("op2 compute_batch"),
                kernel.compute_batch(&op3, p).expect("op3 compute_batch"),
            ];
            let mut batch = kernel.batch(p).expect("jaro batch creation");
            batch.add(&op1);
            batch.add(&op2);
            batch.add(&op3);
            let got = batch.execute().expect("jaro batch");
            for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
                assert_eq!(g.len(), e.len(), "jaro op {i} length");
                for (j, (&gv, &ev)) in g.iter().zip(e).enumerate() {
                    assert!(
                        (gv - ev).abs() < 1e-4,
                        "jaro op {i} pair {j}: batch {gv} vs compute {ev}"
                    );
                }
            }
        }

        // Affine Needleman-Wunsch.
        if let Some(kernel) = gpu_needleman_affine_or_skip() {
            let (m, mm, go, ge) = (2i64, -1i64, -3i64, -1i64);
            let expected = vec![
                kernel
                    .compute_batch(&op1, m, mm, go, ge)
                    .expect("op1 affine"),
                kernel
                    .compute_batch(&op2, m, mm, go, ge)
                    .expect("op2 affine"),
                kernel
                    .compute_batch(&op3, m, mm, go, ge)
                    .expect("op3 affine"),
            ];
            let mut batch = kernel.batch(m, mm, go, ge);
            batch.add(&op1);
            batch.add(&op2);
            batch.add(&op3);
            assert_eq!(
                batch.execute().expect("needleman batch"),
                expected,
                "Needleman batch must equal sequential per-op compute_batch"
            );
        }
    }
}
