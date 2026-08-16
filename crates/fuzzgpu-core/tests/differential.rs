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
    damerau_levenshtein_distance, jaro, jaro_optimized, jaro_winkler,
    levenshtein_distance_raw, levenshtein_myers, needleman_wunsch, needleman_wunsch_affine,
    needleman_wunsch_striped,
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

fn any_pair(max_len: usize) -> impl Strategy<Value = (String, String)> {
    (any_string(max_len), any_string(max_len))
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
            let score = if a[i - 1] == b[j - 1] { match_score } else { mismatch_score };
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
    if m == 0 && n == 0 { return 0; }
    if m == 0 { return gap_open + (n as i64) * gap_extend; }
    if n == 0 { return gap_open + (m as i64) * gap_extend; }
    if a == b { return (m as i64) * match_score; }

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
            let score = if a[i - 1] == b[j - 1] { match_score } else { mismatch_score };
            mtx[i][j] = mtx[i - 1][j - 1].max(ix[i - 1][j - 1]).max(iy[i - 1][j - 1]) + score;
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
    use std::sync::{Mutex, MutexGuard};
    use fuzzgpu_core::jaro::gpu_ext::GpuJaroKernel;
    use fuzzgpu_core::jaro::jaro_winkler_cdist_cpu;
    use fuzzgpu_core::levenshtein::gpu_ext::GpuLevenshteinKernel;
    use fuzzgpu_core::levenshtein::levenshtein_cdist_cpu;
    use fuzzgpu_core::needleman::gpu_ext::GpuNeedlemanAffineKernel;

    /// Serializes GPU access across the differential tests. The lib suite's
    /// equivalent (`gpu::GPU_TEST_DISPATCH_LOCK`) is `#[cfg(test)]`-only and
    /// invisible to integration tests, so this is a local copy — the same
    /// workaround for the same wgpu/driver crash (gfx-rs/wgpu#10085) observed
    /// under >=3 concurrent dispatchers on the shared device (Intel Iris Xe).
    static GPU_TEST_DISPATCH_LOCK: Mutex<()> = Mutex::new(());

    fn gpu_test_lock() -> MutexGuard<'static, ()> {
        GPU_TEST_DISPATCH_LOCK.lock().unwrap_or_else(|e| e.into_inner())
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

    /// GPU computes in f32, CPU in f64 — compare with epsilon.
    fn assert_close(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).abs() < 1e-4, "GPU Jaro result {} differs: {} vs {}", i, x, y);
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

        /// Jaro-Winkler batch: 1000 random pairs, with appended > 128-char
        /// pairs so the CPU-fallback routing inside `compute_batch` stays
        /// exercised alongside the GPU kernel.
        #[test]
        fn jaro_gpu_batch_matches_cpu(pairs in prop::collection::vec(any_pair(32), 1000..=1000)) {
            let _gpu_guard = gpu_test_lock();
            let Some(kernel) = gpu_jaro_or_skip() else { return Ok(()); };
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
            let refs_a: Vec<&str> = a.iter().map(|s| s.as_str()).collect();
            let refs_b: Vec<&str> = b.iter().map(|s| s.as_str()).collect();
            let gpu = kernel.compute_matrix(&refs_a, &refs_b, 0.1).expect("GPU Jaro matrix should succeed");
            let cpu = jaro_winkler_cdist_cpu(&refs_a, &refs_b, 0.1);
            for (grow, crow) in gpu.iter().zip(cpu.iter()) {
                assert_close(grow, crow);
            }
        }
    }
}
