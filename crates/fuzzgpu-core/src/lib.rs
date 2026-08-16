pub mod levenshtein;
pub mod needleman;
pub mod jaro;
pub mod fuzz;
pub mod simd;
pub mod damerau;
#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(feature = "gpu")]
pub use gpu::{GpuEngine, GpuInfo, Result};

#[cfg(not(feature = "gpu"))]
pub type Result<T> = std::result::Result<T, String>;

pub use levenshtein::levenshtein_distance_raw;
pub use levenshtein::LevenshteinKernel;
pub use needleman::{needleman_wunsch, needleman_wunsch_batch, needleman_wunsch_affine, needleman_wunsch_affine_batch};
pub use jaro::{jaro, jaro_winkler, jaro_winkler_batch};
pub use fuzz::{ratio, partial_ratio, token_sort_ratio, token_set_ratio, wratio, ratio_batch, extract, extract_one};
pub use simd::{levenshtein_myers, needleman_wunsch_striped, jaro_optimized};
pub use damerau::{damerau_levenshtein_distance, damerau_levenshtein_batch, damerau_levenshtein_cdist, damerau_ratio};

/// Saturating `i64` addition that also detects overflow in debug builds.
///
/// Release builds keep saturating semantics (no panics, no UB) so production
/// never crashes on pathological inputs; debug builds assert so genuine
/// overflow is caught during development instead of silently producing a
/// wrong alignment score.
#[inline]
pub(crate) fn sat_add(a: i64, b: i64) -> i64 {
    let r = a.saturating_add(b);
    debug_assert!(
        a.checked_add(b).is_some(),
        "fuzzgpu: i64 overflow in score computation ({} + {})",
        a, b
    );
    r
}

/// Saturating `i64` multiplication with debug-build overflow detection.
/// See [`sat_add`].
#[inline]
pub(crate) fn sat_mul(a: i64, b: i64) -> i64 {
    let r = a.saturating_mul(b);
    debug_assert!(
        a.checked_mul(b).is_some(),
        "fuzzgpu: i64 overflow in score computation ({} * {})",
        a, b
    );
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sat_helpers_preserve_values() {
        assert_eq!(sat_add(3, 4), 7);
        assert_eq!(sat_add(-3, 4), 1);
        assert_eq!(sat_mul(6, 7), 42);
        assert_eq!(sat_mul(-6, 7), -42);
    }

    // Production (release) behavior: saturate silently, never UB or panic.
    // (In debug builds the same inputs trip the overflow `debug_assert!` above.)
    #[cfg(not(debug_assertions))]
    #[test]
    fn test_sat_helpers_saturate_instead_of_panicking() {
        assert_eq!(sat_add(i64::MAX, 1), i64::MAX);
        assert_eq!(sat_add(i64::MIN, -1), i64::MIN);
        assert_eq!(sat_mul(i64::MAX, 2), i64::MAX);
        assert_eq!(sat_mul(i64::MIN, 2), i64::MIN);
    }

    // Debug builds must flag the overflow so silent wrong scores are caught in dev.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "i64 overflow in score computation")]
    fn test_sat_add_detects_overflow_in_debug() {
        let _ = sat_add(i64::MAX, 1);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "i64 overflow in score computation")]
    fn test_sat_mul_detects_overflow_in_debug() {
        let _ = sat_mul(i64::MAX, 2);
    }
}
