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
