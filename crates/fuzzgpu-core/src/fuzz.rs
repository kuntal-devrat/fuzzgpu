#![allow(
    clippy::manual_clamp,
    clippy::manual_div_ceil,
    clippy::needless_range_loop,
    clippy::unnecessary_map_or
)]

use rayon::prelude::*;
use std::collections::{BTreeSet, HashSet};

/// Standard Indel / Levenshtein (substitution cost = 2) edit distance used for fuzzy ratios.
/// Supports both ASCII fast-path and full Unicode characters.
///
/// For ASCII inputs with pattern ≤ 64 bytes, routes through the Myers bit-vector
/// for O(n) performance instead of O(mn) DP.
#[inline]
pub fn indel_distance(a: &str, b: &str) -> u32 {
    if a.is_ascii() && b.is_ascii() {
        indel_distance_bytes(a.as_bytes(), b.as_bytes())
    } else {
        let a_chars: Vec<char> = a.chars().collect();
        let b_chars: Vec<char> = b.chars().collect();
        indel_distance_slice(&a_chars, &b_chars)
    }
}

/// ASCII fast path for indel distance using standard single-row DP.
/// Substitution cost = 2 (matching rapidfuzz's indel model).
fn indel_distance_bytes(a: &[u8], b: &[u8]) -> u32 {
    indel_distance_slice(a, b)
}

fn indel_distance_slice<T: PartialEq>(a: &[T], b: &[T]) -> u32 {
    let (m, n) = (a.len(), b.len());
    if m == 0 {
        return n as u32;
    }
    if n == 0 {
        return m as u32;
    }
    if a == b {
        return 0;
    }

    let mut row = vec![0u32; n + 1];
    for (j, item) in row.iter_mut().enumerate() {
        *item = j as u32;
    }

    for i in 1..=m {
        let mut prev_diag = row[0];
        row[0] = i as u32;
        let ai = &a[i - 1];
        for j in 1..=n {
            let old = row[j];
            let cost = if ai == &b[j - 1] { 0 } else { 2 };
            row[j] = (prev_diag + cost).min(row[j] + 1).min(row[j - 1] + 1);
            prev_diag = old;
        }
    }
    row[n]
}

/// Similarity ratio 0.0–100.0 using the standard 2*M / (|a| + |b|) Sørensen–Dice Levenshtein coefficient.
///
/// Formula: `(1 - indel_distance / (|a| + |b|)) × 100` — evaluated in exactly
/// rapidfuzz's order (`indel_normalized_similarity * 100`) for bit-identical
/// floats.
///
/// Matches RapidFuzz and FuzzyWuzzy identically across all ASCII and multi-byte Unicode test cases.
pub fn ratio(s1: &str, s2: &str) -> f64 {
    ratio_with_cutoff(s1, s2, 0.0)
}

/// `ratio` with rapidfuzz's cutoff semantics: returns 0.0 when the score is
/// below `score_cutoff`, otherwise the raw score.
pub fn ratio_with_cutoff(s1: &str, s2: &str, score_cutoff: f64) -> f64 {
    let len_a = if s1.is_ascii() {
        s1.len()
    } else {
        s1.chars().count()
    };
    let len_b = if s2.is_ascii() {
        s2.len()
    } else {
        s2.chars().count()
    };
    let total = len_a + len_b;
    let score = if total == 0 {
        100.0
    } else {
        let dist = indel_distance(s1, s2) as f64;
        (1.0 - dist / total as f64) * 100.0
    };
    if score >= score_cutoff {
        score
    } else {
        0.0
    }
}

/// rapidfuzz `norm_distance`: similarity from an edit distance normalized by
/// the length sum, with cutoff semantics (0 when below cutoff).
#[inline]
fn norm_distance(dist: usize, lensum: usize, score_cutoff: f64) -> f64 {
    let score = if lensum > 0 {
        100.0 - 100.0 * dist as f64 / lensum as f64
    } else {
        100.0
    };
    if score >= score_cutoff {
        score
    } else {
        0.0
    }
}

/// rapidfuzz `score_cutoff_to_distance`: the maximum edit distance that can
/// still meet the cutoff.
#[inline]
fn score_cutoff_to_distance(score_cutoff: f64, lensum: usize) -> usize {
    (lensum as f64 * (1.0 - score_cutoff / 100.0)).ceil() as usize
}

/// Alignment result, rapidfuzz-compatible: `(score, src_start, src_end,
/// dest_start, dest_end)` — identical field order to rapidfuzz's
/// `ScoreAlignment`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Alignment {
    pub score: f64,
    pub src_start: usize,
    pub src_end: usize,
    pub dest_start: usize,
    pub dest_end: usize,
}

/// `ratio` over slices with cutoff semantics (indel distance normalized by the
/// length sum, like rapidfuzz's `indel_normalized_similarity`).
fn ratio_slices_with_cutoff<T: PartialEq>(a: &[T], b: &[T], score_cutoff: f64) -> f64 {
    let lensum = a.len() + b.len();
    let score = if lensum == 0 {
        100.0
    } else {
        let dist = indel_distance_slice(a, b) as f64;
        (1.0 - dist / lensum as f64) * 100.0
    };
    if score >= score_cutoff {
        score
    } else {
        0.0
    }
}

/// Port of rapidfuzz's `partial_ratio_impl` (fuzz_impl.hpp): the branch-and-bound
/// window search over the longer string plus the prefix/suffix search, with the
/// char-set pruning and running-cutoff semantics exactly as rapidfuzz threads
/// them.
///
/// Preconditions: `a.len() <= b.len()`, both non-empty. Returns
/// `(score, dest_start, dest_end)`; `src` is always `0..a.len()`.
fn partial_ratio_impl<T: PartialEq + Eq + std::hash::Hash + Copy>(
    a: &[T],
    b: &[T],
    score_cutoff: f64,
) -> (f64, usize, usize) {
    let len1 = a.len();
    let len2 = b.len();
    let mut cutoff = score_cutoff;
    let mut res_score = 0.0f64;
    let mut dest_start = 0usize;
    let mut dest_end = len1;

    // Char-set of the (shorter) pattern; prefixes/suffixes whose boundary char
    // is absent from it cannot beat the running best and are skipped — the
    // exact pruning rapidfuzz applies.
    let char_set: HashSet<T> = a.iter().copied().collect();

    if len2 > len1 {
        let maximum = len1 * 2;
        // rapidfuzz `NormSim_to_NormDist(score_cutoff / 100)` — note the 1e-5
        // imprecision term, which is load-bearing for the pruning bound.
        let norm_cutoff_sim = (1.0 - cutoff / 100.0 + 0.00001).min(1.0);
        let mut cutoff_dist = (maximum as f64 * norm_cutoff_sim).ceil() as usize;
        let mut best_dist = usize::MAX;
        let mut scores: Vec<usize> = vec![usize::MAX; len2 - len1];
        let mut windows: Vec<(usize, usize)> = vec![(0, len2 - len1 - 1)];
        let mut new_windows: Vec<(usize, usize)> = Vec::new();

        while !windows.is_empty() {
            for &(first, second) in windows.iter() {
                if scores[first] == usize::MAX {
                    scores[first] = indel_distance_slice(a, &b[first..first + len1]) as usize;
                    if scores[first] < cutoff_dist {
                        cutoff_dist = scores[first];
                        best_dist = scores[first];
                        dest_start = first;
                        dest_end = first + len1;
                        if best_dist == 0 {
                            return (100.0, dest_start, dest_end);
                        }
                    }
                }
                if scores[second] == usize::MAX {
                    scores[second] = indel_distance_slice(a, &b[second..second + len1]) as usize;
                    if scores[second] < cutoff_dist {
                        cutoff_dist = scores[second];
                        best_dist = scores[second];
                        dest_start = second;
                        dest_end = second + len1;
                        if best_dist == 0 {
                            return (100.0, dest_start, dest_end);
                        }
                    }
                }

                let cell_diff = second - first;
                if cell_diff == 1 {
                    continue;
                }

                // Bound: no window strictly between `first` and `second` can
                // beat the current best if the minimum possible distance in
                // the range is already at/above cutoff_dist.
                let known_edits = scores[first].abs_diff(scores[second]);
                let max_score_improvement = ((cell_diff - known_edits / 2) / 2) * 2;
                let min_score =
                    scores[first].min(scores[second]) as i64 - max_score_improvement as i64;
                if min_score < cutoff_dist as i64 {
                    let center = cell_diff / 2;
                    new_windows.push((first, first + center));
                    new_windows.push((first + center, second));
                }
            }
            std::mem::swap(&mut windows, &mut new_windows);
            new_windows.clear();
        }

        let score = (1.0 - best_dist as f64 / maximum as f64) * 100.0;
        if score >= cutoff {
            cutoff = score;
            res_score = score;
        }
    }

    // Prefixes of b (length 1..len1), pruning when the boundary char is not in
    // the pattern's char set (rapidfuzz-exact).
    for i in 1..len1 {
        if !char_set.contains(&b[i - 1]) {
            continue;
        }
        let ls_ratio = ratio_slices_with_cutoff(a, &b[..i], cutoff);
        if ls_ratio > res_score {
            cutoff = ls_ratio;
            res_score = ls_ratio;
            dest_start = 0;
            dest_end = i;
            if res_score == 100.0 {
                return (100.0, dest_start, dest_end);
            }
        }
    }

    // Suffixes of b (length len1 down to 1, i.e. starts len2-len1 .. len2-1),
    // same pruning. The suffix of length len1 is the last full window.
    for i in (len2 - len1)..len2 {
        if !char_set.contains(&b[i]) {
            continue;
        }
        let ls_ratio = ratio_slices_with_cutoff(a, &b[i..], cutoff);
        if ls_ratio > res_score {
            cutoff = ls_ratio;
            res_score = ls_ratio;
            dest_start = i;
            dest_end = len2;
            if res_score == 100.0 {
                return (100.0, dest_start, dest_end);
            }
        }
    }

    (res_score, dest_start, dest_end)
}

/// Partial ratio over slices, returning the full rapidfuzz-compatible
/// alignment `(score, src_start, src_end, dest_start, dest_end)`.
fn partial_ratio_alignment_slices<T: PartialEq + Eq + std::hash::Hash + Copy + Clone>(
    a: &[T],
    b: &[T],
    score_cutoff: f64,
) -> Alignment {
    let len1 = a.len();
    let len2 = b.len();

    // Swap so the first argument is always the shorter string, then swap the
    // alignment back (rapidfuzz: `src` is relative to the FIRST argument).
    if len1 > len2 {
        let r = partial_ratio_alignment_slices(b, a, score_cutoff);
        return Alignment {
            score: r.score,
            src_start: r.dest_start,
            src_end: r.dest_end,
            dest_start: r.src_start,
            dest_end: r.src_end,
        };
    }

    if score_cutoff > 100.0 {
        return Alignment {
            score: 0.0,
            src_start: 0,
            src_end: len1,
            dest_start: 0,
            dest_end: len1,
        };
    }
    if len1 == 0 || len2 == 0 {
        let score = if len1 == len2 { 100.0 } else { 0.0 };
        return Alignment {
            score,
            src_start: 0,
            src_end: len1,
            dest_start: 0,
            dest_end: len1,
        };
    }

    let (score, ds, de) = partial_ratio_impl(a, b, score_cutoff);
    let mut alignment = Alignment {
        score,
        src_start: 0,
        src_end: len1,
        dest_start: ds,
        dest_end: de,
    };

    // Equal-length second pass in the other direction (rapidfuzz-exact: the
    // reverse-direction search can find a better window/prefix/suffix match).
    if alignment.score != 100.0 && len1 == len2 {
        let c = score_cutoff.max(alignment.score);
        let (score2, ds2, de2) = partial_ratio_impl(b, a, c);
        if score2 > alignment.score {
            alignment = Alignment {
                score: score2,
                src_start: ds2,
                src_end: de2,
                dest_start: 0,
                dest_end: len1,
            };
        }
    }

    alignment
}

/// Partial ratio: best ratio of the shorter string against all windows of the
/// longer string of length len(shorter), plus all its prefixes (1..len-1) and
/// suffixes (1..len) — the exact rapidfuzz algorithm (branch-and-bound window
/// search + prefix/suffix search with char-set pruning and running cutoffs).
pub fn partial_ratio_alignment(s1: &str, s2: &str, score_cutoff: f64) -> Alignment {
    if s1.is_ascii() && s2.is_ascii() {
        partial_ratio_alignment_slices(s1.as_bytes(), s2.as_bytes(), score_cutoff)
    } else {
        let a: Vec<char> = s1.chars().collect();
        let b: Vec<char> = s2.chars().collect();
        partial_ratio_alignment_slices(&a, &b, score_cutoff)
    }
}

/// Partial ratio with rapidfuzz's cutoff semantics.
pub fn partial_ratio(s1: &str, s2: &str, score_cutoff: f64) -> f64 {
    partial_ratio_alignment(s1, s2, score_cutoff).score
}

/// Split on whitespace and sort lexicographically (rapidfuzz `sorted_split`).
fn sorted_split(s: &str) -> Vec<&str> {
    let mut tokens: Vec<&str> = s.split_whitespace().collect();
    tokens.sort_unstable();
    tokens
}

/// Sorted unique-token set decomposition (rapidfuzz `set_decomposition`).
struct SetDecomposition<'a> {
    intersection: Vec<&'a str>,
    diff_ab: Vec<&'a str>,
    diff_ba: Vec<&'a str>,
}

fn set_decomposition<'a>(tokens_a: &'a [&'a str], tokens_b: &'a [&'a str]) -> SetDecomposition<'a> {
    let set_a: BTreeSet<&str> = tokens_a.iter().copied().collect();
    let set_b: BTreeSet<&str> = tokens_b.iter().copied().collect();
    SetDecomposition {
        intersection: set_a.intersection(&set_b).copied().collect(),
        diff_ab: set_a.difference(&set_b).copied().collect(),
        diff_ba: set_b.difference(&set_a).copied().collect(),
    }
}

/// Total char length of the intersection tokens as joined with separators —
/// rapidfuzz's `SplittedSentenceView::length()` includes the whitespace
/// between words (`sum(word lengths) + (word_count - 1)`).
fn intersection_len(intersection: &[&str]) -> usize {
    if intersection.is_empty() {
        return 0;
    }
    let word_len: usize = intersection.iter().map(|t| t.chars().count()).sum();
    word_len + intersection.len() - 1
}

/// rapidfuzz `token_set_ratio` over token lists (fuzz_impl.hpp): empty token
/// sets score 0 (FuzzyWuzzy compatibility), a non-empty intersection with one
/// empty difference scores 100, otherwise a norm_distance over the joined
/// differences plus the section ratios.
fn token_set_ratio_inner(tokens_a: &[&str], tokens_b: &[&str], score_cutoff: f64) -> f64 {
    if tokens_a.is_empty() || tokens_b.is_empty() {
        return 0.0;
    }

    let decomp = set_decomposition(tokens_a, tokens_b);
    let (intersect, diff_ab, diff_ba) = (&decomp.intersection, &decomp.diff_ab, &decomp.diff_ba);

    // One sentence is part of the other one.
    if !intersect.is_empty() && (diff_ab.is_empty() || diff_ba.is_empty()) {
        return 100.0;
    }

    let diff_ab_joined = diff_ab.join(" ");
    let diff_ba_joined = diff_ba.join(" ");

    let ab_len = diff_ab_joined.chars().count();
    let ba_len = diff_ba_joined.chars().count();
    let sect_len = intersection_len(intersect);

    // String length sect+ab <-> sect and sect+ba <-> sect.
    let sect_ab_len = sect_len + usize::from(sect_len > 0) + ab_len;
    let sect_ba_len = sect_len + usize::from(sect_len > 0) + ba_len;

    let mut result = 0.0;
    let cutoff_distance = score_cutoff_to_distance(score_cutoff, sect_ab_len + sect_ba_len);
    let dist = indel_distance(&diff_ab_joined, &diff_ba_joined) as usize;
    if dist <= cutoff_distance {
        result = norm_distance(dist, sect_ab_len + sect_ba_len, score_cutoff);
    }

    // Exit early since the other ratios are 0.
    if sect_len == 0 {
        return result;
    }

    // The intersection is shared verbatim, so the distance to the full
    // "sect+ab"/"sect+ba" strings reduces to a length difference.
    let sect_ab_dist = usize::from(sect_len > 0) + ab_len;
    let sect_ab_ratio = norm_distance(sect_ab_dist, sect_len + sect_ab_len, score_cutoff);

    let sect_ba_dist = usize::from(sect_len > 0) + ba_len;
    let sect_ba_ratio = norm_distance(sect_ba_dist, sect_len + sect_ba_len, score_cutoff);

    result.max(sect_ab_ratio).max(sect_ba_ratio)
}

/// Token sort ratio: sort tokens alphabetically, then compare.
pub fn token_sort_ratio(s1: &str, s2: &str, score_cutoff: f64) -> f64 {
    if score_cutoff > 100.0 {
        return 0.0;
    }
    let t1 = sorted_split(s1);
    let t2 = sorted_split(s2);
    ratio_with_cutoff(&t1.join(" "), &t2.join(" "), score_cutoff)
}

/// Token set ratio (rapidfuzz-exact, including the FuzzyWuzzy-compatible 0 for
/// empty token sets).
pub fn token_set_ratio(s1: &str, s2: &str, score_cutoff: f64) -> f64 {
    if score_cutoff > 100.0 {
        return 0.0;
    }
    let t1 = sorted_split(s1);
    let t2 = sorted_split(s2);
    token_set_ratio_inner(&t1, &t2, score_cutoff)
}

/// rapidfuzz `token_ratio`: max of the sorted-joined ratio and the token-set
/// decomposition ratios (used by WRatio). Note: unlike `token_set_ratio` this
/// has NO empty-input guard — rapidfuzz scores two empty strings 100 here.
pub fn token_ratio(s1: &str, s2: &str, score_cutoff: f64) -> f64 {
    if score_cutoff > 100.0 {
        return 0.0;
    }
    let t1 = sorted_split(s1);
    let t2 = sorted_split(s2);

    let decomp = set_decomposition(&t1, &t2);
    let (intersect, diff_ab, diff_ba) = (&decomp.intersection, &decomp.diff_ab, &decomp.diff_ba);

    if !intersect.is_empty() && (diff_ab.is_empty() || diff_ba.is_empty()) {
        return 100.0;
    }

    let diff_ab_joined = diff_ab.join(" ");
    let diff_ba_joined = diff_ba.join(" ");

    let ab_len = diff_ab_joined.chars().count();
    let ba_len = diff_ba_joined.chars().count();
    let sect_len = intersection_len(intersect);

    let mut result = ratio_with_cutoff(&t1.join(" "), &t2.join(" "), score_cutoff);

    let sect_ab_len = sect_len + usize::from(sect_len > 0) + ab_len;
    let sect_ba_len = sect_len + usize::from(sect_len > 0) + ba_len;

    let cutoff_distance = score_cutoff_to_distance(score_cutoff, sect_ab_len + sect_ba_len);
    let dist = indel_distance(&diff_ab_joined, &diff_ba_joined) as usize;
    if dist <= cutoff_distance {
        result = result.max(norm_distance(dist, sect_ab_len + sect_ba_len, score_cutoff));
    }

    if sect_len == 0 {
        return result;
    }

    let sect_ab_dist = usize::from(sect_len > 0) + ab_len;
    let sect_ab_ratio = norm_distance(sect_ab_dist, sect_len + sect_ab_len, score_cutoff);

    let sect_ba_dist = usize::from(sect_len > 0) + ba_len;
    let sect_ba_ratio = norm_distance(sect_ba_dist, sect_len + sect_ba_len, score_cutoff);

    result.max(sect_ab_ratio).max(sect_ba_ratio)
}

/// Partial token sort ratio: sort tokens, then partial_ratio.
pub fn partial_token_sort_ratio(s1: &str, s2: &str, score_cutoff: f64) -> f64 {
    if score_cutoff > 100.0 {
        return 0.0;
    }
    let t1 = sorted_split(s1);
    let t2 = sorted_split(s2);
    partial_ratio(&t1.join(" "), &t2.join(" "), score_cutoff)
}

/// rapidfuzz `partial_token_ratio`: 100 when the token intersection is
/// non-empty, otherwise partial_ratio of the joined token lists (and of the
/// joined differences when they differ from the full lists). No empty-input
/// guard (rapidfuzz: two empty strings score 100).
pub fn partial_token_ratio(s1: &str, s2: &str, score_cutoff: f64) -> f64 {
    if score_cutoff > 100.0 {
        return 0.0;
    }
    let t1 = sorted_split(s1);
    let t2 = sorted_split(s2);

    let decomp = set_decomposition(&t1, &t2);

    // Exit early when there is a common word in both sequences.
    if !decomp.intersection.is_empty() {
        return 100.0;
    }

    let result = partial_ratio(&t1.join(" "), &t2.join(" "), score_cutoff);

    // Do not calculate the same partial_ratio twice.
    if t1.len() == decomp.diff_ab.len() && t2.len() == decomp.diff_ba.len() {
        return result;
    }

    let c = score_cutoff.max(result);
    result.max(partial_ratio(
        &decomp.diff_ab.join(" "),
        &decomp.diff_ba.join(" "),
        c,
    ))
}

/// Partial token set ratio (rapidfuzz-exact).
pub fn partial_token_set_ratio(s1: &str, s2: &str, score_cutoff: f64) -> f64 {
    if score_cutoff > 100.0 {
        return 0.0;
    }
    let t1 = sorted_split(s1);
    let t2 = sorted_split(s2);
    if t1.is_empty() || t2.is_empty() {
        return 0.0;
    }

    let decomp = set_decomposition(&t1, &t2);

    // Exit early when there is a common word in both sequences.
    if !decomp.intersection.is_empty() {
        return 100.0;
    }

    partial_ratio(
        &decomp.diff_ab.join(" "),
        &decomp.diff_ba.join(" "),
        score_cutoff,
    )
}

/// WRatio: rapidfuzz's weighted combination — length-ratio-scaled partial
/// ratios, token ratios, and partial token ratios (fuzz_impl.hpp, exact
/// constants and cutoff threading).
pub fn wratio(s1: &str, s2: &str, score_cutoff: f64) -> f64 {
    if score_cutoff > 100.0 {
        return 0.0;
    }

    const UNBASE_SCALE: f64 = 0.95;

    let len1 = s1.chars().count();
    let len2 = s2.chars().count();

    // FuzzyWuzzy compatibility: empty strings score 0.
    if len1 == 0 || len2 == 0 {
        return 0.0;
    }

    let len_ratio = if len1 > len2 {
        len1 as f64 / len2 as f64
    } else {
        len2 as f64 / len1 as f64
    };

    // rapidfuzz threads the *mutated* cutoff forward: each step rescales the
    // running cutoff (already divided by the previous step's scale) by its own
    // scale factor, so a low end_ratio at a high user cutoff escalates the
    // effective cutoff past 100 and suppresses the token/partial-token terms.
    // Reusing the original score_cutoff at every step instead let those terms
    // leak through (WRatio score_cutoff parity).
    let mut cutoff = score_cutoff;
    let mut end_ratio = ratio_with_cutoff(s1, s2, cutoff);

    if len_ratio < 1.5 {
        cutoff = cutoff.max(end_ratio) / UNBASE_SCALE;
        return end_ratio.max(token_ratio(s1, s2, cutoff) * UNBASE_SCALE);
    }

    let partial_scale = if len_ratio <= 8.0 { 0.9 } else { 0.6 };

    cutoff = cutoff.max(end_ratio) / partial_scale;
    end_ratio = end_ratio.max(partial_ratio(s1, s2, cutoff) * partial_scale);

    cutoff = cutoff.max(end_ratio) / UNBASE_SCALE;
    end_ratio.max(partial_token_ratio(s1, s2, cutoff) * UNBASE_SCALE * partial_scale)
}

/// QRatio: rapidfuzz's quick ratio — 0 for empty inputs (FuzzyWuzzy
/// compatibility), otherwise plain `ratio` with cutoff.
pub fn qratio(s1: &str, s2: &str, score_cutoff: f64) -> f64 {
    if s1.is_empty() || s2.is_empty() {
        return 0.0;
    }
    ratio_with_cutoff(s1, s2, score_cutoff)
}

/// Batch ratio: one query vs many candidates, parallelized with Rayon.
/// Uses the shared-query Myers SIMD fast path when the query is ASCII ≤ 64 bytes
/// and all candidates are ASCII — same acceleration as `levenshtein_batch_auto`.
pub fn ratio_batch(query: &str, candidates: &[&str]) -> Vec<f64> {
    // Per-pair ratio over Rayon; each pair takes the ASCII Myers / char-DP
    // fast paths inside indel_distance.
    candidates.par_iter().map(|c| ratio(query, c)).collect()
}

/// Extract top matches with partial-sort optimization.
///
/// Returns `Vec<(match_string, score, original_index)>` sorted by score descending.
pub fn extract(
    query: &str,
    choices: &[&str],
    score_cutoff: f64,
    limit: usize,
) -> Vec<(String, f64, usize)> {
    if choices.is_empty() || limit == 0 {
        return vec![];
    }

    let mut results: Vec<(String, f64, usize)> = if choices.len() > 1000 {
        choices
            .par_iter()
            .enumerate()
            .filter_map(|(i, c)| {
                let score = ratio(query, c);
                if score >= score_cutoff {
                    Some((c.to_string(), score, i))
                } else {
                    None
                }
            })
            .collect()
    } else {
        choices
            .iter()
            .enumerate()
            .filter_map(|(i, c)| {
                let score = ratio(query, c);
                if score >= score_cutoff {
                    Some((c.to_string(), score, i))
                } else {
                    None
                }
            })
            .collect()
    };

    if results.is_empty() {
        return results;
    }

    if limit < results.len() {
        results.select_nth_unstable_by(limit, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit);
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    } else {
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    }

    results
}

/// Extract the single best match above the score cutoff.
///
/// Returns `Some((match_string, score, index))` or `None`.
pub fn extract_one(
    query: &str,
    choices: &[&str],
    score_cutoff: f64,
) -> Option<(String, f64, usize)> {
    if choices.is_empty() {
        return None;
    }

    let mut best: Option<(String, f64, usize)> = None;

    for (i, c) in choices.iter().enumerate() {
        let score = ratio(query, c);
        if score >= score_cutoff {
            let is_better = best.as_ref().is_none_or(|(_, bs, _)| score > *bs);
            if is_better {
                // Update best before the early-exit check so an exact match
                // at index i is always returned rather than an earlier non-exact.
                best = Some((c.to_string(), score, i));
                if score == 100.0 {
                    break;
                }
            }
        }
    }

    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ratio_sorensen_dice() {
        let r = ratio("hello", "hallo");
        assert!((r - 80.0).abs() < 0.01, "Expected 80.0, got {}", r);
    }

    #[test]
    fn test_ratio_unicode() {
        let r = ratio("café", "cafe");
        // "café" (4 chars), "cafe" (4 chars), indel distance = 2
        // (4 + 4 - 2) / (4 + 4) * 100 = 75.0%
        assert!((r - 75.0).abs() < 0.01, "Expected 75.0, got {}", r);
    }

    #[test]
    fn test_ratio_asymmetric() {
        let r = ratio("a", "abc");
        assert!((r - 50.0).abs() < 0.01, "Expected 50.0, got {}", r);
    }

    #[test]
    fn test_partial_ratio() {
        let r = partial_ratio("hello", "oh hello there", 0.0);
        assert!(r >= 100.0 - 0.01, "Expected ~100.0, got {}", r);
    }

    #[test]
    fn test_partial_ratio_empty() {
        // rapidfuzz: both empty -> 100, exactly one empty -> 0.
        assert_eq!(partial_ratio("", "", 0.0), 100.0);
        assert_eq!(partial_ratio("abc", "", 0.0), 0.0);
        assert_eq!(partial_ratio("", "abc", 0.0), 0.0);
    }

    #[test]
    fn test_partial_ratio_prefix_suffix_search() {
        // The suffix "opacg" of the second string wins (rapidfuzz: 40.0).
        let r = partial_ratio("ctgdctopacg", "afqofpvaus", 0.0);
        assert!((r - 40.0).abs() < 1e-9, "Expected 40.0, got {}", r);
    }

    #[test]
    fn test_partial_ratio_alignment_fields() {
        // Rapidfuzz-compatible 5-field shape: score, src_start, src_end,
        // dest_start, dest_end.
        let a = partial_ratio_alignment("hello", "oh hello there", 0.0);
        assert_eq!(a.score, 100.0);
        assert_eq!(a.src_start, 0);
        assert_eq!(a.src_end, 5);
        assert_eq!(a.dest_start, 3);
        assert_eq!(a.dest_end, 8);
    }

    #[test]
    fn test_partial_ratio_alignment_empty() {
        let a = partial_ratio_alignment("", "", 0.0);
        assert_eq!(a.score, 100.0);
        assert_eq!(
            (a.src_start, a.src_end, a.dest_start, a.dest_end),
            (0, 0, 0, 0)
        );
        let a = partial_ratio_alignment("abc", "", 0.0);
        assert_eq!(a.score, 0.0);
        assert_eq!(
            (a.src_start, a.src_end, a.dest_start, a.dest_end),
            (0, 0, 0, 0)
        );
    }

    #[test]
    fn test_token_set_ratio_empty() {
        // rapidfuzz FuzzyWuzzy compatibility: any empty token set -> 0.
        assert_eq!(token_set_ratio("", "", 0.0), 0.0);
        assert_eq!(token_set_ratio("abc", "", 0.0), 0.0);
        assert_eq!(token_set_ratio("", "abc", 0.0), 0.0);
    }

    #[test]
    fn test_token_set_ratio_contained() {
        // One token set contained in the other -> 100.
        assert_eq!(token_set_ratio("hello", "hello world", 0.0), 100.0);
    }

    #[test]
    fn test_qratio_empty() {
        assert_eq!(qratio("", "", 0.0), 0.0);
        assert_eq!(qratio("abc", "", 0.0), 0.0);
        assert_eq!(qratio("", "abc", 0.0), 0.0);
        assert!((qratio("hello", "hallo", 0.0) - 80.0).abs() < 0.01);
    }

    #[test]
    fn test_wratio_empty() {
        assert_eq!(wratio("", "", 0.0), 0.0);
        assert_eq!(wratio("abc", "", 0.0), 0.0);
    }

    #[test]
    fn test_wratio_example() {
        // Verified against rapidfuzz 3.14: ('ctgdctopacg', 'ncb') -> 45.0.
        let r = wratio("ctgdctopacg", "ncb", 0.0);
        assert!((r - 45.0).abs() < 1e-9, "Expected 45.0, got {}", r);
    }

    #[test]
    fn test_partial_token_set_ratio_common_word() {
        // Non-empty intersection -> 100 (rapidfuzz).
        assert_eq!(
            partial_token_set_ratio("hello world", "hello there", 0.0),
            100.0
        );
    }

    #[test]
    fn test_dbg_token_set_components() {
        let s1 = "The quick brown fox jumps over the lazy dog";
        let s2 = "A quick brown fox jumps over the lazy dog!";
        let t1 = sorted_split(s1);
        let t2 = sorted_split(s2);
        let d = set_decomposition(&t1, &t2);
        println!("t1={:?}", t1);
        println!("t2={:?}", t2);
        println!(
            "inter={:?} diff_ab={:?} diff_ba={:?}",
            d.intersection, d.diff_ab, d.diff_ba
        );
        let ab = d.diff_ab.join(" ");
        let ba = d.diff_ba.join(" ");
        println!(
            "ab='{}'({}) ba='{}'({})",
            ab,
            ab.chars().count(),
            ba,
            ba.chars().count()
        );
        println!("sect_len(no sep)={}", intersection_len(&d.intersection));
        println!("indel={}", indel_distance(&ab, &ba));
        let s1b = "new york mets";
        let s2b = "new york yankees";
        let u1 = sorted_split(s1b);
        let u2 = sorted_split(s2b);
        let e = set_decomposition(&u1, &u2);
        println!(
            "mets: inter={:?} diff_ab={:?} diff_ba={:?}",
            e.intersection, e.diff_ab, e.diff_ba
        );
        let mab = e.diff_ab.join(" ");
        let mba = e.diff_ba.join(" ");
        println!(
            "mets: ab='{}'({}) ba='{}'({}) sect_len={} indel={}",
            mab,
            mab.chars().count(),
            mba,
            mba.chars().count(),
            intersection_len(&e.intersection),
            indel_distance(&mab, &mba)
        );
    }

    #[test]
    fn test_extract_one_returns_best() {
        let choices = vec!["apple", "apply", "ape", "banana"];
        let result = extract_one("apple", &choices, 50.0);
        assert!(result.is_some());
        let (m, s, _) = result.unwrap();
        assert_eq!(m, "apple");
        assert!((s - 100.0).abs() < 0.01);
    }

    // ── Comprehensive additional tests ──────────────────────────────────────

    #[test]
    fn test_ratio_identical() {
        assert_eq!(ratio("hello", "hello"), 100.0);
        assert_eq!(ratio("", ""), 100.0);
    }

    #[test]
    fn test_ratio_completely_different() {
        // "abc" (3) vs "xyz" (3): no common chars in indel model.
        // indel_distance = 6 (3 del + 3 ins), total = 6, ratio = 0.0
        let r = ratio("abc", "xyz");
        assert!((0.0..=100.0).contains(&r));
    }

    #[test]
    fn test_ratio_with_cutoff_below_returns_zero() {
        let r = ratio_with_cutoff("hello", "hallo", 90.0);
        assert_eq!(r, 0.0, "Score below cutoff must return 0.0");
    }

    #[test]
    fn test_ratio_with_cutoff_above_returns_score() {
        let r = ratio_with_cutoff("hello", "hallo", 70.0);
        assert!((r - 80.0).abs() < 0.01, "Expected 80.0, got {r}");
    }

    #[test]
    fn test_ratio_symmetry() {
        let pairs = [("hello", "hallo"), ("kitten", "sitting"), ("a", "")];
        for (a, b) in &pairs {
            let ab = ratio(a, b);
            let ba = ratio(b, a);
            assert!(
                (ab - ba).abs() < 1e-12,
                "ratio not symmetric for ({a:?}, {b:?}): {ab} vs {ba}"
            );
        }
    }

    #[test]
    fn test_ratio_batch_matches_serial() {
        let query = "benchmark";
        let candidates = vec!["benchmarks", "bench", "mark", "benchmark", "", "BENCHMARK"];
        let batch = ratio_batch(query, &candidates);
        assert_eq!(batch.len(), candidates.len());
        for (i, c) in candidates.iter().enumerate() {
            let expected = ratio(query, c);
            assert!(
                (batch[i] - expected).abs() < 1e-12,
                "ratio_batch[{i}] ({c:?}) mismatch: {} vs {expected}",
                batch[i]
            );
        }
    }

    #[test]
    fn test_ratio_batch_large() {
        let query = "test_query";
        let candidates: Vec<String> = (0..2000).map(|i| format!("item_{}", i % 40)).collect();
        let refs: Vec<&str> = candidates.iter().map(|s| s.as_str()).collect();
        let batch = ratio_batch(query, &refs);
        assert_eq!(batch.len(), 2000);
        assert!(batch.iter().all(|&r| (0.0..=100.0).contains(&r)));
    }

    #[test]
    fn test_partial_ratio_longer_second() {
        let r = partial_ratio("hello", "oh hello there", 0.0);
        assert!((r - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_partial_ratio_swap_gives_same() {
        let ab = partial_ratio("hello", "oh hello there", 0.0);
        let ba = partial_ratio("oh hello there", "hello", 0.0);
        assert!(
            (ab - ba).abs() < 1e-12,
            "partial_ratio not symmetric: {ab} vs {ba}"
        );
    }

    #[test]
    fn test_partial_ratio_cutoff_filters() {
        assert_eq!(partial_ratio("hello", "hallo", 90.0), 0.0);
    }

    #[test]
    fn test_token_sort_ratio_reordered() {
        assert_eq!(token_sort_ratio("b a c", "a b c", 0.0), 100.0);
        assert_eq!(token_sort_ratio("hello world", "world hello", 0.0), 100.0);
    }

    #[test]
    fn test_token_set_ratio_subset() {
        assert_eq!(token_set_ratio("a b c", "a b c d e", 0.0), 100.0);
        assert_eq!(token_set_ratio("a b c d e", "a b c", 0.0), 100.0);
    }

    #[test]
    fn test_token_ratio_empty_both_is_100() {
        // rapidfuzz: token_ratio("","") = 100 (no FuzzyWuzzy empty guard)
        assert_eq!(token_ratio("", "", 0.0), 100.0);
    }

    #[test]
    fn test_token_ratio_one_empty_is_0() {
        assert_eq!(token_ratio("abc", "", 0.0), 0.0);
        assert_eq!(token_ratio("", "abc", 0.0), 0.0);
    }

    #[test]
    fn test_partial_token_sort_ratio_substring_sort() {
        let r = partial_token_sort_ratio("york new", "new york mets", 0.0);
        assert!((r - 100.0).abs() < 0.01, "Expected ~100.0, got {r}");
    }

    #[test]
    fn test_wratio_identical_is_100() {
        assert_eq!(wratio("hello", "hello", 0.0), 100.0);
    }

    #[test]
    fn test_wratio_cutoff_above_100_returns_zero() {
        assert_eq!(wratio("hello", "hello", 101.0), 0.0);
    }

    #[test]
    fn test_wratio_large_length_ratio() {
        let long = "a".repeat(100);
        let short = "a".repeat(5);
        let r = wratio(&short, &long, 0.0);
        assert!((0.0..=100.0).contains(&r));
    }

    #[test]
    fn test_qratio_identical_is_100() {
        assert_eq!(qratio("hello", "hello", 0.0), 100.0);
    }

    #[test]
    fn test_extract_empty_choices() {
        assert_eq!(extract("q", &[], 0.0, 10), vec![]);
    }

    #[test]
    fn test_extract_limit_zero() {
        assert_eq!(extract("q", &["a", "b"], 0.0, 0), vec![]);
    }

    #[test]
    fn test_extract_cutoff_above_100() {
        assert_eq!(extract("apple", &["apple"], 101.0, 5), vec![]);
    }

    #[test]
    fn test_extract_sorted_descending() {
        let choices = vec!["apple", "apply", "ape", "banana"];
        let results = extract("apple", &choices, 0.0, 10);
        assert!(!results.is_empty());
        for w in results.windows(2) {
            assert!(w[0].1 >= w[1].1, "Not sorted: {} < {}", w[0].1, w[1].1);
        }
    }

    #[test]
    fn test_extract_limit_truncates() {
        let choices: Vec<&str> = (0..50).map(|_| "apple").collect();
        let results = extract("apple", &choices, 0.0, 5);
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_extract_preserves_original_index() {
        let choices = vec!["banana", "apple", "cherry"];
        let results = extract("apple", &choices, 90.0, 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].2, 1, "Index must be 1 (position of 'apple')");
    }

    #[test]
    fn test_extract_one_below_cutoff_is_none() {
        assert!(extract_one("apple", &["zzz"], 90.0).is_none());
    }

    #[test]
    fn test_extract_one_exact_match_wins() {
        let choices = vec!["appel", "apply", "apple"];
        let result = extract_one("apple", &choices, 0.0).unwrap();
        assert_eq!(result.0, "apple");
        assert_eq!(result.1, 100.0);
    }

    #[test]
    fn test_extract_one_correct_index() {
        let choices = vec!["banana", "cherry", "apple"];
        let result = extract_one("apple", &choices, 0.0).unwrap();
        assert_eq!(choices[result.2], "apple");
    }

    #[test]
    fn test_indel_distance_ascii() {
        // substitution cost = 2 in the indel model
        assert_eq!(indel_distance("hello", "hallo"), 2);
        assert_eq!(indel_distance("abc", "abc"), 0);
        assert_eq!(indel_distance("", "abc"), 3);
        assert_eq!(indel_distance("abc", ""), 3);
    }

    #[test]
    fn test_indel_distance_unicode() {
        let d = indel_distance("café", "cafe");
        assert!(d > 0, "café vs cafe must differ");
    }

    #[test]
    fn test_ratio_unicode_large() {
        let a = "日本語テスト".repeat(5);
        assert_eq!(ratio(&a, &a), 100.0);
        assert!((0.0..=100.0).contains(&ratio(&a, "hello")));
    }

    #[test]
    fn test_partial_token_set_ratio_empty_intersection() {
        let r = partial_token_set_ratio("abc def", "xyz uvw", 0.0);
        assert!((0.0..=100.0).contains(&r));
    }

    #[test]
    fn test_wratio_len_ratio_border_1_5() {
        // len_ratio < 1.5 → token_ratio branch
        let r = wratio("new york mets", "new york", 0.0);
        assert!((0.0..=100.0).contains(&r));
    }

    #[test]
    fn test_wratio_cutoff_threads_rescaled_cutoff() {
        // rapidfuzz WRatio rescales the running cutoff at each step, so a high
        // user cutoff suppresses the token terms even when a perfect substring
        // exists. Regression: fuzzgpu returned 57.0/85.5 where rapidfuzz
        // returns 0.0 for these inputs.
        assert_eq!(wratio("new york yankees", "a", 90.0), 0.0);
        assert_eq!(wratio("a", "ab", 95.0), 0.0);
        assert_eq!(wratio("hello world", "world", 95.0), 0.0);
        // The cutoff=0 path is unaffected: the unscaled partial ratio still wins.
        assert!((wratio("new york yankees", "a", 0.0) - 60.0).abs() < 1e-9);
    }
}
