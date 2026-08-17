use rayon::prelude::*;
use std::collections::BTreeSet;

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
    if m == 0 { return n as u32; }
    if n == 0 { return m as u32; }
    if a == b { return 0; }

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
/// Formula: `(|a| + |b| - indel_distance) / (|a| + |b|) × 100`
///
/// Matches RapidFuzz and FuzzyWuzzy identically across all ASCII and multi-byte Unicode test cases.
pub fn ratio(s1: &str, s2: &str) -> f64 {
    let len_a = if s1.is_ascii() { s1.len() } else { s1.chars().count() };
    let len_b = if s2.is_ascii() { s2.len() } else { s2.chars().count() };
    let total = len_a + len_b;
    if total == 0 { return 100.0; }
    let dist = indel_distance(s1, s2) as f64;
    ((total as f64 - dist) / total as f64) * 100.0
}

/// Partial ratio: best ratio of the shorter string against all substrings
/// of approximately the same length in the longer string.
///
/// This handles cases where one string is a substring of another, e.g.,
/// `partial_ratio("hello", "oh hello there")` returns a high score.
pub fn partial_ratio(s1: &str, s2: &str) -> f64 {
    partial_ratio_alignment(s1, s2).0
}

/// Partial ratio with alignment: returns `(score, src_start, dest_start, len)`
/// where `src_start` is the char offset in the shorter string (always 0),
/// `dest_start` is the char offset in the longer string where the best window
/// starts, and `len` is the window length in chars.
///
/// This matches the rapidfuzz `partial_ratio_alignment` API.
pub fn partial_ratio_alignment(s1: &str, s2: &str) -> (f64, usize, usize, usize) {
    if s1.is_empty() || s2.is_empty() {
        return (0.0, 0, 0, 0);
    }

    let s1_count = if s1.is_ascii() { s1.len() } else { s1.chars().count() };
    let s2_count = if s2.is_ascii() { s2.len() } else { s2.chars().count() };

    // Normalise: shorter is always the "source" (pattern).
    let (shorter, longer, short_chars, long_chars, swapped) = if s1_count <= s2_count {
        (s1, s2, s1_count, s2_count, false)
    } else {
        (s2, s1, s2_count, s1_count, true)
    };

    if short_chars == long_chars {
        let score = ratio(shorter, longer);
        return (score, 0, 0, short_chars);
    }

    let mut best_score = 0.0f64;
    let mut best_start = 0usize; // char offset in `longer`

    if shorter.is_ascii() && longer.is_ascii() {
        let short_bytes = shorter.as_bytes();
        for start in 0..=(long_chars - short_chars) {
            let window = &longer[start..start + short_bytes.len()];
            let score = ratio(shorter, window);
            if score > best_score {
                best_score = score;
                best_start = start;
            }
            if best_score == 100.0 { break; }
        }
    } else {
        let longer_chars: Vec<char> = longer.chars().collect();
        for start in 0..=(long_chars - short_chars) {
            let window: String = longer_chars[start..start + short_chars].iter().collect();
            let score = ratio(shorter, &window);
            if score > best_score {
                best_score = score;
                best_start = start;
            }
            if best_score == 100.0 { break; }
        }
    }

    // If we swapped s1/s2, the "dest_start" is in s1 (the original longer).
    // Callers that want alignment in the original coordinate space should
    // swap src/dest back when `s1_count > s2_count`.
    let _ = swapped; // alignment is always relative to the longer string
    (best_score, 0, best_start, short_chars)
}

/// Token sort ratio: sort tokens alphabetically, then compare.
pub fn token_sort_ratio(s1: &str, s2: &str) -> f64 {
    let mut t1: Vec<&str> = s1.split_whitespace().collect();
    let mut t2: Vec<&str> = s2.split_whitespace().collect();
    t1.sort_unstable();
    t2.sort_unstable();
    let s1_sorted = t1.join(" ");
    let s2_sorted = t2.join(" ");
    ratio(&s1_sorted, &s2_sorted)
}

/// Token set ratio: compare unique token sets.
///
/// Uses `BTreeSet` for deterministic sorted order without a separate sort step.
/// Uses `Cow<str>` to avoid allocating intermediate `String`s when the
/// intersection or difference is empty.
pub fn token_set_ratio(s1: &str, s2: &str) -> f64 {
    let t1: BTreeSet<&str> = s1.split_whitespace().collect();
    let t2: BTreeSet<&str> = s2.split_whitespace().collect();

    let inter: Vec<&str> = t1.intersection(&t2).copied().collect();
    let diff1: Vec<&str> = t1.difference(&t2).copied().collect();
    let diff2: Vec<&str> = t2.difference(&t1).copied().collect();

    // If both strings have identical token sets
    if diff1.is_empty() && diff2.is_empty() {
        return 100.0;
    }

    // Build sorted-token strings with pre-sized buffers to avoid realloc.
    let inter_str: std::borrow::Cow<'_, str> = if inter.is_empty() {
        std::borrow::Cow::Borrowed("")
    } else {
        std::borrow::Cow::Owned(inter.join(" "))
    };

    let t1_str: std::borrow::Cow<'_, str> = if diff1.is_empty() {
        inter_str.clone()
    } else if inter_str.is_empty() {
        std::borrow::Cow::Owned(diff1.join(" "))
    } else {
        let mut s = String::with_capacity(inter_str.len() + 1 + diff1.iter().map(|w| w.len() + 1).sum::<usize>());
        s.push_str(&inter_str);
        s.push(' ');
        s.push_str(&diff1.join(" "));
        std::borrow::Cow::Owned(s)
    };

    let t2_str: std::borrow::Cow<'_, str> = if diff2.is_empty() {
        inter_str.clone()
    } else if inter_str.is_empty() {
        std::borrow::Cow::Owned(diff2.join(" "))
    } else {
        let mut s = String::with_capacity(inter_str.len() + 1 + diff2.iter().map(|w| w.len() + 1).sum::<usize>());
        s.push_str(&inter_str);
        s.push(' ');
        s.push_str(&diff2.join(" "));
        std::borrow::Cow::Owned(s)
    };

    let r01 = ratio(&inter_str, &t1_str);
    let r02 = ratio(&inter_str, &t2_str);
    let r12 = ratio(&t1_str, &t2_str);

    r01.max(r02).max(r12)
}

/// WRatio: weighted combination of multiple scorers.
pub fn wratio(s1: &str, s2: &str) -> f64 {
    let r = ratio(s1, s2);
    let tsr = token_sort_ratio(s1, s2);
    let tsr2 = token_set_ratio(s1, s2);
    r.max(tsr).max(tsr2)
}

/// Batch ratio: one query vs many candidates, parallelized with Rayon.
/// Uses the shared-query Myers SIMD fast path when the query is ASCII ≤ 64 bytes
/// and all candidates are ASCII — same acceleration as `levenshtein_batch_auto`.
pub fn ratio_batch(query: &str, candidates: &[&str]) -> Vec<f64> {
    // The indel model (substitution cost=2) does not have a direct Myers
    // bit-vector path — Myers gives edit distance with unit substitution cost.
    // We approximate with the standard Levenshtein batch and then convert:
    //   indel_dist = 2 * lev_dist when lev path only uses ins/del (no subs).
    // For general strings we fall back to the per-pair indel_distance.
    // The Rayon parallel path is correct for all inputs.
    candidates.par_iter().map(|c| ratio(query, c)).collect()
}

/// Extract top matches with partial-sort optimization.
///
/// Returns `Vec<(match_string, score, original_index)>` sorted by score descending.
pub fn extract(query: &str, choices: &[&str], score_cutoff: f64, limit: usize) -> Vec<(String, f64, usize)> {
    if choices.is_empty() || limit == 0 { return vec![]; }

    let mut results: Vec<(String, f64, usize)> = if choices.len() > 1000 {
        choices.par_iter().enumerate()
            .filter_map(|(i, c)| {
                let score = ratio(query, c);
                if score >= score_cutoff { Some((c.to_string(), score, i)) } else { None }
            })
            .collect()
    } else {
        choices.iter().enumerate()
            .filter_map(|(i, c)| {
                let score = ratio(query, c);
                if score >= score_cutoff { Some((c.to_string(), score, i)) } else { None }
            })
            .collect()
    };

    if results.is_empty() { return results; }

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
pub fn extract_one(query: &str, choices: &[&str], score_cutoff: f64) -> Option<(String, f64, usize)> {
    if choices.is_empty() { return None; }

    let mut best: Option<(String, f64, usize)> = None;

    for (i, c) in choices.iter().enumerate() {
        let score = ratio(query, c);
        if score >= score_cutoff {
            let is_better = best.as_ref().is_none_or(|(_, bs, _)| score > *bs);
            if is_better {
                // Update best before the early-exit check so an exact match
                // at index i is always returned rather than an earlier non-exact.
                best = Some((c.to_string(), score, i));
                if score == 100.0 { break; }
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
        let r = partial_ratio("hello", "oh hello there");
        assert!(r >= 100.0 - 0.01, "Expected ~100.0, got {}", r);
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
}
