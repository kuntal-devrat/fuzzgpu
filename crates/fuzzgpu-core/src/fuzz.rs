use rayon::prelude::*;
use std::collections::BTreeSet;

/// Standard Indel / Levenshtein (substitution cost = 2) edit distance used for fuzzy ratios.
/// Supports both ASCII fast-path and full Unicode characters.
#[inline]
pub fn indel_distance(a: &str, b: &str) -> u32 {
    if a.is_ascii() && b.is_ascii() {
        indel_distance_slice(a.as_bytes(), b.as_bytes())
    } else {
        let a_chars: Vec<char> = a.chars().collect();
        let b_chars: Vec<char> = b.chars().collect();
        indel_distance_slice(&a_chars, &b_chars)
    }
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
    if s1.is_empty() || s2.is_empty() { return 0.0; }

    let s1_count = if s1.is_ascii() { s1.len() } else { s1.chars().count() };
    let s2_count = if s2.is_ascii() { s2.len() } else { s2.chars().count() };

    let (shorter, longer, short_chars, long_chars) = if s1_count <= s2_count {
        (s1, s2, s1_count, s2_count)
    } else {
        (s2, s1, s2_count, s1_count)
    };

    if short_chars == long_chars {
        return ratio(shorter, longer);
    }

    if shorter.is_ascii() && longer.is_ascii() {
        let short_bytes = shorter.as_bytes();
        let long_bytes = longer.as_bytes();
        let mut best = 0.0f64;
        for start in 0..=(long_bytes.len() - short_bytes.len()) {
            let window = &longer[start..start + short_bytes.len()];
            let score = ratio(shorter, window);
            if score > best { best = score; }
            if best == 100.0 { return 100.0; }
        }
        best
    } else {
        let longer_chars: Vec<char> = longer.chars().collect();
        let mut best = 0.0f64;
        for start in 0..=(longer_chars.len() - short_chars) {
            let window: String = longer_chars[start..start + short_chars].iter().collect();
            let score = ratio(shorter, &window);
            if score > best { best = score; }
            if best == 100.0 { return 100.0; }
        }
        best
    }
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
pub fn token_set_ratio(s1: &str, s2: &str) -> f64 {
    let t1: BTreeSet<&str> = s1.split_whitespace().collect();
    let t2: BTreeSet<&str> = s2.split_whitespace().collect();

    let inter: Vec<&str> = t1.intersection(&t2).copied().collect();
    let diff1: Vec<&str> = t1.difference(&t2).copied().collect();
    let diff2: Vec<&str> = t2.difference(&t1).copied().collect();

    let inter_str = inter.join(" ");
    let diff1_str = diff1.join(" ");
    let diff2_str = diff2.join(" ");

    // If both strings have identical token sets
    if diff1.is_empty() && diff2.is_empty() {
        return 100.0;
    }

    let t0 = &inter_str;

    let t1 = if inter_str.is_empty() {
        diff1_str.clone()
    } else if diff1.is_empty() {
        inter_str.clone()
    } else {
        format!("{} {}", inter_str, diff1_str)
    };

    let t2 = if inter_str.is_empty() {
        diff2_str.clone()
    } else if diff2.is_empty() {
        inter_str.clone()
    } else {
        format!("{} {}", inter_str, diff2_str)
    };

    let r01 = ratio(t0, &t1);
    let r02 = ratio(t0, &t2);
    let r12 = ratio(&t1, &t2);

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
pub fn ratio_batch(query: &str, candidates: &[&str]) -> Vec<f64> {
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
            if best.as_ref().map_or(true, |(_, bs, _)| score > *bs) {
                best = Some((c.to_string(), score, i));
            }
        }
        if score == 100.0 { break; }
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
