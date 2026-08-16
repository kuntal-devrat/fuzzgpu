use rayon::prelude::*;
use std::collections::HashMap;

/// True unrestricted Damerau-Levenshtein distance (Lowrance & Wagner 1975).
/// Computes edit distance allowing insertions, deletions, substitutions, and transpositions of any characters
/// (including non-adjacent transpositions where characters were inserted/deleted in between).
/// Supports both ASCII (fast array path) and full Unicode characters.
pub fn damerau_levenshtein_distance(a: &str, b: &str) -> u32 {
    if a.is_ascii() && b.is_ascii() {
        damerau_bytes(a.as_bytes(), b.as_bytes())
    } else {
        let a_chars: Vec<char> = a.chars().collect();
        let b_chars: Vec<char> = b.chars().collect();
        damerau_chars(&a_chars, &b_chars)
    }
}

fn damerau_bytes(a: &[u8], b: &[u8]) -> u32 {
    let (m, n) = (a.len(), b.len());
    if m == 0 { return n as u32; }
    if n == 0 { return m as u32; }
    if a == b { return 0; }

    let max_dist = (m + n) as u32;
    let cols = n + 2;
    let mut h = vec![0u32; (m + 2) * cols];

    let idx = |i: isize, j: isize| -> usize {
        ((i + 1) as usize) * cols + ((j + 1) as usize)
    };

    h[idx(-1, -1)] = max_dist;
    for i in 0..=m {
        h[idx(i as isize, -1)] = max_dist;
        h[idx(i as isize, 0)] = i as u32;
    }
    for j in 0..=n {
        h[idx(-1, j as isize)] = max_dist;
        h[idx(0, j as isize)] = j as u32;
    }

    let mut da = [0usize; 256];

    for i in 1..=m {
        let mut db = 0usize;
        let ai = a[i - 1];

        for j in 1..=n {
            let bj = b[j - 1];
            let k = da[bj as usize];
            let l = db;

            let cost = if ai == bj {
                db = j;
                0u32
            } else {
                1u32
            };

            let sub = h[idx((i - 1) as isize, (j - 1) as isize)] + cost;
            let ins = h[idx(i as isize, (j - 1) as isize)] + 1;
            let del = h[idx((i - 1) as isize, j as isize)] + 1;

            let trans = if k > 0 && l > 0 {
                h[idx((k - 1) as isize, (l - 1) as isize)] + ((i - k - 1) as u32) + 1 + ((j - l - 1) as u32)
            } else {
                max_dist
            };

            h[idx(i as isize, j as isize)] = sub.min(ins).min(del).min(trans);
        }

        da[ai as usize] = i;
    }

    h[idx(m as isize, n as isize)]
}

fn damerau_chars(a: &[char], b: &[char]) -> u32 {
    let (m, n) = (a.len(), b.len());
    if m == 0 { return n as u32; }
    if n == 0 { return m as u32; }
    if a == b { return 0; }

    let max_dist = (m + n) as u32;
    let cols = n + 2;
    let mut h = vec![0u32; (m + 2) * cols];

    let idx = |i: isize, j: isize| -> usize {
        ((i + 1) as usize) * cols + ((j + 1) as usize)
    };

    h[idx(-1, -1)] = max_dist;
    for i in 0..=m {
        h[idx(i as isize, -1)] = max_dist;
        h[idx(i as isize, 0)] = i as u32;
    }
    for j in 0..=n {
        h[idx(-1, j as isize)] = max_dist;
        h[idx(0, j as isize)] = j as u32;
    }

    let mut da: HashMap<char, usize> = HashMap::with_capacity(a.len());

    for i in 1..=m {
        let mut db = 0usize;
        let ai = a[i - 1];

        for j in 1..=n {
            let bj = b[j - 1];
            let k = da.get(&bj).copied().unwrap_or(0);
            let l = db;

            let cost = if ai == bj {
                db = j;
                0u32
            } else {
                1u32
            };

            let sub = h[idx((i - 1) as isize, (j - 1) as isize)] + cost;
            let ins = h[idx(i as isize, (j - 1) as isize)] + 1;
            let del = h[idx((i - 1) as isize, j as isize)] + 1;

            let trans = if k > 0 && l > 0 {
                h[idx((k - 1) as isize, (l - 1) as isize)] + ((i - k - 1) as u32) + 1 + ((j - l - 1) as u32)
            } else {
                max_dist
            };

            h[idx(i as isize, j as isize)] = sub.min(ins).min(del).min(trans);
        }

        da.insert(ai, i);
    }

    h[idx(m as isize, n as isize)]
}

/// Batch Damerau-Levenshtein: one query vs many candidates.
pub fn damerau_levenshtein_batch(query: &str, candidates: &[&str]) -> Vec<u32> {
    candidates.par_iter().map(|c| damerau_levenshtein_distance(query, c)).collect()
}

/// Cross-product matrix for Damerau-Levenshtein.
pub fn damerau_levenshtein_cdist(list_a: &[&str], list_b: &[&str]) -> Vec<Vec<u32>> {
    if list_a.is_empty() || list_b.is_empty() {
        return vec![];
    }
    list_a.par_iter().map(|a| {
        list_b.iter().map(|b| damerau_levenshtein_distance(a, b)).collect()
    }).collect()
}

/// Damerau-Levenshtein normalized ratio (0.0 to 100.0) based on standard edit distance similarity formula:
/// `((total_len - dist) / total_len) * 100.0`.
pub fn damerau_ratio(s1: &str, s2: &str) -> f64 {
    let len_a = if s1.is_ascii() { s1.len() } else { s1.chars().count() };
    let len_b = if s2.is_ascii() { s2.len() } else { s2.chars().count() };
    let total = len_a + len_b;
    if total == 0 { return 100.0; }
    let dist = damerau_levenshtein_distance(s1, s2) as f64;
    ((total as f64 - dist) / total as f64) * 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_damerau_transposition() {
        assert_eq!(damerau_levenshtein_distance("ab", "ba"), 1);
        assert_eq!(damerau_levenshtein_distance("ca", "abc"), 2);
        assert_eq!(damerau_levenshtein_distance("kitten", "sitting"), 3);
    }

    #[test]
    fn test_damerau_unicode() {
        assert_eq!(damerau_levenshtein_distance("café", "cafe"), 1);
        assert_eq!(damerau_levenshtein_distance("naïve", "naive"), 1);
        assert_eq!(damerau_levenshtein_distance("🚀", ""), 1);
        assert_eq!(damerau_levenshtein_distance("中文", "中问"), 1);
    }

    #[test]
    fn test_damerau_identical_and_empty() {
        assert_eq!(damerau_levenshtein_distance("", ""), 0);
        assert_eq!(damerau_levenshtein_distance("hello", "hello"), 0);
        assert_eq!(damerau_levenshtein_distance("hello", ""), 5);
        assert_eq!(damerau_levenshtein_distance("", "world"), 5);
    }
}
