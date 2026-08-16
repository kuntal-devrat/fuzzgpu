use rayon::prelude::*;

/// True unrestricted Damerau-Levenshtein distance (Lowrance & Wagner 1975).
/// Computes edit distance allowing insertions, deletions, substitutions, and transpositions of any characters
/// (including non-adjacent transpositions where characters were inserted/deleted in between).
pub fn damerau_levenshtein_distance(a: &str, b: &str) -> u32 {
    let a = a.as_bytes();
    let b = b.as_bytes();
    let (m, n) = (a.len(), b.len());

    if m == 0 { return n as u32; }
    if n == 0 { return m as u32; }
    if a == b { return 0; }

    let max_dist = (m + n) as u32;

    // 2D distance matrix of size (m + 2) x (n + 2)
    let rows = m + 2;
    let cols = n + 2;
    let mut h = vec![0u32; rows * cols];

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

    // da[c] stores the last row where character c appeared in a
    let mut da = [0usize; 256];

    for i in 1..=m {
        let mut db = 0usize; // last column where character appeared in b
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

/// Batch Damerau-Levenshtein: one query vs many candidates.
pub fn damerau_levenshtein_batch(query: &str, candidates: &[&str]) -> Vec<u32> {
    candidates.par_iter().map(|c| damerau_levenshtein_distance(query, c)).collect()
}

/// Cross-product matrix for Damerau-Levenshtein.
pub fn damerau_levenshtein_cdist(list_a: &[&str], list_b: &[&str]) -> Vec<Vec<u32>> {
    list_a.par_iter().map(|a| {
        list_b.iter().map(|b| damerau_levenshtein_distance(a, b)).collect()
    }).collect()
}

/// Damerau-Levenshtein normalized ratio (0.0 to 100.0) based on Sørensen-Dice formula.
pub fn damerau_ratio(s1: &str, s2: &str) -> f64 {
    let (len_a, len_b) = (s1.len(), s2.len());
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
        // "ab" to "ba" is 1 transposition in Damerau-Levenshtein, but 2 edits in Levenshtein
        assert_eq!(damerau_levenshtein_distance("ab", "ba"), 1);
        // "ca" to "abc" is 2 in true Damerau-Levenshtein (transposition of 'c' and 'a', plus insertion of 'b')
        assert_eq!(damerau_levenshtein_distance("ca", "abc"), 2);
        assert_eq!(damerau_levenshtein_distance("kitten", "sitting"), 3);
    }

    #[test]
    fn test_damerau_identical_and_empty() {
        assert_eq!(damerau_levenshtein_distance("", ""), 0);
        assert_eq!(damerau_levenshtein_distance("hello", "hello"), 0);
        assert_eq!(damerau_levenshtein_distance("hello", ""), 5);
        assert_eq!(damerau_levenshtein_distance("", "world"), 5);
    }
}
