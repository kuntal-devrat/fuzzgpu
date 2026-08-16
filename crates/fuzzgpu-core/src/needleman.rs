use rayon::prelude::*;

/// Needleman-Wunsch global alignment score with linear gap penalty.
///
/// Uses single-row DP + scalar diagonal for minimal memory.
/// Supports both ASCII fast-path and full Unicode characters.
pub fn needleman_wunsch(a: &str, b: &str, match_score: i32, mismatch_score: i32, gap_penalty: i32) -> i32 {
    if a.is_ascii() && b.is_ascii() {
        needleman_wunsch_bytes(a.as_bytes(), b.as_bytes(), match_score, mismatch_score, gap_penalty)
    } else {
        let a_chars: Vec<char> = a.chars().collect();
        let b_chars: Vec<char> = b.chars().collect();
        needleman_wunsch_chars(&a_chars, &b_chars, match_score, mismatch_score, gap_penalty)
    }
}

fn needleman_wunsch_bytes(a: &[u8], b: &[u8], match_score: i32, mismatch_score: i32, gap_penalty: i32) -> i32 {
    needleman_wunsch_slice(a, b, match_score, mismatch_score, gap_penalty)
}

fn needleman_wunsch_chars(a: &[char], b: &[char], match_score: i32, mismatch_score: i32, gap_penalty: i32) -> i32 {
    needleman_wunsch_slice(a, b, match_score, mismatch_score, gap_penalty)
}

fn needleman_wunsch_slice<T: PartialEq>(a: &[T], b: &[T], match_score: i32, mismatch_score: i32, gap_penalty: i32) -> i32 {
    let (m, n) = (a.len(), b.len());

    if m == 0 { return (n as i32) * gap_penalty; }
    if n == 0 { return (m as i32) * gap_penalty; }
    if a == b { return (m as i32) * match_score; }

    let mut row = vec![0i32; n + 1];
    for (j, item) in row.iter_mut().enumerate() {
        *item = (j as i32) * gap_penalty;
    }

    for i in 1..=m {
        let mut prev_diag = row[0];
        row[0] = (i as i32) * gap_penalty;
        let ai = &a[i - 1];
        for j in 1..=n {
            let old = row[j];
            let score = if ai == &b[j - 1] { match_score } else { mismatch_score };
            row[j] = (prev_diag + score)
                .max(row[j] + gap_penalty)
                .max(row[j - 1] + gap_penalty);
            prev_diag = old;
        }
    }
    row[n]
}

/// Batch Needleman-Wunsch with linear gap penalty.
pub fn needleman_wunsch_batch(
    query: &str, candidates: &[&str],
    match_score: i32, mismatch_score: i32, gap_penalty: i32,
) -> Vec<i32> {
    candidates.par_iter().map(|c| {
        needleman_wunsch(query, c, match_score, mismatch_score, gap_penalty)
    }).collect()
}

const NEG_INF: i32 = -1_000_000_000;

/// Needleman-Wunsch global alignment score with affine gap penalties (Gotoh 1982 algorithm).
///
/// Affine model: gap of length k costs `gap_open + k * gap_extend`.
pub fn needleman_wunsch_affine(
    a: &str,
    b: &str,
    match_score: i32,
    mismatch_score: i32,
    gap_open: i32,
    gap_extend: i32,
) -> i32 {
    if a.is_ascii() && b.is_ascii() {
        needleman_wunsch_affine_slice(a.as_bytes(), b.as_bytes(), match_score, mismatch_score, gap_open, gap_extend)
    } else {
        let a_chars: Vec<char> = a.chars().collect();
        let b_chars: Vec<char> = b.chars().collect();
        needleman_wunsch_affine_slice(&a_chars, &b_chars, match_score, mismatch_score, gap_open, gap_extend)
    }
}

fn needleman_wunsch_affine_slice<T: PartialEq>(
    a: &[T],
    b: &[T],
    match_score: i32,
    mismatch_score: i32,
    gap_open: i32,
    gap_extend: i32,
) -> i32 {
    let (m, n) = (a.len(), b.len());

    if m == 0 && n == 0 { return 0; }
    if m == 0 { return gap_open + (n as i32) * gap_extend; }
    if n == 0 { return gap_open + (m as i32) * gap_extend; }
    if a == b { return (m as i32) * match_score; }

    let mut m_row = vec![NEG_INF; n + 1];
    let mut ix_row = vec![NEG_INF; n + 1];
    let mut iy_row = vec![NEG_INF; n + 1];

    m_row[0] = 0;
    for j in 1..=n {
        let gap_cost = gap_open + (j as i32) * gap_extend;
        iy_row[j] = gap_cost;
        m_row[j] = gap_cost;
    }

    for i in 1..=m {
        let mut prev_m_diag = m_row[0];
        let mut prev_ix_diag = ix_row[0];
        let mut prev_iy_diag = iy_row[0];

        let gap_cost_i = gap_open + (i as i32) * gap_extend;
        ix_row[0] = gap_cost_i;
        m_row[0] = gap_cost_i;
        iy_row[0] = NEG_INF;

        let ai = &a[i - 1];

        for j in 1..=n {
            let bj = &b[j - 1];
            let sub_score = if ai == bj { match_score } else { mismatch_score };

            let prev_diag_best = prev_m_diag.max(prev_ix_diag).max(prev_iy_diag);
            let new_m = prev_diag_best + sub_score;

            let new_ix = (ix_row[j] + gap_extend).max(m_row[j] + gap_open + gap_extend).max(iy_row[j] + gap_open + gap_extend);
            let new_iy = (iy_row[j - 1] + gap_extend).max(m_row[j - 1] + gap_open + gap_extend).max(ix_row[j - 1] + gap_open + gap_extend);

            prev_m_diag = m_row[j];
            prev_ix_diag = ix_row[j];
            prev_iy_diag = iy_row[j];

            m_row[j] = new_m;
            ix_row[j] = new_ix;
            iy_row[j] = new_iy;
        }
    }

    m_row[n].max(ix_row[n]).max(iy_row[n])
}

/// Batch Needleman-Wunsch with affine gap penalty.
pub fn needleman_wunsch_affine_batch(
    query: &str,
    candidates: &[&str],
    match_score: i32,
    mismatch_score: i32,
    gap_open: i32,
    gap_extend: i32,
) -> Vec<i32> {
    candidates.par_iter().map(|c| {
        needleman_wunsch_affine(query, c, match_score, mismatch_score, gap_open, gap_extend)
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_affine_gap() {
        let s1 = "AGCT";
        let s2 = "AGCT";
        assert_eq!(needleman_wunsch_affine(s1, s2, 2, -1, -3, -1), 8);

        let score = needleman_wunsch_affine("ACGT", "AT", 2, -1, -3, -1);
        assert!(score < 8);
    }
}
