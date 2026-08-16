/// True Myers 1999 bit-vector Levenshtein distance.
///
/// For pattern length ≤ 64: O(n) time using ~15 bitwise operations per text character
/// with **zero inner loop**. This is the theoretically optimal bit-parallel algorithm.
///
/// For pattern length > 64: falls back to optimized single-row DP.
///
/// Reference: Gene Myers, "A Fast Bit-Vector Algorithm for Approximate String
/// Matching Based on Dynamic Programming", JACM 1999.
pub fn levenshtein_myers(a: &[u8], b: &[u8]) -> u32 {
    let (m, n) = (a.len(), b.len());
    if m == 0 { return n as u32; }
    if n == 0 { return m as u32; }

    // Ensure `a` is the shorter string (pattern) for the bitmask.
    // Myers algorithm is O(n × ⌈m/64⌉), so shorter pattern = faster.
    if m > n {
        return levenshtein_myers(b, a);
    }

    // Pattern must fit in a single u64 bitmask (≤ 64 chars).
    if m > 64 {
        return levenshtein_dp_optimized(a, b);
    }

    // Build character bitmasks: Peq[c] has bit j set iff pattern[j] == c.
    let mut peq = [0u64; 256];
    for (j, &ch) in a.iter().enumerate() {
        peq[ch as usize] |= 1u64 << j;
    }

    // Initialize bit-vectors.
    // Pv = all 1s in the pattern-length mask (positive vertical delta = +1 everywhere)
    // Mv = all 0s (negative vertical delta = 0 everywhere)
    let mask = if m == 64 { u64::MAX } else { (1u64 << m) - 1 };
    let mut pv: u64 = mask;  // positive vertical
    let mut mv: u64 = 0u64;  // negative vertical
    let mut score = m as u32;

    // Process each character of text `b` with zero inner loop.
    // Each iteration does ~15 bitwise ops regardless of pattern length.
    //
    // Exact formulation from Myers 1999, Section 4:
    //   Eq  = Peq[b_j]              (match vector)
    //   Xv  = Eq | Mv               (candidates for vertical change)
    //   Eq  = Eq | Mh (from prev)   — not needed for first formulation
    //   Xh  = (((Eq & Pv) + Pv) ^ Pv) | Eq
    //   Ph  = Mv | ~(Xh | Pv)
    //   Mh  = Pv & Xh
    //   — update score from last bit of Ph, Mh —
    //   Ph <<= 1; Ph |= 1           (shift + boundary)
    //   Mh <<= 1
    //   Pv  = (Mh | ~(Xv | Ph)) & mask
    //   Mv  = (Ph & Xv) & mask
    for &ch in b {
        let eq = peq[ch as usize];

        // Core Myers recurrence:
        let xv = eq | mv;
        let eq_and_pv = eq & pv;
        let xh = (eq_and_pv.wrapping_add(pv) ^ pv) | eq;

        let ph = mv | !(xh | pv);
        let mh = pv & xh;

        // Update score: check the m-th bit (last row) of Ph and Mh.
        let last_bit = 1u64 << (m - 1);
        if ph & last_bit != 0 {
            score += 1;
        }
        if mh & last_bit != 0 {
            score -= 1;
        }

        // Shift Ph and Mh left, feed into Pv and Mv for next column.
        let ph_shifted = (ph << 1) | 1;
        let mh_shifted = mh << 1;

        pv = (mh_shifted | !(xv | ph_shifted)) & mask;
        mv = (ph_shifted & xv) & mask;
    }

    score
}

/// Optimized single-row DP for strings > 64 bytes.
fn levenshtein_dp_optimized(a: &[u8], b: &[u8]) -> u32 {
    let (m, n) = (a.len(), b.len());
    let mut row = vec![0u32; n + 1];
    for (j, item) in row.iter_mut().enumerate() {
        *item = j as u32;
    }
    for i in 1..=m {
        let mut prev_diag = row[0];
        row[0] = i as u32;
        let ai = a[i - 1];
        for j in 1..=n {
            let old = row[j];
            let cost = if ai == b[j - 1] { 0 } else { 1 };
            row[j] = (prev_diag + cost).min(row[j] + 1).min(row[j - 1] + 1);
            prev_diag = old;
        }
    }
    row[n]
}

/// Optimized Needleman-Wunsch with cache-friendly loop tiling.
/// (Renamed from `needleman_wunsch_simd` — no actual SIMD intrinsics are used.)
pub fn needleman_wunsch_striped(a: &[u8], b: &[u8], match_score: i32, mismatch_score: i32, gap_penalty: i32) -> i32 {
    let (m, n) = (a.len(), b.len());
    if m == 0 { return (n as i32) * gap_penalty; }
    if n == 0 { return (m as i32) * gap_penalty; }

    // Fast path: identical strings.
    if a == b { return (m as i32) * match_score; }

    // Single-row + diagonal optimization.
    let mut row = vec![0i32; n + 1];
    for (j, item) in row.iter_mut().enumerate() {
        *item = (j as i32) * gap_penalty;
    }

    for i in 1..=m {
        let mut prev_diag = row[0];
        row[0] = (i as i32) * gap_penalty;
        let ai = a[i - 1];

        // Process in cache-friendly blocks of 8.
        let mut j = 1;
        while j + 7 <= n {
            for k in 0..8usize {
                let jj = j + k;
                let old = row[jj];
                let cost = if ai == b[jj - 1] { match_score } else { mismatch_score };
                row[jj] = (prev_diag + cost)
                    .max(row[jj] + gap_penalty)
                    .max(row[jj - 1] + gap_penalty);
                prev_diag = old;
            }
            j += 8;
        }
        // Remainder.
        while j <= n {
            let old = row[j];
            let cost = if ai == b[j - 1] { match_score } else { mismatch_score };
            row[j] = (prev_diag + cost)
                .max(row[j] + gap_penalty)
                .max(row[j - 1] + gap_penalty);
            prev_diag = old;
            j += 1;
        }
    }
    row[n]
}

/// Optimized Jaro similarity.
/// (Renamed from `jaro_simd` — no actual SIMD intrinsics are used.)
pub fn jaro_optimized(a: &[u8], b: &[u8]) -> f64 {
    let (m, n) = (a.len(), b.len());
    if m == 0 && n == 0 { return 1.0; }
    if m == 0 || n == 0 { return 0.0; }
    if a == b { return 1.0; }

    let match_distance = (m.max(n) / 2).saturating_sub(1);

    // Use stack-allocated flags for short strings to avoid heap allocation.
    let mut a_matches_buf = [false; 128];
    let mut b_matches_buf = [false; 128];
    let (a_matches, b_matches): (&mut [bool], &mut [bool]) = if m <= 128 && n <= 128 {
        (&mut a_matches_buf[..m], &mut b_matches_buf[..n])
    } else {
        // This branch leaks into heap only for very long strings.
        // We use Box to avoid the borrow checker issue with conditional vec allocation.
        // For simplicity, just use vec for > 128.
        // (In practice strings > 128 are rare for fuzzy matching.)
        return jaro_optimized_heap(a, b, match_distance);
    };

    let mut matches = 0u32;

    for i in 0..m {
        let lo = i.saturating_sub(match_distance);
        let hi = (i + match_distance + 1).min(n);
        let ai = a[i];
        for j in lo..hi {
            if b_matches[j] || ai != b[j] { continue; }
            a_matches[i] = true;
            b_matches[j] = true;
            matches += 1;
            break;
        }
    }

    if matches == 0 { return 0.0; }

    let mut transpositions = 0u32;
    let mut k = 0;
    for i in 0..m {
        if !a_matches[i] { continue; }
        while !b_matches[k] { k += 1; }
        if a[i] != b[k] { transpositions += 1; }
        k += 1;
    }

    (matches as f64 / m as f64
        + matches as f64 / n as f64
        + (matches as f64 - transpositions as f64 / 2.0) / matches as f64) / 3.0
}

/// Heap-allocating path for strings > 128 bytes.
fn jaro_optimized_heap(a: &[u8], b: &[u8], match_distance: usize) -> f64 {
    let (m, n) = (a.len(), b.len());
    let mut a_matches = vec![false; m];
    let mut b_matches = vec![false; n];
    let mut matches = 0u32;

    for i in 0..m {
        let lo = i.saturating_sub(match_distance);
        let hi = (i + match_distance + 1).min(n);
        let ai = a[i];
        for j in lo..hi {
            if b_matches[j] || ai != b[j] { continue; }
            a_matches[i] = true;
            b_matches[j] = true;
            matches += 1;
            break;
        }
    }

    if matches == 0 { return 0.0; }

    let mut transpositions = 0u32;
    let mut k = 0;
    for i in 0..m {
        if !a_matches[i] { continue; }
        while !b_matches[k] { k += 1; }
        if a[i] != b[k] { transpositions += 1; }
        k += 1;
    }

    (matches as f64 / m as f64
        + matches as f64 / n as f64
        + (matches as f64 - transpositions as f64 / 2.0) / matches as f64) / 3.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_myers_basic() {
        assert_eq!(levenshtein_myers(b"kitten", b"sitting"), 3);
        assert_eq!(levenshtein_myers(b"hello", b"hello"), 0);
        assert_eq!(levenshtein_myers(b"abc", b""), 3);
        assert_eq!(levenshtein_myers(b"", b"xyz"), 3);
        assert_eq!(levenshtein_myers(b"", b""), 0);
    }

    #[test]
    fn test_myers_asymmetric() {
        // Short vs long — tests the swap logic.
        assert_eq!(levenshtein_myers(b"a", b"abcdef"), 5);
        assert_eq!(levenshtein_myers(b"abcdef", b"a"), 5);
    }

    #[test]
    fn test_myers_64_char_boundary() {
        let a: Vec<u8> = (0..64).map(|i| b'a' + (i % 26)).collect();
        let b: Vec<u8> = (0..64).map(|i| b'b' + (i % 26)).collect();
        let result_myers = levenshtein_myers(&a, &b);
        let result_dp = levenshtein_dp_optimized(&a, &b);
        assert_eq!(result_myers, result_dp);
    }

    #[test]
    fn test_myers_fallback_long_strings() {
        let a: Vec<u8> = (0..100).map(|i| b'a' + (i % 26)).collect();
        let b: Vec<u8> = (0..100).map(|i| b'b' + (i % 26)).collect();
        // Should fall back to DP for > 64 chars. Just verify it doesn't panic.
        let _ = levenshtein_myers(&a, &b);
    }
}
