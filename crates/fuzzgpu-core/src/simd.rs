use crate::{sat_add, sat_mul};

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
    // Documented contract: this is the ASCII fast path. `peq` is indexed by raw byte
    // value (always in-bounds for u8), but byte-level matching of non-ASCII input
    // would silently compute distances over UTF-8 continuation bytes instead of
    // characters. Callers (Python/JS bindings) gate on `is_ascii()` first and fall
    // back to the Unicode path. Assert it here so direct Rust callers can't
    // silently get byte-semantics results.
    assert!(a.is_ascii() && b.is_ascii(), "levenshtein_myers requires ASCII inputs");
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

/// Precomputed Myers pattern state (Peq table + mask bits), built once and
/// reused across many texts. The batch/cdist paths build this a single time
/// per shared query instead of once per text (or per 4 texts), which matters
/// at 100k-pair scale.
pub(crate) struct MyersPattern {
    peq: [u64; 256],
    m: usize,
    mask: u64,
    last_bit: u64,
}

impl MyersPattern {
    /// Build pattern state. `pattern` must be non-empty ASCII ≤ 64 bytes.
    pub(crate) fn new(pattern: &[u8]) -> Self {
        debug_assert!(!pattern.is_empty() && pattern.len() <= 64 && pattern.is_ascii());
        let mut peq = [0u64; 256];
        for (j, &ch) in pattern.iter().enumerate() {
            peq[ch as usize] |= 1u64 << j;
        }
        let m = pattern.len();
        let mask = if m == 64 { u64::MAX } else { (1u64 << m) - 1 };
        MyersPattern { peq, m, mask, last_bit: 1u64 << (m - 1) }
    }
}

/// 4-way Myers bit-vector: one shared ASCII pattern (≤ 64 bytes) against four
/// ASCII texts, computed in parallel u64 lanes.
///
/// This is the SIMD-amplified form of [`levenshtein_myers`] (the same idea as
/// rapidfuzz's `MyersSIMD`): each lane runs the identical recurrence, so four
/// independent distances are produced per vector width. On AVX2 the recurrence
/// (add/xor/or/and/shift) maps one-to-one onto `__m256i` u64 lanes and the Peq
/// lookup uses a single gather — about 4× the per-thread throughput of the
/// scalar Myers loop.
///
/// `pattern` must be non-empty ASCII ≤ 64 bytes (the Peq mask must fit one
/// u64). Texts may be any length, ASCII. Callers must gate on these
/// preconditions (e.g. `levenshtein_batch_auto`); this function debug-asserts
/// them.
pub fn levenshtein_myers_4way(pattern: &[u8], texts: [&[u8]; 4]) -> [u32; 4] {
    let pat = MyersPattern::new(pattern);
    levenshtein_myers_4way_pat(&pat, texts)
}

/// Cached AVX2 availability — `is_x86_feature_detected!` runs a CPUID per
/// call (~150-300 cycles), which would cost more than the kernel itself at
/// 100k-pair scale; the dispatch hot path must be a single cached load.
#[cfg(target_arch = "x86_64")]
fn avx2_available() -> bool {
    use std::sync::OnceLock;
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| std::arch::is_x86_feature_detected!("avx2"))
}

/// Cached AVX512F availability (both SIMD kernels only need AVX512F ops).
#[cfg(target_arch = "x86_64")]
fn avx512_available() -> bool {
    use std::sync::OnceLock;
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| std::arch::is_x86_feature_detected!("avx512f"))
}

/// Optional `FUZZGPU_SIMD=portable|neon|avx2|avx512` override, so users can
/// pin the ISA (e.g. disable AVX512 on parts where the 512-bit datapath
/// downclocks the cores, or force a specific width for benchmarking). The
/// value is leaked once; the process is the lifetime.
fn simd_isa_override() -> Option<&'static str> {
    use std::sync::OnceLock;
    static CACHE: OnceLock<Option<&'static str>> = OnceLock::new();
    *CACHE.get_or_init(|| match std::env::var("FUZZGPU_SIMD") {
        Ok(v) => match v.as_str() {
            "portable" | "neon" | "avx2" | "avx512" => Some(Box::leak(v.into_boxed_str())),
            _ => None,
        },
        Err(_) => None,
    })
}

/// Width implied by the `FUZZGPU_SIMD` override, if set.
fn overridden_width() -> Option<usize> {
    simd_isa_override().map(|isa| match isa {
        "avx512" => 8,
        "avx2" => 4,
        "neon" => 2,
        _ => 1,
    })
}

/// Preferred SIMD width for the Myers 4-way family: 8 on AVX512 (8 u64 lanes
/// per 512-bit vector), 4 on AVX2, 2 on NEON (aarch64), else 1 (scalar). The
/// batch paths chunk texts by this width.
#[allow(unused_assignments)] // the initial 1 is dead on aarch64 (w is set below)
pub(crate) fn myers_simd_width() -> usize {
    if let Some(w) = overridden_width() {
        return w;
    }
    let mut w = 1usize;
    #[cfg(target_arch = "x86_64")]
    {
        if avx512_available() {
            w = 8;
        } else if avx2_available() {
            w = 4;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        w = 2;
    }
    w
}

/// Myers batch over any 1..=8 texts at the caller's chosen width: dispatches
/// to AVX512 (8), AVX2 (4), NEON (2) or the scalar fallback. `texts` must be
/// at most 8 entries (chunked by [`myers_simd_width`]).
pub(crate) fn levenshtein_myers_width(pat: &MyersPattern, texts: &[&[u8]]) -> Vec<u32> {
    match texts.len() {
        8 => levenshtein_myers_8way(pat, [
            texts[0], texts[1], texts[2], texts[3], texts[4], texts[5], texts[6], texts[7],
        ])
        .to_vec(),
        4 => levenshtein_myers_4way_pat(pat, [texts[0], texts[1], texts[2], texts[3]]).to_vec(),
        2 => {
            #[cfg(target_arch = "aarch64")]
            {
                // SAFETY: NEON is architecturally mandatory on aarch64.
                return unsafe { levenshtein_myers_2way_neon(pat, [texts[0], texts[1]]) }.to_vec();
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                vec![
                    levenshtein_myers_pattern(pat, texts[0]),
                    levenshtein_myers_pattern(pat, texts[1]),
                ]
            }
        }
        1 => vec![levenshtein_myers_pattern(pat, texts[0])],
        _ => texts.iter().map(|t| levenshtein_myers_pattern(pat, t)).collect(),
    }
}

/// 8-way Myers: AVX512 when available, else two 4-way calls.
pub(crate) fn levenshtein_myers_8way(pat: &MyersPattern, texts: [&[u8]; 8]) -> [u32; 8] {
    #[cfg(target_arch = "x86_64")]
    {
        if avx512_available() {
            // SAFETY: `avx512_available()` guarantees AVX512F.
            return unsafe { levenshtein_myers_8way_avx512(pat, texts) };
        }
    }
    let lo = levenshtein_myers_4way_pat(pat, [texts[0], texts[1], texts[2], texts[3]]);
    let hi = levenshtein_myers_4way_pat(pat, [texts[4], texts[5], texts[6], texts[7]]);
    [lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]]
}

/// AVX512 implementation of the 8-way Myers kernel: eight u64 bit-vector
/// lanes in one `__m512i`, the same recurrence as the AVX2 4-way with
/// mask-register comparisons for the branchless score deltas. Only AVX512F
/// instructions are used, so it runs on any AVX512-capable chip (Intel
/// Ice Lake+/Tiger Lake+, and Zen 4/5 which lack AVX512BW but have F).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn levenshtein_myers_8way_avx512(pat: &MyersPattern, texts: [&[u8]; 8]) -> [u32; 8] {
    use std::arch::x86_64::*;
    let MyersPattern { peq, m, mask, last_bit } = pat;
    let zero = _mm512_setzero_si512();
    let one = _mm512_set1_epi64(1);
    let ones = _mm512_set1_epi64(-1);
    let mask_bcast = _mm512_set1_epi64(*mask as i64);
    let last_bcast = _mm512_set1_epi64(*last_bit as i64);

    let lens = [
        texts[0].len(), texts[1].len(), texts[2].len(), texts[3].len(),
        texts[4].len(), texts[5].len(), texts[6].len(), texts[7].len(),
    ];
    let min_len = lens.iter().copied().min().unwrap_or(0);
    let max_len = lens.iter().copied().max().unwrap_or(0);

    let mut pv = _mm512_set1_epi64(*mask as i64);
    let mut mv = zero;
    let mut score = _mm512_set1_epi64(*m as i64);

    // Main loop: all eight lanes active — no masks, no bounds checks.
    for j in 0..min_len {
        let eq = _mm512_set_epi64(
            peq[*texts[7].get_unchecked(j) as usize] as i64,
            peq[*texts[6].get_unchecked(j) as usize] as i64,
            peq[*texts[5].get_unchecked(j) as usize] as i64,
            peq[*texts[4].get_unchecked(j) as usize] as i64,
            peq[*texts[3].get_unchecked(j) as usize] as i64,
            peq[*texts[2].get_unchecked(j) as usize] as i64,
            peq[*texts[1].get_unchecked(j) as usize] as i64,
            peq[*texts[0].get_unchecked(j) as usize] as i64,
        );

        let xv = _mm512_or_si512(eq, mv);
        let eq_and_pv = _mm512_and_si512(eq, pv);
        let xh = _mm512_or_si512(_mm512_xor_si512(_mm512_add_epi64(eq_and_pv, pv), pv), eq);
        let ph = _mm512_or_si512(mv, _mm512_andnot_si512(_mm512_or_si512(xh, pv), ones));
        let mh = _mm512_and_si512(pv, xh);

        // Branchless score deltas via mask registers.
        let hit_k = _mm512_cmpeq_epi64_mask(_mm512_and_si512(ph, last_bcast), zero);
        score = _mm512_add_epi64(score, _mm512_maskz_mov_epi64(hit_k ^ 0xFF, one));
        let miss_k = _mm512_cmpeq_epi64_mask(_mm512_and_si512(mh, last_bcast), zero);
        score = _mm512_sub_epi64(score, _mm512_maskz_mov_epi64(miss_k ^ 0xFF, one));

        let ph_shifted = _mm512_or_si512(_mm512_slli_epi64(ph, 1), one);
        let mh_shifted = _mm512_slli_epi64(mh, 1);
        pv = _mm512_and_si512(
            _mm512_or_si512(mh_shifted, _mm512_andnot_si512(_mm512_or_si512(xv, ph_shifted), ones)),
            mask_bcast,
        );
        mv = _mm512_and_si512(_mm512_and_si512(ph_shifted, xv), mask_bcast);
    }

    // Tail: shorter lanes freeze their score via the activity mask.
    if min_len < max_len {
        let mut remaining = _mm512_set_epi64(
            (lens[7] - min_len) as i64,
            (lens[6] - min_len) as i64,
            (lens[5] - min_len) as i64,
            (lens[4] - min_len) as i64,
            (lens[3] - min_len) as i64,
            (lens[2] - min_len) as i64,
            (lens[1] - min_len) as i64,
            (lens[0] - min_len) as i64,
        );
        for j in min_len..max_len {
            let is_zero = _mm512_cmpeq_epi64_mask(remaining, zero);
            let active = _mm512_maskz_mov_epi64(is_zero ^ 0xFF, ones);
            let eq = _mm512_and_si512(
                _mm512_set_epi64(
                    peq[*texts[7].get(j).unwrap_or(&0) as usize] as i64,
                    peq[*texts[6].get(j).unwrap_or(&0) as usize] as i64,
                    peq[*texts[5].get(j).unwrap_or(&0) as usize] as i64,
                    peq[*texts[4].get(j).unwrap_or(&0) as usize] as i64,
                    peq[*texts[3].get(j).unwrap_or(&0) as usize] as i64,
                    peq[*texts[2].get(j).unwrap_or(&0) as usize] as i64,
                    peq[*texts[1].get(j).unwrap_or(&0) as usize] as i64,
                    peq[*texts[0].get(j).unwrap_or(&0) as usize] as i64,
                ),
                active,
            );

            let xv = _mm512_or_si512(eq, mv);
            let eq_and_pv = _mm512_and_si512(eq, pv);
            let xh = _mm512_or_si512(_mm512_xor_si512(_mm512_add_epi64(eq_and_pv, pv), pv), eq);
            let ph = _mm512_or_si512(mv, _mm512_andnot_si512(_mm512_or_si512(xh, pv), ones));
            let mh = _mm512_and_si512(pv, xh);

            let hit = _mm512_and_si512(_mm512_and_si512(ph, last_bcast), active);
            let hit_k = _mm512_cmpeq_epi64_mask(hit, zero);
            score = _mm512_add_epi64(score, _mm512_maskz_mov_epi64(hit_k ^ 0xFF, one));
            let miss = _mm512_and_si512(_mm512_and_si512(mh, last_bcast), active);
            let miss_k = _mm512_cmpeq_epi64_mask(miss, zero);
            score = _mm512_sub_epi64(score, _mm512_maskz_mov_epi64(miss_k ^ 0xFF, one));

            let ph_shifted = _mm512_or_si512(_mm512_slli_epi64(ph, 1), one);
            let mh_shifted = _mm512_slli_epi64(mh, 1);
            pv = _mm512_and_si512(
                _mm512_or_si512(mh_shifted, _mm512_andnot_si512(_mm512_or_si512(xv, ph_shifted), ones)),
                mask_bcast,
            );
            mv = _mm512_and_si512(_mm512_and_si512(ph_shifted, xv), mask_bcast);

            remaining = _mm512_and_si512(_mm512_sub_epi64(remaining, one), active);
        }
    }

    let lanes: [i64; 8] = std::mem::transmute(score);
    std::array::from_fn(|i| lanes[i] as u32)
}

/// NEON 2-way Myers for aarch64: two u64 lanes per 128-bit vector, mirroring
/// the portable `[u64; 2]` recurrence (always-masked, saturating per-lane
/// remaining counters via `vqsubq_u64`). NEON is mandatory on aarch64, so no
/// feature gate is needed beyond the architecture.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn levenshtein_myers_2way_neon(pat: &MyersPattern, texts: [&[u8]; 2]) -> [u32; 2] {
    use std::arch::aarch64::*;
    let MyersPattern { peq, m, mask, last_bit } = pat;
    let mask_v = vdupq_n_u64(*mask);
    let last_v = vdupq_n_u64(*last_bit);
    let one = vdupq_n_u64(1);
    let zero = vdupq_n_u64(0);
    let ones = vdupq_n_u64(u64::MAX);

    let mut pv = mask_v;
    let mut mv = zero;
    let mut score = vdupq_n_u64(*m as u64);
    let mut remaining = vsetq_lane_u64(texts[1].len() as u64, vsetq_lane_u64(texts[0].len() as u64, zero, 0), 1);
    let max_len = texts[0].len().max(texts[1].len());

    for j in 0..max_len {
        let is_zero = vceqq_u64(remaining, zero);
        // NEON has no vmvnq_u64; bitwise-NOT is XOR with all-ones.
        let active = veorq_u64(is_zero, ones); // all-ones while this lane still has chars

        let r0 = vgetq_lane_u64(remaining, 0);
        let r1 = vgetq_lane_u64(remaining, 1);
        let eq0 = if r0 > 0 { peq[*texts[0].get(j).unwrap_or(&0) as usize] } else { 0 };
        let eq1 = if r1 > 0 { peq[*texts[1].get(j).unwrap_or(&0) as usize] } else { 0 };
        let eq = vandq_u64(vsetq_lane_u64(eq1, vsetq_lane_u64(eq0, zero, 0), 1), active);

        let xv = vorrq_u64(eq, mv);
        let eq_and_pv = vandq_u64(eq, pv);
        let xh = vorrq_u64(veorq_u64(vaddq_u64(eq_and_pv, pv), pv), eq);
        let ph = vorrq_u64(mv, veorq_u64(vorrq_u64(xh, pv), ones));
        let mh = vandq_u64(pv, xh);

        let hit = vandq_u64(vandq_u64(ph, last_v), active);
        let hit_bool = vshrq_n_u64(veorq_u64(vceqq_u64(hit, zero), ones), 63);
        score = vaddq_u64(score, hit_bool);
        let miss = vandq_u64(vandq_u64(mh, last_v), active);
        let miss_bool = vshrq_n_u64(veorq_u64(vceqq_u64(miss, zero), ones), 63);
        score = vsubq_u64(score, miss_bool);

        let ph_shifted = vorrq_u64(vshlq_n_u64(ph, 1), one);
        let mh_shifted = vshlq_n_u64(mh, 1);
        pv = vandq_u64(vorrq_u64(mh_shifted, veorq_u64(vorrq_u64(xv, ph_shifted), ones)), mask_v);
        mv = vandq_u64(vandq_u64(ph_shifted, xv), mask_v);

        remaining = vqsubq_u64(remaining, one); // saturating: stays 0 once empty
    }

    let lanes: [u64; 2] = std::mem::transmute(score);
    [lanes[0] as u32, lanes[1] as u32]
}

/// 4-way kernel over a prebuilt pattern (see [`MyersPattern`]).
pub(crate) fn levenshtein_myers_4way_pat(pat: &MyersPattern, texts: [&[u8]; 4]) -> [u32; 4] {
    debug_assert!(texts.iter().all(|t| t.is_ascii()));
    #[cfg(target_arch = "x86_64")]
    {
        if avx2_available() {
            // SAFETY: `avx2_available()` guarantees AVX2 availability.
            return unsafe { levenshtein_myers_4way_avx2(pat, texts) };
        }
    }
    levenshtein_myers_4way_portable(pat, texts)
}

/// Scalar Myers over a prebuilt pattern (tail/fallback path of the batch).
pub(crate) fn levenshtein_myers_pattern(pat: &MyersPattern, text: &[u8]) -> u32 {
    debug_assert!(text.is_ascii());
    let MyersPattern { peq, m, mask, last_bit } = pat;
    let mut pv: u64 = *mask;
    let mut mv: u64 = 0;
    let mut score = *m as u32;
    for &ch in text {
        let eq = peq[ch as usize];
        let xv = eq | mv;
        let eq_and_pv = eq & pv;
        let xh = (eq_and_pv.wrapping_add(pv) ^ pv) | eq;
        let ph = mv | !(xh | pv);
        let mh = pv & xh;
        if ph & last_bit != 0 {
            score += 1;
        }
        if mh & last_bit != 0 {
            score -= 1;
        }
        let ph_shifted = (ph << 1) | 1;
        let mh_shifted = mh << 1;
        pv = (mh_shifted | !(xv | ph_shifted)) & mask;
        mv = (ph_shifted & xv) & mask;
    }
    score
}

/// Portable fallback for [`levenshtein_myers_4way`] (and the reference used to
/// validate the AVX2 path). Same recurrence, `[u64; 4]` lanes.
fn levenshtein_myers_4way_portable(pat: &MyersPattern, texts: [&[u8]; 4]) -> [u32; 4] {
    let MyersPattern { peq, m, mask, last_bit } = pat;
    let mut pv = [*mask; 4];
    let mut mv = [0u64; 4];
    let mut score = [*m as u32; 4];
    let mut remaining = [
        texts[0].len(),
        texts[1].len(),
        texts[2].len(),
        texts[3].len(),
    ];
    let max_len = remaining.iter().copied().max().unwrap_or(0);

    for j in 0..max_len {
        for i in 0..4 {
            let active = if remaining[i] > 0 { u64::MAX } else { 0 };
            let eq = if remaining[i] > 0 { peq[texts[i][j] as usize] } else { 0 };
            let xv = eq | mv[i];
            let eq_and_pv = eq & pv[i];
            let xh = (eq_and_pv.wrapping_add(pv[i]) ^ pv[i]) | eq;
            let ph = mv[i] | !(xh | pv[i]);
            let mh = pv[i] & xh;
            if active != 0 && (ph & last_bit) != 0 {
                score[i] += 1;
            }
            if active != 0 && (mh & last_bit) != 0 {
                score[i] -= 1;
            }
            let ph_shifted = (ph << 1) | 1;
            let mh_shifted = mh << 1;
            pv[i] = (mh_shifted | !(xv | ph_shifted)) & mask;
            mv[i] = (ph_shifted & xv) & mask;
        }
        for r in remaining.iter_mut() {
            *r = r.saturating_sub(1);
        }
    }
    score
}

/// AVX2 implementation of [`levenshtein_myers_4way`]: four u64 bit-vector lanes
/// in one `__m256i`, Peq values gathered per text character, branchless score
/// deltas, saturating per-lane activity mask so finished lanes freeze their
/// score while longer texts keep processing.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn levenshtein_myers_4way_avx2(pat: &MyersPattern, texts: [&[u8]; 4]) -> [u32; 4] {
    use std::arch::x86_64::*;
    let MyersPattern { peq, m, mask, last_bit } = pat;
    let zero = _mm256_setzero_si256();
    let ones = _mm256_set1_epi64x(-1);
    let one = _mm256_set1_epi64x(1);
    let mask_bcast = _mm256_set1_epi64x(*mask as i64);
    let last_bcast = _mm256_set1_epi64x(*last_bit as i64);

    let lens = [texts[0].len(), texts[1].len(), texts[2].len(), texts[3].len()];
    let min_len = lens.iter().copied().min().unwrap_or(0);
    let max_len = lens.iter().copied().max().unwrap_or(0);

    let mut pv = _mm256_set1_epi64x(*mask as i64);
    let mut mv = zero;
    let mut score = _mm256_set1_epi64x(*m as i64);

    // Main loop: every lane is active, so there are no bounds checks, no
    // activity masks and no score freezing — the common equal-length case pays
    // nothing extra. The Peq lookup uses four independent scalar loads (L1
    // hits, hoistable by the OOO engine) instead of `vpgatherqq`, which is
    // catastrophically slow (~50 cycles) on the Iris Xe-era CPUs we target.
    for j in 0..min_len {
        let eq = _mm256_set_epi64x(
            peq[*texts[3].get_unchecked(j) as usize] as i64,
            peq[*texts[2].get_unchecked(j) as usize] as i64,
            peq[*texts[1].get_unchecked(j) as usize] as i64,
            peq[*texts[0].get_unchecked(j) as usize] as i64,
        );

        let xv = _mm256_or_si256(eq, mv);
        let eq_and_pv = _mm256_and_si256(eq, pv);
        let xh = _mm256_or_si256(_mm256_xor_si256(_mm256_add_epi64(eq_and_pv, pv), pv), eq);
        let ph = _mm256_or_si256(mv, _mm256_andnot_si256(_mm256_or_si256(xh, pv), ones));
        let mh = _mm256_and_si256(pv, xh);

        let hit_bool = _mm256_andnot_si256(_mm256_cmpeq_epi64(_mm256_and_si256(ph, last_bcast), zero), one);
        score = _mm256_add_epi64(score, hit_bool);
        let miss_bool = _mm256_andnot_si256(_mm256_cmpeq_epi64(_mm256_and_si256(mh, last_bcast), zero), one);
        score = _mm256_sub_epi64(score, miss_bool);

        let ph_shifted = _mm256_or_si256(_mm256_slli_epi64(ph, 1), one);
        let mh_shifted = _mm256_slli_epi64(mh, 1);
        pv = _mm256_and_si256(
            _mm256_or_si256(mh_shifted, _mm256_andnot_si256(_mm256_or_si256(xv, ph_shifted), ones)),
            mask_bcast,
        );
        mv = _mm256_and_si256(_mm256_and_si256(ph_shifted, xv), mask_bcast);
    }

    // Tail loop: lanes that ended early freeze their score (activity mask);
    // lanes still reading use bounds-checked access. Empty when all texts have
    // equal length, which is the common batch shape.
    if min_len < max_len {
        let mut remaining = _mm256_set_epi64x(
            (lens[3] - min_len) as i64,
            (lens[2] - min_len) as i64,
            (lens[1] - min_len) as i64,
            (lens[0] - min_len) as i64,
        );
        for j in min_len..max_len {
            let is_zero = _mm256_cmpeq_epi64(remaining, zero);
            let active = _mm256_andnot_si256(is_zero, ones);
            let eq = _mm256_and_si256(_mm256_set_epi64x(
                peq[*texts[3].get(j).unwrap_or(&0) as usize] as i64,
                peq[*texts[2].get(j).unwrap_or(&0) as usize] as i64,
                peq[*texts[1].get(j).unwrap_or(&0) as usize] as i64,
                peq[*texts[0].get(j).unwrap_or(&0) as usize] as i64,
            ), active);

            let xv = _mm256_or_si256(eq, mv);
            let eq_and_pv = _mm256_and_si256(eq, pv);
            let xh = _mm256_or_si256(_mm256_xor_si256(_mm256_add_epi64(eq_and_pv, pv), pv), eq);
            let ph = _mm256_or_si256(mv, _mm256_andnot_si256(_mm256_or_si256(xh, pv), ones));
            let mh = _mm256_and_si256(pv, xh);

            let hit = _mm256_and_si256(_mm256_and_si256(ph, last_bcast), active);
            let hit_bool = _mm256_andnot_si256(_mm256_cmpeq_epi64(hit, zero), one);
            score = _mm256_add_epi64(score, hit_bool);
            let miss = _mm256_and_si256(_mm256_and_si256(mh, last_bcast), active);
            let miss_bool = _mm256_andnot_si256(_mm256_cmpeq_epi64(miss, zero), one);
            score = _mm256_sub_epi64(score, miss_bool);

            let ph_shifted = _mm256_or_si256(_mm256_slli_epi64(ph, 1), one);
            let mh_shifted = _mm256_slli_epi64(mh, 1);
            pv = _mm256_and_si256(
                _mm256_or_si256(mh_shifted, _mm256_andnot_si256(_mm256_or_si256(xv, ph_shifted), ones)),
                mask_bcast,
            );
            mv = _mm256_and_si256(_mm256_and_si256(ph_shifted, xv), mask_bcast);

            remaining = _mm256_and_si256(_mm256_sub_epi64(remaining, one), active);
        }
    }

    let lanes: [i64; 4] = std::mem::transmute(score);
    [lanes[0] as u32, lanes[1] as u32, lanes[2] as u32, lanes[3] as u32]
}

/// Bit-parallel Jaro similarity (the foundation for the 4-way kernel).
///
/// Replaces the O(m·w) window-scan inner loop of `jaro_inner_slice` with u64
/// mask operations: for each position i in `a`, the candidate matches in `b`
/// are `pos_b[a[i]] & window(i) & ~matched`, and the first match is the lowest
/// set bit (blsi). The transposition count uses the same ordered k-th-pair
/// walk as the reference. ASCII bytes, both inputs ≤ 64 (the position masks
/// must fit one u64).
pub fn jaro_bitpar(a: &[u8], b: &[u8]) -> f64 {
    debug_assert!(a.len() <= 64 && b.len() <= 64);
    debug_assert!(a.is_ascii() && b.is_ascii());
    let (m, n) = (a.len(), b.len());
    if m == 0 && n == 0 {
        return 1.0;
    }
    if m == 0 || n == 0 {
        return 0.0;
    }
    if a == b {
        return 1.0;
    }
    let wd = (m.max(n) / 2).saturating_sub(1);

    // pos_b[c] = bitmask of positions j where b[j] == c.
    let mut pos_b = [0u64; 256];
    for (j, &ch) in b.iter().enumerate() {
        pos_b[ch as usize] |= 1u64 << j;
    }
    let n_mask = if n == 64 { u64::MAX } else { (1u64 << n) - 1 };

    let mut matched_b = 0u64;
    let mut matched_a = 0u64;
    let mut matches = 0u32;

    for i in 0..m {
        let lo = i.saturating_sub(wd);
        let hi = (i + wd).min(n - 1);
        // Window bits [lo, hi] ∩ [0, n): lo ≤ 63 always; hi ≤ n-1 ≤ 63, so
        // hi+1 ≤ 64 — guard the == 64 case.
        let lo_mask = (1u64 << lo) - 1;
        let hi_mask = if hi == 63 { u64::MAX } else { (1u64 << (hi + 1)) - 1 };
        let window = n_mask & !lo_mask & hi_mask;

        let cand = pos_b[a[i] as usize] & window & !matched_b;
        if cand != 0 {
            let lowest = cand & cand.wrapping_neg();
            matched_b |= lowest;
            matched_a |= 1u64 << i;
            matches += 1;
        }
    }

    if matches == 0 {
        return 0.0;
    }

    // Ordered k-th-pair transposition walk (identical semantics to the
    // reference's `while !b_matches[k]` loop).
    let mut transpositions = 0u32;
    let mut ma = matched_a;
    let mut mb = matched_b;
    while ma != 0 {
        let i = ma.trailing_zeros() as usize;
        let j = mb.trailing_zeros() as usize;
        if a[i] != b[j] {
            transpositions += 1;
        }
        ma &= ma - 1;
        mb &= mb - 1;
    }

    (matches as f64 / m as f64
        + matches as f64 / n as f64
        + (matches as f64 - transpositions as f64 / 2.0) / matches as f64)
        / 3.0
}

/// 4-way Jaro: one shared ASCII first string (≤ 64 bytes) against four ASCII
/// texts (≤ 64 bytes).
///
/// The matching-window pass is vectorized: the four lanes share the position
/// `i` into `a`, so the per-character window masks (which depend on `i`, the
/// lane's length, and the lane's match distance) are computed with AVX2
/// variable shifts, and the per-lane `pos_b[a[i]]` lookups are four independent
/// scalar loads packed into one vector (gathers are catastrophically slow on
/// the Iris Xe-class CPUs this targets). Transpositions and scoring run in a
/// short scalar tail per lane.
pub fn jaro_4way(a: &[u8], texts: [&[u8]; 4]) -> [f64; 4] {
    debug_assert!(a.len() <= 64 && a.is_ascii());
    debug_assert!(texts.iter().all(|t| t.len() <= 64 && t.is_ascii()));
    #[cfg(target_arch = "x86_64")]
    {
        if avx2_available() {
            // SAFETY: `avx2_available()` guarantees AVX2 availability.
            return unsafe { jaro_4way_avx2(a, texts) };
        }
    }
    [
        jaro_bitpar(a, texts[0]),
        jaro_bitpar(a, texts[1]),
        jaro_bitpar(a, texts[2]),
        jaro_bitpar(a, texts[3]),
    ]
}

/// Preferred SIMD width for Jaro: 8 on AVX512, 4 on AVX2, 2 on NEON, 1 scalar.
#[allow(unused_assignments)] // the initial 1 is dead on aarch64 (w is set below)
pub(crate) fn jaro_simd_width() -> usize {
    if let Some(w) = overridden_width() {
        return w;
    }
    let mut w = 1usize;
    #[cfg(target_arch = "x86_64")]
    {
        if avx512_available() {
            w = 8;
        } else if avx2_available() {
            w = 4;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        w = 2;
    }
    w
}

/// Jaro over 1..=8 texts at the caller's chosen width (see [`jaro_simd_width`]).
pub(crate) fn jaro_width(a: &[u8], texts: &[&[u8]]) -> Vec<f64> {
    match texts.len() {
        8 => jaro_8way(a, [
            texts[0], texts[1], texts[2], texts[3], texts[4], texts[5], texts[6], texts[7],
        ])
        .to_vec(),
        4 => jaro_4way(a, [texts[0], texts[1], texts[2], texts[3]]).to_vec(),
        2 => {
            #[cfg(target_arch = "aarch64")]
            {
                // SAFETY: NEON is architecturally mandatory on aarch64.
                return unsafe { jaro_2way_neon(a, [texts[0], texts[1]]) }.to_vec();
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                vec![jaro_bitpar(a, texts[0]), jaro_bitpar(a, texts[1])]
            }
        }
        1 => vec![jaro_bitpar(a, texts[0])],
        _ => texts.iter().map(|t| jaro_bitpar(a, t)).collect(),
    }
}

/// 8-way Jaro: AVX512 when available, else two 4-way calls.
pub(crate) fn jaro_8way(a: &[u8], texts: [&[u8]; 8]) -> [f64; 8] {
    #[cfg(target_arch = "x86_64")]
    {
        if avx512_available() {
            // SAFETY: `avx512_available()` guarantees AVX512F.
            return unsafe { jaro_8way_avx512(a, texts) };
        }
    }
    let lo = jaro_4way(a, [texts[0], texts[1], texts[2], texts[3]]);
    let hi = jaro_4way(a, [texts[4], texts[5], texts[6], texts[7]]);
    [lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]]
}

/// AVX512 implementation of the 8-way Jaro kernel: the matching-window pass
/// runs eight lanes per 512-bit vector; window masks use AVX512F variable
/// shifts and unsigned min, score deltas use mask registers. The scalar
/// transposition tail is shared with the AVX2 kernel.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn jaro_8way_avx512(a: &[u8], texts: [&[u8]; 8]) -> [f64; 8] {
    use std::arch::x86_64::*;
    let m = a.len();
    let lens = [
        texts[0].len(), texts[1].len(), texts[2].len(), texts[3].len(),
        texts[4].len(), texts[5].len(), texts[6].len(), texts[7].len(),
    ];

    let mut pos = [[0u64; 256]; 8];
    for lane in 0..8 {
        for (j, &ch) in texts[lane].iter().enumerate() {
            pos[lane][ch as usize] |= 1u64 << j;
        }
    }
    let wd: [u64; 8] = lens.map(|n| (m.max(n) / 2).saturating_sub(1) as u64);
    let n_mask: [u64; 8] = lens.map(|n| if n == 64 { u64::MAX } else { (1u64 << n) - 1 });

    let zero = _mm512_setzero_si512();
    let one = _mm512_set1_epi64(1);
    let ones = _mm512_set1_epi64(-1);
    let wd_vec = _mm512_set_epi64(wd[7] as i64, wd[6] as i64, wd[5] as i64, wd[4] as i64, wd[3] as i64, wd[2] as i64, wd[1] as i64, wd[0] as i64);
    let nmask_vec = _mm512_set_epi64(n_mask[7] as i64, n_mask[6] as i64, n_mask[5] as i64, n_mask[4] as i64, n_mask[3] as i64, n_mask[2] as i64, n_mask[1] as i64, n_mask[0] as i64);
    let nm1_vec = _mm512_set_epi64(
        lens[7].saturating_sub(1) as i64, lens[6].saturating_sub(1) as i64,
        lens[5].saturating_sub(1) as i64, lens[4].saturating_sub(1) as i64,
        lens[3].saturating_sub(1) as i64, lens[2].saturating_sub(1) as i64,
        lens[1].saturating_sub(1) as i64, lens[0].saturating_sub(1) as i64,
    );

    let mut matched_b = zero;
    let mut matched_a = zero;
    let mut counts = zero;

    for i in 0..m {
        let i_vec = _mm512_set1_epi64(i as i64);

        // lo = max(i - wd, 0); underflowed lanes are negative as signed.
        let t = _mm512_sub_epi64(i_vec, wd_vec);
        let under = _mm512_cmpgt_epi64_mask(zero, t);
        let lo = _mm512_mask_mov_epi64(zero, under ^ 0xFF, t);
        // hi = min(i + wd, n - 1) via unsigned min (AVX512F).
        let hi_raw = _mm512_add_epi64(i_vec, wd_vec);
        let hi = _mm512_min_epu64(hi_raw, nm1_vec);
        // hi_mask = (1 << (hi+1)) - 1; variable shift saturates at 64 -> 0.
        let hi_mask = _mm512_sub_epi64(_mm512_sllv_epi64(one, _mm512_add_epi64(hi, one)), one);
        let lo_mask = _mm512_sub_epi64(_mm512_sllv_epi64(one, lo), one);
        let window = _mm512_and_si512(
            _mm512_and_si512(hi_mask, _mm512_andnot_si512(lo_mask, ones)),
            nmask_vec,
        );

        let ai = a[i] as usize;
        let p = _mm512_set_epi64(
            pos[7][ai] as i64, pos[6][ai] as i64, pos[5][ai] as i64, pos[4][ai] as i64,
            pos[3][ai] as i64, pos[2][ai] as i64, pos[1][ai] as i64, pos[0][ai] as i64,
        );
        let cand = _mm512_and_si512(_mm512_and_si512(p, window), _mm512_andnot_si512(matched_b, ones));

        let hit_k = _mm512_cmpeq_epi64_mask(cand, zero);
        let hit = _mm512_maskz_mov_epi64(hit_k ^ 0xFF, one);
        counts = _mm512_add_epi64(counts, hit);
        let lowest = _mm512_and_si512(cand, _mm512_sub_epi64(zero, cand));
        matched_b = _mm512_or_si512(matched_b, lowest);
        matched_a = _mm512_or_si512(matched_a, _mm512_sllv_epi64(hit, i_vec));
    }

    let mb: [u64; 8] = std::mem::transmute(matched_b);
    let ma: [u64; 8] = std::mem::transmute(matched_a);
    let mc: [u64; 8] = std::mem::transmute(counts);
    std::array::from_fn(|lane| {
        jaro_score_from_masks(a, texts[lane], ma[lane], mb[lane], mc[lane], m as u64, lens[lane] as u64)
    })
}

/// NEON 2-way Jaro for aarch64: two lanes per 128-bit vector; window masks use
/// `vminq_u64` and variable `vshlq_u64`, the blsi step is `vandq_u64(x, -x)`
/// via a signed negate, and the underflow clamp uses a signed less-than
/// compare + `vbslq_u64` select.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn jaro_2way_neon(a: &[u8], texts: [&[u8]; 2]) -> [f64; 2] {
    use std::arch::aarch64::*;
    let m = a.len();
    let lens = [texts[0].len(), texts[1].len()];

    let mut pos = [[0u64; 256]; 2];
    for lane in 0..2 {
        for (j, &ch) in texts[lane].iter().enumerate() {
            pos[lane][ch as usize] |= 1u64 << j;
        }
    }
    let wd: [u64; 2] = lens.map(|n| (m.max(n) / 2).saturating_sub(1) as u64);
    let n_mask: [u64; 2] = lens.map(|n| if n == 64 { u64::MAX } else { (1u64 << n) - 1 });

    let zero_u = vdupq_n_u64(0);
    let one_u = vdupq_n_u64(1);
    let ones_u = vdupq_n_u64(u64::MAX);
    let zero_s = vdupq_n_s64(0);
    let wd_v = vsetq_lane_u64(wd[1], vsetq_lane_u64(wd[0], zero_u, 0), 1);
    let nmask_v = vsetq_lane_u64(n_mask[1], vsetq_lane_u64(n_mask[0], zero_u, 0), 1);
    let nm1_v = vsetq_lane_u64(
        lens[1].saturating_sub(1) as u64,
        vsetq_lane_u64(lens[0].saturating_sub(1) as u64, zero_u, 0),
        1,
    );

    let mut matched_b = zero_u;
    let mut matched_a = zero_u;
    let mut counts = zero_u;

    for i in 0..m {
        let i_u = vdupq_n_u64(i as u64);
        let i_s = vdupq_n_s64(i as i64);

        // lo = max(i - wd, 0): underflowed lanes are negative as signed.
        let t = vsubq_u64(i_u, wd_v);
        let is_neg = vcltq_s64(vreinterpretq_s64_u64(t), zero_s);
        let lo = vbslq_u64(is_neg, zero_u, t);
        // hi = min(i + wd, n - 1): ARM has no 64-bit min instruction, so use
        // a less-than compare + bitwise select; the operands are small
        // non-negative, so the signed compare is equivalent.
        let hi_raw = vaddq_u64(i_u, wd_v);
        let lt = vcltq_s64(vreinterpretq_s64_u64(hi_raw), vreinterpretq_s64_u64(nm1_v));
        let hi = vbslq_u64(lt, hi_raw, nm1_v);
        // hi_mask = (1 << (hi+1)) - 1; NEON vshl needs signed shift counts
        // and saturates at >= 64 to 0.
        let hi_mask = vsubq_u64(vshlq_u64(one_u, vreinterpretq_s64_u64(vaddq_u64(hi, one_u))), one_u);
        let lo_mask = vsubq_u64(vshlq_u64(one_u, vreinterpretq_s64_u64(lo)), one_u);
        let window = vandq_u64(vandq_u64(hi_mask, veorq_u64(lo_mask, ones_u)), nmask_v);

        let ai = a[i] as usize;
        let p = vsetq_lane_u64(pos[1][ai], vsetq_lane_u64(pos[0][ai], zero_u, 0), 1);
        let cand = vandq_u64(vandq_u64(p, window), veorq_u64(matched_b, ones_u));

        let hit_bool = vshrq_n_u64(veorq_u64(vceqq_u64(cand, zero_u), ones_u), 63);
        counts = vaddq_u64(counts, hit_bool);
        let lowest = vandq_u64(cand, vreinterpretq_u64_s64(vnegq_s64(vreinterpretq_s64_u64(cand))));
        matched_b = vorrq_u64(matched_b, lowest);
        matched_a = vorrq_u64(matched_a, vshlq_u64(hit_bool, i_s));
    }

    let mb: [u64; 2] = std::mem::transmute(matched_b);
    let ma: [u64; 2] = std::mem::transmute(matched_a);
    let mc: [u64; 2] = std::mem::transmute(counts);
    std::array::from_fn(|lane| {
        jaro_score_from_masks(a, texts[lane], ma[lane], mb[lane], mc[lane], m as u64, lens[lane] as u64)
    })
}

/// Shared scalar tail: ordered transposition count + the Jaro score formula.
fn jaro_score_from_masks(a: &[u8], b: &[u8], matched_a: u64, matched_b: u64, matches: u64, m: u64, n: u64) -> f64 {
    if matches == 0 {
        return 0.0;
    }
    let mut transpositions = 0u64;
    let mut ma = matched_a;
    let mut mb = matched_b;
    while ma != 0 {
        let i = ma.trailing_zeros() as usize;
        let j = mb.trailing_zeros() as usize;
        if a[i] != b[j] {
            transpositions += 1;
        }
        ma &= ma - 1;
        mb &= mb - 1;
    }
    let mf = m as f64;
    let nf = n as f64;
    let mat = matches as f64;
    (mat / mf + mat / nf + (mat - transpositions as f64 / 2.0) / mat) / 3.0
}

/// AVX2 implementation of [`jaro_4way`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn jaro_4way_avx2(a: &[u8], texts: [&[u8]; 4]) -> [f64; 4] {
    use std::arch::x86_64::*;
    let m = a.len();
    let lens = [texts[0].len(), texts[1].len(), texts[2].len(), texts[3].len()];

    // Per-lane position tables: pos[lane][c] = bitmask of positions j in
    // texts[lane] where texts[lane][j] == c (scalar build, one pass per lane).
    let mut pos = [[0u64; 256]; 4];
    for lane in 0..4 {
        for (j, &ch) in texts[lane].iter().enumerate() {
            pos[lane][ch as usize] |= 1u64 << j;
        }
    }
    let wd: [u64; 4] = lens.map(|n| (m.max(n) / 2).saturating_sub(1) as u64);
    let n_mask: [u64; 4] = lens.map(|n| if n == 64 { u64::MAX } else { (1u64 << n) - 1 });

    let zero = _mm256_setzero_si256();
    let one = _mm256_set1_epi64x(1);
    let ones = _mm256_set1_epi64x(-1);
    let wd_vec = _mm256_set_epi64x(wd[3] as i64, wd[2] as i64, wd[1] as i64, wd[0] as i64);
    let nmask_vec = _mm256_set_epi64x(n_mask[3] as i64, n_mask[2] as i64, n_mask[1] as i64, n_mask[0] as i64);
    let nm1_vec = _mm256_set_epi64x(
        lens[3].saturating_sub(1) as i64,
        lens[2].saturating_sub(1) as i64,
        lens[1].saturating_sub(1) as i64,
        lens[0].saturating_sub(1) as i64,
    );

    let mut matched_b = zero;
    let mut matched_a = zero;
    let mut counts = zero;

    for i in 0..m {
        let i_vec = _mm256_set1_epi64x(i as i64);

        // lo = max(i - wd, 0): underflowed lanes are negative as signed.
        let t = _mm256_sub_epi64(i_vec, wd_vec);
        let lo = _mm256_andnot_si256(_mm256_cmpgt_epi64(zero, t), t);
        // hi = min(i + wd, n - 1): both operands are small non-negative, so
        // signed compare == unsigned compare here.
        let hi_raw = _mm256_add_epi64(i_vec, wd_vec);
        let hi = _mm256_blendv_epi8(
            hi_raw,
            nm1_vec,
            _mm256_cmpgt_epi64(hi_raw, nm1_vec),
        );
        // hi_mask = (1 << (hi+1)) - 1; the variable shift saturates: a count
        // of 64 yields 0, so subtract gives u64::MAX (the hi == n == 64 case).
        let hi_mask = _mm256_sub_epi64(_mm256_sllv_epi64(one, _mm256_add_epi64(hi, one)), one);
        let lo_mask = _mm256_sub_epi64(_mm256_sllv_epi64(one, lo), one);
        let window = _mm256_and_si256(
            _mm256_and_si256(hi_mask, _mm256_andnot_si256(lo_mask, ones)),
            nmask_vec,
        );

        // Per-lane pos_b lookups at the shared index a[i] — four independent
        // scalar loads, not a gather.
        let ai = a[i] as usize;
        let p = _mm256_set_epi64x(
            pos[3][ai] as i64,
            pos[2][ai] as i64,
            pos[1][ai] as i64,
            pos[0][ai] as i64,
        );
        let cand = _mm256_and_si256(_mm256_and_si256(p, window), _mm256_andnot_si256(matched_b, ones));

        let hit = _mm256_andnot_si256(_mm256_cmpeq_epi64(cand, zero), one); // 1 if matched
        counts = _mm256_add_epi64(counts, hit);
        let lowest = _mm256_and_si256(cand, _mm256_sub_epi64(zero, cand)); // blsi
        matched_b = _mm256_or_si256(matched_b, lowest);
        matched_a = _mm256_or_si256(matched_a, _mm256_sllv_epi64(hit, i_vec));
    }

    let mb: [u64; 4] = std::mem::transmute(matched_b);
    let ma: [u64; 4] = std::mem::transmute(matched_a);
    let mc: [u64; 4] = std::mem::transmute(counts);
    let mut out = [0.0f64; 4];
    for lane in 0..4 {
        out[lane] = jaro_score_from_masks(a, texts[lane], ma[lane], mb[lane], mc[lane], m as u64, lens[lane] as u64);
    }
    out
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
pub fn needleman_wunsch_striped(a: &[u8], b: &[u8], match_score: i64, mismatch_score: i64, gap_penalty: i64) -> i64 {
    let (m, n) = (a.len(), b.len());
    if m == 0 { return sat_mul(n as i64, gap_penalty); }
    if n == 0 { return sat_mul(m as i64, gap_penalty); }

    // Fast path: identical strings.
    if a == b { return sat_mul(m as i64, match_score); }

    // Single-row + diagonal optimization.
    let mut row = vec![0i64; n + 1];
    for (j, item) in row.iter_mut().enumerate() {
        *item = sat_mul(j as i64, gap_penalty);
    }

    for i in 1..=m {
        let mut prev_diag = row[0];
        row[0] = sat_mul(i as i64, gap_penalty);
        let ai = a[i - 1];

        // Process in cache-friendly blocks of 8.
        let mut j = 1;
        while j + 7 <= n {
            for k in 0..8usize {
                let jj = j + k;
                let old = row[jj];
                let cost = if ai == b[jj - 1] { match_score } else { mismatch_score };
                row[jj] = sat_add(prev_diag, cost)
                    .max(sat_add(row[jj], gap_penalty))
                    .max(sat_add(row[jj - 1], gap_penalty));
                prev_diag = old;
            }
            j += 8;
        }
        // Remainder.
        while j <= n {
            let old = row[j];
            let cost = if ai == b[j - 1] { match_score } else { mismatch_score };
            row[j] = sat_add(prev_diag, cost)
                .max(sat_add(row[j], gap_penalty))
                .max(sat_add(row[j - 1], gap_penalty));
            prev_diag = old;
            j += 1;
        }
    }
    row[n]
}

/// Optimized Jaro similarity.
/// (Renamed from `jaro_simd` — no actual SIMD intrinsics are used.)
///
/// Uses stack-allocated match flags for strings ≤ 128 bytes and a single shared
/// inner implementation for both stack and heap paths (no code duplication).
pub fn jaro_optimized(a: &[u8], b: &[u8]) -> f64 {
    let (m, n) = (a.len(), b.len());
    if m == 0 && n == 0 { return 1.0; }
    if m == 0 || n == 0 { return 0.0; }
    if a == b { return 1.0; }

    let match_distance = (m.max(n) / 2).saturating_sub(1);

    if m <= 128 && n <= 128 {
        // Stack-allocated flags for short strings to avoid heap allocation.
        let mut a_matches = [false; 128];
        let mut b_matches = [false; 128];
        jaro_inner_slice(a, b, &mut a_matches[..m], &mut b_matches[..n], match_distance)
    } else {
        // Heap path only for strings > 128 bytes (rare in fuzzy matching).
        let mut a_matches = vec![false; m];
        let mut b_matches = vec![false; n];
        jaro_inner_slice(a, b, &mut a_matches, &mut b_matches, match_distance)
    }
}

/// Single implementation shared by the stack and heap allocation paths.
#[inline]
fn jaro_inner_slice(a: &[u8], b: &[u8], a_matches: &mut [bool], b_matches: &mut [bool], match_distance: usize) -> f64 {
    let (m, n) = (a.len(), b.len());
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
    fn test_myers_4way_matches_scalar() {
        // Deterministic LCG strings; every lane exercises a different length
        // (including lanes shorter than the shared pattern, which the activity
        // mask must freeze correctly).
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut gen = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let pattern: Vec<u8> = (0..16).map(|_| b'a' + (gen() % 26) as u8).collect();
        let texts: [Vec<u8>; 4] = std::array::from_fn(|i| {
            let len = [0usize, 1, 8, 40][i];
            (0..len).map(|_| b'a' + (gen() % 26) as u8).collect()
        });
        let got = levenshtein_myers_4way(&pattern, [
            &texts[0], &texts[1], &texts[2], &texts[3],
        ]);
        for i in 0..4 {
            let expect = levenshtein_myers(&pattern, &texts[i]);
            assert_eq!(got[i], expect, "lane {} mismatch", i);
        }
    }

    #[test]
    fn test_myers_4way_matches_scalar_random() {
        let mut state = 0x243F6A8885A308D3u64;
        for round in 0..64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let m = 1 + (state % 64) as usize;
            let pattern: Vec<u8> = (0..m).map(|_| b'a' + (state.wrapping_mul(31) % 26) as u8).collect();
            let texts: [Vec<u8>; 4] = std::array::from_fn(|i| {
                let len = (state.wrapping_mul(97 + i as u64) % 80) as usize;
                (0..len).map(|_| b'a' + (state.wrapping_mul(17 + i as u64) % 26) as u8).collect()
            });
            let got = levenshtein_myers_4way(&pattern, [
                &texts[0], &texts[1], &texts[2], &texts[3],
            ]);
            for i in 0..4 {
                let expect = levenshtein_myers(&pattern, &texts[i]);
                assert_eq!(got[i], expect, "round {} lane {} mismatch", round, i);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_myers_8way_avx512_matches_portable() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        let mut state = 0x6A09E667F3BCC909u64;
        for _ in 0..32 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let m = 1 + (state % 64) as usize;
            let pattern: Vec<u8> = (0..m).map(|_| b'a' + (state % 26) as u8).collect();
            let texts: [Vec<u8>; 8] = std::array::from_fn(|i| {
                let len = (state.wrapping_mul(3 + i as u64) % 90) as usize;
                (0..len).map(|_| b'a' + (state.wrapping_mul(5 + i as u64) % 26) as u8).collect()
            });
            let pat = MyersPattern::new(&pattern);
            let expect = levenshtein_myers_8way(&pat, [
                &texts[0], &texts[1], &texts[2], &texts[3], &texts[4], &texts[5], &texts[6], &texts[7],
            ]);
            // SAFETY: guarded by is_x86_feature_detected! above.
            let avx512 = unsafe { levenshtein_myers_8way_avx512(&pat, [
                &texts[0], &texts[1], &texts[2], &texts[3], &texts[4], &texts[5], &texts[6], &texts[7],
            ]) };
            assert_eq!(expect, avx512);
            // And every lane must match the scalar Myers.
            for (i, t) in texts.iter().enumerate() {
                assert_eq!(avx512[i], levenshtein_myers(&pattern, t), "lane {i}");
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_jaro_8way_avx512_matches_bitpar() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        let mut state = 0xBB67AE8584CAA73Bu64;
        for _ in 0..32 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let m = 1 + (state % 64) as usize;
            let a: Vec<u8> = (0..m).map(|_| b'a' + (state % 26) as u8).collect();
            let texts: [Vec<u8>; 8] = std::array::from_fn(|i| {
                let n = (state.wrapping_mul(7 + i as u64) % 65) as usize;
                (0..n).map(|_| b'a' + (state.wrapping_mul(11 + i as u64) % 26) as u8).collect()
            });
            // SAFETY: guarded by is_x86_feature_detected! above.
            let avx512 = unsafe { jaro_8way_avx512(&a, [
                &texts[0], &texts[1], &texts[2], &texts[3], &texts[4], &texts[5], &texts[6], &texts[7],
            ]) };
            for i in 0..8 {
                let expect = jaro_bitpar(&a, &texts[i]);
                assert!((avx512[i] - expect).abs() < 1e-12, "lane {i}: {} != {}", avx512[i], expect);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_myers_2way_neon_matches_portable() {
        let mut state = 0x3C6EF372FE94F82Bu64;
        for _ in 0..64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let m = 1 + (state % 64) as usize;
            let pattern: Vec<u8> = (0..m).map(|_| b'a' + (state % 26) as u8).collect();
            let texts: [Vec<u8>; 2] = std::array::from_fn(|i| {
                let len = (state.wrapping_mul(3 + i as u64) % 90) as usize;
                (0..len).map(|_| b'a' + (state.wrapping_mul(5 + i as u64) % 26) as u8).collect()
            });
            let pat = MyersPattern::new(&pattern);
            // SAFETY: NEON is mandatory on aarch64.
            let neon = unsafe { levenshtein_myers_2way_neon(&pat, [&texts[0], &texts[1]]) };
            for i in 0..2 {
                assert_eq!(neon[i], levenshtein_myers(&pattern, &texts[i]), "lane {i}");
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_jaro_2way_neon_matches_bitpar() {
        let mut state = 0xA54FF53A5F1D36F1u64;
        for _ in 0..64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let m = 1 + (state % 64) as usize;
            let a: Vec<u8> = (0..m).map(|_| b'a' + (state % 26) as u8).collect();
            let texts: [Vec<u8>; 2] = std::array::from_fn(|i| {
                let n = (state.wrapping_mul(7 + i as u64) % 65) as usize;
                (0..n).map(|_| b'a' + (state.wrapping_mul(11 + i as u64) % 26) as u8).collect()
            });
            // SAFETY: NEON is mandatory on aarch64.
            let neon = unsafe { jaro_2way_neon(&a, [&texts[0], &texts[1]]) };
            for i in 0..2 {
                let expect = jaro_bitpar(&a, &texts[i]);
                assert!((neon[i] - expect).abs() < 1e-12, "lane {i}: {} != {}", neon[i], expect);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_myers_4way_avx2_matches_portable() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut state = 0xB7E151628AED2A6Bu64;
        for _ in 0..32 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let m = 1 + (state % 64) as usize;
            let pattern: Vec<u8> = (0..m).map(|_| b'a' + (state % 26) as u8).collect();
            let texts: [Vec<u8>; 4] = std::array::from_fn(|i| {
                let len = (state.wrapping_mul(3 + i as u64) % 100) as usize;
                (0..len).map(|_| b'a' + (state.wrapping_mul(5 + i as u64) % 26) as u8).collect()
            });
            let pat = MyersPattern::new(&pattern);
            let portable = levenshtein_myers_4way_portable(&pat, [
                &texts[0], &texts[1], &texts[2], &texts[3],
            ]);
            // SAFETY: guarded by is_x86_feature_detected! above.
            let avx2 = unsafe { levenshtein_myers_4way_avx2(&pat, [
                &texts[0], &texts[1], &texts[2], &texts[3],
            ]) };
            assert_eq!(portable, avx2);
        }
    }

    #[test]
    fn test_myers_4way_64_char_pattern_boundary() {
        // The m == 64 mask boundary (u64::MAX) must not wrap the shift.
        let pattern: Vec<u8> = (0..64).map(|i| b'a' + (i % 26)).collect();
        let texts: [Vec<u8>; 4] = [
            (0..64).map(|i| b'b' + (i % 26)).collect(),
            (0..63).map(|i| b'c' + (i % 26)).collect(),
            (0..65).map(|i| b'a' + (i % 26)).collect(),
            (0..128).map(|i| b'b' + (i % 26)).collect(),
        ];
        let got = levenshtein_myers_4way(&pattern, [
            &texts[0], &texts[1], &texts[2], &texts[3],
        ]);
        for i in 0..4 {
            assert_eq!(got[i], levenshtein_myers(&pattern, &texts[i]), "lane {}", i);
        }
    }

    #[test]
    fn test_myers_fallback_long_strings() {
        let a: Vec<u8> = (0..100).map(|i| b'a' + (i % 26)).collect();
        let b: Vec<u8> = (0..100).map(|i| b'b' + (i % 26)).collect();
        // Should fall back to DP for > 64 chars. Just verify it doesn't panic.
        let _ = levenshtein_myers(&a, &b);
    }

    #[test]
    #[should_panic(expected = "levenshtein_myers requires ASCII inputs")]
    fn test_myers_rejects_non_ascii() {
        let _ = levenshtein_myers("café".as_bytes(), b"cafe");
    }

    #[test]
    fn test_jaro_bitpar_matches_reference() {
        use crate::jaro;
        let cases = [
            ("MARTHA", "MARHTA"),
            ("DIXON", "DICKSONX"),
            ("kitten", "sitting"),
            ("abc", ""),
            ("", "xyz"),
            ("", ""),
            ("same", "same"),
            ("a", "a"),
            ("ab", "ba"),
            ("dwayne", "duane"),
        ];
        for (a, b) in cases {
            let got = jaro_bitpar(a.as_bytes(), b.as_bytes());
            let expect = jaro(a, b);
            assert!(
                (got - expect).abs() < 1e-12,
                "bitpar mismatch for {a:?} vs {b:?}: {got} != {expect}"
            );
        }
    }

    #[test]
    fn test_jaro_bitpar_random_matches_reference() {
        use crate::jaro;
        let mut state = 0xC2B2AE3D27D4EB4Fu64;
        for _ in 0..256 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let m = (state % 65) as usize;
            let a: Vec<u8> = (0..m).map(|_| b'a' + (state.wrapping_mul(31) % 26) as u8).collect();
            let n = (state.wrapping_mul(97) % 65) as usize;
            let b: Vec<u8> = (0..n).map(|_| b'a' + (state.wrapping_mul(17) % 26) as u8).collect();
            let oa = String::from_utf8(a.clone()).unwrap();
            let ob = String::from_utf8(b.clone()).unwrap();
            let got = jaro_bitpar(&a, &b);
            let expect = jaro(&oa, &ob);
            assert!(
                (got - expect).abs() < 1e-12,
                "random mismatch: {:?} vs {:?}: {got} != {expect}",
                oa, ob
            );
        }
    }

    #[test]
    fn test_jaro_4way_matches_bitpar() {
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut gen = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..128 {
            let m = 1 + (gen() % 64) as usize;
            let a: Vec<u8> = (0..m).map(|_| b'a' + (gen() % 26) as u8).collect();
            let texts: [Vec<u8>; 4] = std::array::from_fn(|i| {
                let n = [0usize, 1, 40, 63][i];
                (0..n).map(|_| b'a' + (gen() % 26) as u8).collect()
            });
            let got = jaro_4way(&a, [&texts[0], &texts[1], &texts[2], &texts[3]]);
            for i in 0..4 {
                let expect = jaro_bitpar(&a, &texts[i]);
                assert!((got[i] - expect).abs() < 1e-12, "lane {i} mismatch: {} != {}", got[i], expect);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_jaro_4way_avx2_matches_bitpar() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut state = 0xB7E151628AED2A6Bu64;
        for _ in 0..64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let m = 1 + (state % 64) as usize;
            let a: Vec<u8> = (0..m).map(|_| b'a' + (state % 26) as u8).collect();
            let texts: [Vec<u8>; 4] = std::array::from_fn(|i| {
                let n = (state.wrapping_mul(3 + i as u64) % 65) as usize;
                (0..n).map(|_| b'a' + (state.wrapping_mul(5 + i as u64) % 26) as u8).collect()
            });
            // SAFETY: guarded by is_x86_feature_detected! above.
            let avx2 = unsafe { jaro_4way_avx2(&a, [&texts[0], &texts[1], &texts[2], &texts[3]]) };
            for i in 0..4 {
                let expect = jaro_bitpar(&a, &texts[i]);
                assert!((avx2[i] - expect).abs() < 1e-12, "lane {i} mismatch: {} != {}", avx2[i], expect);
            }
        }
    }

    #[test]
    fn test_jaro_optimized_matches_reference() {
        use crate::jaro;
        // Stack path (≤ 128 bytes).
        for (a, b) in [("MARTHA", "MARHTA"), ("DIXON", "DICKSONX"), ("kitten", "sitting")] {
            assert!((jaro_optimized(a.as_bytes(), b.as_bytes()) - jaro(a, b)).abs() < 1e-12, "{} vs {}", a, b);
        }
        // Heap path (> 128 bytes) — must agree with the reference too.
        let long_a: Vec<u8> = (0..200).map(|i| b'a' + (i % 26)).collect();
        let long_b: Vec<u8> = (0..200).map(|i| b'b' + (i % 26)).collect();
        let oa = String::from_utf8(long_a).unwrap();
        let ob = String::from_utf8(long_b).unwrap();
        let simd = jaro_optimized(oa.as_bytes(), ob.as_bytes());
        let reference = jaro(&oa, &ob);
        assert!((simd - reference).abs() < 1e-12, "heap path mismatch: {} vs {}", simd, reference);
    }

    #[test]
    fn test_needleman_wunsch_striped_matches_reference() {
        use crate::needleman_wunsch;
        let cases = [("AGCT", "AGCT"), ("kitten", "sitting"), ("ACGT", "AT"), ("hello", "world")];
        for (a, b) in cases {
            assert_eq!(
                needleman_wunsch_striped(a.as_bytes(), b.as_bytes(), 2, -1, -1),
                needleman_wunsch(a, b, 2, -1, -1),
                "{} vs {}", a, b
            );
        }
    }
}
