//! Benchmark harness for fuzzgpu-core.
//!
//! CPU benches always run; GPU benches run when the `gpu` feature is enabled
//! (default) *and* a usable device is present — otherwise they print a note
//! and bench a no-op so the suite still completes (mirroring the test suite's
//! require/skip contract). GPU benches dispatch single-threaded (criterion
//! runs one bench at a time), which is the safe regime for gfx-rs/wgpu#10085.
//!
//! Run: `cargo bench -p fuzzgpu-core` (release profile, real numbers).

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use fuzzgpu_core::damerau::damerau_levenshtein_batch;
use fuzzgpu_core::fuzz::ratio;
use fuzzgpu_core::jaro::{jaro_winkler, jaro_winkler_batch};
use fuzzgpu_core::levenshtein::{
    levenshtein_cdist_cpu, levenshtein_distance_raw, LevenshteinKernel,
};
use fuzzgpu_core::needleman::{needleman_wunsch_affine, needleman_wunsch_affine_batch};
use fuzzgpu_core::simd::levenshtein_myers;

/// Deterministic ASCII strings (LCG, same generator as the test suite).
fn gen_strings(count: usize, len: usize, seed: u64) -> Vec<String> {
    let mut state = seed;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let mut s = String::with_capacity(len);
        for _ in 0..len {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            s.push((b'a' + ((state >> 33) as u8 % 26)) as char);
        }
        out.push(s);
    }
    out
}

fn make_pairs<'a>(q: &'a str, cands: &'a [String]) -> Vec<(&'a str, &'a str)> {
    cands.iter().map(|c| (q, c.as_str())).collect()
}

const MATCH: i64 = 1;
const MISMATCH: i64 = -1;
const GAP_OPEN: i64 = -2;
const GAP_EXTEND: i64 = -1;

fn cpu_benches(c: &mut Criterion) {
    let short = gen_strings(2, 8, 1);
    let medium = gen_strings(2, 64, 2);
    let long = gen_strings(2, 256, 3);

    let mut g = c.benchmark_group("scalar");
    for (name, s) in [
        ("short_8", &short),
        ("medium_64", &medium),
        ("long_256", &long),
    ] {
        let (s1, s2) = (&s[0], &s[1]);
        g.bench_function(format!("levenshtein/{name}"), |b| {
            b.iter(|| black_box(levenshtein_distance_raw(s1, s2)))
        });
        g.bench_function(format!("levenshtein_myers/{name}"), |b| {
            b.iter(|| black_box(levenshtein_myers(s1.as_bytes(), s2.as_bytes())))
        });
        g.bench_function(format!("jaro_winkler/{name}"), |b| {
            b.iter(|| black_box(jaro_winkler(s1, s2, 0.1)))
        });
        g.bench_function(format!("ratio/{name}"), |b| {
            b.iter(|| black_box(ratio(s1, s2)))
        });
        g.bench_function(format!("needleman_affine/{name}"), |b| {
            b.iter(|| {
                black_box(needleman_wunsch_affine(
                    s1, s2, MATCH, MISMATCH, GAP_OPEN, GAP_EXTEND,
                ))
            })
        });
        g.bench_function(format!("damerau/{name}"), |b| {
            b.iter(|| black_box(fuzzgpu_core::damerau::damerau_levenshtein_distance(s1, s2)))
        });
    }
    g.finish();

    // Batch + matrix workloads (1000 pairs / 200x200 grid).
    let cands = gen_strings(1000, 16, 4);
    let query = "benchmark-query-string";
    let pairs = make_pairs(query, &cands);
    let list_a = gen_strings(200, 16, 5);
    let list_b = gen_strings(200, 16, 6);
    let refs_a: Vec<&str> = list_a.iter().map(|s| s.as_str()).collect();
    let refs_b: Vec<&str> = list_b.iter().map(|s| s.as_str()).collect();
    let cand_refs: Vec<&str> = cands.iter().map(|s| s.as_str()).collect();

    let mut g = c.benchmark_group("cpu_batch");
    let kernel = LevenshteinKernel;
    g.bench_function("levenshtein_batch_1000", |b| {
        b.iter(|| black_box(kernel.compute(&pairs)))
    });
    g.bench_function("jaro_winkler_batch_1000", |b| {
        b.iter(|| black_box(jaro_winkler_batch(query, &cand_refs, 0.1)))
    });
    g.bench_function("needleman_affine_batch_1000", |b| {
        b.iter(|| {
            black_box(needleman_wunsch_affine_batch(
                query, &cand_refs, MATCH, MISMATCH, GAP_OPEN, GAP_EXTEND,
            ))
        })
    });
    // Long-string needleman (80 chars): the anti-diagonal wavefront GPU path's
    // target workload; Rayon CPU is the comparison baseline.
    let cands_nw80 = gen_strings(1000, 80, 0xAB);
    let cand_refs_nw80: Vec<&str> = cands_nw80.iter().map(|s| s.as_str()).collect();
    g.bench_function("needleman_affine_batch_1000_long80_large", |b| {
        b.iter(|| {
            black_box(needleman_wunsch_affine_batch(
                query,
                &cand_refs_nw80,
                MATCH,
                MISMATCH,
                GAP_OPEN,
                GAP_EXTEND,
            ))
        })
    });
    g.bench_function("damerau_batch_1000", |b| {
        b.iter(|| black_box(damerau_levenshtein_batch(query, &cand_refs)))
    });
    g.bench_function("levenshtein_cdist_200x200", |b| {
        b.iter(|| black_box(levenshtein_cdist_cpu(&refs_a, &refs_b)))
    });
    g.finish();

    // Large-scale workloads: where GPU parallelism has a chance to beat the
    // sync round-trip floor + Rayon (100k pairs, 256-char strings, 1M-cell grid).
    let cands_100k = gen_strings(100_000, 16, 7);
    let pairs_100k = make_pairs(query, &cands_100k);
    let cands_long = gen_strings(1000, 256, 8);
    let pairs_long = make_pairs(query, &cands_long);
    let list_a_1k = gen_strings(1000, 16, 9);
    let list_b_1k = gen_strings(1000, 16, 10);
    let refs_a_1k: Vec<&str> = list_a_1k.iter().map(|s| s.as_str()).collect();
    let refs_b_1k: Vec<&str> = list_b_1k.iter().map(|s| s.as_str()).collect();

    let mut g = c.benchmark_group("cpu_batch");
    g.bench_function("levenshtein_batch_100k_large", |b| {
        b.iter(|| black_box(kernel.compute(&pairs_100k)))
    });
    g.bench_function("levenshtein_batch_1000_long256_large", |b| {
        b.iter(|| black_box(kernel.compute(&pairs_long)))
    });
    g.bench_function("levenshtein_cdist_1000x1000_large", |b| {
        b.iter(|| black_box(levenshtein_cdist_cpu(&refs_a_1k, &refs_b_1k)))
    });
    // Sequential 10 x 1000 pairs: the CPU baseline for the batched-GPU bench.
    let cands10: Vec<Vec<String>> = (0..10)
        .map(|i| gen_strings(1000, 16, 0x100 + i as u64))
        .collect();
    let ops10: Vec<Vec<(&str, &str)>> = cands10.iter().map(|c| make_pairs(query, c)).collect();
    g.bench_function("levenshtein_10x1000_seq_cpu", |b| {
        b.iter(|| {
            for op in &ops10 {
                black_box(kernel.compute(op).expect("cpu batch"));
            }
        })
    });
    g.finish();
}

#[cfg(feature = "gpu")]
fn gpu_benches(c: &mut Criterion) {
    use fuzzgpu_core::damerau::gpu_ext::GpuDamerauKernel;
    use fuzzgpu_core::jaro::gpu_ext::GpuJaroKernel;
    use fuzzgpu_core::levenshtein::gpu_ext::GpuLevenshteinKernel;
    use fuzzgpu_core::needleman::gpu_ext::GpuNeedlemanAffineKernel;

    let cands = gen_strings(1000, 16, 4);
    let query = "benchmark-query-string";
    let pairs = make_pairs(query, &cands);
    let list_a = gen_strings(200, 16, 5);
    let list_b = gen_strings(200, 16, 6);
    let refs_a: Vec<&str> = list_a.iter().map(|s| s.as_str()).collect();
    let refs_b: Vec<&str> = list_b.iter().map(|s| s.as_str()).collect();

    // Same workload shapes as the cpu_batch group, so the GPU numbers are
    // directly comparable. Strings are 16 chars (< 256) and batches are
    // >= 500 pairs, so the kernels take the GPU dispatch path (no auto-routing
    // to Rayon), and `compute`'s empty/identical short-circuits don't trigger.
    let mut g = c.benchmark_group("gpu_batch");
    match GpuLevenshteinKernel::get() {
        Ok(kernel) => {
            g.bench_function("levenshtein_batch_1000", |b| {
                b.iter(|| black_box(kernel.compute(&pairs)))
            });
            g.bench_function("levenshtein_cdist_200x200", |b| {
                b.iter(|| black_box(kernel.compute_matrix(&refs_a, &refs_b)))
            });
        }
        Err(e) => {
            eprintln!("WARN: no GPU device, skipping GpuLevenshteinKernel benches: {e}");
            g.bench_function("levenshtein_batch_1000", |b| b.iter(|| black_box(0)));
            g.bench_function("levenshtein_cdist_200x200", |b| b.iter(|| black_box(0)));
        }
    }
    match GpuJaroKernel::get() {
        Ok(kernel) => {
            g.bench_function("jaro_winkler_batch_1000", |b| {
                b.iter(|| black_box(kernel.compute_batch(&pairs, 0.1)))
            });
        }
        Err(e) => {
            eprintln!("WARN: no GPU device, skipping GpuJaroKernel benches: {e}");
            g.bench_function("jaro_winkler_batch_1000", |b| b.iter(|| black_box(0)));
        }
    }
    match GpuNeedlemanAffineKernel::get() {
        Ok(kernel) => {
            g.bench_function("needleman_affine_batch_1000", |b| {
                b.iter(|| {
                    black_box(kernel.compute_batch(&pairs, MATCH, MISMATCH, GAP_OPEN, GAP_EXTEND))
                })
            });
            // 80-char pairs: routes to the anti-diagonal wavefront kernel.
            let cands_nw80 = gen_strings(1000, 80, 0xAB);
            let pairs_nw80 = make_pairs(query, &cands_nw80);
            g.bench_function("needleman_affine_batch_1000_long80_large", |b| {
                b.iter(|| {
                    black_box(kernel.compute_batch(
                        &pairs_nw80,
                        MATCH,
                        MISMATCH,
                        GAP_OPEN,
                        GAP_EXTEND,
                    ))
                })
            });
        }
        Err(e) => {
            eprintln!("WARN: no GPU device, skipping GpuNeedlemanAffineKernel benches: {e}");
            g.bench_function("needleman_affine_batch_1000", |b| b.iter(|| black_box(0)));
        }
    }
    match GpuDamerauKernel::get() {
        Ok(kernel) => {
            // 16-char candidates + 22-char query, both <= the 32-char cap, so
            // every pair runs the Lowrance-Wagner SLM kernel.
            g.bench_function("damerau_batch_1000", |b| {
                b.iter(|| black_box(kernel.compute_batch(&pairs)))
            });
            g.bench_function("damerau_cdist_200x200", |b| {
                b.iter(|| black_box(kernel.compute_matrix(&refs_a, &refs_b)))
            });
        }
        Err(e) => {
            eprintln!("WARN: no GPU device, skipping GpuDamerauKernel benches: {e}");
            g.bench_function("damerau_batch_1000", |b| b.iter(|| black_box(0)));
            g.bench_function("damerau_cdist_200x200", |b| b.iter(|| black_box(0)));
        }
    }
    g.finish();

    // Large-scale GPU workloads (mirror the cpu_batch "_large" group).
    let cands_100k = gen_strings(100_000, 16, 7);
    let pairs_100k = make_pairs(query, &cands_100k);
    let cands_long = gen_strings(1000, 256, 8);
    let pairs_long = make_pairs(query, &cands_long);
    let list_a_1k = gen_strings(1000, 16, 9);
    let list_b_1k = gen_strings(1000, 16, 10);
    let refs_a_1k: Vec<&str> = list_a_1k.iter().map(|s| s.as_str()).collect();
    let refs_b_1k: Vec<&str> = list_b_1k.iter().map(|s| s.as_str()).collect();

    let mut g = c.benchmark_group("gpu_batch");
    match GpuLevenshteinKernel::get() {
        Ok(kernel) => {
            g.bench_function("levenshtein_batch_100k_large", |b| {
                b.iter(|| black_box(kernel.compute(&pairs_100k)))
            });
            g.bench_function("levenshtein_batch_1000_long256_large", |b| {
                b.iter(|| black_box(kernel.compute(&pairs_long)))
            });
            g.bench_function("levenshtein_cdist_1000x1000_large", |b| {
                b.iter(|| black_box(kernel.compute_matrix(&refs_a_1k, &refs_b_1k)))
            });
            // 10 x 1000 pairs: one batched dispatch+readback vs 10 sequential
            // sync-round-trips. Same total work as cpu_batch/..._seq_cpu.
            let cands10: Vec<Vec<String>> = (0..10)
                .map(|i| gen_strings(1000, 16, 0x100 + i as u64))
                .collect();
            let ops10: Vec<Vec<(&str, &str)>> =
                cands10.iter().map(|c| make_pairs(query, c)).collect();
            g.bench_function("levenshtein_10x1000_seq", |b| {
                b.iter(|| {
                    for op in &ops10 {
                        black_box(kernel.compute(op).expect("gpu seq"));
                    }
                })
            });
            g.bench_function("levenshtein_10x1000_batched", |b| {
                b.iter(|| {
                    let mut batch = kernel.batch();
                    for op in &ops10 {
                        batch.add(op);
                    }
                    black_box(batch.execute().expect("gpu batch"))
                })
            });
        }
        Err(e) => {
            eprintln!(
                "WARN: no GPU device, skipping large-scale GpuLevenshteinKernel benches: {e}"
            );
            g.bench_function("levenshtein_batch_100k_large", |b| b.iter(|| black_box(0)));
            g.bench_function("levenshtein_batch_1000_long256_large", |b| {
                b.iter(|| black_box(0))
            });
            g.bench_function("levenshtein_cdist_1000x1000_large", |b| {
                b.iter(|| black_box(0))
            });
            g.bench_function("levenshtein_10x1000_seq", |b| b.iter(|| black_box(0)));
            g.bench_function("levenshtein_10x1000_batched", |b| b.iter(|| black_box(0)));
        }
    }
    match GpuDamerauKernel::get() {
        Ok(kernel) => {
            g.bench_function("damerau_batch_100k_large", |b| {
                b.iter(|| black_box(kernel.compute_batch(&pairs_100k)))
            });
            g.bench_function("damerau_cdist_1000x1000_large", |b| {
                b.iter(|| black_box(kernel.compute_matrix(&refs_a_1k, &refs_b_1k)))
            });
        }
        Err(e) => {
            eprintln!("WARN: no GPU device, skipping large-scale GpuDamerauKernel benches: {e}");
            g.bench_function("damerau_batch_100k_large", |b| b.iter(|| black_box(0)));
            g.bench_function("damerau_cdist_1000x1000_large", |b| b.iter(|| black_box(0)));
        }
    }
    g.finish();
}

criterion_group!(cpu_group, cpu_benches);

#[cfg(feature = "gpu")]
criterion_group!(gpu_group, gpu_benches);

#[cfg(feature = "gpu")]
criterion_main!(cpu_group, gpu_group);

#[cfg(not(feature = "gpu"))]
criterion_main!(cpu_group);
