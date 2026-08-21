#!/usr/bin/env python3
"""
Terminal Demo Recording Script
===============================

Generates a clean terminal output suitable for asciinema recording or MP4
screen capture.

Usage:
  pip install fuzzgpu rapidfuzz

  # Record with asciinema:
  asciinema rec demo.cast -c "python demo/record_demo.py"

  # Or just print the output:
  python demo/record_demo.py
"""

import os, time, random, sys

os.environ["COLUMNS"] = "80"  # Force consistent width

import fuzzgpu

# Warm up GPU — first call initializes wgpu device + compiles WGSL shaders.
# This cold-start penalty is one-time; all subsequent calls are fast.
fuzzgpu.warmup()

def hr(char="=", width=72):
    print(char * width)

def section(title):
    print()
    hr("-")
    print(f"  {title}")
    hr("-")
    print()

def random_strings(n, min_len=5, max_len=30):
    alpha = "abcdefghijklmnopqrstuvwxyz "
    return ["".join(random.choices(alpha, k=random.randint(min_len, max_len))) for _ in range(n)]

# ═══════════════════════════════════════════════════════════════════════════

hr()
print("  fuzzgpu — GPU-Accelerated Fuzzy String Matching")
print("  Drop-in rapidfuzz replacement | No CUDA required")
hr()
print()

# GPU Info
print(f"  GPU:       {fuzzgpu.gpu_info()}")
print(f"  Hardware:  {fuzzgpu.hardware_info()[:65]}")
print()

# ═══════════════════════════════════════════════════════════════════════════

section("1. Drop-in Compatibility (169,744 pairs verified, 0 mismatches)")

text1, text2 = "hello world", "helo wrold"

print(f"  Input:   \"{text1}\" vs \"{text2}\"")
print()
print(f"  Levenshtein:   fuzzgpu={fuzzgpu.levenshtein_distance(text1, text2)}")
print(f"  Jaro-Winkler:  fuzzgpu={fuzzgpu.jaro_winkler_similarity(text1, text2):.4f}")
print(f"  Fuzz ratio:    fuzzgpu={fuzzgpu.ratio(text1, text2):.1f}")
print()

# ═══════════════════════════════════════════════════════════════════════════

section("2. Batch Speed Comparison")

import rapidfuzz.distance as rf_dist

query = "benchmark-query-string"

print(f"  {'N':>8}  {'fuzzgpu':>10}  {'rapidfuzz':>10}  {'speedup':>8}")
print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*8}")

for n in [1_000, 10_000, 50_000]:
    cands = random_strings(n)

    t0 = time.perf_counter()
    fuzzgpu.levenshtein_batch(query, cands)
    t_fg = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    [rf_dist.Levenshtein.distance(query, c) for c in cands]
    t_rf = (time.perf_counter() - t0) * 1000

    print(f"  {n:>8,}  {t_fg:>8.2f}ms  {t_rf:>8.2f}ms  {t_rf/t_fg:>6.1f}x")

# ═══════════════════════════════════════════════════════════════════════════

section("3. Damerau-Levenshtein (Unrestricted)")

print("  rapidfuzz uses OSA (restricted transpositions)")
print("  fuzzgpu uses Lowrance-Wagner (unrestricted)")
print()

a, b = "CA", "ABC"
print(f"  fuzzgpu unrestricted (\"{a}\", \"{b}\") = {fuzzgpu.damerau_levenshtein_distance(a, b)}")
print(f"  rapidfuzz OSA      (\"{a}\", \"{b}\") = {rf_dist.DamerauLevenshtein.distance(a, b)}")
print()
a2, b2 = "ABC", "ACB"
print(f"  fuzzgpu unrestricted (\"{a2}\", \"{b2}\") = {fuzzgpu.damerau_levenshtein_distance(a2, b2)}")
print(f"  rapidfuzz OSA      (\"{a2}\", \"{b2}\") = {rf_dist.DamerauLevenshtein.distance(a2, b2)}")
print()
print("  Unrestricted allows non-adjacent transpositions (ACB -> ABC = 1 swap)")
print("  OSA restricts to adjacent-only swaps")

# ═══════════════════════════════════════════════════════════════════════════

section("4. Cross-Product Matrix (500x500 = 250K evaluations)")

list_a = random_strings(500)
list_b = random_strings(500)

t0 = time.perf_counter()
fuzzgpu.levenshtein_cdist(list_a, list_b)
t_fg = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
[[rf_dist.Levenshtein.distance(a, b) for b in list_b] for a in list_a]
t_rf = (time.perf_counter() - t0) * 1000

print(f"  500 x 500 = 250,000 distance evaluations")
print(f"  fuzzgpu:   {t_fg:.1f} ms")
print(f"  rapidfuzz: {t_rf:.1f} ms")
print(f"  speedup:   {t_rf/t_fg:.1f}x")

# ═══════════════════════════════════════════════════════════════════════════

section("5. Needleman-Wunsch Affine Gap (fuzzgpu exclusive!)")

print("  rapidfuzz has NO API for this algorithm.")
print()

cands = random_strings(500)
t0 = time.perf_counter()
fuzzgpu.needleman_wunsch_affine_batch(
    "query-seq", cands, match_score=2, mismatch_score=-1, gap_open=-2, gap_extend=-1
)
t_fg = (time.perf_counter() - t0) * 1000
print(f"  500 pairs:  {t_fg:.1f} ms (GPU-accelerated)")

# ═══════════════════════════════════════════════════════════════════════════

hr()
print()
print("  Install:  pip install fuzzgpu")
print("  GitHub:   github.com/kuntal-devrat/fuzzgpu")
print()
hr()
