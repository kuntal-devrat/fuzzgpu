"""Concurrency, thread-safety, and multi-threaded stress tests for fuzzgpu."""

import concurrent.futures
import random
import string
import pytest
import fuzzgpu
import fuzzgpu.fuzz as fuzz


def random_string(min_len=5, max_len=25):
    length = random.randint(min_len, max_len)
    return "".join(random.choices(string.ascii_lowercase, k=length))


class TestConcurrencyAndThreadSafety:
    """Test thread-safety when multiple worker threads execute GPU and CPU workloads concurrently."""

    def test_concurrent_gpu_batch_dispatches(self):
        """16 worker threads concurrently dispatching GPU batch levenshtein queries."""
        def worker_task(thread_id):
            random.seed(thread_id * 1000)
            query = random_string(10, 20)
            candidates = [random_string(8, 22) for _ in range(600)]
            results = fuzzgpu.levenshtein_batch(query, candidates)
            # Verify results match CPU ground truth
            for idx in range(0, 600, 50):
                expected = fuzzgpu.levenshtein(query, candidates[idx])
                assert results[idx] == expected, f"Thread {thread_id} mismatch at {idx}"
            return len(results)

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(worker_task, i) for i in range(32)]
            for future in concurrent.futures.as_completed(futures):
                count = future.result()
                assert count == 600

    def test_concurrent_mixed_gpu_and_cpu_workloads(self):
        """Concurrently run Levenshtein GPU, Jaro-Winkler GPU, Matrix CDist, and Fuzzy Extract."""
        choices = [f"candidate_item_{i}" for i in range(1000)]

        def task_levenshtein():
            return fuzzgpu.levenshtein_batch("candidate_item_500", choices)

        def task_jaro():
            return fuzzgpu.jaro_winkler_batch("candidate_item_500", choices, 0.1)

        def task_matrix():
            sub_a = choices[:25]
            sub_b = choices[25:50]
            return fuzzgpu.levenshtein_cdist(sub_a, sub_b)

        def task_extract():
            return fuzz.extract("candidate_item_500", choices, score_cutoff=80.0, limit=5)

        tasks = [task_levenshtein, task_jaro, task_matrix, task_extract] * 10

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(fn) for fn in tasks]
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                assert res is not None
