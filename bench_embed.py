import time
import argparse
import tracemalloc
import statistics
from embedding_search import EmbeddingCache
from tqdm import trange


def measure_once(text: str) -> dict:
    tracemalloc.start()

    t0 = time.perf_counter()
    cache = EmbeddingCache()
    emb = cache.compute_embedding(text)
    t1 = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "time_s": t1 - t0,
        "peak_bytes": peak,
        "vector_len": len(emb),
    }


def main():
    p = argparse.ArgumentParser(description="Benchmark embedding computation")
    p.add_argument("--text", "-t", default="hello world", help="Text to embed")
    p.add_argument("--iters", "-n", type=int, default=100, help="Iterations")
    args = p.parse_args()

    results = []
    print(f"Running {args.iters} iterations for text: {args.text!r}")
    for i in trange(args.iters):
        r = measure_once(args.text)
        results.append(r)
        print(
            f"iter {i + 1}: time={r['time_s']:.4f}s peak={r['peak_bytes'] / 1024:.1f} KiB vec={r['vector_len']}"
        )

    times = [r["time_s"] for r in results]
    peaks = [r["peak_bytes"] for r in results]
    print("\nSummary:")
    print(
        f"  mean time: {statistics.mean(times):.4f}s, stdev: {statistics.stdev(times) if len(times) > 1 else 0:.4f}s"
    )
    print(f"  mean peak memory: {statistics.mean(peaks) / 1024:.1f} KiB")


if __name__ == "__main__":
    main()
