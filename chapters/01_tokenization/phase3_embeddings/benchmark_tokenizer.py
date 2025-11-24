"""
Benchmark script to test tokenization performance improvements.

Tests the optimized BPETokenizerLoader against different text sizes.
"""

import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from train_embeddings import BPETokenizerLoader

def format_time(seconds):
    """Format seconds into human-readable string."""
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"

def benchmark_tokenizer(tokenizer, text, name="", num_runs=3):
    """Benchmark tokenizer performance."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"Text size: {len(text):,} characters, {len(text.split()):,} words")
    print(f"{'='*60}")
    
    times = []
    token_count = 0
    
    # Warmup run
    print("Warming up...")
    _ = tokenizer.encode(text)
    
    # Timed runs
    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}...", end=" ", flush=True)
        start = time.time()
        tokens = tokenizer.encode(text)
        elapsed = time.time() - start
        times.append(elapsed)
        token_count = len(tokens)
        print(f"{format_time(elapsed)}")
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    # Calculate throughput
    words = len(text.split())
    words_per_sec = words / avg_time
    chars_per_sec = len(text) / avg_time
    
    print(f"\n{'Results:':-^60}")
    print(f"  Average time:      {format_time(avg_time)}")
    print(f"  Min time:          {format_time(min_time)}")
    print(f"  Max time:          {format_time(max_time)}")
    print(f"  Tokens generated:  {token_count:,}")
    print(f"  Throughput:        {words_per_sec:,.0f} words/sec")
    print(f"                     {chars_per_sec:,.0f} chars/sec")
    print(f"  Tokens/word:       {token_count/words:.2f}")
    
    # Cache stats
    cache_info = tokenizer.tokenize_word.cache_info()
    print(f"\n{'Cache Statistics:':-^60}")
    print(f"  Hits:              {cache_info.hits:,}")
    print(f"  Misses:            {cache_info.misses:,}")
    print(f"  Cache size:        {cache_info.currsize:,} words")
    if cache_info.hits + cache_info.misses > 0:
        hit_rate = cache_info.hits / (cache_info.hits + cache_info.misses) * 100
        print(f"  Hit rate:          {hit_rate:.1f}%")
    
    return {
        'avg_time': avg_time,
        'min_time': min_time,
        'tokens': token_count,
        'words_per_sec': words_per_sec,
        'cache_hits': cache_info.hits,
        'cache_misses': cache_info.misses,
        'hit_rate': cache_info.hits / (cache_info.hits + cache_info.misses) * 100 if cache_info.hits + cache_info.misses > 0 else 0
    }

def main():
    print("\n" + "="*60)
    print("BPE TOKENIZER OPTIMIZATION BENCHMARK")
    print("="*60)
    
    # Load tokenizer
    tokenizer_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'processed' / '01_tokenization'
    
    if not tokenizer_dir.exists():
        print(f"\nError: Tokenizer directory not found: {tokenizer_dir}")
        print("Please train a BPE tokenizer first.")
        return
    
    print(f"\nLoading tokenizer from: {tokenizer_dir}")
    tokenizer = BPETokenizerLoader.load(tokenizer_dir)
    
    # Load test corpus
    corpus_path = Path(__file__).parent.parent.parent.parent / 'data' / 'raw' / '01_tokenization' / 'frankenstein.txt'
    
    if not corpus_path.exists():
        print(f"\nError: Corpus not found: {corpus_path}")
        return
    
    print(f"Loading corpus from: {corpus_path}")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        full_text = f.read()
    
    # Test 1: Small text (1000 words)
    words = full_text.split()
    small_text = ' '.join(words[:1000])
    results_small = benchmark_tokenizer(tokenizer, small_text, "Small Text (1K words)", num_runs=5)
    
    # Clear cache between tests
    tokenizer.clear_cache()
    
    # Test 2: Medium text (10,000 words)
    medium_text = ' '.join(words[:10000])
    results_medium = benchmark_tokenizer(tokenizer, medium_text, "Medium Text (10K words)", num_runs=3)
    
    # Clear cache between tests
    tokenizer.clear_cache()
    
    # Test 3: Full text
    results_full = benchmark_tokenizer(tokenizer, full_text, "Full Frankenstein", num_runs=3)
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Test':<20} {'Time':<15} {'Throughput':<20} {'Hit Rate':<12}")
    print("-" * 70)
    print(f"{'Small (1K words)':<20} {format_time(results_small['avg_time']):<15} {results_small['words_per_sec']:>10,.0f} w/s    {results_small['hit_rate']:>8.1f}%")
    print(f"{'Medium (10K words)':<20} {format_time(results_medium['avg_time']):<15} {results_medium['words_per_sec']:>10,.0f} w/s    {results_medium['hit_rate']:>8.1f}%")
    print(f"{'Full (~78K words)':<20} {format_time(results_full['avg_time']):<15} {results_full['words_per_sec']:>10,.0f} w/s    {results_full['hit_rate']:>8.1f}%")
    
    print(f"\n{'Key Optimizations Applied:':-^60}")
    print("  ✓ LRU cache for word tokenization (100K cache size)")
    print("  ✓ Pre-compiled regex patterns")
    print("  ✓ Merge rank lookup (O(1) instead of O(n))")
    print("  ✓ Optimized merge algorithm")
    print("  ✓ Unknown token caching")
    print("  ✓ Local variable references")
    
    print(f"\n{'Expected Speedup for WikiText-2:':-^60}")
    print(f"  With {results_full['hit_rate']:.1f}% cache hit rate observed")
    print(f"  WikiText-2 (~2M words) estimated time: {format_time(2_000_000 / results_full['words_per_sec'])}")
    print(f"  (vs. ~10-30 minutes with unoptimized version)")
    print()

if __name__ == "__main__":
    main()

