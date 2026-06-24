#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inference Speed Benchmark: all-mpnet-base-v2 vs intfloat/e5-large-v2
Measures actual end-to-end encoding time on real queries
"""

import time
import json
import numpy as np
from typing import List, Dict, Tuple
import torch
from sentence_transformers import SentenceTransformer


def benchmark_model(
    model_name: str,
    queries: List[str],
    batch_sizes: List[int] = [1, 8, 16, 32, 64],
    n_runs: int = 5,
    device: str = "cuda",
) -> Dict[str, any]:
    """
    Benchmark a model across different batch sizes.
    
    Args:
        model_name: HuggingFace model identifier
        queries: List of test queries
        batch_sizes: Batch sizes to test
        n_runs: Number of runs per batch size (for averaging)
        device: 'cuda' or 'cpu'
    
    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'='*80}")
    print(f"Benchmarking: {model_name}")
    print(f"Device: {device.upper()}")
    print(f"{'='*80}\n")
    
    # Load model
    print("Loading model...")
    model = SentenceTransformer(model_name, device=device)
    model.eval()
    
    results = {
        "model_name": model_name,
        "device": device,
        "batch_sizes": {},
    }
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Prepare batches
        n_batches = max(1, len(queries) // batch_size)
        batches = [queries[i*batch_size:(i+1)*batch_size] for i in range(n_batches)]
        
        # Warmup (important for GPU!)
        print("  Warmup...", end=" ")
        for _ in range(3):
            _ = model.encode(batches[0], show_progress_bar=False, normalize_embeddings=True)
        print("✓")
        
        # Benchmark
        times = []
        queries_per_sec = []
        
        for run in range(n_runs):
            print(f"  Run {run+1}/{n_runs}...", end=" ")
            
            start = time.time()
            total_queries = 0
            
            for batch in batches:
                _ = model.encode(
                    batch,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
                total_queries += len(batch)
            
            elapsed = time.time() - start
            qps = total_queries / elapsed
            
            times.append(elapsed)
            queries_per_sec.append(qps)
            
            print(f"{elapsed:.3f}s ({qps:.1f} q/s)")
        
        # Statistics
        results["batch_sizes"][batch_size] = {
            "mean_time": float(np.mean(times)),
            "std_time": float(np.std(times)),
            "min_time": float(np.min(times)),
            "max_time": float(np.max(times)),
            "mean_qps": float(np.mean(queries_per_sec)),
            "std_qps": float(np.std(queries_per_sec)),
            "total_queries": total_queries,
            "n_runs": n_runs,
        }
        
        print(f"  → Mean: {np.mean(times):.3f}s ± {np.std(times):.3f}s")
        print(f"  → Throughput: {np.mean(queries_per_sec):.1f} ± {np.std(queries_per_sec):.1f} queries/sec")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    return results


def compare_speeds(results1: Dict, results2: Dict) -> Dict[str, any]:
    """
    Compare speeds between two models.
    """
    comparison = {
        "model1": results1["model_name"],
        "model2": results2["model_name"],
        "speedup_by_batch_size": {},
    }
    
    print(f"\n{'='*80}")
    print("SPEED COMPARISON")
    print(f"{'='*80}\n")
    print(f"Model 1: {results1['model_name']}")
    print(f"Model 2: {results2['model_name']}")
    print()
    
    for batch_size in results1["batch_sizes"].keys():
        if batch_size not in results2["batch_sizes"]:
            continue
        
        r1 = results1["batch_sizes"][batch_size]
        r2 = results2["batch_sizes"][batch_size]
        
        speedup = r1["mean_time"] / r2["mean_time"]
        qps_ratio = r2["mean_qps"] / r1["mean_qps"]
        
        comparison["speedup_by_batch_size"][batch_size] = {
            "model1_time": r1["mean_time"],
            "model2_time": r2["mean_time"],
            "model2_speedup": float(speedup),  # How much faster is model2?
            "model1_qps": r1["mean_qps"],
            "model2_qps": r2["mean_qps"],
            "qps_ratio": float(qps_ratio),
        }
        
        # Model 2 is slower if speedup < 1.0
        if speedup < 1.0:
            print(f"Batch {batch_size}:")
            print(f"  Model 1: {r1['mean_time']:.3f}s ({r1['mean_qps']:.1f} q/s)")
            print(f"  Model 2: {r2['mean_time']:.3f}s ({r2['mean_qps']:.1f} q/s)")
            print(f"  → Model 2 is {1/speedup:.2f}× SLOWER ({(1/speedup - 1)*100:.1f}% slower)")
        else:
            print(f"Batch {batch_size}:")
            print(f"  Model 1: {r1['mean_time']:.3f}s ({r1['mean_qps']:.1f} q/s)")
            print(f"  Model 2: {r2['mean_time']:.3f}s ({r2['mean_qps']:.1f} q/s)")
            print(f"  → Model 2 is {speedup:.2f}× FASTER")
        print()
    
    return comparison


def generate_speed_table_latex(comparison: Dict, results1: Dict, results2: Dict) -> str:
    """
    Generate LaTeX table for speed comparison.
    """
    latex = []
    latex.append("\\begin{table}[t]")
    latex.append("\\centering")
    latex.append("\\caption{Inference speed comparison (NVIDIA T4 GPU, CounselChat queries).}")
    latex.append("\\label{tab:inference_speed}")
    latex.append("\\begin{tabular}{lccccc}")
    latex.append("\\toprule")
    latex.append("\\textbf{Batch} & \\textbf{mpnet Time} & \\textbf{E5 Time} & \\textbf{mpnet QPS} & \\textbf{E5 QPS} & \\textbf{Speedup} \\\\")
    latex.append("\\textbf{Size} & \\textbf{(sec)} & \\textbf{(sec)} & & & \\textbf{(mpnet/E5)} \\\\")
    latex.append("\\midrule")
    
    for bs in sorted(comparison["speedup_by_batch_size"].keys()):
        data = comparison["speedup_by_batch_size"][bs]
        latex.append(
            f"{bs} & "
            f"{data['model1_time']:.3f} & "
            f"{data['model2_time']:.3f} & "
            f"{data['model1_qps']:.1f} & "
            f"{data['model2_qps']:.1f} & "
            f"{1/data['model2_speedup']:.2f}× \\\\"
        )
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def main():
    # Load sample queries from CounselChat
    print("Loading sample queries...")
    
    # Use actual CounselChat queries (you can load from your data)
    # For now, let's create representative mental health queries
    sample_queries = [
        "How do I deal with anxiety and depression?",
        "I feel very lonely and isolated",
        "My relationship is falling apart",
        "I can't stop thinking negative thoughts",
        "How to cope with stress at work?",
        "I'm having trouble sleeping",
        "How do I build self-confidence?",
        "I feel overwhelmed by everything",
        "How to deal with anger issues?",
        "I'm struggling with self-esteem",
        "How to overcome social anxiety?",
        "I feel sad all the time",
        "How to manage panic attacks?",
        "I can't concentrate on anything",
        "How to deal with grief and loss?",
        "I'm worried about my mental health",
        "How to stop overthinking?",
        "I feel like I'm not good enough",
        "How to deal with toxic relationships?",
        "I'm having suicidal thoughts",
    ] * 50  # 1000 queries total
    
    print(f"Using {len(sample_queries)} test queries")
    
    # Models to benchmark
    baseline = "sentence-transformers/all-mpnet-base-v2"
    best = "intfloat/e5-large-v2"
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Batch sizes to test
    batch_sizes = [1, 8, 16, 32, 64]
    
    # Benchmark both models
    results_baseline = benchmark_model(baseline, sample_queries, batch_sizes, n_runs=5, device=device)
    results_best = benchmark_model(best, sample_queries, batch_sizes, n_runs=5, device=device)
    
    # Compare
    comparison = compare_speeds(results_baseline, results_best)
    
    # Generate LaTeX
    latex_table = generate_speed_table_latex(comparison, results_baseline, results_best)
    
    print("\n" + "="*80)
    print("LATEX TABLE (copy to paper)")
    print("="*80 + "\n")
    print(latex_table)
    
    # Save results
    output = {
        "baseline": results_baseline,
        "best_performer": results_best,
        "comparison": comparison,
        "latex_table": latex_table,
    }
    
    with open("speed_benchmark_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    with open("speed_benchmark_table.tex", "w") as f:
        f.write(latex_table)
    
    print("\n✓ Results saved to:")
    print("  - speed_benchmark_results.json")
    print("  - speed_benchmark_table.tex")
    
    # Summary
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80 + "\n")
    
    # Average speedup across batch sizes
    speedups = [1/v["model2_speedup"] for v in comparison["speedup_by_batch_size"].values()]
    avg_speedup = np.mean(speedups)
    
    print(f"Average speedup (mpnet vs E5): {avg_speedup:.2f}×")
    print(f"mpnet is {avg_speedup:.2f}× faster than E5-large")
    print(f"E5-large is {1/avg_speedup:.2f}× slower than mpnet")
    
    # Best batch size for each
    best_bs_baseline = max(
        results_baseline["batch_sizes"].items(),
        key=lambda x: x[1]["mean_qps"]
    )
    best_bs_best = max(
        results_best["batch_sizes"].items(),
        key=lambda x: x[1]["mean_qps"]
    )
    
    print(f"\nOptimal batch size:")
    print(f"  mpnet: {best_bs_baseline[0]} (throughput: {best_bs_baseline[1]['mean_qps']:.1f} q/s)")
    print(f"  E5-large: {best_bs_best[0]} (throughput: {best_bs_best[1]['mean_qps']:.1f} q/s)")


if __name__ == "__main__":
    main()
