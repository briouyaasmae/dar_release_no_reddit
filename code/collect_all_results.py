#!/usr/bin/env python3
"""
Collect all experimental results into a comprehensive summary
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

def load_report(exp_dir: Path) -> Dict[str, Any]:
    """Load routing_report.json from experiment directory"""
    report_path = exp_dir / "routing_report.json"
    if not report_path.exists():
        return None
    try:
        with open(report_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {report_path}: {e}")
        return None

def extract_key_metrics(report: Dict[str, Any], exp_name: str) -> Dict[str, Any]:
    """Extract key metrics from a report"""
    if not report:
        return {"experiment": exp_name, "status": "FAILED"}
    
    metrics = {"experiment": exp_name, "status": "SUCCESS"}
    
    # Retrieval metrics
    try:
        metrics["dense_ndcg"] = report["retrieval"]["dense"]["mean_ndcg@10"]
        metrics["dense_ci_low"] = report["retrieval"]["dense"]["mean_ndcg@10_ci_low"]
        metrics["dense_ci_high"] = report["retrieval"]["dense"]["mean_ndcg@10_ci_high"]
        metrics["dense_zero_rate"] = report["retrieval"]["dense"]["zero_rate"]
    except:
        metrics["dense_ndcg"] = None
    
    try:
        metrics["sparse_ndcg"] = report["retrieval"]["sparse"]["mean_ndcg@10"]
        metrics["sparse_zero_rate"] = report["retrieval"]["sparse"]["zero_rate"]
    except:
        metrics["sparse_ndcg"] = None
    
    try:
        metrics["hybrid_ndcg"] = report["retrieval"]["hybrid"]["mean_ndcg@10"]
        metrics["hybrid_ci_low"] = report["retrieval"]["hybrid"]["mean_ndcg@10_ci_low"]
        metrics["hybrid_ci_high"] = report["retrieval"]["hybrid"]["mean_ndcg@10_ci_high"]
        metrics["hybrid_zero_rate"] = report["retrieval"]["hybrid"]["zero_rate"]
    except:
        metrics["hybrid_ndcg"] = None
    
    try:
        metrics["verified_ndcg"] = report["retrieval"]["hybrid_verified"]["mean_ndcg@10"]
        metrics["verified_ci_low"] = report["retrieval"]["hybrid_verified"]["mean_ndcg@10_ci_low"]
        metrics["verified_ci_high"] = report["retrieval"]["hybrid_verified"]["mean_ndcg@10_ci_high"]
        metrics["verified_zero_rate"] = report["retrieval"]["hybrid_verified"]["zero_rate"]
    except:
        metrics["verified_ndcg"] = None
    
    # Predictor metrics
    try:
        metrics["sparse_predictor_pearson"] = report["predictor"]["sparse"]["pearson_cal"]
        metrics["sparse_predictor_pearson_ci_low"] = report["predictor"]["sparse"]["pearson_cal_ci_low"]
        metrics["sparse_predictor_pearson_ci_high"] = report["predictor"]["sparse"]["pearson_cal_ci_high"]
        metrics["sparse_predictor_spearman"] = report["predictor"]["sparse"]["spearman_cal"]
    except:
        metrics["sparse_predictor_pearson"] = None
    
    try:
        metrics["hybrid_predictor_pearson"] = report["predictor"]["hybrid"]["pearson_cal"]
        metrics["hybrid_predictor_pearson_ci_low"] = report["predictor"]["hybrid"]["pearson_cal_ci_low"]
        metrics["hybrid_predictor_pearson_ci_high"] = report["predictor"]["hybrid"]["pearson_cal_ci_high"]
        metrics["hybrid_predictor_spearman"] = report["predictor"]["hybrid"]["spearman_cal"]
    except:
        metrics["hybrid_predictor_pearson"] = None
    
    # Routing metrics
    try:
        metrics["router_single_ndcg"] = report["routing"]["single"]["avg_ndcg@10_auto"]
        metrics["router_single_cost"] = report["routing"]["single"]["avg_cost"]
        metrics["router_single_efficiency"] = report["routing"]["single"]["efficiency"]
        metrics["router_single_auto_ratio"] = report["routing"]["single"]["auto_ratio"]
    except:
        metrics["router_single_ndcg"] = None
    
    try:
        metrics["router_dual_ndcg"] = report["routing"]["dual"]["avg_ndcg@10_auto"]
        metrics["router_dual_cost"] = report["routing"]["dual"]["avg_cost"]
        metrics["router_dual_efficiency"] = report["routing"]["dual"]["efficiency"]
        metrics["router_dual_auto_ratio"] = report["routing"]["dual"]["auto_ratio"]
        metrics["router_dual_crisis_ratio"] = report["routing"]["dual"].get("crisis_ratio", 0.0)
    except:
        metrics["router_dual_ndcg"] = None
    
    # Config info
    try:
        cfg = report["config"]
        metrics["chunk_mode"] = cfg.get("chunk_mode", "unknown")
        metrics["hybrid_mode"] = cfg.get("hybrid_mode", "unknown")
        metrics["verified_source"] = cfg.get("verified_source", "unknown")
        metrics["rrf_w_bm25"] = cfg.get("rrf_w_bm25", None)
        metrics["rrf_w_dense"] = cfg.get("rrf_w_dense", None)
        metrics["min_chunk_words"] = cfg.get("min_chunk_words", None)
        metrics["max_chunk_words"] = cfg.get("max_chunk_words", None)
        metrics["topk_bm25"] = cfg.get("topk_bm25", None)
        metrics["topk_dense"] = cfg.get("topk_dense", None)
    except:
        pass
    
    # Baseline comparisons
    try:
        baselines = report.get("baselines", {})
        if "NQC" in baselines:
            metrics["baseline_nqc_pearson"] = baselines["NQC"]["pearson"]
    except:
        pass
    
    # Ablations
    try:
        ablations = report.get("ablations", {})
        if "- Dual Predictor (Single)" in ablations:
            metrics["ablation_no_dual"] = ablations["- Dual Predictor (Single)"]["avg_ndcg@10_auto"]
        if "- Safety Override" in ablations:
            metrics["ablation_no_safety"] = ablations["- Safety Override"]["avg_ndcg@10_auto"]
        if "- Emotion Features (retrained)" in ablations:
            metrics["ablation_no_emotion"] = ablations["- Emotion Features (retrained)"]["avg_ndcg@10_auto"]
    except:
        pass
    
    return metrics

def load_zero_ndcg_stats(exp_dir: Path) -> Dict[str, Any]:
    """Load zero-nDCG statistics if available"""
    stats_path = exp_dir / "zero_ndcg_stats.json"
    if not stats_path.exists():
        return {}
    try:
        with open(stats_path, 'r') as f:
            return json.load(f)
    except:
        return {}

def main():
    # Find all experiment directories
    runs_dir = Path("paper_experiments/runs")
    if not runs_dir.exists():
        print("Error: 'runs' directory not found")
        print("Make sure you're running this from the directory containing 'runs/'")
        sys.exit(1)
    
    exp_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()])
    
    if not exp_dirs:
        print("Error: No experiment directories found in 'runs/'")
        sys.exit(1)
    
    print(f"Found {len(exp_dirs)} experiment directories")
    print("="*80)
    
    # Collect all results
    all_metrics = []
    
    for exp_dir in exp_dirs:
        exp_name = exp_dir.name
        print(f"Processing: {exp_name}")
        
        report = load_report(exp_dir)
        metrics = extract_key_metrics(report, exp_name)
        
        # Add zero-nDCG stats if available
        zero_stats = load_zero_ndcg_stats(exp_dir)
        if zero_stats:
            metrics["zero_overall_rate"] = zero_stats.get("Overall", {}).get("zero_rate", None)
            metrics["zero_mean_query_len"] = zero_stats.get("Query_Length", {}).get("zero_mean", None)
            metrics["zero_mean_coverage"] = zero_stats.get("Corpus_Coverage", {}).get("zero_mean_relevant", None)
            metrics["high_risk_zeros"] = zero_stats.get("Safety", {}).get("high_risk_zeros", None)
        
        all_metrics.append(metrics)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_metrics)
    
    # Save to CSV
    output_csv = "all_results_summary.csv"
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Saved summary to: {output_csv}")
    
    # Create a formatted summary for main experiments
    print("\n" + "="*80)
    print("MAIN EXPERIMENTS SUMMARY")
    print("="*80)
    
    main_cols = [
        "experiment", "dense_ndcg", "hybrid_ndcg", "verified_ndcg", 
        "router_dual_ndcg", "dense_zero_rate",
        "sparse_predictor_pearson", "hybrid_predictor_pearson"
    ]
    
    main_experiments = df[df["experiment"].str.contains("exp[0-9]", regex=True)]
    if not main_experiments.empty:
        print(main_experiments[main_cols].to_string(index=False))
    
    # Ablations summary
    print("\n" + "="*80)
    print("ABLATION EXPERIMENTS SUMMARY")
    print("="*80)
    
    ablation_experiments = df[df["experiment"].str.contains("ablation", regex=True)]
    if not ablation_experiments.empty:
        ablation_cols = [
            "experiment", "dense_ndcg", "hybrid_ndcg", "verified_ndcg", 
            "hybrid_mode", "verified_source"
        ]
        print(ablation_experiments[ablation_cols].to_string(index=False))
    
    # Create comparison table for paper
    print("\n" + "="*80)
    print("PAPER TABLE FORMAT")
    print("="*80)
    
    paper_data = []
    for _, row in df.iterrows():
        if "exp" in row["experiment"] or "ablation" in row["experiment"]:
            paper_data.append({
                "Experiment": row["experiment"],
                "Dense": f"{row['dense_ndcg']:.3f}" if row['dense_ndcg'] else "N/A",
                "Hybrid": f"{row['hybrid_ndcg']:.3f}" if row['hybrid_ndcg'] else "N/A",
                "Verified": f"{row['verified_ndcg']:.3f}" if row['verified_ndcg'] else "N/A",
                "Router": f"{row['router_dual_ndcg']:.3f}" if row['router_dual_ndcg'] else "N/A",
                "Zero%": f"{row['dense_zero_rate']*100:.1f}%" if row['dense_zero_rate'] else "N/A",
                "Sp.Pred": f"{row['sparse_predictor_pearson']:.3f}" if row['sparse_predictor_pearson'] else "N/A",
                "Hy.Pred": f"{row['hybrid_predictor_pearson']:.3f}" if row['hybrid_predictor_pearson'] else "N/A"
            })
    
    paper_df = pd.DataFrame(paper_data)
    print(paper_df.to_string(index=False))
    
    # Save paper table
    paper_csv = "paper_results_table.csv"
    paper_df.to_csv(paper_csv, index=False)
    print(f"\n✅ Saved paper table to: {paper_csv}")
    
    # Configuration summary
    print("\n" + "="*80)
    print("CONFIGURATION SUMMARY")
    print("="*80)
    
    config_cols = [
        "experiment", "chunk_mode", "hybrid_mode", "verified_source",
        "rrf_w_bm25", "rrf_w_dense", "min_chunk_words", "max_chunk_words"
    ]
    
    config_df = df[config_cols]
    print(config_df.to_string(index=False))
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    if not main_experiments.empty:
        print(f"\nMain Experiments (n={len(main_experiments)}):")
        print(f"  Dense nDCG@10:        {main_experiments['dense_ndcg'].mean():.3f} ± {main_experiments['dense_ndcg'].std():.3f}")
        print(f"  Hybrid nDCG@10:       {main_experiments['hybrid_ndcg'].mean():.3f} ± {main_experiments['hybrid_ndcg'].std():.3f}")
        print(f"  Router nDCG@10:       {main_experiments['router_dual_ndcg'].mean():.3f} ± {main_experiments['router_dual_ndcg'].std():.3f}")
        print(f"  Zero-rate:            {main_experiments['dense_zero_rate'].mean()*100:.1f}% ± {main_experiments['dense_zero_rate'].std()*100:.1f}%")
        print(f"  Sparse Predictor:     {main_experiments['sparse_predictor_pearson'].mean():.3f} ± {main_experiments['sparse_predictor_pearson'].std():.3f}")
        print(f"  Hybrid Predictor:     {main_experiments['hybrid_predictor_pearson'].mean():.3f} ± {main_experiments['hybrid_predictor_pearson'].std():.3f}")
    
    # Best results
    print("\n" + "="*80)
    print("BEST RESULTS")
    print("="*80)
    
    best_dense = df.loc[df['dense_ndcg'].idxmax()] if df['dense_ndcg'].notna().any() else None
    best_router = df.loc[df['router_dual_ndcg'].idxmax()] if df['router_dual_ndcg'].notna().any() else None
    best_predictor = df.loc[df['sparse_predictor_pearson'].idxmax()] if df['sparse_predictor_pearson'].notna().any() else None
    
    if best_dense is not None:
        print(f"\nBest Dense Performance:")
        print(f"  Experiment: {best_dense['experiment']}")
        print(f"  nDCG@10: {best_dense['dense_ndcg']:.4f}")
    
    if best_router is not None:
        print(f"\nBest Router Performance:")
        print(f"  Experiment: {best_router['experiment']}")
        print(f"  nDCG@10: {best_router['router_dual_ndcg']:.4f}")
        print(f"  vs Dense: +{(best_router['router_dual_ndcg'] - best_router['dense_ndcg'])*100:.1f}%")
    
    if best_predictor is not None:
        print(f"\nBest Predictor Performance:")
        print(f"  Experiment: {best_predictor['experiment']}")
        print(f"  Sparse Pearson: {best_predictor['sparse_predictor_pearson']:.4f}")
        print(f"  Hybrid Pearson: {best_predictor['hybrid_predictor_pearson']:.4f}")
    
    # Fusion degradation analysis
    print("\n" + "="*80)
    print("FUSION DEGRADATION ANALYSIS")
    print("="*80)
    
    for _, row in df.iterrows():
        if row['dense_ndcg'] and row['hybrid_ndcg'] and row['dense_ndcg'] > 0:
            degradation = ((row['hybrid_ndcg'] - row['dense_ndcg']) / row['dense_ndcg']) * 100
            if abs(degradation) > 1:  # Only show significant differences
                direction = "↑" if degradation > 0 else "↓"
                print(f"  {row['experiment']:30s}: {degradation:+6.1f}% {direction} (Hybrid: {row['hybrid_ndcg']:.3f} vs Dense: {row['dense_ndcg']:.3f})")
    
    print("\n" + "="*80)
    print(f"✅ Collection complete! Generated files:")
    print(f"   - {output_csv}")
    print(f"   - {paper_csv}")
    print("="*80)

if __name__ == "__main__":
    main()
