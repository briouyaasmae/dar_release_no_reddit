#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_all_experiments.py

Master script to run complete DAR evaluation:
1. CounselChat baseline (full analysis)
2. Reddit dataset preparation
3. Cross-domain evaluation (Reddit)
4. Comparison report

This is the ONE COMMAND to run everything for your paper.

Usage:
    python run_all_experiments.py --device auto

Outputs:
    paper_experiments/
      ├── runs/
      │   ├── counselchat_baseline/  (main experiment)
      │   └── reddit_transfer/       (cross-domain)
      ├── data/
      │   └── reddit_mental_health/  (BEIR-style dataset)
      └── final_comparison.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def run_cmd(cmd: list, description: str) -> bool:
    """Run command and return True if successful"""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")
    
    start = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - start
    
    if result.returncode == 0:
        print(f"\n✅ SUCCESS ({elapsed:.1f}s)")
        return True
    else:
        print(f"\n❌ FAILED (code {result.returncode})")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run all DAR experiments")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--skip_reddit", action="store_true", help="Skip Reddit experiments")
    parser.add_argument("--quick", action="store_true", help="Quick run (limit queries, no sweep)")
    
    args = parser.parse_args()
    
    base_dir = Path("paper_experiments")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("  DAR COMPLETE EXPERIMENT SUITE")
    print("="*70)
    print(f"\nBase directory: {base_dir.resolve()}")
    print(f"Device: {args.device}")
    print(f"Quick mode: {args.quick}")
    print(f"Skip Reddit: {args.skip_reddit}")
    print("\n" + "="*70)
    
    results = {}
    
    # =====================================================
    # Experiment 1: CounselChat Baseline
    # =====================================================
    counselchat_dir = base_dir / "runs" / "counselchat_baseline"
    
    cmd = [
        sys.executable, "run_counselchat_baseline.py",
        "--workdir", str(counselchat_dir),
        "--device", args.device,
        "--full_analysis"
    ]
    
    if args.quick:
        cmd.extend(["--limit_queries", "500", "--no_sweep"])
    
    success = run_cmd(cmd, "EXPERIMENT 1: CounselChat Baseline")
    results["counselchat"] = {
        "success": success,
        "workdir": str(counselchat_dir)
    }
    
    if not success:
        print("\n⚠️  CounselChat baseline failed. Continuing anyway...")
    
    # =====================================================
    # Experiment 2: Reddit Dataset Preparation
    # =====================================================
    if not args.skip_reddit:
        reddit_data_dir = base_dir / "data" / "reddit_mental_health"
        
        cmd = [
            sys.executable, "reddit_mental_health_adapter.py",
            "--output_dir", str(reddit_data_dir),
            "--num_queries", "2000" if not args.quick else "500",
            "--seed", "42"
        ]
        
        success = run_cmd(cmd, "EXPERIMENT 2: Reddit Dataset Preparation")
        results["reddit_prep"] = {
            "success": success,
            "data_dir": str(reddit_data_dir)
        }
        
        # =====================================================
        # Experiment 3: Reddit Cross-Domain Evaluation
        # =====================================================
        if success:
            reddit_eval_dir = base_dir / "runs" / "reddit_transfer"
            
            cmd = [
                sys.executable, "dar_router_main.py",
                "--workdir", str(reddit_eval_dir),
                "--device", args.device,
                "--data_root", str(reddit_data_dir),
                "--chunk_mode", "sentence",
                "--no_sweep"
            ]
            
            success = run_cmd(cmd, "EXPERIMENT 3: Reddit Cross-Domain Evaluation")
            results["reddit_eval"] = {
                "success": success,
                "workdir": str(reddit_eval_dir)
            }
        else:
            print("\n⚠️  Skipping Reddit evaluation (dataset prep failed)")
            results["reddit_eval"] = {"success": False, "reason": "prep_failed"}
    else:
        print("\n⏭️  Skipping Reddit experiments (--skip_reddit)")
        results["reddit_prep"] = {"success": False, "reason": "skipped"}
        results["reddit_eval"] = {"success": False, "reason": "skipped"}
    
    # =====================================================
    # Generate Final Comparison Report
    # =====================================================
    print(f"\n{'='*70}")
    print("  GENERATING FINAL COMPARISON REPORT")
    print(f"{'='*70}\n")
    
    comparison = {
        "experiments": results,
        "summary": {}
    }
    
    # Load CounselChat results
    cc_report_path = counselchat_dir / "routing_report.json"
    if cc_report_path.exists():
        with open(cc_report_path) as f:
            cc_report = json.load(f)
        
        comparison["counselchat"] = {
            "retrieval": cc_report.get("retrieval", {}),
            "predictor": cc_report.get("predictor", {}),
            "routing": cc_report.get("routing", {}),
            "extended_analysis": cc_report.get("extended_analysis", {})
        }
        
        # Extract key metrics
        if "routing" in cc_report and "dual" in cc_report["routing"]:
            dual = cc_report["routing"]["dual"]
            comparison["summary"]["counselchat"] = {
                "ndcg@10": dual.get("avg_ndcg@10_auto", 0),
                "cost": dual.get("avg_cost", 0),
                "efficiency": dual.get("efficiency", 0),
                "crisis_ratio": dual.get("crisis_ratio", 0)
            }
    
    # Load Reddit results (if available)
    if not args.skip_reddit:
        reddit_report_path = base_dir / "runs" / "reddit_transfer" / "routing_report.json"
        if reddit_report_path.exists():
            with open(reddit_report_path) as f:
                reddit_report = json.load(f)
            
            comparison["reddit"] = {
                "retrieval": reddit_report.get("retrieval", {}),
                "routing": reddit_report.get("routing", {})
            }
            
            if "routing" in reddit_report and "dual" in reddit_report["routing"]:
                dual = reddit_report["routing"]["dual"]
                comparison["summary"]["reddit"] = {
                    "ndcg@10": dual.get("avg_ndcg@10_auto", 0),
                    "cost": dual.get("avg_cost", 0),
                    "efficiency": dual.get("efficiency", 0),
                    "crisis_ratio": dual.get("crisis_ratio", 0)
                }
            
            # Compute transfer gap
            if "counselchat" in comparison["summary"] and "reddit" in comparison["summary"]:
                cc_ndcg = comparison["summary"]["counselchat"]["ndcg@10"]
                reddit_ndcg = comparison["summary"]["reddit"]["ndcg@10"]
                comparison["summary"]["transfer_gap"] = {
                    "ndcg_drop": cc_ndcg - reddit_ndcg,
                    "relative_drop": (cc_ndcg - reddit_ndcg) / cc_ndcg if cc_ndcg > 0 else 0
                }
    
    # Save comparison
    comparison_path = base_dir / "final_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    
    # =====================================================
    # Print Summary
    # =====================================================
    print(f"\n{'='*70}")
    print("  EXPERIMENT SUITE COMPLETE")
    print(f"{'='*70}\n")
    
    print("📊 SUMMARY\n")
    
    if "counselchat" in comparison.get("summary", {}):
        cc = comparison["summary"]["counselchat"]
        print("CounselChat (In-Domain):")
        print(f"  nDCG@10:    {cc['ndcg@10']:.3f}")
        print(f"  Efficiency: {cc['efficiency']:.3f}")
        print(f"  Crisis:     {cc['crisis_ratio']:.1%}")
    
    if "reddit" in comparison.get("summary", {}):
        rd = comparison["summary"]["reddit"]
        print("\nReddit (Cross-Domain):")
        print(f"  nDCG@10:    {rd['ndcg@10']:.3f}")
        print(f"  Efficiency: {rd['efficiency']:.3f}")
        print(f"  Crisis:     {rd['crisis_ratio']:.1%}")
    
    if "transfer_gap" in comparison.get("summary", {}):
        tg = comparison["summary"]["transfer_gap"]
        print("\nTransfer Performance:")
        print(f"  nDCG drop:      {tg['ndcg_drop']:.3f}")
        print(f"  Relative drop:  {tg['relative_drop']:.1%}")
    
    print(f"\n📁 All results in: {base_dir.resolve()}")
    print(f"\nKey files:")
    print(f"  ✓ {comparison_path}")
    print(f"  ✓ {counselchat_dir}/routing_report.json")
    print(f"  ✓ {counselchat_dir}/table*.csv (publication tables)")
    if not args.skip_reddit:
        reddit_report = base_dir / "runs" / "reddit_transfer" / "routing_report.json"
        if reddit_report.exists():
            print(f"  ✓ {reddit_report}")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
