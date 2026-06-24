#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate LaTeX tables for baseline comparisons.
"""

import pandas as pd
from pathlib import Path


def generate_qpp_comparison_table(csv_path: Path, output_path: Path):
    """
    Generate LaTeX table comparing QPP methods.
    """
    df = pd.read_csv(csv_path)
    
    # Sort by Pearson correlation (descending)
    df = df.sort_values('pearson', ascending=False)
    
    latex = []
    latex.append("\\begin{table}[t]")
    latex.append("\\centering")
    latex.append("\\caption{Query Performance Prediction baseline comparison.}")
    latex.append("\\label{tab:qpp_baselines}")
    latex.append("\\begin{tabular}{lccc}")
    latex.append("\\toprule")
    latex.append("\\textbf{Method} & \\textbf{Year} & \\textbf{Pearson} & \\textbf{Spearman} \\\\")
    latex.append("\\midrule")
    
    # Group by category
    latex.append("\\multicolumn{4}{l}{\\textit{Classic Methods (Pre-2019)}} \\\\")
    for idx, row in df[df.index.str.contains("SCQ|Clarity|WIG|NQC")].iterrows():
        year = "2012" if "NQC" in idx else "2008"
        latex.append(f"{idx} & {year} & {row['pearson']:.3f} & {row['spearman']:.3f} \\\\")
    
    latex.append("\\midrule")
    latex.append("\\multicolumn{4}{l}{\\textit{Neural Methods (2019-2023)}} \\\\")
    for idx, row in df[~df.index.str.contains("SCQ|Clarity|WIG|NQC")].iterrows():
        year = idx.split("_")[-1] if "_" in idx else "2020"
        latex.append(f"{idx} & {year} & {row['pearson']:.3f} & {row['spearman']:.3f} \\\\")
    
    latex.append("\\midrule")
    latex.append("\\multicolumn{4}{l}{\\textit{Ours}} \\\\")
    latex.append(f"Dual-tier QPP & 2025 & \\textbf{{{df.loc['Dual_tier', 'pearson']:.3f}}} & \\textbf{{{df.loc['Dual_tier', 'spearman']:.3f}}} \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    with open(output_path, 'w') as f:
        f.write("\n".join(latex))
    
    print(f"✓ LaTeX table saved: {output_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python generate_baseline_tables.py <workdir>")
        sys.exit(1)
    
    workdir = Path(sys.argv[1])
    csv_path = workdir / "qpp_baseline_comparison.csv"
    output_path = workdir / "qpp_baselines_table.tex"
    
    generate_qpp_comparison_table(csv_path, output_path)
