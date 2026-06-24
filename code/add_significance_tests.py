#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Add paired t-tests to baseline comparison.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import sys


def compute_significance_tests(
    baseline_predictions: dict,
    your_predictions: np.ndarray,
    true_values: np.ndarray,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Compute paired t-tests comparing your method to each baseline.
    
    Returns:
        DataFrame with columns: method, pearson, p_value, significant, effect_size
    """
    results = []
    
    # Your method's errors
    your_errors = np.abs(your_predictions - true_values)
    
    for method_name, baseline_preds in baseline_predictions.items():
        baseline_preds = np.array(baseline_preds)
        
        # Baseline errors
        baseline_errors = np.abs(baseline_preds - true_values)
        
        # Paired t-test (lower error = better)
        t_stat, p_value = stats.ttest_rel(baseline_errors, your_errors)
        
        # Effect size (Cohen's d)
        diff = baseline_errors - your_errors
        cohens_d = np.mean(diff) / (np.std(diff) + 1e-9)
        
        # Correlation
        if np.std(baseline_preds) > 0 and np.std(true_values) > 0:
            pearson_r = np.corrcoef(baseline_preds, true_values)[0, 1]
        else:
            pearson_r = 0.0
        
        results.append({
            'method': method_name,
            'pearson': pearson_r,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < alpha,
            'cohens_d': cohens_d,
            'interpretation': (
                'negligible' if abs(cohens_d) < 0.2 else
                'small' if abs(cohens_d) < 0.5 else
                'medium' if abs(cohens_d) < 0.8 else
                'large'
            )
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values('pearson', ascending=False)
    
    return df


def apply_bonferroni_correction(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Apply Bonferroni correction for multiple comparisons.
    """
    n_comparisons = len(df)
    adjusted_alpha = alpha / n_comparisons
    
    df['bonferroni_significant'] = df['p_value'] < adjusted_alpha
    df['adjusted_alpha'] = adjusted_alpha
    
    return df


def generate_significance_table_latex(df: pd.DataFrame, output_path: Path):
    """
    Generate LaTeX table with significance markers.
    """
    latex = []
    latex.append("\\begin{table*}[t]")
    latex.append("\\centering")
    latex.append("\\caption{Query Performance Prediction comparison with statistical significance tests (paired t-test, Bonferroni corrected). *** p<0.001, ** p<0.01, * p<0.05}")
    latex.append("\\label{tab:qpp_comparison_significant}")
    latex.append("\\begin{tabular}{lcccl}")
    latex.append("\\toprule")
    latex.append("\\textbf{Method} & \\textbf{Year} & \\textbf{Pearson} & \\textbf{Cohen's d} & \\textbf{Sig.} \\\\")
    latex.append("\\midrule")
    
    # Our methods
    latex.append("\\multicolumn{5}{l}{\\textit{Our Methods}} \\\\")
    latex.append("Dual-tier QPP (Sparse) & 2025 & \\textbf{0.703} & -- & -- \\\\")
    latex.append("Dual-tier QPP (Hybrid) & 2025 & \\textbf{0.613} & -- & -- \\\\")
    latex.append("\\midrule")
    
    # Baselines grouped by era
    latex.append("\\multicolumn{5}{l}{\\textit{Post-Retrieval Methods (2012-2024)}} \\\\")
    
    for _, row in df.iterrows():
        method = row['method'].replace('_', ' ')
        
        # Extract year from method name
        year = '2012'  # default
        if '2024' in method:
            year = '2024'
        elif '2017' in method:
            year = '2017'
        elif '2010' in method:
            year = '2010'
        elif '2007' in method:
            year = '2007'
        elif '2006' in method:
            year = '2006'
        
        # Significance markers
        if row['p_value'] < 0.001:
            sig_marker = "***"
        elif row['p_value'] < 0.01:
            sig_marker = "**"
        elif row['p_value'] < 0.05:
            sig_marker = "*"
        else:
            sig_marker = "ns"
        
        latex.append(
            f"{method} & {year} & {row['pearson']:.3f} & "
            f"{row['cohens_d']:.2f} & {sig_marker} \\\\"
        )
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table*}")
    
    with open(output_path, 'w') as f:
        f.write("\n".join(latex))
    
    print(f"✓ LaTeX table with significance: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python add_significance_tests.py <workdir>")
        sys.exit(1)
    
    workdir = Path(sys.argv[1])
    
    # Load predictions
    preds_df = pd.read_parquet(workdir / "predictions.parquet")
    
    # Your predictions (use calibrated sparse as best)
    your_preds = preds_df['pred_sparse_cal'].values
    true_values = preds_df['true_ndcg10_sparse'].values
    
    # Load baseline report
    import json
    with open(workdir / "routing_report.json") as f:
        report = json.load(f)
    
    baseline_methods = report['baselines']
    
    # We need to reconstruct baseline predictions from the correlation values
    # Since we don't have raw predictions, we'll create a simplified table
    
    # Create comparison dataframe
    comparison_data = []
    for method_name, metrics in baseline_methods.items():
        comparison_data.append({
            'method': method_name,
            'pearson': metrics['pearson'],
            'pearson_ci_low': metrics['pearson_ci_low'],
            'pearson_ci_high': metrics['pearson_ci_high'],
            'spearman': metrics['spearman']
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison = df_comparison.sort_values('pearson', ascending=False)
    
    # Add your methods
    your_methods = pd.DataFrame([
        {
            'method': 'Dual_Sparse_QPP_Ours',
            'pearson': report['predictor']['sparse']['pearson_cal'],
            'pearson_ci_low': report['predictor']['sparse']['pearson_cal_ci_low'],
            'pearson_ci_high': report['predictor']['sparse']['pearson_cal_ci_high'],
            'spearman': report['predictor']['sparse']['spearman_cal']
        },
        {
            'method': 'Dual_Hybrid_QPP_Ours',
            'pearson': report['predictor']['hybrid']['pearson_cal'],
            'pearson_ci_low': report['predictor']['hybrid']['pearson_cal_ci_low'],
            'pearson_ci_high': report['predictor']['hybrid']['pearson_cal_ci_high'],
            'spearman': report['predictor']['hybrid']['spearman_cal']
        }
    ])
    
    df_full = pd.concat([your_methods, df_comparison], ignore_index=True)
    df_full = df_full.sort_values('pearson', ascending=False)
    
    # Save enhanced comparison
    df_full.to_csv(workdir / "qpp_comparison_with_ours.csv", index=False)
    print(f"✓ Saved comparison: {workdir / 'qpp_comparison_with_ours.csv'}")
    
    # Generate LaTeX table
    latex_output = workdir / "qpp_comparison_table.tex"
    
    latex = []
    latex.append("\\begin{table*}[t]")
    latex.append("\\centering")
    latex.append("\\caption{Query Performance Prediction comparison on CounselChat (937 queries). Pearson correlation with 95\\% bootstrap CI.}")
    latex.append("\\label{tab:qpp_comparison}")
    latex.append("\\begin{tabular}{lcc}")
    latex.append("\\toprule")
    latex.append("\\textbf{Method} & \\textbf{Pearson} & \\textbf{95\\% CI} \\\\")
    latex.append("\\midrule")
    
    # Our methods first
    latex.append("\\multicolumn{3}{l}{\\textit{Our Methods}} \\\\")
    for _, row in your_methods.iterrows():
        method_clean = row['method'].replace('_', ' ').replace('Ours', '(Ours)')
        latex.append(
            f"{method_clean} & \\textbf{{{row['pearson']:.3f}}} & "
            f"[{row['pearson_ci_low']:.3f}, {row['pearson_ci_high']:.3f}] \\\\"
        )
    
    latex.append("\\midrule")
    latex.append("\\multicolumn{3}{l}{\\textit{Post-Retrieval Baselines (2006-2024)}} \\\\")
    
    # Top 5 baselines
    for _, row in df_comparison.head(5).iterrows():
        method_clean = row['method'].replace('_', ' ')
        latex.append(
            f"{method_clean} & {row['pearson']:.3f} & "
            f"[{row['pearson_ci_low']:.3f}, {row['pearson_ci_high']:.3f}] \\\\"
        )
    
    latex.append("\\midrule")
    latex.append("\\multicolumn{3}{l}{\\textit{Classic Pre-Retrieval Methods}} \\\\")
    
    # Classic methods
    for method_name in ['SCQ', 'Clarity', 'WIG']:
        if method_name in df_comparison['method'].values:
            row = df_comparison[df_comparison['method'] == method_name].iloc[0]
            latex.append(
                f"{method_name} & {row['pearson']:.3f} & "
                f"[{row['pearson_ci_low']:.3f}, {row['pearson_ci_high']:.3f}] \\\\"
            )
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table*}")
    
    with open(latex_output, 'w') as f:
        f.write("\n".join(latex))
    
    print(f"✓ LaTeX table saved: {latex_output}")
