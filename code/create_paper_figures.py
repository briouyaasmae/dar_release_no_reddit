#!/usr/bin/env python3
"""
Generate all publication-quality figures from experimental results.
Creates figures for paper submission.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plotting style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("colorblind")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13

# Color scheme
COLORS = {
    'dense': '#2E7D32',      # Green
    'hybrid': '#1976D2',     # Blue
    'verified': '#D32F2F',   # Red
    'router': '#F57C00',     # Orange
    'sparse': '#7B1FA2',     # Purple
    'baseline': '#757575',   # Gray
}

def load_experiment_results(runs_dir: Path) -> pd.DataFrame:
    """Load all experiment results into a DataFrame"""
    experiments = []
    
    for exp_dir in sorted(runs_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
            
        report_path = exp_dir / "routing_report.json"
        if not report_path.exists():
            continue
            
        try:
            with open(report_path, 'r') as f:
                report = json.load(f)
            
            exp_name = exp_dir.name
            
            # Extract key metrics
            exp_data = {
                'experiment': exp_name,
                'dense_ndcg': report['retrieval']['dense']['mean_ndcg@10'],
                'dense_ci_low': report['retrieval']['dense']['mean_ndcg@10_ci_low'],
                'dense_ci_high': report['retrieval']['dense']['mean_ndcg@10_ci_high'],
                'hybrid_ndcg': report['retrieval']['hybrid']['mean_ndcg@10'],
                'hybrid_ci_low': report['retrieval']['hybrid']['mean_ndcg@10_ci_low'],
                'hybrid_ci_high': report['retrieval']['hybrid']['mean_ndcg@10_ci_high'],
                'verified_ndcg': report['retrieval']['hybrid_verified']['mean_ndcg@10'],
                'verified_ci_low': report['retrieval']['hybrid_verified']['mean_ndcg@10_ci_low'],
                'verified_ci_high': report['retrieval']['hybrid_verified']['mean_ndcg@10_ci_high'],
                'sparse_ndcg': report['retrieval']['sparse']['mean_ndcg@10'],
                'router_ndcg': report['routing']['dual']['avg_ndcg@10_auto'],
                'router_cost': report['routing']['dual']['avg_cost'],
                'router_efficiency': report['routing']['dual']['efficiency'],
                'dense_zero_rate': report['retrieval']['dense']['zero_rate'],
                'hybrid_zero_rate': report['retrieval']['hybrid']['zero_rate'],
                'verified_zero_rate': report['retrieval']['hybrid_verified']['zero_rate'],
                'sparse_predictor_pearson': report['predictor']['sparse']['pearson_cal'],
                'hybrid_predictor_pearson': report['predictor']['hybrid']['pearson_cal'],
                'chunk_mode': report['config'].get('chunk_mode', 'unknown'),
                'hybrid_mode': report['config'].get('hybrid_mode', 'unknown'),
                'rrf_w_bm25': report['config'].get('rrf_w_bm25', 1.0),
                'rrf_w_dense': report['config'].get('rrf_w_dense', 1.0),
            }
            
            experiments.append(exp_data)
            
        except Exception as e:
            print(f"Warning: Could not load {exp_dir.name}: {e}")
            continue
    
    return pd.DataFrame(experiments)


def figure1_fusion_degradation(df: pd.DataFrame, output_path: Path):
    """Figure 1: Fusion degradation with BM25 weight"""
    
    # Filter experiments with RRF fusion (overlap chunks, mpnet model)
    fusion_exps = df[
        (df['experiment'].isin(['exp1_dense_only', 'exp2_rrf_equal', 'exp3_rrf_downweight', 'ablation_rrf_stress'])) |
        ((df['chunk_mode'] == 'overlap') & (df['hybrid_mode'] == 'rrf'))
    ].copy()
    
    # Map experiments to weights
    weight_map = {
        'exp1_dense_only': 0.0,
        'exp3_rrf_downweight': 0.2,
        'exp2_rrf_equal': 1.0,
        'ablation_rrf_stress': 5.0,
    }
    
    fusion_exps['w_bm25'] = fusion_exps['experiment'].map(weight_map)
    fusion_exps = fusion_exps.dropna(subset=['w_bm25']).sort_values('w_bm25')
    
    if fusion_exps.empty:
        print("Warning: No fusion experiments found for Figure 1")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot dense baseline
    dense_baseline = fusion_exps['dense_ndcg'].iloc[0]
    ax.axhline(y=dense_baseline, color=COLORS['dense'], linestyle='--', 
               linewidth=2, label='Dense baseline', zorder=1)
    
    # Plot hybrid performance
    ax.errorbar(fusion_exps['w_bm25'], fusion_exps['hybrid_ndcg'],
                yerr=[
                    fusion_exps['hybrid_ndcg'] - fusion_exps['hybrid_ci_low'],
                    fusion_exps['hybrid_ci_high'] - fusion_exps['hybrid_ndcg']
                ],
                marker='o', markersize=10, linewidth=2.5, capsize=6, capthick=2,
                color=COLORS['hybrid'], label='RRF Hybrid', zorder=3)
    
    # Fit exponential decay model
    weights = fusion_exps['w_bm25'].values
    hybrid_vals = fusion_exps['hybrid_ndcg'].values
    
    # Model: y = a * exp(-b * x)
    from scipy.optimize import curve_fit
    def exp_decay(x, a, b):
        return a * np.exp(-b * x)
    
    try:
        params, _ = curve_fit(exp_decay, weights, hybrid_vals, p0=[dense_baseline, 0.35])
        
        # Plot fitted curve
        x_smooth = np.linspace(0, 5, 100)
        y_fit = exp_decay(x_smooth, *params)
        ax.plot(x_smooth, y_fit, 'r--', linewidth=1.5, alpha=0.7,
                label=f'Exponential fit: y={params[0]:.3f}×exp(-{params[1]:.3f}×x)')
        
        # Calculate R²
        residuals = hybrid_vals - exp_decay(weights, *params)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((hybrid_vals - np.mean(hybrid_vals))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Add R² text
        ax.text(0.98, 0.02, f'R² = {r_squared:.3f}', 
                transform=ax.transAxes, fontsize=11, 
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
    except Exception as e:
        print(f"Warning: Could not fit exponential model: {e}")
    
    # Add degradation percentages
    for idx, row in fusion_exps.iterrows():
        if row['w_bm25'] > 0:
            degradation = ((row['hybrid_ndcg'] - dense_baseline) / dense_baseline) * 100
            ax.annotate(f'{degradation:.1f}%', 
                       xy=(row['w_bm25'], row['hybrid_ndcg']),
                       xytext=(10, -10), textcoords='offset points',
                       fontsize=9, color='red', fontweight='bold')
    
    ax.set_xlabel('BM25 Weight in RRF (w_BM25)', fontsize=12, fontweight='bold')
    ax.set_ylabel('nDCG@10', fontsize=12, fontweight='bold')
    ax.set_title('Fusion Quality Degradation vs BM25 Weight\n(Overlap Chunking, all-mpnet-base-v2)', 
                 fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0.35, 0.75)
    
    plt.tight_layout()
    plt.savefig(output_path / 'figure1_fusion_degradation.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'figure1_fusion_degradation.pdf', bbox_inches='tight')
    plt.close()
    print(f"✅ Saved Figure 1: {output_path / 'figure1_fusion_degradation.png'}")


def figure2_architecture_hierarchy(df: pd.DataFrame, output_path: Path):
    """Figure 2: Architecture impact hierarchy (Model > Fusion > Chunk)"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define comparisons
    comparisons = []
    
    # Model impact: mpnet vs PubMed (sentence chunks)
    mpnet_sent = df[(df['chunk_mode'] == 'sentence') & 
                    (df['experiment'].str.contains('exp5|exp4.*mpnet'))]['dense_ndcg'].mean()
    pubmed_sent = df[(df['experiment'] == 'exp4_sentence_chunks')]['dense_ndcg'].iloc[0] if \
                     len(df[df['experiment'] == 'exp4_sentence_chunks']) > 0 else None
    
    if mpnet_sent and pubmed_sent:
        model_impact = ((mpnet_sent - pubmed_sent) / pubmed_sent) * 100
        comparisons.append(('Model\n(mpnet vs PubMed)', model_impact, mpnet_sent, pubmed_sent))
    
    # Fusion impact: Dense vs RRF (overlap, w=1.0)
    exp2 = df[df['experiment'] == 'exp2_rrf_equal']
    if not exp2.empty:
        dense_val = exp2['dense_ndcg'].iloc[0]
        hybrid_val = exp2['hybrid_ndcg'].iloc[0]
        fusion_impact = ((hybrid_val - dense_val) / dense_val) * 100
        comparisons.append(('Fusion\n(Dense vs RRF)', fusion_impact, dense_val, hybrid_val))
    
    # Chunk impact: Sentence vs Overlap (mpnet, dense-only)
    sent_dense = df[(df['chunk_mode'] == 'sentence') & 
                    (df['hybrid_mode'] == 'rrf') &
                    (df['experiment'].str.contains('exp5|exp4.*mpnet'))]['dense_ndcg'].mean()
    overlap_dense = df[(df['chunk_mode'] == 'overlap') & 
                       (df['experiment'] == 'exp1_dense_only')]['dense_ndcg'].mean()
    
    if sent_dense and overlap_dense:
        chunk_impact = ((sent_dense - overlap_dense) / overlap_dense) * 100
        comparisons.append(('Chunk\n(Sentence vs Overlap)', chunk_impact, sent_dense, overlap_dense))
    
    # Plot bars
    labels = [c[0] for c in comparisons]
    impacts = [c[1] for c in comparisons]
    colors_list = [COLORS['dense'], COLORS['hybrid'], COLORS['verified']]
    
    bars = ax.bar(labels, impacts, color=colors_list[:len(impacts)], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, comp in zip(bars, comparisons):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:+.1f}%\n({comp[2]:.3f} vs {comp[3]:.3f})',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=10, fontweight='bold')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_ylabel('Quality Impact (%)', fontsize=12, fontweight='bold')
    ax.set_title('Architecture Impact Hierarchy in Mental Health QA\n(Relative Effect Sizes)', 
                 fontsize=13, fontweight='bold', pad=20)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # Add ranking annotations
    ax.text(0.02, 0.98, '🥇 1st: Model selection', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='gold', alpha=0.3))
    ax.text(0.02, 0.90, '🥈 2nd: Fusion strategy', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='silver', alpha=0.3))
    ax.text(0.02, 0.82, '🥉 3rd: Chunk mode', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#CD7F32', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_path / 'figure2_architecture_hierarchy.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'figure2_architecture_hierarchy.pdf', bbox_inches='tight')
    plt.close()
    print(f"✅ Saved Figure 2: {output_path / 'figure2_architecture_hierarchy.png'}")


def figure3_retrieval_comparison(df: pd.DataFrame, output_path: Path):
    """Figure 3: Retrieval method comparison across experiments"""
    
    # Select main experiments
    main_exps = df[df['experiment'].isin([
        'exp1_dense_only', 'exp2_rrf_equal', 'exp3_rrf_downweight',
        'exp5_sentence_clean', 'exp4_sentence_chunks_mpnet', 'exp4_sentence_chunks'
    ])].copy()
    
    if main_exps.empty:
        print("Warning: No main experiments found for Figure 3")
        return
    
    # Shorten names for display
    name_map = {
        'exp1_dense_only': 'Overlap\nDense-only',
        'exp2_rrf_equal': 'Overlap\nRRF (w=1.0)',
        'exp3_rrf_downweight': 'Overlap\nRRF (w=0.2)',
        'exp5_sentence_clean': 'Sentence\n(mpnet)',
        'exp4_sentence_chunks_mpnet': 'Sentence\n(mpnet-v2)',
        'exp4_sentence_chunks': 'Sentence\n(PubMed)',
    }
    
    main_exps['display_name'] = main_exps['experiment'].map(name_map)
    main_exps = main_exps.sort_values('dense_ndcg', ascending=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel A: Retrieval quality comparison
    x = np.arange(len(main_exps))
    width = 0.2
    
    ax1.bar(x - 1.5*width, main_exps['dense_ndcg'], width, 
            label='Dense', color=COLORS['dense'], alpha=0.8, edgecolor='black')
    ax1.bar(x - 0.5*width, main_exps['hybrid_ndcg'], width,
            label='Hybrid', color=COLORS['hybrid'], alpha=0.8, edgecolor='black')
    ax1.bar(x + 0.5*width, main_exps['verified_ndcg'], width,
            label='Verified', color=COLORS['verified'], alpha=0.8, edgecolor='black')
    ax1.bar(x + 1.5*width, main_exps['router_ndcg'], width,
            label='Router', color=COLORS['router'], alpha=0.8, edgecolor='black')
    
    ax1.set_ylabel('nDCG@10', fontsize=11, fontweight='bold')
    ax1.set_title('(A) Retrieval Quality Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(main_exps['display_name'], fontsize=9)
    ax1.legend(loc='upper right', ncol=2)
    ax1.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 0.8)
    
    # Panel B: Zero-rate comparison
    ax2.bar(x, main_exps['dense_zero_rate'] * 100, 
            color=COLORS['dense'], alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, (idx, row) in enumerate(main_exps.iterrows()):
        ax2.text(i, row['dense_zero_rate'] * 100 + 1, 
                f"{row['dense_zero_rate']*100:.1f}%",
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.set_ylabel('Zero-nDCG Rate (%)', fontsize=11, fontweight='bold')
    ax2.set_title('(B) Zero-nDCG Rate (Dense Retrieval)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(main_exps['display_name'], fontsize=9)
    ax2.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(0, 50)
    
    # Add horizontal line at 10%
    ax2.axhline(y=10, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='10% threshold')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / 'figure3_retrieval_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'figure3_retrieval_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"✅ Saved Figure 3: {output_path / 'figure3_retrieval_comparison.png'}")


def figure4_predictor_performance(df: pd.DataFrame, output_path: Path):
    """Figure 4: QPP predictor performance"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel A: Sparse predictor across experiments
    exps = df[df['experiment'].str.contains('exp')].sort_values('sparse_predictor_pearson', ascending=False)
    
    if not exps.empty:
        ax1.barh(range(len(exps)), exps['sparse_predictor_pearson'], 
                color=COLORS['sparse'], alpha=0.8, edgecolor='black')
        ax1.set_yticks(range(len(exps)))
        ax1.set_yticklabels(exps['experiment'], fontsize=9)
        ax1.set_xlabel('Pearson Correlation (Calibrated)', fontsize=11, fontweight='bold')
        ax1.set_title('(A) Sparse Quality Predictor', fontsize=12, fontweight='bold')
        ax1.axvline(x=0.8, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Strong (0.8)')
        ax1.axvline(x=0.23, color='red', linestyle='--', linewidth=2, alpha=0.5, label='NQC baseline')
        ax1.legend()
        ax1.grid(True, axis='x', alpha=0.3, linestyle='--')
        ax1.set_xlim(0, 1.0)
        
        # Add value labels
        for i, val in enumerate(exps['sparse_predictor_pearson']):
            ax1.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
    
    # Panel B: Hybrid predictor across experiments
    if not exps.empty:
        ax2.barh(range(len(exps)), exps['hybrid_predictor_pearson'],
                color=COLORS['hybrid'], alpha=0.8, edgecolor='black')
        ax2.set_yticks(range(len(exps)))
        ax2.set_yticklabels(exps['experiment'], fontsize=9)
        ax2.set_xlabel('Pearson Correlation (Calibrated)', fontsize=11, fontweight='bold')
        ax2.set_title('(B) Hybrid Quality Predictor', fontsize=12, fontweight='bold')
        ax2.axvline(x=0.6, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Good (0.6)')
        ax2.legend()
        ax2.grid(True, axis='x', alpha=0.3, linestyle='--')
        ax2.set_xlim(0, 1.0)
        
        # Add value labels
        for i, val in enumerate(exps['hybrid_predictor_pearson']):
            ax2.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / 'figure4_predictor_performance.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'figure4_predictor_performance.pdf', bbox_inches='tight')
    plt.close()
    print(f"✅ Saved Figure 4: {output_path / 'figure4_predictor_performance.png'}")


def figure5_efficiency_frontier(df: pd.DataFrame, output_path: Path):
    """Figure 5: Cost-quality efficiency frontier"""
    
    # Calculate baselines from actual data (excluding NaN)
    sparse_mean = df['sparse_ndcg'].dropna().mean()
    dense_mean = df['dense_ndcg'].dropna().mean()
    hybrid_mean = df['hybrid_ndcg'].dropna().mean()
    verified_mean = df['verified_ndcg'].dropna().mean()
    
    # Static baselines with correct means
    baselines = [
        {'name': 'Sparse', 'cost': 1.0, 'ndcg': sparse_mean, 'marker': 's', 'size': 300},
        {'name': 'Dense', 'cost': 4.0, 'ndcg': dense_mean, 'marker': 'o', 'size': 300},
        {'name': 'Hybrid', 'cost': 4.0, 'ndcg': hybrid_mean, 'marker': '^', 'size': 300},
        {'name': 'Verified', 'cost': 10.0, 'ndcg': verified_mean, 'marker': 'd', 'size': 300},
    ]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Debug
    print(f"\nDEBUG Figure 5:")
    print(f"  Baselines: {len(baselines)}")
    for b in baselines:
        print(f"    {b['name']}: ({b['cost']:.1f}, {b['ndcg']:.3f})")
    
    # Plot baselines
    for b in baselines:
        ax.scatter(b['cost'], b['ndcg'], s=b['size'], marker=b['marker'],
                  color=COLORS.get(b['name'].lower(), 'gray'), 
                  alpha=0.7, edgecolors='black', linewidth=2,
                  label=f"{b['name']} (always)", zorder=3)
        
        # Add text label with offset to avoid overlap
        offset_x = 0.3 if b['cost'] < 5 else -0.3
        offset_y = 0.02
        ax.annotate(b['name'], (b['cost'], b['ndcg']), 
                   xytext=(offset_x, offset_y), textcoords='offset fontsize',
                   fontsize=10, fontweight='bold', ha='left' if b['cost'] < 5 else 'right')
    
    # Plot routers
    router_exps = df[df['experiment'].str.contains('exp')].copy()
    
    print(f"  Router experiments: {len(router_exps)}")
    if not router_exps.empty:
        print(f"  Cost range: {router_exps['router_cost'].min():.2f} - {router_exps['router_cost'].max():.2f}")
        print(f"  nDCG range: {router_exps['router_ndcg'].min():.2f} - {router_exps['router_ndcg'].max():.2f}")
    
    for idx, row in router_exps.iterrows():
        ax.scatter(row['router_cost'], row['router_ndcg'], s=400, marker='*',
                  color=COLORS['router'], alpha=0.9, edgecolors='black', 
                  linewidth=2, zorder=4)
        
        # Label - shortened
        label = row['experiment'].replace('exp', 'E').replace('_', '\n')
        ax.annotate(label, (row['router_cost'], row['router_ndcg']),
                   xytext=(0, -20), textcoords='offset points',
                   fontsize=7, style='italic', ha='center')
    
    # Add efficiency lines (nDCG/cost) - more visible
    efficiency_levels = [0.05, 0.10, 0.15, 0.20]
    x_line = np.linspace(0.5, 11, 100)
    
    for eff in efficiency_levels:
        y_line = eff * x_line
        # Only plot where y is in valid range
        valid_mask = (y_line >= 0.15) & (y_line <= 0.85)
        ax.plot(x_line[valid_mask], y_line[valid_mask], 'k--', 
                alpha=0.15, linewidth=1, zorder=1)
        
        # Label at right edge
        y_label = eff * 11
        if 0.15 < y_label < 0.85:
            ax.text(11.2, y_label, f'ε={eff:.2f}', 
                   fontsize=8, alpha=0.4, va='center')
    
    ax.set_xlabel('Average Cost (arbitrary units)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average nDCG@10', fontsize=12, fontweight='bold')
    ax.set_title('Cost-Quality Efficiency Frontier\n(Routers vs Static Baselines)', 
                 fontsize=13, fontweight='bold', pad=20)
    
    # Adjust legend position to avoid data
    ax.legend(loc='upper left', framealpha=0.95, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
    
    # Set limits with padding
    ax.set_xlim(0, 12)
    ax.set_ylim(0.25, 0.75)
    
    plt.tight_layout()
    plt.savefig(output_path / 'figure5_efficiency_frontier.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'figure5_efficiency_frontier.pdf', bbox_inches='tight')
    plt.close()
    print(f"✅ Saved Figure 5: {output_path / 'figure5_efficiency_frontier.png'}")
def figure6_reranker_interaction(df: pd.DataFrame, output_path: Path):
    """Figure 6: Reranker effectiveness vs base model quality"""
    
    # Collect data points: (base_quality, reranker_delta)
    data_points = []
    
    for idx, row in df.iterrows():
        if row['dense_ndcg'] and row['verified_ndcg']:
            delta = ((row['verified_ndcg'] - row['dense_ndcg']) / row['dense_ndcg']) * 100
            data_points.append({
                'base_quality': row['dense_ndcg'],
                'reranker_delta': delta,
                'experiment': row['experiment'],
                'chunk_mode': row['chunk_mode'],
            })
    
    if not data_points:
        print("Warning: No reranker data for Figure 6")
        return
    
    reranker_df = pd.DataFrame(data_points)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Separate by chunk mode
    for chunk in reranker_df['chunk_mode'].unique():
        subset = reranker_df[reranker_df['chunk_mode'] == chunk]
        marker = 'o' if chunk == 'overlap' else 's'
        ax.scatter(subset['base_quality'], subset['reranker_delta'],
                  s=150, marker=marker, alpha=0.7, edgecolors='black',
                  linewidth=1.5, label=f'{chunk.capitalize()} chunks')
        
        # Annotate
        for _, row in subset.iterrows():
            ax.annotate(row['experiment'].replace('_', '\n'), 
                       (row['base_quality'], row['reranker_delta']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=7, alpha=0.7)
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
    
    # Add trend line
    from scipy.stats import linregress
    slope, intercept, r_value, _, _ = linregress(reranker_df['base_quality'], 
                                                  reranker_df['reranker_delta'])
    x_fit = np.array([reranker_df['base_quality'].min(), reranker_df['base_quality'].max()])
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, 'r--', linewidth=2, alpha=0.7,
            label=f'Trend: R²={r_value**2:.3f}')
    
    # Shade regions
    ax.axhspan(-50, 0, alpha=0.1, color='red', label='Reranker hurts')
    ax.axhspan(0, 50, alpha=0.1, color='green', label='Reranker helps')
    
    ax.set_xlabel('Base Dense Retrieval Quality (nDCG@10)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Reranker Effect (%)', fontsize=12, fontweight='bold')
    ax.set_title('Cross-Encoder Reranking: Quality-Dependent Effectiveness\n(Helps Weak Models, Hurts Strong Models)', 
                 fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path / 'figure6_reranker_interaction.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'figure6_reranker_interaction.pdf', bbox_inches='tight')
    plt.close()
    print(f"✅ Saved Figure 6: {output_path / 'figure6_reranker_interaction.png'}")


def figure7_chunk_mode_comparison(df: pd.DataFrame, output_path: Path):
    """Figure 7: Chunk mode impact (controlled comparison)"""
    
    # Filter for all-mpnet experiments with different chunk modes
    mpnet_exps = df[
        ((df['experiment'].str.contains('exp1')) & (df['chunk_mode'] == 'overlap')) |
        ((df['experiment'].str.contains('exp5|exp4.*mpnet')) & (df['chunk_mode'] == 'sentence'))
    ].copy()
    
    if len(mpnet_exps) < 2:
        print("Warning: Not enough chunk mode experiments for Figure 7")
        return
    
    # Aggregate by chunk mode
    chunk_summary = mpnet_exps.groupby('chunk_mode').agg({
        'dense_ndcg': 'mean',
        'hybrid_ndcg': 'mean',
        'verified_ndcg': 'mean',
        'router_ndcg': 'mean',
        'dense_zero_rate': 'mean',
    }).reset_index()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel A: Quality comparison
    x = np.arange(len(chunk_summary))
    width = 0.2
    
    ax1.bar(x - 1.5*width, chunk_summary['dense_ndcg'], width,
            label='Dense', color=COLORS['dense'], alpha=0.8, edgecolor='black')
    ax1.bar(x - 0.5*width, chunk_summary['hybrid_ndcg'], width,
            label='Hybrid', color=COLORS['hybrid'], alpha=0.8, edgecolor='black')
    ax1.bar(x + 0.5*width, chunk_summary['verified_ndcg'], width,
            label='Verified', color=COLORS['verified'], alpha=0.8, edgecolor='black')
    ax1.bar(x + 1.5*width, chunk_summary['router_ndcg'], width,
            label='Router', color=COLORS['router'], alpha=0.8, edgecolor='black')
    
    ax1.set_ylabel('nDCG@10', fontsize=11, fontweight='bold')
    ax1.set_title('(A) Retrieval Quality by Chunk Mode\n(all-mpnet-base-v2)', 
                  fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.capitalize() for c in chunk_summary['chunk_mode']], fontsize=10)
    ax1.legend(loc='upper right')
    ax1.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 0.8)
    
    # Add value labels
    for i, (idx, row) in enumerate(chunk_summary.iterrows()):
        for j, metric in enumerate(['dense_ndcg', 'hybrid_ndcg', 'verified_ndcg', 'router_ndcg']):
            x_pos = i + (j - 1.5) * width
            ax1.text(x_pos, row[metric] + 0.02, f'{row[metric]:.3f}',
                    ha='center', va='bottom', fontsize=8, rotation=90)
    
    # Panel B: Zero-rate comparison
    bars = ax2.bar(x, chunk_summary['dense_zero_rate'] * 100,
                   color=[COLORS['dense'], COLORS['hybrid']], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax2.set_ylabel('Zero-nDCG Rate (%)', fontsize=11, fontweight='bold')
    ax2.set_title('(B) Coverage by Chunk Mode', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([c.capitalize() for c in chunk_summary['chunk_mode']], fontsize=10)
    ax2.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(0, 15)
    
    # Add value labels
    for bar, val in zip(bars, chunk_summary['dense_zero_rate'] * 100):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.3,
                f'{val:.1f}%', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
    
    # Calculate improvement
    if len(chunk_summary) == 2:
        overlap_zero = chunk_summary[chunk_summary['chunk_mode'] == 'overlap']['dense_zero_rate'].iloc[0]
        sentence_zero = chunk_summary[chunk_summary['chunk_mode'] == 'sentence']['dense_zero_rate'].iloc[0]
        improvement = ((overlap_zero - sentence_zero) / overlap_zero) * 100
        
        ax2.text(0.5, 0.95, f'Sentence improvement: {improvement:.1f}% reduction',
                transform=ax2.transAxes, ha='center', va='top',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path / 'figure7_chunk_mode_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'figure7_chunk_mode_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"✅ Saved Figure 7: {output_path / 'figure7_chunk_mode_comparison.png'}")


def create_summary_table(df: pd.DataFrame, output_path: Path):
    """Create LaTeX table for main results"""
    
    # Select main experiments
    main_exps = df[df['experiment'].isin([
        'exp1_dense_only', 'exp2_rrf_equal', 'exp3_rrf_downweight',
        'exp5_sentence_clean', 'exp4_sentence_chunks_mpnet', 'exp4_sentence_chunks'
    ])].copy()
    
    if main_exps.empty:
        return
    
    # Prepare table data
    table_data = []
    for _, row in main_exps.iterrows():
        table_data.append({
            'Experiment': row['experiment'].replace('_', ' ').title(),
            'Chunk': row['chunk_mode'].capitalize(),
            'w_{BM25}': f"{row['rrf_w_bm25']:.1f}",
            'Dense': f"{row['dense_ndcg']:.3f}",
            'Hybrid': f"{row['hybrid_ndcg']:.3f}",
            'Verified': f"{row['verified_ndcg']:.3f}",
            'Router': f"{row['router_ndcg']:.3f}",
            'Zero%': f"{row['dense_zero_rate']*100:.1f}",
        })
    
    # Create DataFrame
    table_df = pd.DataFrame(table_data)
    
    # Save as CSV
    table_df.to_csv(output_path / 'table_main_results.csv', index=False)
    
    # Create LaTeX
    latex_str = table_df.to_latex(index=False, escape=False, 
                                    caption='Main Experimental Results',
                                    label='tab:main_results')
    
    with open(output_path / 'table_main_results.tex', 'w') as f:
        f.write(latex_str)
    
    print(f"✅ Saved Table: {output_path / 'table_main_results.csv'}")
    print(f"✅ Saved LaTeX: {output_path / 'table_main_results.tex'}")


def create_all_figures(runs_dir: str = "runs", output_dir: str = "figures"):
    """Main function to create all figures"""
    
    runs_path = Path(runs_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("="*80)
    print("CREATING PUBLICATION FIGURES")
    print("="*80)
    
    # Load data
    print("\n📂 Loading experimental results...")
    df = load_experiment_results(runs_path)
    
    if df.empty:
        print("❌ Error: No experimental results found!")
        return
    
    print(f"✅ Loaded {len(df)} experiments")
    print(f"   Experiments: {', '.join(df['experiment'].tolist())}")
    
    # Create figures
    print("\n📊 Generating figures...")
    
    try:
        figure1_fusion_degradation(df, output_path)
    except Exception as e:
        print(f"⚠️  Figure 1 failed: {e}")
    
    try:
        figure2_architecture_hierarchy(df, output_path)
    except Exception as e:
        print(f"⚠️  Figure 2 failed: {e}")
    
    try:
        figure3_retrieval_comparison(df, output_path)
    except Exception as e:
        print(f"⚠️  Figure 3 failed: {e}")
    
    try:
        figure4_predictor_performance(df, output_path)
    except Exception as e:
        print(f"⚠️  Figure 4 failed: {e}")
    
    try:
        figure5_efficiency_frontier(df, output_path)
    except Exception as e:
        print(f"⚠️  Figure 5 failed: {e}")
    
    try:
        figure6_reranker_interaction(df, output_path)
    except Exception as e:
        print(f"⚠️  Figure 6 failed: {e}")
    
    try:
        figure7_chunk_mode_comparison(df, output_path)
    except Exception as e:
        print(f"⚠️  Figure 7 failed: {e}")
    
    try:
        create_summary_table(df, output_path)
    except Exception as e:
        print(f"⚠️  Table creation failed: {e}")
    
    print("\n" + "="*80)
    print("✅ FIGURE GENERATION COMPLETE")
    print("="*80)
    print(f"\n📁 All figures saved to: {output_path.resolve()}/")
    print("\nGenerated files:")
    for f in sorted(output_path.glob('figure*.png')):
        print(f"  ✓ {f.name}")
    for f in sorted(output_path.glob('table*.csv')):
        print(f"  ✓ {f.name}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate publication figures from experimental results')
    parser.add_argument('--runs_dir', type=str, default='runs',
                       help='Directory containing experimental results (default: runs)')
    parser.add_argument('--output_dir', type=str, default='figures',
                       help='Output directory for figures (default: figures)')
    
    args = parser.parse_args()
    
    create_all_figures(runs_dir=args.runs_dir, output_dir=args.output_dir)
