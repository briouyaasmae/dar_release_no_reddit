#!/usr/bin/env python3
"""
Zero-nDCG Visualizations
Creates plots to understand retrieval failures.
Robust to: route category mismatches, missing seaborn, categorical FutureWarnings,
and zero-coverage bins.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# seaborn optional
try:
    import seaborn as sns
    HAS_SNS = True
except Exception:
    HAS_SNS = False


def _set_style():
    if HAS_SNS:
        sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (16, 12)


def _aligned_value_counts(series: pd.Series, categories: list) -> pd.Series:
    """Return value_counts aligned to supplied categories (fill 0s)."""
    vc = series.value_counts(dropna=False)
    return vc.reindex(categories, fill_value=0)


def create_zero_ndcg_visualizations(workdir: Path = Path("mhqpp_out")):
    """
    Create comprehensive visualizations of zero-nDCG patterns.
    Requires: predictions.parquet, routes.csv, qrels.parquet
    """
    # ---------- Load ----------
    preds = pd.read_parquet(workdir / "predictions.parquet")
    routes = pd.read_csv(workdir / "routes.csv")
    qrels = pd.read_parquet(workdir / "qrels.parquet")

    # Dtypes/merge safety
    for df in (preds, routes, qrels):
        if "qid" in df.columns:
            df["qid"] = df["qid"].astype(str)

    # pick route column
    route_col = "route_dual" if "route_dual" in routes.columns else (
        "route_single" if "route_single" in routes.columns else None
    )
    if route_col is None:
        raise ValueError("routes.csv must contain either 'route_dual' or 'route_single'.")

    preds = preds.merge(routes[["qid", route_col]], on="qid", how="left")
    preds.rename(columns={route_col: "route"}, inplace=True)

    # Derived features
    preds["query"] = preds["query"].astype(str)
    preds["query_len"] = preds["query"].str.split().str.len()
    preds["is_zero"] = (preds["true_ndcg10_hybrid"] == 0).astype(int)

    # Relevant chunk counts per qid
    rel_counts = qrels.groupby("qid", observed=False).size().to_dict()
    preds["n_relevant"] = preds["qid"].map(rel_counts).fillna(0).astype(int)

    # ---------- Plots ----------
    _set_style()
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle("Zero-nDCG Query Analysis Dashboard", fontsize=16, fontweight="bold")

    # 1) Query length distribution
    ax = axes[0, 0]
    zero_q = preds.loc[preds["is_zero"] == 1, "query_len"]
    nonzero_q = preds.loc[preds["is_zero"] == 0, "query_len"]
    ax.hist([nonzero_q, zero_q], bins=30, label=["Non-zero", "Zero"], alpha=0.7)
    if len(zero_q):
        ax.axvline(zero_q.mean(), linestyle="--", linewidth=2, label=f"Zero mean: {zero_q.mean():.1f}")
    if len(nonzero_q):
        ax.axvline(nonzero_q.mean(), linestyle="--", linewidth=2, label=f"Non-zero mean: {nonzero_q.mean():.1f}")
    ax.set_xlabel("Query Length (words)")
    ax.set_ylabel("Count")
    ax.set_title("Query Length Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2) Relevant chunks vs nDCG
    ax = axes[0, 1]
    scatter_data = preds.sample(min(500, len(preds)), random_state=42)
    ax.scatter(
        scatter_data["n_relevant"], scatter_data["true_ndcg10_hybrid"],
        c=scatter_data["is_zero"].map({0: "#2ecc71", 1: "#e74c3c"}), alpha=0.5, s=30
    )
    ax.axhline(0, linestyle="--", alpha=0.5, linewidth=2)
    ax.set_xlabel("Number of Relevant Chunks")
    ax.set_ylabel("nDCG@10 (Hybrid)")
    ax.set_title("Corpus Coverage vs Retrieval Quality")
    ax.grid(True, alpha=0.3)

    # 3) Calibration
    ax = axes[0, 2]
    ax.scatter(
        preds["pred_hybrid_cal"], preds["true_ndcg10_hybrid"],
        c=preds["is_zero"].map({0: "#2ecc71", 1: "#e74c3c"}), alpha=0.3, s=20
    )
    ax.plot([0, 1], [0, 1], "k--", linewidth=2, label="Perfect calibration")
    ax.set_xlabel("Predicted nDCG@10")
    ax.set_ylabel("True nDCG@10")
    ax.set_title("Prediction Calibration (Zero vs Non-zero)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4) Zero rate by query length bins
    ax = axes[1, 0]
    preds["len_bin"] = pd.cut(
        preds["query_len"], bins=[0, 10, 20, 30, 50, 100, 1000],
        labels=["<10", "10-20", "20-30", "30-50", "50-100", ">100"],
        include_lowest=True, right=True
    )
    zero_rate_by_len = preds.groupby("len_bin", observed=False)["is_zero"].mean()
    counts_by_len = preds.groupby("len_bin", observed=False).size()
    bars = ax.bar(range(len(zero_rate_by_len)), zero_rate_by_len.values, alpha=0.7)
    ax.set_xticks(range(len(zero_rate_by_len)))
    ax.set_xticklabels(zero_rate_by_len.index.astype(str), rotation=45)
    ax.set_ylabel("Zero nDCG Rate")
    ax.set_xlabel("Query Length Bin")
    ax.set_title("Zero Rate by Query Length")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, count in zip(bars, counts_by_len):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"n={count}",
                ha="center", va="bottom", fontsize=8)

    # 5) Zero rate by relevant chunk count (include 0)
    ax = axes[1, 1]
    preds["rel_bin"] = pd.cut(
        preds["n_relevant"],
        bins=[-0.1, 0.5, 1.5, 2.5, 3.5, 5.5, 100],
        labels=["0", "1", "2", "3", "4-5", ">5"],
        include_lowest=True, right=True,
    )
    zero_rate_by_rel = preds.groupby("rel_bin", observed=False)["is_zero"].mean()
    counts_by_rel = preds.groupby("rel_bin", observed=False).size()
    bars = ax.bar(range(len(zero_rate_by_rel)), zero_rate_by_rel.values, alpha=0.7)
    ax.set_xticks(range(len(zero_rate_by_rel)))
    ax.set_xticklabels(zero_rate_by_rel.index.astype(str))
    ax.set_ylabel("Zero nDCG Rate")
    ax.set_xlabel("Relevant Chunks")
    ax.set_title("Zero Rate by Corpus Coverage")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, count in zip(bars, counts_by_rel):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"n={count}",
                ha="center", va="bottom", fontsize=8)

    # 6) Prediction error distribution
    ax = axes[1, 2]
    preds["pred_error"] = preds["pred_hybrid_cal"] - preds["true_ndcg10_hybrid"]
    zero_err = preds.loc[preds["is_zero"] == 1, "pred_error"]
    nonzero_err = preds.loc[preds["is_zero"] == 0, "pred_error"]
    data = [
        nonzero_err.values if len(nonzero_err) else [0.0],
        zero_err.values if len(zero_err) else [0.0],
    ]
    bp = ax.boxplot(data, labels=["Non-zero", "Zero"], patch_artist=True)
    if len(bp["boxes"]) >= 2:
        bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_alpha(0.7)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Prediction Error")
    ax.set_title("Prediction Error Distribution")
    ax.grid(True, alpha=0.3, axis="y")

    # 7) High-risk distribution
    ax = axes[2, 0]
    zero_risk = preds.loc[preds["is_zero"] == 1, "high_risk_prob"]
    nonzero_risk = preds.loc[preds["is_zero"] == 0, "high_risk_prob"]
    ax.hist([nonzero_risk, zero_risk], bins=20, label=["Non-zero", "Zero"], alpha=0.7)
    ax.axvline(0.3, linestyle="--", linewidth=2, label="Safety threshold (0.3)")
    ax.set_xlabel("High-Risk Probability")
    ax.set_ylabel("Count")
    ax.set_title("High-Risk Score Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 8) Routing distribution (align categories)
    ax = axes[2, 1]
    zero_routes = preds.loc[preds["is_zero"] == 1, "route"]
    nonzero_routes = preds.loc[preds["is_zero"] == 0, "route"]
    cats = sorted(set(zero_routes.dropna().unique()).union(set(nonzero_routes.dropna().unique())))
    if len(cats) == 0:
        ax.text(0.5, 0.5, "No routing data", ha="center", va="center")
        ax.set_axis_off()
    else:
        zr = _aligned_value_counts(zero_routes, cats)
        nr = _aligned_value_counts(nonzero_routes, cats)
        x = np.arange(len(cats))
        w = 0.35
        ax.bar(x - w / 2, zr.values, w, label="Zero", alpha=0.7)
        ax.bar(x + w / 2, nr.values, w, label="Non-zero", alpha=0.7)
        ax.set_xlabel("Route")
        ax.set_ylabel("Count")
        ax.set_title("Routing Distribution by Zero Status")
        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    # 9) Cumulative distribution
    ax = axes[2, 2]
    if len(preds):
        sorted_preds = preds.sort_values("true_ndcg10_hybrid")
        cum_zero = (sorted_preds["is_zero"] == 1).cumsum() / len(sorted_preds)
        cum_all = np.arange(1, len(sorted_preds) + 1) / len(sorted_preds)
        ax.plot(sorted_preds["true_ndcg10_hybrid"].values, cum_all, label="All queries", linewidth=2)
        ax.fill_between(sorted_preds["true_ndcg10_hybrid"].values, 0, cum_zero.values,
                        alpha=0.3, label="Zero queries (shaded)")
        ax.axhline(cum_zero.iloc[-1], linestyle="--", linewidth=2, label=f"Zero rate: {cum_zero.iloc[-1]:.1%}")
        ax.set_xlabel("nDCG@10 (Hybrid)")
        ax.set_ylabel("Cumulative Proportion")
        ax.set_title("Cumulative Distribution of Retrieval Quality")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center"); ax.set_axis_off()

    plt.tight_layout()
    out_png = workdir / "zero_ndcg_visualizations.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"\n✅ Saved visualizations to: {out_png}")

    # ---------- Text report ----------
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("ZERO-nDCG QUERY EXAMPLES BY PATTERN")
    report_lines.append("=" * 80)

    # very short zeros
    short_zeros = preds[(preds["is_zero"] == 1) & (preds["query_len"] < 10)]
    if len(short_zeros):
        report_lines.append(f"\n1. VERY SHORT QUERIES (n={len(short_zeros)}, {100*len(short_zeros)/len(preds):.1f}% of all)")
        report_lines.append("-" * 80)
        for _, r in short_zeros.head(5).iterrows():
            report_lines.append(f"\n   Query ({r['query_len']} w): {str(r['query'])[:150]}")
            report_lines.append(f"   Pred nDCG: {r['pred_hybrid_cal']:.3f} | Relevant: {int(r['n_relevant'])}")

    # long with few relevant
    long_few = preds[(preds["is_zero"] == 1) & (preds["query_len"] > 50) & (preds["n_relevant"] <= 2)]
    if len(long_few):
        report_lines.append(f"\n\n2. LONG QUERIES WITH FEW RELEVANT (n={len(long_few)})")
        report_lines.append("-" * 80)
        for _, r in long_few.head(5).iterrows():
            report_lines.append(f"\n   Query ({r['query_len']} w): {str(r['query'])[:150]}...")
            report_lines.append(f"   Pred nDCG: {r['pred_hybrid_cal']:.3f} | Relevant: {int(r['n_relevant'])}")

    # high confidence but zero
    high_conf_zero = preds[(preds["is_zero"] == 1) & (preds["pred_hybrid_cal"] > 0.5)]
    if len(high_conf_zero):
        report_lines.append(f"\n\n3. PREDICTION FAILURES (pred>0.5 but zero, n={len(high_conf_zero)})")
        report_lines.append("-" * 80)
        for _, r in high_conf_zero.head(5).iterrows():
            report_lines.append(f"\n   Query ({r['query_len']} w): {str(r['query'])[:150]}")
            report_lines.append(f"   Pred nDCG: {r['pred_hybrid_cal']:.3f} | Relevant: {int(r['n_relevant'])}")

    # only 1 relevant
    one_chunk = preds[(preds["is_zero"] == 1) & (preds["n_relevant"] == 1)]
    if len(one_chunk):
        report_lines.append(f"\n\n4. QUERIES WITH ONLY 1 RELEVANT CHUNK (n={len(one_chunk)})")
        report_lines.append("-" * 80)
        for _, r in one_chunk.head(5).iterrows():
            report_lines.append(f"\n   Query ({r['query_len']} w): {str(r['query'])[:150]}")
            report_lines.append(f"   Pred nDCG: {r['pred_hybrid_cal']:.3f} | Route: {r['route']}")

    # high-risk zeros
    high_risk_zero = preds[(preds["is_zero"] == 1) & (preds["high_risk_prob"] > 0.2)]
    if len(high_risk_zero):
        report_lines.append(f"\n\n5. HIGH-RISK QUERIES WITH ZERO nDCG (n={len(high_risk_zero)})")
        report_lines.append("-" * 80)
        report_lines.append("   ⚠️  These are concerning - high-risk queries need good retrieval!")
        for _, r in high_risk_zero.head(5).iterrows():
            report_lines.append(f"\n   Query ({r['query_len']} w): {str(r['query'])[:150]}")
            report_lines.append(f"   Risk: {r['high_risk_prob']:.3f} | Route: {r['route']} | Relevant: {int(r['n_relevant'])}")

    (workdir / "zero_ndcg_examples.txt").write_text("\n".join(report_lines))
    print(f"✅ Saved example queries to: {workdir}/zero_ndcg_examples.txt")

    # ---------- Statistical summary ----------
    summary = {
        "Overall": {
            "total_queries": int(len(preds)),
            "zero_count": int((preds['is_zero'] == 1).sum()),
            "zero_rate": float((preds['is_zero'] == 1).mean()),
        },
        "Query_Length": {
            "zero_mean": float(zero_q.mean() if len(zero_q) else 0.0),
            "nonzero_mean": float(nonzero_q.mean() if len(nonzero_q) else 0.0),
            "very_short_zero_rate": float(
                preds.loc[(preds["is_zero"] == 1) & (preds["query_len"] < 10)].shape[0] /
                max(1, preds.loc[preds["query_len"] < 10].shape[0])
            ),
        },
        "Corpus_Coverage": {
            "zero_mean_relevant": float(preds.loc[preds["is_zero"] == 1, "n_relevant"].mean() if len(zero_q) else 0.0),
            "nonzero_mean_relevant": float(preds.loc[preds["is_zero"] == 0, "n_relevant"].mean() if len(nonzero_q) else 0.0),
            "one_chunk_zero_rate": float(
                preds.loc[(preds["is_zero"] == 1) & (preds["n_relevant"] == 1)].shape[0] /
                max(1, preds.loc[preds["n_relevant"] == 1].shape[0])
            ),
        },
        "Prediction_Quality": {
            "zero_mean_predicted": float(preds.loc[preds["is_zero"] == 1, "pred_hybrid_cal"].mean() if len(zero_q) else 0.0),
            "nonzero_mean_predicted": float(preds.loc[preds["is_zero"] == 0, "pred_hybrid_cal"].mean() if len(nonzero_q) else 0.0),
            "overconfident_zeros": int(((preds["is_zero"] == 1) & (preds["pred_hybrid_cal"] > 0.5)).sum()),
        },
        "Safety": {
            "high_risk_zeros": int(((preds["is_zero"] == 1) & (preds["high_risk_prob"] > 0.2)).sum()),
            "high_risk_zero_rate": float(
                ((preds["is_zero"] == 1) & (preds["high_risk_prob"] > 0.2)).sum() /
                max(1, (preds["high_risk_prob"] > 0.2).sum())
            ),
        },
    }
    (workdir / "zero_ndcg_stats.json").write_text(json.dumps(summary, indent=2))
    print(f"✅ Saved statistical summary to: {workdir}/zero_ndcg_stats.json")

    # Key findings (stdout)
    print(f"\n{'='*80}\nKEY FINDINGS\n{'='*80}")
    print(f"\n📊 Overall: {summary['Overall']['zero_rate']:.1%} queries get zero nDCG@10")
    print(f"\n📏 Query Length:")
    print(f"   - Zero queries: {summary['Query_Length']['zero_mean']:.1f} words (avg)")
    print(f"   - Non-zero queries: {summary['Query_Length']['nonzero_mean']:.1f} words (avg)")
    print(f"   - Very short (<10 words) zero rate: {summary['Query_Length']['very_short_zero_rate']:.1%}")
    print(f"\n📚 Corpus Coverage:")
    print(f"   - Zero queries: {summary['Corpus_Coverage']['zero_mean_relevant']:.1f} relevant chunks (avg)")
    print(f"   - Non-zero queries: {summary['Corpus_Coverage']['nonzero_mean_relevant']:.1f} relevant chunks (avg)")
    print(f"   - Queries with 1 chunk zero rate: {summary['Corpus_Coverage']['one_chunk_zero_rate']:.1%}")
    print(f"\n🎯 Prediction Quality:")
    print(f"   - Model predicts {summary['Prediction_Quality']['zero_mean_predicted']:.3f} for zero queries")
    print(f"   - Model predicts {summary['Prediction_Quality']['nonzero_mean_predicted']:.3f} for non-zero queries")
    print(f"   - Overconfident failures (pred>0.5 but zero): {summary['Prediction_Quality']['overconfident_zeros']}")
    print(f"\n⚠️  Safety:")
    print(f"   - High-risk queries with zero nDCG: {summary['Safety']['high_risk_zeros']}")
    print(f"   - High-risk zero rate: {summary['Safety']['high_risk_zero_rate']:.1%}")

    return summary


if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    workdir = Path(args[0]) if args else Path("mhqpp_out")

    if not workdir.exists():
        print(f"❌ Error: Directory not found: {workdir}")
        raise SystemExit(1)

    req = ["predictions.parquet", "routes.csv", "qrels.parquet"]
    missing = [f for f in req if not (workdir / f).exists()]
    if missing:
        print(f"❌ Error: Missing required files: {', '.join(missing)}")
        raise SystemExit(1)

    create_zero_ndcg_visualizations(workdir)
