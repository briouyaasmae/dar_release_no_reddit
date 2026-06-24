import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd

def _safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

def find_run_dirs(roots: List[str]) -> List[Path]:
    run_dirs: List[Path] = []
    for root in roots:
        p = Path(root)
        if not p.exists():
            continue
        # If root itself is a run folder
        if (p / "routing_report.json").exists():
            run_dirs.append(p)
        # Otherwise search recursively
        for rp in p.rglob("routing_report.json"):
            run_dirs.append(rp.parent)
    # Deduplicate while preserving order
    seen = set()
    unique: List[Path] = []
    for d in run_dirs:
        s = str(d.resolve())
        if s in seen:
            continue
        seen.add(s)
        unique.append(d)
    return unique

def load_report(run_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    routing_report = {}
    zero_stats = {}
    f = run_dir / "routing_report.json"
    if f.exists():
        with open(f, "r", encoding="utf-8") as fh:
            routing_report = json.load(fh)
    fz = run_dir / "zero_ndcg_stats.json"
    if fz.exists():
        with open(fz, "r", encoding="utf-8") as fh:
            zero_stats = json.load(fh)
    return routing_report, zero_stats

def flatten_retrieval(run_name: str, report: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    retr = report.get("retrieval", {}) or {}
    for tier in ["sparse", "hybrid", "hybrid_verified"]:
        if tier in retr:
            m = retr[tier]
            rows.append({
                "run": run_name,
                "tier": tier,
                "mean_ndcg@10": m.get("mean_ndcg@10"),
                "mean_ndcg@10_ci_low": m.get("mean_ndcg@10_ci_low"),
                "mean_ndcg@10_ci_high": m.get("mean_ndcg@10_ci_high"),
                "precision@10": m.get("precision@10"),
                "recall@10": m.get("recall@10"),
                "zero_rate": m.get("zero_rate"),
            })
    return rows

def flatten_routing(run_name: str, report: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    routing = report.get("routing", {}) or {}
    for strat_key, strat in routing.items():
        if not isinstance(strat, dict):
            continue
        rows.append({
            "run": run_name,
            "strategy": strat_key,
            "avg_cost": strat.get("avg_cost"),
            "avg_ndcg@10_auto": strat.get("avg_ndcg@10_auto"),
            "avg_ndcg@10_overall": strat.get("avg_ndcg@10_overall"),
            "efficiency": strat.get("efficiency"),
            "auto_ratio": strat.get("auto_ratio"),
            "crisis_ratio": strat.get("crisis_ratio"),
        })
    return rows

def flatten_predictors(run_name: str, report: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    pred = report.get("predictor", {}) or {}
    for name, m in pred.items():
        if not isinstance(m, dict):
            continue
        rows.append({
            "run": run_name,
            "predictor": name,  # 'hybrid' | 'sparse'
            "rmse_raw": m.get("rmse_raw"),
            "mae_raw": m.get("mae_raw"),
            "rmse_cal": m.get("rmse_cal"),
            "mae_cal": m.get("mae_cal"),
            "pearson_raw": m.get("pearson_raw"),
            "pearson_cal": m.get("pearson_cal"),
            "spearman_raw": m.get("spearman_raw"),
            "spearman_cal": m.get("spearman_cal"),
            "pearson_raw_ci_low": m.get("pearson_raw_ci_low"),
            "pearson_raw_ci_high": m.get("pearson_raw_ci_high"),
            "pearson_cal_ci_low": m.get("pearson_cal_ci_low"),
            "pearson_cal_ci_high": m.get("pearson_cal_ci_high"),
            "spearman_raw_ci_low": m.get("spearman_raw_ci_low"),
            "spearman_raw_ci_high": m.get("spearman_raw_ci_high"),
            "spearman_cal_ci_low": m.get("spearman_cal_ci_low"),
            "spearman_cal_ci_high": m.get("spearman_cal_ci_high"),
        })
    return rows

def flatten_config(run_name: str, report: Dict[str, Any]) -> Dict[str, Any]:
    cfg = report.get("config", {}) or {}
    # Pick a focused subset that is useful for comparison
    keep_keys = [
        "chunk_mode","min_chunk_words","max_chunk_words",
        "overlap_chunk_words","overlap_size",
        "topk_bm25","topk_dense","topk_fusion","rrf_k",
        "verified_topn","verified_max_len","qpp_post_k",
        "hybrid_mode","verified_source","rrf_w_bm25","rrf_w_dense",
        "thr_sparse","thr_hybrid","safety_override_thr","dual_delta",
        "cost_sparse","cost_hybrid","cost_verified",
        "enable_crisis_tier","user_country","device","seed"
    ]
    out = {k: cfg.get(k) for k in keep_keys if k in cfg}
    out["run"] = run_name
    return out

def flatten_crisis(run_name: str, report: Dict[str, Any]) -> Dict[str, Any]:
    cr = report.get("crisis", {}) or {}
    return {
        "run": run_name,
        "crisis_enabled": cr.get("enabled"),
        "user_country": cr.get("user_country"),
        "crisis_docs_count": cr.get("crisis_docs_count"),
        "crisis_outputs_json": cr.get("crisis_outputs_json"),
    }

def flatten_zero_stats(run_name: str, zero_stats: Dict[str, Any]) -> Dict[str, Any]:
    overall = zero_stats.get("Overall", {}) if isinstance(zero_stats, dict) else {}
    safety = zero_stats.get("Safety", {}) if isinstance(zero_stats, dict) else {}
    qlen = zero_stats.get("Query_Length", {}) if isinstance(zero_stats, dict) else {}
    cov = zero_stats.get("Corpus_Coverage", {}) if isinstance(zero_stats, dict) else {}
    pred = zero_stats.get("Prediction_Quality", {}) if isinstance(zero_stats, dict) else {}
    return {
        "run": run_name,
        "zero_total_queries": overall.get("total_queries"),
        "zero_count": overall.get("zero_count"),
        "zero_rate": overall.get("zero_rate"),
        "high_risk_zeros": safety.get("high_risk_zeros"),
        "high_risk_zero_rate": safety.get("high_risk_zero_rate"),
        "zero_len_mean": qlen.get("zero_mean"),
        "nonzero_len_mean": qlen.get("nonzero_mean"),
        "very_short_zero_rate": qlen.get("very_short_zero_rate"),
        "zero_mean_relevant": cov.get("zero_mean_relevant"),
        "nonzero_mean_relevant": cov.get("nonzero_mean_relevant"),
        "one_chunk_zero_rate": cov.get("one_chunk_zero_rate"),
        "zero_mean_predicted": pred.get("zero_mean_predicted"),
        "nonzero_mean_predicted": pred.get("nonzero_mean_predicted"),
        "overconfident_zeros": pred.get("overconfident_zeros"),
    }

def flatten_chunk_ablation(run_name: str, report: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    ab = report.get("chunk_ablation", {}) or {}
    for mode, metrics in ab.items():
        # expect the metrics to look like run-level retrieval stats for the chosen tier
        row = {"run": run_name, "chunk_mode_ablation": mode}
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                row[k] = v
        out.append(row)
    return out

def flatten_ablations(run_name: str, report: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    abl = report.get("ablations", {}) or {}
    for name, metrics in abl.items():
        row = {"run": run_name, "ablation": name}
        if isinstance(metrics, dict):
            row.update(metrics)
        out.append(row)
    return out

def aggregate(roots: List[str]) -> Dict[str, pd.DataFrame]:
    run_dirs = find_run_dirs(roots)
    if not run_dirs:
        # Build empty frames with expected columns
        return {
            "retrieval": pd.DataFrame(columns=["run","tier","mean_ndcg@10","precision@10","recall@10","zero_rate"]),
            "routing": pd.DataFrame(columns=["run","strategy","avg_cost","avg_ndcg@10_auto","avg_ndcg@10_overall","efficiency","auto_ratio","crisis_ratio"]),
            "predictors": pd.DataFrame(columns=["run","predictor","rmse_raw","rmse_cal","pearson_cal","spearman_cal"]),
            "config": pd.DataFrame(columns=["run"]),
            "crisis": pd.DataFrame(columns=["run"]),
            "zero_stats": pd.DataFrame(columns=["run","zero_rate"]),
            "chunk_ablation": pd.DataFrame(columns=["run","chunk_mode_ablation","mean_ndcg@10"]),
            "ablations": pd.DataFrame(columns=["run","ablation","avg_ndcg@10_auto","efficiency"]),
        }

    all_retr, all_rout, all_pred, all_cfg, all_cr, all_zero, all_chunkabl, all_abl = [], [], [], [], [], [], [], []

    for d in run_dirs:
        run_name = d.name
        report, zero_stats = load_report(d)

        all_retr += flatten_retrieval(run_name, report)
        all_rout += flatten_routing(run_name, report)
        all_pred += flatten_predictors(run_name, report)
        all_cfg.append(flatten_config(run_name, report))
        all_cr.append(flatten_crisis(run_name, report))
        if zero_stats:
            all_zero.append(flatten_zero_stats(run_name, zero_stats))
        all_chunkabl += flatten_chunk_ablation(run_name, report)
        all_abl += flatten_ablations(run_name, report)

    df_retrieval = pd.DataFrame(all_retr).sort_values(["run","tier"])
    df_routing   = pd.DataFrame(all_rout).sort_values(["run","strategy"])
    df_predict   = pd.DataFrame(all_pred).sort_values(["run","predictor"])
    df_config    = pd.DataFrame(all_cfg).sort_values(["run"])
    df_crisis    = pd.DataFrame(all_cr).sort_values(["run"])
    df_zero      = pd.DataFrame(all_zero).sort_values(["run"]) if all_zero else pd.DataFrame(columns=["run"])
    df_chunkabl  = pd.DataFrame(all_chunkabl).sort_values(["run","chunk_mode_ablation"]) if all_chunkabl else pd.DataFrame(columns=["run"])
    df_abl       = pd.DataFrame(all_abl).sort_values(["run","ablation"]) if all_abl else pd.DataFrame(columns=["run"])

    return {
        "retrieval": df_retrieval,
        "routing": df_routing,
        "predictors": df_predict,
        "config": df_config,
        "crisis": df_crisis,
        "zero_stats": df_zero,
        "chunk_ablation": df_chunkabl,
        "ablations": df_abl,
    }

def save_outputs(dfs: Dict[str, pd.DataFrame], outdir: Path) -> List[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    # CSVs
    for name, df in dfs.items():
        p = outdir / f"{name}.csv"
        df.to_csv(p, index=False)
        paths.append(p)
    # Excel workbook with one sheet per table
    xlsx = outdir / "aggregated_results.xlsx"
    with pd.ExcelWriter(xlsx, engine="xlsxwriter") as w:
        for name, df in dfs.items():
            # Excel limits sheet names to 31 chars
            sheet = name[:31]
            df.to_excel(w, sheet_name=sheet, index=False)
    paths.append(xlsx)
    return paths

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=["paper_experiments/runs", "runs"], help="Root folders to search for run outputs")
    ap.add_argument("--outdir", default="/mnt/data/aggregated_results", help="Where to write CSVs/XLSX")
    args = ap.parse_args()

    dfs = aggregate(args.roots)
    paths = save_outputs(dfs, Path(args.outdir))

    # Print a tiny summary to stdout
    print("Found tables:")
    for name, df in dfs.items():
        print(f"- {name}: {len(df)} rows")

    print("\\nSaved files:")
    for p in paths:
        print(f"  {p}")

if __name__ == "__main__":
    main()
