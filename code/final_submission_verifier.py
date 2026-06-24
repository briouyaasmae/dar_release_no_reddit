#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import shlex
import zipfile
from pathlib import Path

import pandas as pd


OUT = Path("paper_experiments/reviewer_revision_outputs")
OUT.mkdir(parents=True, exist_ok=True)


def parse_flags(cmd):
    toks = shlex.split(cmd)
    out = {}
    i = 0

    while i < len(toks):
        if toks[i].startswith("--"):
            key = toks[i][2:].replace("-", "_")

            if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                out[key] = toks[i + 1]
                i += 2
            else:
                out[key] = True
                i += 1
        else:
            i += 1

    return out


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_workdir(raw_workdir):
    """
    Resolve the workdir from the manifest.

    Some older commands saved dense_comparison_tier3 as:
        paper_experiments/dense_comparison_tier3

    but the actual folder may now be:
        paper_experiments/runs/dense_comparison_tier3

    This function first tries the manifest path, then falls back to
    paper_experiments/runs/<folder_name>.
    """
    wd = Path(raw_workdir)

    if (wd / "routing_report.json").exists():
        return wd, False

    alt = Path("paper_experiments/runs") / wd.name
    if (alt / "routing_report.json").exists():
        return alt, True

    return wd, False


def values_match(a, b):
    """
    Compare manifest command values with report config values.
    Handles exact string matches and numeric equivalence.
    """
    a = str(a)
    b = str(b)

    if a.lower() == b.lower():
        return True

    if a.lower() == b.lower() + ".0":
        return True

    if b.lower() == a.lower() + ".0":
        return True

    try:
        return abs(float(a) - float(b)) <= 1e-9
    except Exception:
        return False


manifest_path = Path("paper_experiments/experiment_manifest.json")
if not manifest_path.exists():
    raise FileNotFoundError(
        "Missing paper_experiments/experiment_manifest.json"
    )

with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)


summary = []
verify = []

for name, cmd in manifest.items():
    flags = parse_flags(cmd)

    if "workdir" not in flags:
        verify.append(
            {
                "experiment": name,
                "status": "missing_workdir_in_manifest",
                "workdir": "",
                "resolved_from_runs": False,
                "mismatches": "",
            }
        )
        continue

    wd, resolved_from_runs = resolve_workdir(flags["workdir"])
    report_path = wd / "routing_report.json"

    if not report_path.exists():
        verify.append(
            {
                "experiment": name,
                "status": "missing_report",
                "workdir": str(wd),
                "resolved_from_runs": resolved_from_runs,
                "mismatches": "",
            }
        )
        continue

    rep = load_json(report_path)
    cfg = rep.get("config", {})

    keys = [
        "chunk_mode",
        "topk_bm25",
        "topk_dense",
        "topk_fusion",
        "rrf_k",
        "verified_topn",
        "verified_max_len",
        "qpp_post_k",
        "hybrid_mode",
        "verified_source",
        "rrf_w_bm25",
        "rrf_w_dense",
        "thr_sparse",
        "thr_hybrid",
        "safety_override_thr",
        "dual_delta",
        "limit_queries",
    ]

    mismatches = []

    for k in keys:
        if k in flags and k in cfg:
            if not values_match(flags[k], cfg[k]):
                mismatches.append(
                    f"{k}: command={flags[k]}, report={cfg[k]}"
                )

    verify.append(
        {
            "experiment": name,
            "status": "ok" if not mismatches else "mismatch",
            "workdir": str(wd),
            "resolved_from_runs": resolved_from_runs,
            "mismatches": "; ".join(mismatches),
        }
    )

    routing = rep.get("routing", {})
    retrieval = rep.get("retrieval", {})
    pred = rep.get("predictor", {})

    dual = routing.get("dual", {})
    single = routing.get("single", {})

    row = {
        "experiment": name,
        "workdir": str(wd),
        "resolved_from_runs": resolved_from_runs,
        "chunk_mode": cfg.get("chunk_mode"),
        "hybrid_mode": cfg.get("hybrid_mode"),
        "verified_source": cfg.get("verified_source"),
    }

    for run_name, rv in retrieval.items():
        if isinstance(rv, dict):
            row[f"{run_name}_ndcg"] = rv.get("mean_ndcg@10")

    row.update(
        {
            "dar_dual_ndcg_overall": dual.get("avg_ndcg@10_overall"),
            "dar_dual_ndcg_auto": dual.get("avg_ndcg@10_auto"),
            "dar_dual_cost": dual.get(
                "avg_cost_overall",
                dual.get("avg_cost"),
            ),
            "dar_dual_efficiency": dual.get(
                "efficiency_overall",
                dual.get("efficiency"),
            ),
            "dar_dual_crisis_ratio": dual.get("crisis_ratio"),
            "single_ndcg_overall": single.get("avg_ndcg@10_overall"),
        }
    )

    if "sparse" in pred:
        row["sparse_qpp_pearson_cal"] = pred["sparse"].get("pearson_cal")

    if "hybrid" in pred:
        row["hybrid_qpp_pearson_cal"] = pred["hybrid"].get("pearson_cal")

    # Optional dense model comparison block.
    dense_cmp = rep.get("dense_model_comparison", {})
    if not dense_cmp:
        dense_cmp = rep.get("contributions", {}).get("dense_model_comparison", {})

    if dense_cmp:
        row["has_dense_model_comparison"] = True

        if isinstance(dense_cmp, dict):
            for k, v in dense_cmp.items():
                if isinstance(v, (int, float, str, bool)) or v is None:
                    row[f"dense_cmp_{k}"] = v
    else:
        row["has_dense_model_comparison"] = False

    paired_path = wd / "paired_significance_tests_oldconfig.json"
    if paired_path.exists():
        paired = load_json(paired_path)
        best = paired.get("dar_vs_best_static", {})

        row["paired_best_baseline"] = best.get("baseline")
        row["paired_mean_delta"] = best.get("mean_delta")
        row["paired_ci_low"] = best.get("ci_low")
        row["paired_ci_high"] = best.get("ci_high")
        row["paired_bootstrap_p"] = best.get("bootstrap_p_two_sided")
        row["paired_wilcoxon_p"] = best.get("wilcoxon_p")
        row["paired_sig_bootstrap"] = best.get("significant_95_bootstrap")

    ans_path = wd / "answer_level_sanity_check_summary_oldconfig.csv"
    if ans_path.exists():
        ans = pd.read_csv(ans_path)

        for _, r in ans.iterrows():
            m = r["method"]

            row[f"answer_{m}_relevance"] = r.get(
                "answer_relevance_tokenF1",
                r.get("answer_relevance_rougeL_or_tokenF1"),
            )
            row[f"answer_{m}_unsupported"] = r.get(
                "unsupported_sentence_rate"
            )

    summary.append(row)


verify_df = pd.DataFrame(verify)
summary_df = pd.DataFrame(summary)

verify_csv = OUT / "configuration_verification.csv"
summary_csv = OUT / "all_experiment_numbers_verified.csv"

verify_df.to_csv(verify_csv, index=False)
summary_df.to_csv(summary_csv, index=False)

print("\nCONFIGURATION VERIFICATION")
print(verify_df.to_string(index=False))

print("\nSUMMARY NUMBERS")
print(summary_df.to_string(index=False))


# LaTeX export.
tex_path = OUT / "verified_experiment_summary.tex"

latex_cols = [
    "experiment",
    "sparse_ndcg",
    "dense_ndcg",
    "hybrid_ndcg",
    "hybrid_verified_ndcg",
    "dar_dual_ndcg_overall",
    "dar_dual_cost",
    "dar_dual_efficiency",
    "paired_best_baseline",
    "paired_mean_delta",
    "paired_bootstrap_p",
    "paired_wilcoxon_p",
]

latex_cols = [c for c in latex_cols if c in summary_df.columns]

with open(tex_path, "w", encoding="utf-8") as f:
    if latex_cols:
        f.write(
            summary_df[latex_cols].to_latex(
                index=False,
                float_format="%.4f",
                escape=False,
            )
        )
    else:
        f.write("% No matching columns available for LaTeX export.\n")


# Response-letter numeric bullets.
bullets_path = OUT / "response_letter_numeric_bullets.txt"

with open(bullets_path, "w", encoding="utf-8") as f:
    for _, r in summary_df.iterrows():
        f.write(
            "Experiment "
            f"{r.get('experiment')} "
            f"({r.get('chunk_mode')}, {r.get('hybrid_mode')}): "
            f"DAR overall nDCG={r.get('dar_dual_ndcg_overall')}, "
            f"cost={r.get('dar_dual_cost')}, "
            f"efficiency={r.get('dar_dual_efficiency')}, "
            f"paired best baseline={r.get('paired_best_baseline')}, "
            f"mean delta={r.get('paired_mean_delta')}, "
            f"bootstrap p={r.get('paired_bootstrap_p')}, "
            f"Wilcoxon p={r.get('paired_wilcoxon_p')}\n"
        )


# Optional compact reviewer-ready table for routing rows.
routing_table_path = OUT / "routing_results_compact.csv"

compact_cols = [
    "experiment",
    "workdir",
    "chunk_mode",
    "hybrid_mode",
    "verified_source",
    "sparse_ndcg",
    "dense_ndcg",
    "hybrid_ndcg",
    "hybrid_verified_ndcg",
    "dar_dual_ndcg_overall",
    "dar_dual_cost",
    "dar_dual_efficiency",
    "paired_best_baseline",
    "paired_mean_delta",
    "paired_ci_low",
    "paired_ci_high",
    "paired_bootstrap_p",
    "paired_wilcoxon_p",
    "paired_sig_bootstrap",
]

compact_cols = [c for c in compact_cols if c in summary_df.columns]
summary_df[compact_cols].to_csv(routing_table_path, index=False)


# Collect important files into ZIP.
zip_path = OUT / "dar_final_revision_verified_outputs.zip"

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in OUT.glob("*"):
        if p.name != zip_path.name:
            z.write(p, arcname=p.name)

    for _, row in verify_df.iterrows():
        wd = Path(row["workdir"])

        for fname in [
            "routing_report.json",
            "paired_significance_tests_oldconfig.json",
            "answer_level_sanity_check_summary_oldconfig.csv",
            "manual_answer_annotation_template_blinded_oldconfig.csv",
            "routes.csv",
            "version_info.json",
        ]:
            p = wd / fname
            if p.exists():
                z.write(p, arcname=f"{wd.name}/{fname}")

print("\n[done] outputs:", OUT.resolve())
print("[configuration]", verify_csv.resolve())
print("[summary]", summary_csv.resolve())
print("[latex]", tex_path.resolve())
print("[bullets]", bullets_path.resolve())
print("[compact]", routing_table_path.resolve())
print("[zip]", zip_path.resolve())
