# dar_router_main.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Driver script for Domain-Aware Router (DAR)

Pipeline:
  0) Data → CounselChat or BEIR-style JSONL
  1) Chunking → docs_df, qrels_df
  2) Indices → BM25, Dense, Cross-Encoder (verified)
  3) Retrieval → sparse/dense/hybrid/(verified)
  4) Features → pre + embedding + post + emotion/risk
  5) Predictors → LightGBM + isotonic (hybrid & sparse)
  6) Routing → single & dual (safety-aware)
     * Optional CRISIS tier (region-aware hotlines + NIMH snippets)
  7) Sweeps, baselines, ablations, simulations
  8) Transfer hooks (if csvs are present)
  9) Report → routing_report.json (+ artifacts)

Examples:

  # Default CSV mode (CounselChat-style)
  python dar_router_main.py \
      --workdir runs/exp1 --device auto

  # BEIR-style JSONL dataset
  python dar_router_main.py \
      --workdir runs/medpsych --device cpu \
      --data_root paper_experiments/data/medpsych_online \
      --no_sweep
"""

from __future__ import annotations

import argparse
import json
import gc
from pathlib import Path
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch

# Core / Crisis / Utils
from dar_utils import (
    SEED, NumpyPathEncoder, Config, get_version_info
)
from dar_core import (
    download_counselchat_csv, load_and_clean, build_corpus,
    BM25Index, DenseIndex, CrossEncoderReranker,
    build_runs, evaluate_runs, ndcg_at_k,
    EmotionDetector, build_feature_frame, cv_fit_predictor,
    simulate_routing, route_one_single, route_one_dual,
    compute_qpp_baselines, compare_chunk_modes,
    make_predict_fn, counterfactual_regret, adversarial_consistency
)
from dar_crisis import (
    build_crisis_components, CrisisResponder, estimate_low_coverage
)


DENSE_MODELS_TIER_1 = [
    # General-purpose SOTA
    "sentence-transformers/all-mpnet-base-v2",       # Your baseline
    "intfloat/e5-base-v2",                           # Microsoft 2023
    "BAAI/bge-base-en-v1.5",                         # MTEB top 2024
    
    # Efficient variants
    "sentence-transformers/all-MiniLM-L6-v2",        # Tiny (22M)
    "sentence-transformers/all-distilroberta-v1",    # Mid (82M)
    
    # Biomedical
    "pritamdeka/S-PubMedBert-MS-MARCO",              # Biomedical MS-MARCO
    "allenai/scibert_scivocab_uncased",              # Scientific
    
    # Advanced
    "hkunlp/instructor-base",                        # Instruction-following
    "thenlper/gte-base",                             # Alibaba GTE
]

# Tier 2: Extended (14 models total)
DENSE_MODELS_TIER_2 = DENSE_MODELS_TIER_1 + [
    "intfloat/e5-large-v2",                          # Larger E5
    "BAAI/bge-large-en-v1.5",                        # Larger BGE
    "sentence-transformers/gtr-t5-base",             # Google T5
    "jinaai/jina-embeddings-v2-base-en",             # Jina 8K context
    "michiyasunaga/BioLinkBERT-base",                # Biomedical links
]

# Tier 3: Full (17 models)
DENSE_MODELS_TIER_3 = DENSE_MODELS_TIER_2 + [
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",  # PubMed
    "emilyalsentzer/Bio_ClinicalBERT",               # Clinical notes
    "facebook/contriever-msmarco",                   # Contriever (mean pooling warning)
    # Note: facebook/dpr-question_encoder-single-nq-base may fail, excluded
]
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=str, required=True, help="Output working directory (new or empty)")
    ap.add_argument("--device", type=str, default="auto", help="'cpu' | 'cuda' | 'auto'")
    ap.add_argument("--limit_queries", type=int, default=0, help="Optional cap on #queries for quick runs")
    ap.add_argument("--chunk_mode", type=str, default="sentence", choices=["sentence", "semantic", "overlap"])
    ap.add_argument("--run_chunk_ablation", action="store_true", help="Compare chunking modes (retrieval only)")

    # Data root override (CSV or BEIR-style JSONL directory)
    ap.add_argument(
        "--data_root",
        type=str,
        default=None,
        help=(
            "Optional dataset root. "
            "If a CSV file: load via load_and_clean (CounselChat-style). "
            "If a directory with corpus.jsonl/queries.jsonl/qrels.jsonl: load as BEIR-style triple."
        ),
    )

    # Augmentation & crisis tier
    ap.add_argument("--augment_nimh", action="store_true", help="Fetch a small NIMH corpus (public domain) for crisis tier")
    ap.add_argument("--user_country", type=str, default=None, help="ISO country code for region-specific crisis info (e.g., US, GB, CA, MA)")
    ap.add_argument("--crisis_resources", type=str, default=None, help="Path to crisis_resources.json")

    # Optional overrides for cost
    ap.add_argument("--cost_sparse", type=float, default=None)
    ap.add_argument("--cost_hybrid", type=float, default=None)
    ap.add_argument("--cost_verified", type=float, default=None)
    # Optional overrides for threshold
    ap.add_argument("--thr_sparse", type=float, default=None)
    ap.add_argument("--thr_hybrid", type=float, default=None)
    ap.add_argument("--safety_override_thr", type=float, default=None)
    ap.add_argument("--dual_delta", type=float, default=None)
    ap.add_argument("--no_sweep", action="store_true")

    ap.add_argument("--dense_model", type=str, default="pritamdeka/S-PubMedBert-MS-MARCO")
    ap.add_argument("--verified_model", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")

    ap.add_argument("--topk_bm25", type=int, default=None)
    ap.add_argument("--topk_dense", type=int, default=None)
    ap.add_argument("--topk_fusion", type=int, default=None)
    ap.add_argument("--rrf_k", type=int, default=None)
    ap.add_argument("--verified_topn", type=int, default=None)
    ap.add_argument("--qpp_post_k", type=int, default=None)

    ap.add_argument("--min_chunk_words", type=int, default=None)
    ap.add_argument("--max_chunk_words", type=int, default=None)
    ap.add_argument("--overlap_chunk_words", type=int, default=None)
    ap.add_argument("--overlap_size", type=int, default=None)

    ap.add_argument("--hybrid_mode", type=str, default=None, choices=["rrf", "dense_only", "bm25_only"])
    ap.add_argument("--verified_source", type=str, default=None, choices=["hybrid", "dense", "sparse"])
    ap.add_argument("--rrf_w_bm25", type=float, default=None)
    ap.add_argument("--rrf_w_dense", type=float, default=None)
    ap.add_argument("--verified_max_len", type=int, default=None)
    ap.add_argument(
        "--run_dense_comparison", 
        action="store_true", 
        help="Compare multiple dense retrieval models (Tier 1: 9 models)"
    )
    ap.add_argument(
        "--dense_comparison_tier", 
        type=int, 
        default=1, 
        choices=[1, 2, 3],
        help="Which tier of models to test (1=essential, 2=extended, 3=all)"
    )
    args = ap.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # ---------------- Config ----------------
    cfg = Config(workdir=workdir)
    cfg.chunk_mode = args.chunk_mode
    cfg.limit_queries = args.limit_queries
    cfg.run_chunk_ablation = args.run_chunk_ablation
    cfg.user_country = args.user_country
    cfg.crisis_resources_path = args.crisis_resources

    # Enforce updated routing thresholds explicitly (can be overridden by CLI)
    cfg.thr_sparse = 0.50
    cfg.thr_hybrid = 0.55
    cfg.safety_override_thr = 0.55
    cfg.dual_delta = 0.00

    if args.cost_sparse is not None:
        cfg.cost_sparse = args.cost_sparse
    if args.cost_hybrid is not None:
        cfg.cost_hybrid = args.cost_hybrid
    if args.cost_verified is not None:
        cfg.cost_verified = args.cost_verified
    if args.thr_sparse is not None:
        cfg.thr_sparse = args.thr_sparse
    if args.thr_hybrid is not None:
        cfg.thr_hybrid = args.thr_hybrid
    if args.safety_override_thr is not None:
        cfg.safety_override_thr = args.safety_override_thr
    if args.dual_delta is not None:
        cfg.dual_delta = args.dual_delta
    if args.no_sweep:
        cfg.sweep = False
    if args.topk_bm25 is not None:
        cfg.topk_bm25 = args.topk_bm25
    if args.topk_dense is not None:
        cfg.topk_dense = args.topk_dense
    if args.topk_fusion is not None:
        cfg.topk_fusion = args.topk_fusion
    if args.rrf_k is not None:
        cfg.rrf_k = args.rrf_k
    if args.verified_topn is not None:
        cfg.verified_topn = args.verified_topn
    if args.qpp_post_k is not None:
        cfg.qpp_post_k = args.qpp_post_k

    if args.min_chunk_words is not None:
        cfg.min_chunk_words = args.min_chunk_words
    if args.max_chunk_words is not None:
        cfg.max_chunk_words = args.max_chunk_words
    if args.overlap_chunk_words is not None:
        cfg.overlap_chunk_words = args.overlap_chunk_words
    if args.overlap_size is not None:
        cfg.overlap_size = args.overlap_size

    if args.hybrid_mode is not None:
        cfg.hybrid_mode = args.hybrid_mode
    if args.verified_source is not None:
        cfg.verified_source = args.verified_source
    if args.rrf_w_bm25 is not None:
        cfg.rrf_w_bm25 = args.rrf_w_bm25
    if args.rrf_w_dense is not None:
        cfg.rrf_w_dense = args.rrf_w_dense
    if args.verified_max_len is not None:
        cfg.verified_max_len = args.verified_max_len

    # ---------------- Device ----------------
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[cfg] device = {device}")

    # ---------------- Version stamp ----------------
    ver_info = get_version_info(device)
    with open(workdir / "version_info.json", "w") as vf:
        json.dump(ver_info, vf, indent=2, cls=NumpyPathEncoder)
    print(f"[env] version info recorded -> {workdir / 'version_info.json'}")

    # =====================================================
    # 0) Data
    # =====================================================
    beir_mode = False
    docs_df_beir = None
    qrels_df_beir = None

    if args.data_root is not None:
        data_root = Path(args.data_root)
        if data_root.is_file():
            # CSV path (CounselChat-style)
            print(f"[data] loading CSV from {data_root}")
            df_raw = load_and_clean(data_root)
            print(f"[data] {len(df_raw)} QA pairs after cleaning")
        elif data_root.is_dir():
            print(f"[data] loading from directory {data_root}")
            corpus_path = data_root / "corpus.jsonl"
            queries_path = data_root / "queries.jsonl"
            qrels_path = data_root / "qrels.jsonl"

            if corpus_path.exists() and queries_path.exists() and qrels_path.exists():
                print("[data] detected BEIR-style JSONL triple (corpus/queries/qrels); constructing dataset in-memory ...")
                corpus_df = pd.read_json(corpus_path, lines=True)
                queries_df = pd.read_json(queries_path, lines=True)
                qrels_df_raw = pd.read_json(qrels_path, lines=True)

                # Map corpus string IDs ('d0', 'd1', ...) to int64 doc_id for our indices
                doc_id_map = {cid: i for i, cid in enumerate(corpus_df["id"].tolist())}
                corpus_df["doc_id"] = corpus_df["id"].map(doc_id_map).astype(np.int64)

                # docs_df used by indices and retrieval
                docs_df_beir = corpus_df[["doc_id", "text"]].rename(columns={"text": "text"}).copy()

                # qrels: 'query-id', 'corpus-id', 'score' -> qid, doc_id, rel
                qrels_df_raw["qid"] = qrels_df_raw["query-id"].astype(str)
                qrels_df_raw["doc_id"] = qrels_df_raw["corpus-id"].map(doc_id_map).astype(np.int64)
                if "score" in qrels_df_raw.columns:
                    qrels_df_raw["rel"] = qrels_df_raw["score"].astype(float)
                else:
                    qrels_df_raw["rel"] = 1.0
                qrels_df_beir = qrels_df_raw[["qid", "doc_id", "rel"]].copy()

                # df_raw only needs (qid, qtext) for the rest of the pipeline
                queries_df = queries_df.rename(columns={"id": "qid", "text": "qtext"})
                df_raw = queries_df[["qid", "qtext"]].copy()

                print(
                    f"[data] BEIR dataset: {df_raw['qid'].nunique()} queries, "
                    f"{docs_df_beir['doc_id'].nunique()} documents, {len(qrels_df_beir)} qrels"
                )
                beir_mode = True
            else:
                raise FileNotFoundError(
                    f"{data_root} is a directory but I couldn't find corpus.jsonl / queries.jsonl / qrels.jsonl"
                )
        else:
            raise FileNotFoundError(f"--data_root={data_root} not found")
    else:
        # Default CSV: combined-data.csv in /kaggle/working
        csv_path = Path("/kaggle/working/combined-data.csv")
        if not csv_path.exists():
            download_counselchat_csv(csv_path)
        df_raw = load_and_clean(csv_path)
        print(f"[data] {len(df_raw)} QA pairs after cleaning")

    # Optionally limit queries
    q_meta_all = df_raw[["qid", "qtext"]].drop_duplicates("qid").reset_index(drop=True)
    if cfg.limit_queries and cfg.limit_queries < len(q_meta_all):
        samp_qids = set(q_meta_all.sample(cfg.limit_queries, random_state=SEED)["qid"].tolist())
        df_raw = df_raw[df_raw["qid"].isin(samp_qids)]
        print(f"[data] limited to {df_raw['qid'].nunique()} queries for this run")

    # Optional retrieval-only chunk-mode ablation
    if cfg.run_chunk_ablation:
        if beir_mode:
            print("[ablation] chunk_modes skipped for BEIR-style dataset")
            chunk_results = {}
        else:
            print("[ablation] chunk_modes (retrieval-only) ...")
            chunk_results = compare_chunk_modes(df_raw, cfg, device)
    else:
        chunk_results = {}

    # =====================================================
    # 1) Chunking
    # =====================================================
    if beir_mode:
        docs_df = docs_df_beir
        qrels_df = qrels_df_beir
        print(f"[chunk] using pre-built BEIR docs; {len(docs_df)} docs; qrels rows={len(qrels_df)}")
    else:
        docs_df, qrels_df = build_corpus(df_raw, cfg)
        print(f"[chunk] mode={cfg.chunk_mode}; {len(docs_df)} chunks; qrels rows={len(qrels_df)}")

    docs_df.to_parquet(workdir / "docs_corpus.parquet", index=False)
    qrels_df.to_parquet(workdir / "qrels.parquet", index=False)

    # =====================================================
    # 2) Indices
    # =====================================================
    print("[index] building BM25...")
    bm25 = BM25Index(docs_df["text"].tolist(), docs_df["doc_id"].tolist())

    print("[index] loading dense model (domain) ...")
    dense_model_name = args.dense_model
    dense_idx = DenseIndex(dense_model_name, device=device)
    dense_idx.build(docs_df["text"].tolist(), docs_df["doc_id"].tolist())

    print("[index] loading cross-encoder reranker (verified tier) ...")
    reranker = CrossEncoderReranker(args.verified_model, device=device, max_len=cfg.verified_max_len) if cfg.use_verified else None
    
    # =====================================================
    # 3) Retrieval
    # =====================================================
    q_meta = df_raw[["qid", "qtext"]].drop_duplicates("qid").reset_index(drop=True)
    qids = q_meta["qid"].astype(str).tolist()
    queries = q_meta["qtext"].astype(str).tolist()

    print("[retrieval] running searches...")
    runs = build_runs(queries, docs_df, bm25, dense_idx, reranker, cfg)

    print("[retrieval] evaluating per run (with 95% bootstrap CI for mean nDCG@10)...")
    run_eval = evaluate_runs(runs, qids, qrels_df, cfg)
    print(json.dumps(run_eval, indent=2, cls=NumpyPathEncoder))

    # Label vectors for QPP regression targets
    rels_map = {
        qid: {int(r.doc_id): float(r.rel) for r in g.itertuples(index=False)}
        for qid, g in qrels_df.groupby("qid")
    }
    y_hybrid, y_sparse = [], []
    for i, qid in enumerate(qids):
        g_h = [rels_map.get(qid, {}).get(doc, 0.0) for doc, _ in runs["hybrid"][i]]
        g_s = [rels_map.get(qid, {}).get(doc, 0.0) for doc, _ in runs["sparse"][i]]
        y_hybrid.append(ndcg_at_k(g_h, cfg.eval_k))
        y_sparse.append(ndcg_at_k(g_s, cfg.eval_k))
    y_hybrid = np.array(y_hybrid, dtype=float)
    y_sparse = np.array(y_sparse, dtype=float)
    # =====================================================
    # 3.5) Optional: Dense Model Comparison
    # =====================================================
    dense_model_comparison_report = {}
    
    if args.run_dense_comparison:
        print("\n" + "="*80)
        print("DENSE MODEL COMPARISON")
        print("="*80 + "\n")
        
        # Select tier
        if args.dense_comparison_tier == 1:
            models_to_test = DENSE_MODELS_TIER_1
        elif args.dense_comparison_tier == 2:
            models_to_test = DENSE_MODELS_TIER_2
        else:
            models_to_test = DENSE_MODELS_TIER_3
        
        print(f"Testing {len(models_to_test)} models (Tier {args.dense_comparison_tier})")
        
        # Run comparison
        from dar_core import compare_dense_models
        
        comparison_df = compare_dense_models(
            model_names=models_to_test,
            docs_df=docs_df,
            qrels_df=qrels_df,
            queries=queries,
            qids=qids,
            cfg=cfg,
            device=device,
            reranker=None,  # Dense-only comparison
        )
        
        # Save results
        comparison_csv = workdir / "dense_model_comparison.csv"
        comparison_df.to_csv(comparison_csv, index=False)
        print(f"\n✓ Comparison saved to: {comparison_csv}")
        
        # Generate LaTeX table
        latex_path = workdir / "dense_model_comparison.tex"
        with open(latex_path, 'w') as f:
            # Write table header
            f.write("\\begin{table*}[t]\n")
            f.write("\\centering\n")
            f.write("\\caption{Dense retriever comparison on CounselChat (937 queries).}\n")
            f.write("\\label{tab:dense_comparison}\n")
            f.write("\\begin{tabular}{llcccc}\n")  # 6 columns: Model, Category, Params, nDCG@10, Recall@10, Zero%
            f.write("\\toprule\n")
            f.write("\\textbf{Model} & \\textbf{Category} & \\textbf{Params} & \\textbf{nDCG@10} & \\textbf{Recall@10} & \\textbf{Zero\\%} \\\\\n")
            f.write("\\midrule\n")
            
            # Write BM25 baseline first
            f.write("\\multicolumn{6}{l}{\\textit{Sparse Baseline}} \\\\\n")  # Match 6 columns
            f.write(f"BM25 & Lexical & - & {run_eval['sparse']['mean_ndcg@10']:.3f} & {run_eval['sparse']['recall@10']:.3f} & {run_eval['sparse']['zero_rate']*100:.1f}\\% \\\\\n")
            f.write("\\midrule\n")
            
            # Group by category
            for category in ['General', 'Biomedical', 'Mental Health', 'Efficient']:
                cat_df = comparison_df[comparison_df['category'] == category]
                if len(cat_df) == 0:
                    continue
                
                f.write(f"\\multicolumn{{6}}{{l}}{{\\textit{{{category} Dense Models}}}} \\\\\n")  # Match 6 columns
                
                for _, row in cat_df.iterrows():
                    model_short = row['model'].split('/')[-1]  # Just the model name
                    f.write(f"{model_short} & {row['category']} & {row['parameters']} & ")
                    f.write(f"{row['ndcg@10']:.3f} & ")  # Removed CI for space
                    f.write(f"{row['recall@10']:.3f} & {row['zero_rate']*100:.1f}\\% \\\\\n")
                
                f.write("\\midrule\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table*}\n")
        
        print(f"✓ LaTeX table saved to: {latex_path}")
        
        # Add to final report
        dense_model_comparison_report = {
            "tier": args.dense_comparison_tier,
            "models_tested": len(models_to_test),
            "csv_path": str(comparison_csv),
            "latex_path": str(latex_path),
            "best_model": comparison_df.iloc[0]["model"],
            "best_ndcg": float(comparison_df.iloc[0]["ndcg@10"]),
        }
        
        print("\n" + "="*80 + "\n")
    # =====================================================
    # 4) Features
    # =====================================================
    print("[features] emotion & risk detection ...")
    emo = EmotionDetector(device=device)
    emo_feats = emo.features_for_texts(queries)
    n_highrisk = int(np.sum(np.array(emo_feats["high_risk_prob"]) >= cfg.high_risk_prob_thr))
    print(f"[safety] high-risk queries detected = {n_highrisk} (thr={cfg.high_risk_prob_thr})")

    print("[features] building stacked features (pre + emotion + post + complementarity) ...")
    X = build_feature_frame(queries, dense_idx.model, emo_feats, runs["sparse"], runs["dense"], cfg)
    X.to_parquet(workdir / "features_stacked.parquet", index=False)
    feat_columns = list(X.columns)

    # =====================================================
    # 5) Predictors
    # =====================================================
    print("[predictor] training LightGBM (hybrid) + isotonic (5-fold CV) ...")
    model_h, iso_h, metrics_h = cv_fit_predictor(X, y_hybrid, cfg)
    print(json.dumps(metrics_h, indent=2, cls=NumpyPathEncoder))

    print("[predictor] training LightGBM (sparse) + isotonic (5-fold CV) ...")
    model_s, iso_s, metrics_s = cv_fit_predictor(X, y_sparse, cfg)
    print(json.dumps({"sparse_predictor": metrics_s}, indent=2, cls=NumpyPathEncoder))

    # Predictions (raw + calibrated)
    y_raw_h = model_h.predict(X.values)
    y_cal_h = iso_h.predict(y_raw_h)
    y_raw_s = model_s.predict(X.values)
    y_cal_s = iso_s.predict(y_raw_s)

    preds_df = pd.DataFrame(
        {
            "qid": qids,
            "query": queries,
            "true_ndcg10_hybrid": y_hybrid,
            "true_ndcg10_sparse": y_sparse,
            "pred_hybrid_raw": y_raw_h,
            "pred_hybrid_cal": y_cal_h,
            "pred_sparse_raw": y_raw_s,
            "pred_sparse_cal": y_cal_s,
            "high_risk_prob": emo_feats["high_risk_prob"],
        }
    )
    preds_df.to_parquet(workdir / "predictions.parquet", index=False)

    # =====================================================
    # Crisis components (resources + optional NIMH corpus)
    # =====================================================
    crisis_res, crisis_docs_df, crisis_bm25 = build_crisis_components(
        augment_nimh=args.augment_nimh,
        workdir=workdir,
        resources_path=args.crisis_resources,
        cfg=cfg,
    )
    crisis = CrisisResponder(crisis_res, crisis_docs_df, crisis_bm25, cfg)

    # =====================================================
    # 6) Routing simulation
    # =====================================================
    print("[routing] simulating safety-aware (single-predictor) ...")
    routes_single = [
        route_one_single(float(y_cal_h[i]), float(emo_feats["high_risk_prob"][i]), cfg)
        for i in range(len(queries))
    ]
    stats_single = simulate_routing(qids, routes_single, runs, qrels_df, cfg)
    print(json.dumps(stats_single, indent=2, cls=NumpyPathEncoder))

    print("[routing] dual-predictor safety-aware strategy ...")
    routes_dual = [
        route_one_dual(float(y_cal_s[i]), float(y_cal_h[i]), float(emo_feats["high_risk_prob"][i]), cfg)
        for i in range(len(queries))
    ]
    n_crisis = sum(1 for r in routes_dual if r == "crisis_response")
    print(f"[crisis] promoted to crisis tier = {n_crisis}")

    # Optional promotion to crisis tier if high-risk AND BM25 looks low-coverage
    routes_dual_promoted = []
    for i, r in enumerate(routes_dual):
        if r != "crisis_response" and cfg.enable_crisis_tier:
            if float(emo_feats["high_risk_prob"][i]) >= cfg.high_risk_prob_thr:
                if estimate_low_coverage(runs["sparse"][i]):
                    r = "crisis_response"
        routes_dual_promoted.append(r)
    routes_dual = routes_dual_promoted

    stats_dual = simulate_routing(qids, routes_dual, runs, qrels_df, cfg)
    print(json.dumps(stats_dual, indent=2, cls=NumpyPathEncoder))

    # Save routes for auditing
    routes_df = pd.DataFrame({"qid": qids, "route_single": routes_single, "route_dual": routes_dual})
    routes_df.to_csv(workdir / "routes.csv", index=False)

    # Generate crisis fallbacks (if any)
    crisis_out = []
    if cfg.enable_crisis_tier and any(r == "crisis_response" for r in routes_dual):
        print("[crisis] rendering crisis responses for flagged queries ...")
        for i, (q, r) in enumerate(zip(queries, routes_dual)):
            if r != "crisis_response":
                crisis_out.append(None)
                continue
            crisis_out.append(crisis.respond(q, qid=qids[i], country=cfg.user_country))
        with open(workdir / "crisis_fallbacks.json", "w") as f:
            json.dump(crisis_out, f, indent=2, cls=NumpyPathEncoder)
        print(f"[crisis] wrote {workdir / 'crisis_fallbacks.json'}")

    # =====================================================
    # 7) Threshold sweeps
    # =====================================================
    sweep = {}
    _thr_keep = (cfg.thr_sparse, cfg.thr_hybrid, cfg.safety_override_thr, cfg.dual_delta)

    if cfg.sweep:
        print("[routing] grid-sweeping thresholds (single + dual) ...")
        best_single = {"best_ndcg": None, "best_efficiency": None, "best_balanced": None}
        best_dual = {"best_ndcg": None, "best_efficiency": None, "best_balanced": None}

        # Single predictor sweep
        recs_single = []
        for ts in np.linspace(0.5, 0.85, 8):
            for th in np.linspace(0.45, 0.7, 6):
                for so in np.linspace(0.45, 0.6, 4):
                    cfg.thr_sparse, cfg.thr_hybrid, cfg.safety_override_thr = float(ts), float(th), float(so)
                    routes_s = [
                        route_one_single(float(y_cal_h[i]), float(emo_feats["high_risk_prob"][i]), cfg)
                        for i in range(len(queries))
                    ]
                    st = simulate_routing(qids, routes_s, runs, qrels_df, cfg)
                    recs_single.append(
                        {
                            **st,
                            "thr_sparse": cfg.thr_sparse,
                            "thr_hybrid": cfg.thr_hybrid,
                            "safety_override_thr": cfg.safety_override_thr,
                        }
                    )
        df_single = pd.DataFrame(recs_single)
        df_single.to_csv(workdir / "routing_threshold_sweep_single.csv", index=False)
        nd_norm = (df_single["avg_ndcg@10_auto"] - df_single["avg_ndcg@10_auto"].min()) / (
            df_single["avg_ndcg@10_auto"].max() - df_single["avg_ndcg@10_auto"].min() + 1e-9
        )
        ef_norm = (df_single["efficiency"] - df_single["efficiency"].min()) / (
            df_single["efficiency"].max() - df_single["efficiency"].min() + 1e-9
        )
        df_single["balanced"] = 0.5 * (nd_norm + ef_norm)
        best_single["best_ndcg"] = df_single.sort_values("avg_ndcg@10_auto", ascending=False).iloc[0].to_dict()
        best_single["best_efficiency"] = df_single.sort_values("efficiency", ascending=False).iloc[0].to_dict()
        best_single["best_balanced"] = df_single.sort_values("balanced", ascending=False).iloc[0].to_dict()

        # Dual predictor sweep
        recs_dual = []
        for ts in np.linspace(0.5, 0.8, 7):
            for th in np.linspace(0.5, 0.7, 5):
                for so in np.linspace(0.45, 0.6, 4):
                    for dd in [0.0, 0.01, 0.02, 0.03]:
                        cfg.thr_sparse, cfg.thr_hybrid, cfg.safety_override_thr, cfg.dual_delta = (
                            float(ts),
                            float(th),
                            float(so),
                            float(dd),
                        )
                        routes_d = [
                            route_one_dual(
                                float(y_cal_s[i]),
                                float(y_cal_h[i]),
                                float(emo_feats["high_risk_prob"][i]),
                                cfg,
                            )
                            for i in range(len(queries))
                        ]
                        st = simulate_routing(qids, routes_d, runs, qrels_df, cfg)
                        recs_dual.append(
                            {
                                **st,
                                "thr_sparse": cfg.thr_sparse,
                                "thr_hybrid": cfg.thr_hybrid,
                                "safety_override_thr": cfg.safety_override_thr,
                                "dual_delta": cfg.dual_delta,
                            }
                        )
        df_dual = pd.DataFrame(recs_dual)
        df_dual.to_csv(workdir / "routing_threshold_sweep_dual.csv", index=False)
        nd_norm = (df_dual["avg_ndcg@10_auto"] - df_dual["avg_ndcg@10_auto"].min()) / (
            df_dual["avg_ndcg@10_auto"].max() - df_dual["avg_ndcg@10_auto"].min() + 1e-9
        )
        ef_norm = (df_dual["efficiency"] - df_dual["efficiency"].min()) / (
            df_dual["efficiency"].max() - df_dual["efficiency"].min() + 1e-9
        )
        df_dual["balanced"] = 0.5 * (nd_norm + ef_norm)
        best_dual["best_ndcg"] = df_dual.sort_values("avg_ndcg@10_auto", ascending=False).iloc[0].to_dict()
        best_dual["best_efficiency"] = df_dual.sort_values("efficiency", ascending=False).iloc[0].to_dict()
        best_dual["best_balanced"] = df_dual.sort_values("balanced", ascending=False).iloc[0].to_dict()

        sweep = {
            "single": best_single,
            "dual": best_dual,
            "single_csv": str(workdir / "routing_threshold_sweep_single.csv"),
            "dual_csv": str(workdir / "routing_threshold_sweep_dual.csv"),
        }

        cfg.thr_sparse, cfg.thr_hybrid, cfg.safety_override_thr, cfg.dual_delta = _thr_keep

    # =====================================================
    # 8) Baselines & Ablations
    # =====================================================
    # =====================================================
    # 8) Baselines & Ablations
    # =====================================================
    baselines = {}
    if cfg.run_baselines:
        print("[baselines] computing CLASSIC QPP baselines (SCQ, Clarity, WIG, NQC) ...")
        classic_vals = compute_qpp_baselines(
            queries, runs["sparse"], bm25, docs_df["text"].tolist(), cfg
        )
        
        print("[baselines] computing PRACTICAL POST-RETRIEVAL baselines (proven methods) ...")
        try:
            from dar_practical_baselines import compute_practical_qpp_baselines
            practical_vals = compute_practical_qpp_baselines(
                queries=queries,
                sparse_runs=runs["sparse"],
                dense_runs=runs["dense"],
                docs_texts=docs_df["text"].tolist(),
                device=device
            )
        except ImportError:
            print("[warning] dar_practical_baselines.py not found")
            practical_vals = {}
        
        # Combine all
        all_vals = {**classic_vals, **practical_vals}
        
        # Correlate with y_hybrid + bootstrap CIs
        def corr(a, b):
            a = np.array(a)
            b = np.array(b)
            if a.std() == 0 or b.std() == 0:
                return 0.0, 0.0
            pear = float(np.corrcoef(a, b)[0, 1])
            ra = a.argsort().argsort()
            rb = b.argsort().argsort()
            spear = float(np.corrcoef(ra, rb)[0, 1])
            return pear, spear

        from dar_utils import bootstrap_ci_corr as _ci

        for name, vals in all_vals.items():
            p, s = corr(vals, y_hybrid)
            p_low, p_high = _ci(
                np.array(vals, dtype=float),
                y_hybrid,
                kind="pearson",
                n_resamples=1000,
                alpha=0.05,
                seed=SEED,
            )
            s_low, s_high = _ci(
                np.array(vals, dtype=float),
                y_hybrid,
                kind="spearman",
                n_resamples=1000,
                alpha=0.05,
                seed=SEED,
            )
            baselines[name] = {
                "pearson": p,
                "pearson_ci_low": p_low,
                "pearson_ci_high": p_high,
                "spearman": s,
                "spearman_ci_low": s_low,
                "spearman_ci_high": s_high,
            }
        
        # Save comparison table
        baseline_df = pd.DataFrame(baselines).T
        baseline_df = baseline_df.sort_values('pearson', ascending=False)
        baseline_df.to_csv(workdir / "qpp_baseline_comparison.csv")
        print(f"✓ Baseline comparison saved: {workdir / 'qpp_baseline_comparison.csv'}")
        
        # Print top performers
        print("\n[baselines] Top 5 performers:")
        print(baseline_df[['pearson', 'spearman']].head())
        
    ablations = {}
    if cfg.run_ablations:
        print("[ablations] running ...")
        # Full (dual + safety + verified)
        ablations["Full System (Dual + Safety + Verified)"] = stats_dual
        # - Dual Predictor (Single)
        ablations["- Dual Predictor (Single)"] = stats_single
        # - Safety Override: disable override by forcing safety threshold to 1.0 (never triggers)
        keep = (cfg.thr_sparse, cfg.thr_hybrid, cfg.safety_override_thr)
        cfg.safety_override_thr = 1.0
        routes_nosafe = [
            route_one_dual(
                float(y_cal_s[i]),
                float(y_cal_h[i]),
                float(emo_feats["high_risk_prob"][i]),
                cfg,
            )
            for i in range(len(queries))
        ]
        ablations["- Safety Override"] = simulate_routing(qids, routes_nosafe, runs, qrels_df, cfg)
        cfg.thr_sparse, cfg.thr_hybrid, cfg.safety_override_thr = keep

        # - Emotion Features (retrained without emotion columns)
        emo_cols = ["emo_entropy", "emo_dom", "emo_maxprob", "high_risk_prob"]
        X_noemo = X.drop(columns=emo_cols, errors="ignore")
        m_h2, iso_h2, met_h2 = cv_fit_predictor(X_noemo, y_hybrid, cfg)
        y_cal_h2 = iso_h2.predict(m_h2.predict(X_noemo.values))
        routes_noemo = [route_one_single(float(y_cal_h2[i]), 0.0, cfg) for i in range(len(queries))]
        ablations["- Emotion Features (retrained)"] = simulate_routing(qids, routes_noemo, runs, qrels_df, cfg)

        # Sparse only baseline (sanity)
        ablations["Sparse Only"] = {
            "avg_cost": cfg.cost_sparse,
            "avg_ndcg@10_auto": float(np.mean(y_sparse)),
            "avg_ndcg@10_overall": None,
            "efficiency": (float(np.mean(y_sparse)) / cfg.cost_sparse) if cfg.cost_sparse > 0 else 0.0,
        }

    # =====================================================
    # 9) Simulations / Analyses
    # =====================================================
    analysis = {}
    if cfg.run_sims:
        print("[transfer] cross-domain (file-based) ...")
        # Counterfactual regret (for chosen dual routes)
        analysis["counterfactual"] = counterfactual_regret(qids, runs, qrels_df, routes_dual, cfg)

        # Failure heuristics (toy buckets)
        fails = {
            "Ambiguous": [],
            "Out_of_domain": [],
            "Highly_specific": [],
            "Implicit_risk": [],
            "Negation_confusion": [],
        }
        for i, q in enumerate(queries):
            ql = q.lower()
            if len(q.split()) < 5 or (
                any(t in ql for t in ["help", "advice", "what do i do"]) and len(q.split()) < 12
            ):
                fails["Ambiguous"].append(i)
            if any(t in ql for t in ["pizza", "restaurant", "nyc", "car insurance"]):
                fails["Out_of_domain"].append(i)
            if any(t in ql for t in ["i'm not suicidal", "not suicidal"]):
                fails["Negation_confusion"].append(i)
        analysis["failures"] = fails

        # Adversarial consistency using safe predictors
        predict_h = make_predict_fn(model_h, iso_h, feat_columns, bm25, dense_idx, emo, dense_idx.model, cfg)
        predict_s = make_predict_fn(model_s, iso_s, feat_columns, bm25, dense_idx, emo, dense_idx.model, cfg)

        def router_dual_from_models(q: str, **_):
            y_h, risk = predict_h(q)
            y_s, _ = predict_s(q)
            return route_one_dual(y_s, y_h, risk, cfg)

        test_q = queries[min(10, len(queries) - 1)] if len(queries) else "I feel very anxious at night"
        analysis["adversarial_one"] = adversarial_consistency(test_q, router_dual_from_models, {})

        # Cost sensitivity grid (example ratios)
        rows = []
        orig_costs = (cfg.cost_sparse, cfg.cost_hybrid, cfg.cost_verified, cfg.cost_crisis)
        for cs, ch, cv in [(1, 2, 3), (1, 3, 5), (1, 4, 10)]:
            cfg.cost_sparse = cs
            cfg.cost_hybrid = ch
            cfg.cost_verified = cv
            # leave cfg.cost_crisis as-is; it's typically set by config / CLI
            st = simulate_routing(qids, routes_dual, runs, qrels_df, cfg)
            st.update(
                {
                    "cost_sparse": cs,
                    "cost_hybrid": ch,
                    "cost_verified": cv,
                    "cost_crisis": cfg.cost_crisis,
                }
            )
            rows.append(st)
        cost_csv = workdir / "cost_sensitivity.csv"
        pd.DataFrame(rows).to_csv(cost_csv, index=False)
        analysis["cost_sensitivity_csv"] = str(cost_csv)

        # restore original costs so the rest of the report uses the configured values
        cfg.cost_sparse, cfg.cost_hybrid, cfg.cost_verified, cfg.cost_crisis = orig_costs

    # =====================================================
    # 10) Transfer hooks (if csvs exist)
    # =====================================================
    transfer = {}
    if cfg.run_transfer:
        datasets = {
            "healthtap": workdir / "healthtap.csv",
            "legaladvice": workdir / "legaladvice.csv",
            "stackoverflow": workdir / "stackoverflow.csv",
        }
        for name, path in datasets.items():
            if not path.exists():
                transfer[name] = {"status": "skipped (csv not found)"}
            else:
                try:
                    df_t = pd.read_csv(path)
                    df_t = df_t.rename(columns={"qid": "qid", "question": "qtext", "answer": "answer"})
                    df_t["qid"] = df_t["qid"].astype(str)
                    df_t = df_t.dropna(subset=["qtext", "answer"])
                    docs_t, qrels_t = build_corpus(df_t, cfg)
                    bm25_t = BM25Index(docs_t["text"].tolist())
                    dense_t = DenseIndex(dense_model_name, device=device)
                    dense_t.build(docs_t["text"].tolist(), docs_t["doc_id"].tolist())
                    rer_t = reranker
                    qmeta_t = df_t[["qid", "qtext"]].drop_duplicates("qid").reset_index(drop=True)
                    qids_t = qmeta_t["qid"].astype(str).tolist()
                    queries_t = qmeta_t["qtext"].astype(str).tolist()
                    runs_t = build_runs(queries_t, docs_t, bm25_t, dense_t, rer_t, cfg)
                    evals_t = evaluate_runs(runs_t, qids_t, qrels_t, cfg)
                    transfer[name] = {"status": "ok", "retrieval": evals_t}
                except Exception as e:
                    transfer[name] = {"status": f"failed: {e}"}

    # =====================================================
    # 11) Assemble report
    # =====================================================
    base_sparse = {
        "avg_cost": cfg.cost_sparse,
        "avg_ndcg@10": run_eval["sparse"]["mean_ndcg@10"],
        "avg_ndcg@10_ci_low": run_eval["sparse"]["mean_ndcg@10_ci_low"],
        "avg_ndcg@10_ci_high": run_eval["sparse"]["mean_ndcg@10_ci_high"],
        "efficiency": run_eval["sparse"]["mean_ndcg@10"] / cfg.cost_sparse if cfg.cost_sparse > 0 else 0.0,
        "auto_ratio": 1.0,
    }
    base_hybrid = {
        "avg_cost": cfg.cost_hybrid,
        "avg_ndcg@10": run_eval["hybrid"]["mean_ndcg@10"],
        "avg_ndcg@10_ci_low": run_eval["hybrid"]["mean_ndcg@10_ci_low"],
        "avg_ndcg@10_ci_high": run_eval["hybrid"]["mean_ndcg@10_ci_high"],
        "efficiency": run_eval["hybrid"]["mean_ndcg@10"] / cfg.cost_hybrid if cfg.cost_hybrid > 0 else 0.0,
        "auto_ratio": 1.0,
    }
    base_verified = {
        "avg_cost": cfg.cost_verified,
        "avg_ndcg@10": run_eval["hybrid_verified"]["mean_ndcg@10"],
        "avg_ndcg@10_ci_low": run_eval["hybrid_verified"]["mean_ndcg@10_ci_low"],
        "avg_ndcg@10_ci_high": run_eval["hybrid_verified"]["mean_ndcg@10_ci_high"],
        "efficiency": run_eval["hybrid_verified"]["mean_ndcg@10"] / cfg.cost_verified if cfg.cost_verified > 0 else 0.0,
        "auto_ratio": 1.0,
    }

    report = {
        "position_statement": (
            "We study query performance prediction and adaptive routing in domain-specific "
            "information retrieval, using mental health as a challenging test case. Our focus "
            "is on predicting retrieval difficulty and optimizing cost-quality tradeoffs, not "
            "on clinical outcomes or therapeutic effectiveness."
        ),
        "contributions": {
            "qpp": "Dual-predictor architecture with post-retrieval dispersion and complementarity features",
            "safety": "Integrates risk detection into routing decisions with crisis fallback",
            "calibration": "Global isotonic calibration for multi-tier retrieval systems",
            "dense_model_comparison": dense_model_comparison_report,
            "efficiency": "Pareto-optimal threshold selection via grid search",
        },
        "retrieval": run_eval,
        "predictor": {"hybrid": metrics_h, "sparse": metrics_s},
        "routing": {
            "single": stats_single,
            "dual": stats_dual,
            "always_sparse": base_sparse,
            "always_hybrid": base_hybrid,
            "always_hybrid_verified": base_verified,
        },
        "crisis": {
            "enabled": cfg.enable_crisis_tier,
            "user_country": cfg.user_country,
            "crisis_outputs_json": str(workdir / "crisis_fallbacks.json")
            if any(r == "crisis_response" for r in routes_dual)
            else None,
            "crisis_docs_count": int(len(crisis_docs_df)),
        },
        "routing_sweep": sweep,
        "baselines": baselines,
        "ablations": ablations,
        "analysis": analysis,
        "transfer": transfer,
        "chunk_ablation": chunk_results,
        "config": asdict(cfg),
        "version_info": ver_info,
    }

    with open(workdir / "routing_report.json", "w") as f:
        json.dump(report, f, indent=2, cls=NumpyPathEncoder)

    print("\n=== SUMMARY ===")
    print(json.dumps(report, indent=2, cls=NumpyPathEncoder))
    print(f"\n[done] artifacts in {workdir.resolve()}")


if __name__ == "__main__":
    main()
