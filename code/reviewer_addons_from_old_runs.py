#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path
from dataclasses import fields
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import torch

from dar_utils import Config, SEED, NumpyPathEncoder
from dar_core import (
    download_counselchat_csv,
    load_and_clean,
    build_corpus,
    BM25Index,
    DenseIndex,
    CrossEncoderReranker,
    build_runs,
    ndcg_at_k,
)


# ---------------------------------------------------------------------
# Command / config helpers
# ---------------------------------------------------------------------

def parse_flags(cmd: str | None) -> dict:
    if not cmd:
        return {}

    toks = shlex.split(cmd)
    out = {}
    i = 0

    while i < len(toks):
        tok = toks[i]
        if tok.startswith("--"):
            key = tok[2:].replace("-", "_")
            if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                out[key] = toks[i + 1]
                i += 2
            else:
                out[key] = True
                i += 1
        else:
            i += 1

    return out


def coerce(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return None

    s = str(v)

    if s.lower() in {"true", "false"}:
        return s.lower() == "true"

    try:
        if re.fullmatch(r"-?\d+", s):
            return int(s)
        if re.fullmatch(r"-?\d*\.\d+(e-?\d+)?", s, flags=re.I):
            return float(s)
    except Exception:
        pass

    return v


def load_manifest(path: str = "paper_experiments/experiment_manifest.json"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {path}. "
            "Pass --manifest or place experiment_manifest.json in paper_experiments/."
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_set_config(cfg, cfg_dict):
    valid = {f.name for f in fields(Config)}

    for k, v in cfg_dict.items():
        if k in valid:
            try:
                setattr(cfg, k, coerce(v))
            except Exception:
                pass

    return cfg


def build_cfg_from_report_and_manifest(workdir: Path, manifest: dict):
    report_path = workdir / "routing_report.json"
    if not report_path.exists():
        raise FileNotFoundError(report_path)

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    cfg = Config(workdir=workdir)
    cfg = safe_set_config(cfg, report.get("config", {}))
    cfg.workdir = workdir

    cmd = None
    flags = {}

    for _, c in manifest.items():
        trial_flags = parse_flags(c)
        if Path(trial_flags.get("workdir", "")) == workdir:
            cmd = c
            flags = trial_flags
            break

    # Apply CLI flags from the manifest because these are often more complete
    # than routing_report["config"].
    for k, v in flags.items():
        if hasattr(cfg, k):
            try:
                setattr(cfg, k, coerce(v))
            except Exception:
                pass

    return cfg, report, flags, cmd


def infer_device(x: str) -> str:
    if x == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return x


# ---------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------

def _pick_text_column(corpus: pd.DataFrame) -> pd.Series:
    if "text" in corpus.columns:
        return corpus["text"].astype(str)

    # Robust fallback for BEIR-like corpora with title/text split.
    parts = []
    if "title" in corpus.columns:
        parts.append(corpus["title"].astype(str))
    if "body" in corpus.columns:
        parts.append(corpus["body"].astype(str))
    if "contents" in corpus.columns:
        parts.append(corpus["contents"].astype(str))

    if parts:
        s = parts[0]
        for p in parts[1:]:
            s = s + " " + p
        return s.astype(str)

    raise ValueError(f"Could not find text column in corpus columns: {list(corpus.columns)}")


def _pick_query_text_column(queries: pd.DataFrame) -> pd.Series:
    if "text" in queries.columns:
        return queries["text"].astype(str)
    if "qtext" in queries.columns:
        return queries["qtext"].astype(str)
    if "query" in queries.columns:
        return queries["query"].astype(str)
    if "title" in queries.columns:
        return queries["title"].astype(str)

    raise ValueError(f"Could not find query text column in query columns: {list(queries.columns)}")


def _pick_qrels_score(qrels: pd.DataFrame) -> pd.Series:
    for col in ["score", "rel", "relevance", "label"]:
        if col in qrels.columns:
            return qrels[col].astype(float)
    return pd.Series(np.ones(len(qrels), dtype=float), index=qrels.index)


def load_beir_jsonl(data_root: Path):
    corpus_path = data_root / "corpus.jsonl"
    queries_path = data_root / "queries.jsonl"
    qrels_path = data_root / "qrels.jsonl"

    for p in [corpus_path, queries_path, qrels_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    corpus = pd.read_json(corpus_path, lines=True)
    queries = pd.read_json(queries_path, lines=True)
    qrels = pd.read_json(qrels_path, lines=True)

    if "id" not in corpus.columns:
        raise ValueError(f"corpus.jsonl must contain id column. Columns: {list(corpus.columns)}")
    if "id" not in queries.columns:
        raise ValueError(f"queries.jsonl must contain id column. Columns: {list(queries.columns)}")

    if "query-id" not in qrels.columns or "corpus-id" not in qrels.columns:
        raise ValueError(
            "qrels.jsonl must contain query-id and corpus-id columns. "
            f"Columns: {list(qrels.columns)}"
        )

    corpus_ids = corpus["id"].astype(str).tolist()
    doc_map = {cid: i for i, cid in enumerate(corpus_ids)}

    docs_df = pd.DataFrame(
        {
            "doc_id": [doc_map[x] for x in corpus_ids],
            "text": _pick_text_column(corpus),
        }
    )

    q_meta = pd.DataFrame(
        {
            "qid": queries["id"].astype(str),
            "qtext": _pick_query_text_column(queries),
        }
    )

    # Drop qrels with query/doc ids that are not available.
    qrels = qrels.copy()
    qrels["query-id"] = qrels["query-id"].astype(str)
    qrels["corpus-id"] = qrels["corpus-id"].astype(str)

    valid_qids = set(q_meta["qid"].astype(str).tolist())
    valid_docids = set(doc_map.keys())

    before = len(qrels)
    qrels = qrels[
        qrels["query-id"].isin(valid_qids)
        & qrels["corpus-id"].isin(valid_docids)
    ].copy()
    dropped = before - len(qrels)

    if dropped:
        print(f"[data] dropped {dropped} qrels with missing query/doc ids")

    qrels_df = pd.DataFrame(
        {
            "qid": qrels["query-id"].astype(str),
            "doc_id": qrels["corpus-id"].map(doc_map).astype(int),
            "rel": _pick_qrels_score(qrels),
        }
    )

    return q_meta.reset_index(drop=True), docs_df.reset_index(drop=True), qrels_df.reset_index(drop=True)


def apply_limit_queries_to_beir(q_meta: pd.DataFrame, qrels_df: pd.DataFrame, cfg):
    limit = int(getattr(cfg, "limit_queries", 0) or 0)

    if limit <= 0 or limit >= len(q_meta):
        return q_meta.reset_index(drop=True), qrels_df.reset_index(drop=True)

    # Match the old non-BEIR behavior: random sample with fixed SEED.
    q_meta_limited = q_meta.sample(limit, random_state=SEED).reset_index(drop=True)
    keep_qids = set(q_meta_limited["qid"].astype(str).tolist())

    qrels_limited = qrels_df[
        qrels_df["qid"].astype(str).isin(keep_qids)
    ].reset_index(drop=True)

    print(
        f"[data] applied limit_queries={limit}: "
        f"{len(q_meta)} -> {len(q_meta_limited)} queries, "
        f"{len(qrels_df)} -> {len(qrels_limited)} qrels"
    )

    return q_meta_limited, qrels_limited


def load_dataset_from_old_config(cfg, flags):
    data_root = flags.get("data_root", None)

    if data_root:
        p = Path(data_root)

        if p.is_dir():
            q_meta, docs_df, qrels_df = load_beir_jsonl(p)
            q_meta, qrels_df = apply_limit_queries_to_beir(q_meta, qrels_df, cfg)
            return q_meta, docs_df, qrels_df

        if p.is_file():
            df = load_and_clean(p)
        else:
            raise FileNotFoundError(p)

    else:
        p = Path("/kaggle/working/combined-data.csv")
        if not p.exists():
            download_counselchat_csv(p)
        df = load_and_clean(p)

    # Non-BEIR / CounselChat path.
    if getattr(cfg, "limit_queries", 0):
        q_meta_all = df[["qid", "qtext"]].drop_duplicates("qid").reset_index(drop=True)

        if cfg.limit_queries < len(q_meta_all):
            sampled_qids = set(
                q_meta_all.sample(cfg.limit_queries, random_state=SEED)["qid"].tolist()
            )
            df = df[df["qid"].isin(sampled_qids)].copy()

            print(
                f"[data] applied limit_queries={cfg.limit_queries}: "
                f"{len(q_meta_all)} -> {df['qid'].nunique()} queries"
            )

    docs_df, qrels_df = build_corpus(df, cfg)
    q_meta = df[["qid", "qtext"]].drop_duplicates("qid").reset_index(drop=True)

    return q_meta, docs_df, qrels_df


# ---------------------------------------------------------------------
# Per-query metrics
# ---------------------------------------------------------------------

def make_rels(qrels_df: pd.DataFrame):
    return {
        str(q): {int(r.doc_id): float(r.rel) for r in g.itertuples(index=False)}
        for q, g in qrels_df.groupby("qid")
    }


def per_query_ndcg(qids, run, qrels_df, k):
    rels = make_rels(qrels_df)
    vals = []

    for i, qid in enumerate(qids):
        gains = [rels.get(str(qid), {}).get(int(d), 0.0) for d, _ in run[i]]
        vals.append(ndcg_at_k(gains, k))

    return np.asarray(vals, float)


def per_query_for_routes(qids, routes, runs, qrels_df, cfg):
    rels = make_rels(qrels_df)

    route_map = {
        "sparse": "sparse",
        "primary": "hybrid",
        "hybrid": "hybrid",
        "dense": "dense",
        "verified": "hybrid_verified",
        "hybrid_verified": "hybrid_verified",
    }

    vals = []

    for i, qid in enumerate(qids):
        r = str(routes[i])

        if r in {"crisis_response", "human_review", "fallback"}:
            vals.append(0.0)
            continue

        rn = route_map.get(r, "hybrid_verified")

        if rn not in runs:
            rn = "hybrid_verified" if "hybrid_verified" in runs else "hybrid"

        gains = [rels.get(str(qid), {}).get(int(d), 0.0) for d, _ in runs[rn][i]]
        vals.append(ndcg_at_k(gains, cfg.eval_k))

    return np.asarray(vals, float)


def paired_bootstrap(a, b, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    diff = np.asarray(a) - np.asarray(b)
    m = len(diff)
    means = np.empty(n)

    for i in range(n):
        means[i] = diff[rng.integers(0, m, m)].mean()

    lo, hi = np.quantile(means, [0.025, 0.975])
    obs = diff.mean()

    p = 2 * min(np.mean(means <= 0), np.mean(means >= 0))
    p = float(min(max(p, 0), 1))

    try:
        if np.allclose(diff, 0):
            stat, wp = 0.0, 1.0
        else:
            stat, wp = wilcoxon(diff)
    except Exception:
        stat, wp = None, None

    return {
        "mean_delta": float(obs),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "bootstrap_p_two_sided": p,
        "wilcoxon_stat": None if stat is None else float(stat),
        "wilcoxon_p": None if wp is None else float(wp),
        "n_queries": int(m),
        "significant_95_bootstrap": bool(lo > 0 or hi < 0),
    }


# ---------------------------------------------------------------------
# Answer-level sanity check
# ---------------------------------------------------------------------

def norm_text(x):
    return re.sub(r"\s+", " ", str(x)).strip()


def token_f1(a, b):
    A = re.findall(r"\w+", str(a).lower())
    B = re.findall(r"\w+", str(b).lower())

    if not A or not B:
        return 0.0

    ca, cb = Counter(A), Counter(B)
    ov = sum((ca & cb).values())

    if ov == 0:
        return 0.0

    p = ov / len(A)
    r = ov / len(B)

    return 2 * p * r / (p + r + 1e-12)


def ref_answers(qids, docs_df, qrels_df, max_words=250):
    docs = dict(zip(docs_df.doc_id.astype(int), docs_df.text.astype(str)))
    out = {}

    for qid, g in qrels_df.groupby("qid"):
        txt = " ".join(
            docs.get(int(r.doc_id), "")
            for r in g.itertuples(index=False)
            if float(r.rel) > 0
        )
        out[str(qid)] = " ".join(norm_text(txt).split()[:max_words])

    return out


def extract_answer(run_items, docs_df, max_words=180):
    docs = dict(zip(docs_df.doc_id.astype(int), docs_df.text.astype(str)))

    txt = " ".join(
        norm_text(docs.get(int(d), ""))
        for d, _ in run_items[:3]
    )

    return " ".join(txt.split()[:max_words])


def selected_run(method, i, routes, runs):
    if method == "always_sparse":
        return runs["sparse"][i]

    if method == "always_hybrid":
        return runs["hybrid"][i]

    if method == "always_verified":
        return runs["hybrid_verified"][i]

    if method == "dar":
        r = str(routes[i])

        if r == "sparse":
            return runs["sparse"][i]

        if r in {"hybrid", "primary"}:
            return runs["hybrid"][i]

        if r == "dense":
            if "dense" in runs:
                return runs["dense"][i]
            return runs["hybrid"][i]

        if r in {"verified", "hybrid_verified"}:
            return runs["hybrid_verified"][i]

        return []

    raise ValueError(method)


def answer_sanity(q_meta, qids, routes, runs, docs_df, qrels_df, outdir, sample_n=60, seed=42):
    rng = np.random.default_rng(seed)
    refs = ref_answers(qids, docs_df, qrels_df)
    qlookup = dict(zip(q_meta.qid.astype(str), q_meta.qtext.astype(str)))

    methods = ["always_hybrid", "always_verified", "dar"]
    labels = {
        "always_hybrid": "A",
        "always_verified": "B",
        "dar": "C",
    }

    rows = []
    ann = []

    sample = rng.choice(len(qids), min(sample_n, len(qids)), replace=False)

    for method in methods:
        rel = []
        sup = []
        uns = []
        lens = []
        fb = []

        for i, qid in enumerate(qids):
            fallback = (
                method == "dar"
                and str(routes[i]) in {"crisis_response", "human_review", "fallback"}
            )

            ans = "" if fallback else extract_answer(selected_run(method, i, routes, runs), docs_df)

            rel.append(token_f1(ans, refs.get(str(qid), "")))
            sup.append(1.0 if ans else 0.0)
            uns.append(0.0)
            lens.append(len(ans.split()))
            fb.append(1.0 if fallback else 0.0)

        rows.append(
            {
                "method": method,
                "answer_relevance_tokenF1": float(np.mean(rel)),
                "evidence_supported_sentence_rate": float(np.mean(sup)),
                "unsupported_sentence_rate": float(np.mean(uns)),
                "answer_length_words": float(np.mean(lens)),
                "fallback": float(np.mean(fb)),
            }
        )

    for i in sample:
        for method in methods:
            run = selected_run(method, i, routes, runs)
            ans = extract_answer(run, docs_df)
            ev = extract_answer(run[:1], docs_df, 120)

            ann.append(
                {
                    "qid": str(qids[i]),
                    "query": qlookup.get(str(qids[i]), ""),
                    "method_blinded": labels[method],
                    "answer": ans,
                    "top_evidence": ev,
                    "manual_relevance_1_to_5": "",
                    "manual_grounding_1_to_5": "",
                    "unsupported_claims_present_yes_no": "",
                    "safety_concern_yes_no": "",
                    "notes": "",
                }
            )

    sdf = pd.DataFrame(rows)
    adf = pd.DataFrame(ann)

    sdf.to_csv(outdir / "answer_level_sanity_check_summary_oldconfig.csv", index=False)
    adf.to_csv(outdir / "manual_answer_annotation_template_blinded_oldconfig.csv", index=False)

    return sdf, adf


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------

def load_routes(workdir: Path, qids):
    routes_path = workdir / "routes.csv"

    if not routes_path.exists():
        raise FileNotFoundError(routes_path)

    rdf = pd.read_csv(routes_path)

    candidate_cols = [
        "route_dual",
        "route",
        "selected_route",
        "decision",
        "route_final",
    ]

    route_col = None
    for c in candidate_cols:
        if c in rdf.columns:
            route_col = c
            break

    if route_col is None:
        raise KeyError(
            f"No route column found in {routes_path}. "
            f"Available columns: {list(rdf.columns)}"
        )

    # Best case: align by qid instead of row order.
    if "qid" in rdf.columns:
        rdf = rdf.copy()
        rdf["qid"] = rdf["qid"].astype(str)
        route_map = dict(zip(rdf["qid"], rdf[route_col].astype(str)))

        qids_str = [str(q) for q in qids]
        missing = [q for q in qids_str if q not in route_map]

        if missing:
            raise ValueError(
                f"routes.csv is missing {len(missing)} qids for {workdir}. "
                f"First missing qids: {missing[:10]}. "
                "This means the loaded query subset does not match the old run."
            )

        routes = [route_map[q] for q in qids_str]

        print(
            f"[routes] loaded {len(routes)} routes from column '{route_col}' "
            "aligned by qid"
        )

        return routes

    # Fallback only for old files without qid.
    routes = rdf[route_col].astype(str).tolist()

    if len(routes) != len(qids):
        raise ValueError(
            f"routes.csv length mismatch for {workdir}: "
            f"{len(routes)} routes, but {len(qids)} queries after loading. "
            "This usually means limit_queries or dataset sampling does not match the old run."
        )

    print(
        f"[routes] loaded {len(routes)} routes from column '{route_col}' "
        "by row order"
    )

    return routes


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--manifest", default="paper_experiments/experiment_manifest.json")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--bootstrap_resamples", type=int, default=10000)
    ap.add_argument("--sample_n", type=int, default=60)

    args = ap.parse_args()

    workdir = Path(args.workdir)

    manifest = load_manifest(args.manifest)
    cfg, report, flags, cmd = build_cfg_from_report_and_manifest(workdir, manifest)

    device = infer_device(args.device)

    print("[verify] command:", cmd)
    print(
        "[cfg]",
        {
            k: getattr(cfg, k, None)
            for k in [
                "chunk_mode",
                "limit_queries",
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
            ]
        },
    )

    q_meta, docs_df, qrels_df = load_dataset_from_old_config(cfg, flags)

    qids = q_meta.qid.astype(str).tolist()
    queries = q_meta.qtext.astype(str).tolist()

    print("[data]", len(qids), "queries", len(docs_df), "docs", len(qrels_df), "qrels")

    bm25 = BM25Index(docs_df.text.tolist(), docs_df.doc_id.tolist())

    dense_model = flags.get(
        "dense_model",
        getattr(cfg, "dense_model", "sentence-transformers/all-mpnet-base-v2"),
    )

    verified_model = flags.get(
        "verified_model",
        getattr(cfg, "verified_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    )

    print("[models] dense=", dense_model, "verified=", verified_model)

    dense = DenseIndex(dense_model, device=device)
    dense.build(docs_df.text.tolist(), docs_df.doc_id.tolist())

    reranker = (
        CrossEncoderReranker(
            verified_model,
            device=device,
            max_len=cfg.verified_max_len,
        )
        if getattr(cfg, "use_verified", True)
        else None
    )

    runs = build_runs(queries, docs_df, bm25, dense, reranker, cfg)

    routes = load_routes(workdir, qids)

    dar = per_query_for_routes(qids, routes, runs, qrels_df, cfg)
    sparse = per_query_ndcg(qids, runs["sparse"], qrels_df, cfg.eval_k)
    hybrid = per_query_ndcg(qids, runs["hybrid"], qrels_df, cfg.eval_k)
    verified = per_query_ndcg(qids, runs["hybrid_verified"], qrels_df, cfg.eval_k)

    static = {
        "always_sparse": sparse,
        "always_hybrid": hybrid,
        "always_verified": verified,
    }

    best = max(static, key=lambda k: np.mean(static[k]))

    paired = {
        "dar_vs_best_static": {
            **paired_bootstrap(dar, static[best], args.bootstrap_resamples, SEED),
            "method": "dar",
            "baseline": best,
        },
        "dar_vs_always_sparse": {
            **paired_bootstrap(dar, sparse, args.bootstrap_resamples, SEED),
            "method": "dar",
            "baseline": "always_sparse",
        },
        "dar_vs_always_hybrid": {
            **paired_bootstrap(dar, hybrid, args.bootstrap_resamples, SEED),
            "method": "dar",
            "baseline": "always_hybrid",
        },
        "dar_vs_always_verified": {
            **paired_bootstrap(dar, verified, args.bootstrap_resamples, SEED),
            "method": "dar",
            "baseline": "always_verified",
        },
    }

    out_path = workdir / "paired_significance_tests_oldconfig.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(paired, f, indent=2, cls=NumpyPathEncoder)

    print(json.dumps(paired, indent=2, cls=NumpyPathEncoder))
    print("[saved]", out_path)

    sdf, adf = answer_sanity(
        q_meta,
        qids,
        routes,
        runs,
        docs_df,
        qrels_df,
        workdir,
        args.sample_n,
        SEED,
    )

    print(sdf)
    print("[manual template]", adf.shape)


if __name__ == "__main__":
    main()
