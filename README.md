# Dual-Predictor Adaptive Routing for Mental Health Question Answering

This repository contains code and verified experimental artifacts for the paper:

**Complementarity-Aware Adaptive Retrieval Routing for Mental Health Question Answering**

The project studies retrieval strategy selection as a calibrated adaptive routing problem. The proposed method, Dual-Predictor Adaptive Routing (DAR), uses query performance prediction and sparse-dense complementarity features to route each query to sparse, hybrid, verified, or fallback retrieval tiers.

This GitHub release package excludes Reddit transfer artifacts. The Reddit MH-QA experiment was run separately because of runtime constraints.

## Contents

```text
code/                         Core implementation and analysis scripts
runs/                         Verified non-Reddit experiment outputs
metadata/                     Environment, manifest, and verified summary CSVs
reviewer_outputs/             Final reviewer-oriented verification outputs
dataset_metadata/             Dataset notes and excluded large-file explanation
requirements.txt              Python dependency list
README.md                     This file
MANIFEST.md                   File inventory
```

## Verified experiments included

- exp1_dense_only
- exp2_rrf_equal
- exp3_rrf_downweight
- exp4_sentence_chunks
- exp4_sentence_chunks_mpnet
- dense_comparison_tier3
- medpsych_sentence_chunks
- nfcorpus_sentence_chunks

## Key verified results

The table below is generated from `routing_report.json` and paired significance outputs.

| experiment                 |   sparse_ndcg |   dense_ndcg |   hybrid_ndcg |   hybrid_verified_ndcg |   dar_dual_ndcg_overall |   dar_dual_cost |   dar_dual_efficiency | paired_best_baseline   |   paired_mean_delta |   paired_ci_low |   paired_ci_high |   paired_wilcoxon_p |
|:---------------------------|--------------:|-------------:|--------------:|-----------------------:|------------------------:|----------------:|----------------------:|:-----------------------|--------------------:|----------------:|-----------------:|--------------------:|
| exp1_dense_only            |        0.299  |       0.709  |        0.709  |                 0.4643 |                  0.7158 |          4.9749 |                0.1439 | always_hybrid          |              0.0069 |         -0.0043 |           0.0182 |              0.5388 |
| exp2_rrf_equal             |        0.299  |       0.709  |        0.4756 |                 0.4744 |                  0.5659 |          7.2866 |                0.0777 | always_hybrid          |              0.0903 |          0.0716 |           0.109  |              0      |
| exp3_rrf_downweight        |        0.3096 |       0.709  |        0.641  |                 0.4622 |                  0.6357 |          5.7689 |                0.1102 | always_hybrid          |             -0.0052 |         -0.0192 |           0.0088 |              0.1633 |
| exp4_sentence_chunks       |        0.3149 |       0.3964 |        0.4113 |                 0.5377 |                  0.5673 |          6.8335 |                0.083  | always_verified        |              0.0296 |          0.0176 |           0.0422 |              0.0002 |
| exp4_sentence_chunks_mpnet |        0.3149 |       0.7347 |        0.5147 |                 0.5292 |                  0.6224 |          6.207  |                0.1003 | always_verified        |              0.0932 |          0.0764 |           0.1099 |              0      |
| dense_comparison_tier3     |        0.3169 |       0.4022 |        0.4126 |                 0.5383 |                  0.5687 |          6.6644 |                0.0853 | always_verified        |              0.0332 |          0.0205 |           0.0467 |              0.0001 |
| medpsych_sentence_chunks   |        0.5456 |       0.6256 |        0.625  |                 0.699  |                  0.7432 |          5.503  |                0.1351 | always_verified        |              0.0442 |          0.0369 |           0.0517 |              0      |
| nfcorpus_sentence_chunks   |        0.4303 |       0.4115 |        0.4345 |                 0.5029 |                  0.5201 |          6.1827 |                0.0841 | always_verified        |              0.0172 |          0.006  |           0.0291 |              0.0043 |

## Interpretation of paired tests

Paired bootstrap confidence intervals and Wilcoxon signed-rank tests were computed using per-query nDCG@10. A positive `paired_mean_delta` means DAR outperformed the best static baseline for that experiment.

The verified outputs show that DAR significantly improves over the best static baseline in multiple settings, including equal-weight RRF, sentence-chunk CounselChat, MedPsych-Online, NFCorpus, and the dense comparison tier. DAR is statistically comparable to the best static baseline in dense-only and downweighted-RRF settings, where the static retriever is already strong.

## Datasets

This release does not include large dataset files. The original experiments used:

- CounselChat
- MedPsych-Online
- NFCorpus
- Reddit MH-QA, excluded from this no-Reddit package

Expected local structure for full reproduction:

```text
paper_experiments/data/medpsych_online/
paper_experiments/data/nfcorpus/
paper_experiments/data/reddit_mental_health/
combined-data.csv
```

## Main commands

Run a single experiment:

```bash
python dar_router_main.py \
  --workdir paper_experiments/runs/exp4_sentence_chunks \
  --device auto \
  --chunk_mode sentence \
  --min_chunk_words 120 \
  --max_chunk_words 260 \
  --topk_bm25 200 \
  --topk_dense 200 \
  --topk_fusion 200 \
  --rrf_k 60 \
  --verified_topn 50 \
  --verified_max_len 256 \
  --qpp_post_k 50 \
  --dense_model pritamdeka/S-PubMedBert-MS-MARCO \
  --verified_model cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --hybrid_mode rrf \
  --rrf_w_bm25 1.0 \
  --rrf_w_dense 1.0 \
  --verified_source hybrid \
  --thr_sparse 0.50 \
  --thr_hybrid 0.55 \
  --safety_override_thr 0.55 \
  --dual_delta 0.00 \
  --augment_nimh \
  --user_country US \
  --no_sweep
```

Run reviewer add-on for paired tests and answer sanity checks:

```bash
python reviewer_addons_from_old_runs.py \
  --workdir paper_experiments/runs/exp4_sentence_chunks \
  --device auto \
  --bootstrap_resamples 10000 \
  --sample_n 60
```

Run final verifier:

```bash
python final_submission_verifier.py
```

Run this GitHub packager:

```bash
python github_release_packager_no_reddit.py
```

## Important notes

- Reddit transfer is intentionally excluded from this package.
- Large raw datasets are excluded to keep the repository lightweight.
- Answer-level sanity checks are proxy checks based on retrieved evidence token overlap, not human clinical evaluation.
- This work evaluates retrieval-layer behavior, not direct mental-health outcomes or therapeutic effectiveness.

## Generated files

The main generated files are:

```text
metadata/verified_no_reddit_summary.csv
metadata/missing_files_report.csv
metadata/environment_info.json
reviewer_outputs/all_experiment_numbers_verified.csv
reviewer_outputs/routing_results_compact.csv
runs/*/routing_report.json
runs/*/paired_significance_tests_oldconfig.json
runs/*/answer_level_sanity_check_summary_oldconfig.csv
```
