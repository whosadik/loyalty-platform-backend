# ML Training Pipelines

## 1) Install ML dependencies

```bash
pip install -r requirements-ml.txt
```

## 2) Recs pipeline (external datasets)

This pipeline builds and evaluates a recommendation reranker with:

- user-level train/test split,
- no train/eval leakage (transition map is built only from train users),
- deterministic reports for baseline and reranker.

```bash
python ml/training/run_recs_pipeline.py \
  --python .venv/Scripts/python.exe \
  --data_source cosmetics_raw \
  --raw_glob "data/raw/cosmetics/*.csv" \
  --processed_dir data/processed/cosmetics \
  --models_dir models/recs_reranker_v2 \
  --reports_dir reports \
  --top_m 1500 \
  --context_k 10 \
  --product_type_fallback_topn 400 \
  --category_fallback_topn 400 \
  --brand_fallback_topn 400 \
  --behavior_event_types add_to_cart,click,purchase_attributed \
  --behavior_weight 0.25 \
  --estimator hgb \
  --model_version recs_reranker_vnext_hgb \
  --neg_per_pos 40 \
  --test_size 0.2 \
  --seed 42
```

Outputs:

- `models/recs_reranker_v2/model.pkl`
- `models/recs_reranker_v2/train_users.txt`
- `models/recs_reranker_v2/test_users.txt`
- `reports/cooc_baseline_test.txt`
- `reports/reranker_test.txt`
- `reports/cooc_baseline_test.json`
- `reports/reranker_test.json`

JSON reports now include additional business proxies:

- `revenue_recall_at_10`, `revenue_recall_at_20`
- `catalog_coverage_at_10`, `catalog_coverage_at_20`
- `avg_recommended_price_at_10`, `avg_recommended_price_at_20`

## 3) Recs pipeline (project DB tables)

Use this mode to build train data from backend DB tables:

- `transactions` / `transaction_items`
- `recs_analytics_recommendationevent`
- `offers_offerevent`

```bash
python ml/training/run_recs_pipeline.py \
  --python .venv/Scripts/python.exe \
  --data_source project_db \
  --db_days 365 \
  --processed_dir data/processed/project \
  --models_dir models/recs_reranker_project_v1 \
  --reports_dir reports \
  --top_m 1500 \
  --context_k 10 \
  --product_type_fallback_topn 400 \
  --category_fallback_topn 400 \
  --brand_fallback_topn 400 \
  --behavior_event_types add_to_cart,click,purchase_attributed \
  --behavior_weight 0.25 \
  --estimator hgb \
  --model_version recs_reranker_project_vnext_hgb \
  --neg_per_pos 40 \
  --test_size 0.2 \
  --seed 42
```

Additional exported artifacts (for future feature engineering):

- `data/processed/project/interactions.parquet`
- `data/processed/project/items.parquet`
- `data/processed/project/transactions_items.parquet`
- `data/processed/project/rec_events.parquet`
- `data/processed/project/offer_events.parquet`
- `data/processed/project/offer_assignments.parquet`

## 4) Offer pipeline (project DB events)

This pipeline exports training data from your current backend DB and trains
an offer redemption probability model.

```bash
python ml/training/run_offer_pipeline.py \
  --python .venv/Scripts/python.exe \
  --processed_dir data/processed/project \
  --models_dir models/offer_redemption_lr_v1 \
  --days 180 \
  --seed 42
```

Outputs:

- `data/processed/project/offer_train.parquet`
- `models/offer_redemption_lr_v1/model.pkl`
- `models/offer_redemption_lr_v1/metadata.json`

## 5) Recommended retrain cadence

- Recs reranker: daily or every 2-3 days.
- Offer redemption model: weekly.
- Always compare against previous model on fixed holdout before promote.

## 6) Retrieval sanity check (coverage first)

Before tuning reranker features, validate candidate retrieval quality:

```bash
python ml/training/sanity_check_candidates.py \
  --interactions data/processed/cosmetics/interactions.parquet \
  --items data/processed/cosmetics/items.parquet \
  --ds data/processed/cosmetics/next_purchase_ds.parquet \
  --train_users models/recs_reranker_v2/train_users.txt \
  --eval_users models/recs_reranker_v2/test_users.txt \
  --top_m 1500 \
  --context_k 10 \
  --product_type_fallback_topn 400 \
  --category_fallback_topn 400 \
  --brand_fallback_topn 400 \
  --behavior_event_types add_to_cart,click,purchase_attributed \
  --behavior_weight 0.25
```

Target: `coverage_with_fallback >= 0.40`.
