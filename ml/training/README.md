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
  --raw_glob "data/raw/cosmetics/*.csv" \
  --processed_dir data/processed/cosmetics \
  --models_dir models/recs_reranker_v2 \
  --reports_dir reports \
  --top_m 200 \
  --context_k 3 \
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

## 3) Offer pipeline (project DB events)

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

## 4) Recommended retrain cadence

- Recs reranker: daily or every 2-3 days.
- Offer redemption model: weekly.
- Always compare against previous model on fixed holdout before promote.
