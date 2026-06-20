# XGBoost Training Pipeline — Prompt Injection Detection

**Date:** 2026-05-20  
**Status:** Approved  
**Primary Metric:** AUC-ROC

---

## Context

399,741 labeled prompts (balanced: 50.8% benign / 49.2% injection), 35 hand-crafted features from `features_engineered.csv`. GPU available: RTX 5070 (12 GB VRAM), CUDA 13.2. XGBoost not yet installed; will be installed in cell 1.

---

## Approach

**5-fold Stratified CV inside Optuna + holdout test evaluation.**

Optuna (100 trials, TPE sampler + MedianPruner) tunes XGBoost hyperparameters by maximizing mean AUC-ROC across 5 stratified folds on the 80% train set. The final model is trained on the full 80% train set (with 20% of that as early-stopping val), then evaluated on the 20% holdout test set — which is never seen during any optimization step.

---

## Data

- **Source:** `/home/linkezio/Projects/IA-SI/data/features_engineered.csv`
- **Features (35):** text_length, word_count, unique_words, sentence_count, avg_word_length, char_count_no_space, uppercase_ratio, lowercase_ratio, digit_ratio, special_char_ratio, punctuation_ratio, space_ratio, newline_count, type_token_ratio, lexical_diversity, avg_word_frequency, word_length_variance, entropy, has_ignore_keyword, count_ignore_keyword, has_act_as_keyword, count_act_as_keyword, has_system_keyword, count_system_keyword, has_override_keyword, count_override_keyword, has_execute_keyword, count_execute_keyword, total_injection_keywords, keyword_density, colon_count, bracket_count, parenthesis_count, quote_count, comma_to_period_ratio
- **Target:** `label` (0=benign, 1=injection)
- **Preprocessing:** fill 124 nulls in `lexical_diversity` with column median; clip `comma_to_period_ratio` outliers at 99th percentile

---

## Notebook Structure (training.ipynb)

### Cell 1 — Imports & Setup
- `pip install xgboost optuna shap` inside notebook
- All library imports
- Global `SEED = 42`, `DEVICE = 'cuda'`
- Suppress non-essential warnings

### Cell 2 — Configuration
Single `CONFIG` dict with all paths, hyperparameter bounds, and constants. No magic numbers elsewhere.

```python
CONFIG = {
    "data_path": "/home/linkezio/Projects/IA-SI/data/features_engineered.csv",
    "original_data_path": "/home/linkezio/Datasets/prompt_injection_dataset.csv",
    "models_dir": "/home/linkezio/Projects/IA-SI/models",
    "test_size": 0.20,
    "cv_folds": 5,
    "n_trials": 100,
    "early_stopping_rounds": 50,
    "seed": 42,
    "device": "cuda",
    # hyperparameter bounds...
}
```

### Cell 3 — Data Loading & Preprocessing
- Load CSV, assert shape and columns
- Fill `lexical_diversity` nulls with median
- Clip `comma_to_period_ratio` at 99th percentile
- Print class distribution, null counts

### Cell 4 — Train/Test Split
- `train_test_split` stratified, 80/20, seed=42
- Print shapes and class distributions for both sets
- Assert no leakage

### Cell 5 — Optuna Objective Function
```
def objective(trial):
    params = {sample from search space}
    cv_scores = StratifiedKFold(5) → XGBClassifier.fit → roc_auc
    return mean(cv_scores)
```
- `device='cuda'`, `tree_method='hist'` for GPU
- `eval_metric='auc'`
- MedianPruner for early trial termination

**Hyperparameter search space:**

| Parameter | Range | Scale |
|---|---|---|
| n_estimators | 300–3000 | int |
| max_depth | 3–10 | int |
| learning_rate | 0.005–0.3 | log-uniform |
| subsample | 0.5–1.0 | float |
| colsample_bytree | 0.5–1.0 | float |
| min_child_weight | 1–15 | int |
| gamma | 1e-8–5 | log-uniform |
| reg_alpha | 1e-8–10 | log-uniform |
| reg_lambda | 1e-8–10 | log-uniform |

### Cell 6 — Optuna Search
- `study = optuna.create_study(direction='maximize', sampler=TPESampler, pruner=MedianPruner)`
- `study.optimize(objective, n_trials=100, show_progress_bar=True)`
- Print best AUC-ROC and best params
- Plot: optimization history, param importances (built-in Optuna visualizations)

### Cell 7 — Final Model Training
- Merge best params with fixed GPU params
- Train on 80% train set, using 20% of train as internal early-stopping validation (not the holdout)
- Use `verbose_eval=100` to show training progress
- Print final number of trees used (after early stopping)

### Cell 8 — Evaluation on Holdout Test Set
Compute and display:
- **AUC-ROC** (primary) with ROC curve plot
- **AUC-PR** with Precision-Recall curve plot
- **Confusion matrix** (heatmap, both raw counts and normalized)
- **Classification report:** accuracy, precision, recall, F1 at threshold=0.5
- **Calibration curve** (reliability diagram, 10 bins)
- **Threshold analysis plot:** precision, recall, F1 vs. threshold from 0.1 to 0.9

### Cell 9 — Error Analysis
- Find false positives (benign predicted as injection): top 20 by confidence
- Find false negatives (injection predicted as benign): top 20 by confidence
- For each group: show the original text snippet + feature values that likely drove the error
- Print summary statistics of misclassified samples

*Note: requires re-joining with original text. Load `/home/linkezio/Datasets/prompt_injection_dataset.csv` (column `text`), apply same 80/20 stratified split with seed=42 to get aligned test indices, join by position.*

### Cell 10 — Feature Importance
- XGBoost built-in: plot top 20 by `gain`, `weight`, and `cover` (3 plots)
- SHAP beeswarm plot (global, all features)
- SHAP waterfall plot for 3 examples: highest-confidence injection, highest-confidence benign, and most uncertain (score ≈ 0.5)

### Cell 11 — Save Artifacts
Save to `/home/linkezio/Projects/IA-SI/models/`:
- `xgboost_model.json` — trained model (native XGBoost format)
- `best_params.json` — Optuna best params
- `evaluation_metrics.json` — all holdout metrics (AUC-ROC, F1, etc.)
- `optuna_study.pkl` — full Optuna study object

---

## Artifacts Output

```
models/
├── xgboost_model.json
├── best_params.json
├── evaluation_metrics.json
└── optuna_study.pkl
```

---

## Expected Results

Based on feature correlation analysis (top features: `has_override_keyword` r=0.40, `keyword_density` r=0.39, `entropy` r=0.33):

- Expected AUC-ROC: **0.93–0.97** after full Optuna search
- Training time: ~30–40 min on RTX 5070

---

## Dependencies to Install

```
xgboost>=2.0   (for device='cuda' support)
optuna>=3.0
shap>=0.44
```

---

## Risks & Notes

- `lexical_diversity` has 124 nulls — fill with median before training, not dropping rows
- RTX 5070 uses CUDA 13.2; XGBoost 2.x supports this natively with `device='cuda'`
- Error analysis requires loading original dataset to show text snippets; adds ~2 GB RAM temporarily
- Optuna `MedianPruner` may prune aggressively — if too many trials are pruned, switch to `NopPruner`
