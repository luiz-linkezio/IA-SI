# XGBoost Training Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a professional-grade XGBoost training notebook for prompt injection detection, with GPU acceleration (RTX 5070 / CUDA 13.2), Optuna hyperparameter search (5-fold CV, 100 trials), full evaluation suite, error analysis, and SHAP explainability.

**Architecture:** Single Jupyter notebook (`notebooks/training.ipynb`) with 11 independent, sequentially-runnable cells. Each cell has one responsibility. All paths and tunable constants live in a single `CONFIG` dict (Cell 2). Optuna runs 5-fold stratified CV inside each trial to produce a reliable AUC-ROC estimate; the final model is trained on the full 80% train set and evaluated on a 20% holdout that is never seen during optimization.

**Tech Stack:** XGBoost ≥ 2.0 (`device='cuda'`, `tree_method='hist'`), Optuna ≥ 3.0 (TPE sampler + MedianPruner), SHAP ≥ 0.44, scikit-learn, pandas, matplotlib, seaborn.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `notebooks/training.ipynb` | **Modify** (currently empty) | The complete training pipeline |
| `models/xgboost_model.json` | **Create** (output) | Trained XGBoost model |
| `models/best_params.json` | **Create** (output) | Optuna best hyperparameters |
| `models/evaluation_metrics.json` | **Create** (output) | All holdout evaluation metrics |
| `models/optuna_study.pkl` | **Create** (output) | Full Optuna study object |
| `models/evaluation_plots.png` | **Create** (output) | 6-panel evaluation figure |
| `models/feature_importance.png` | **Create** (output) | Feature importance (gain/weight/cover) |
| `models/shap_beeswarm.png` | **Create** (output) | SHAP global beeswarm |

---

## Task 1: Cell 1 — Imports & Setup

**Files:**
- Modify: `notebooks/training.ipynb` (cell 1)

- [ ] **Step 1.1: Write Cell 1 — install packages and verify GPU**

Replace the empty notebook with this first cell:

```python
import subprocess
import sys

# Install required packages (idempotent)
packages = ['xgboost>=2.0', 'optuna>=3.5', 'shap>=0.44']
for pkg in packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import optuna
import shap
import json
import pickle
import os
import time
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve,
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
np.random.seed(42)

print(f"XGBoost:  {xgb.__version__}")
print(f"Optuna:   {optuna.__version__}")
print(f"SHAP:     {shap.__version__}")

# Verify GPU
print("\nVerifying GPU...")
try:
    _dm = xgb.DMatrix([[1, 2], [3, 4]], label=[0, 1])
    xgb.train({'device': 'cuda', 'tree_method': 'hist', 'eval_metric': 'auc'},
               _dm, num_boost_round=1, verbose_eval=False)
    print("✓ XGBoost GPU (cuda) OK")
except Exception as e:
    print(f"⚠ GPU unavailable: {e}")
    print("  Falling back to CPU — set DEVICE='cpu' in CONFIG")
```

- [ ] **Step 1.2: Run Cell 1 and verify output**

Expected output:
```
XGBoost:  2.x.x
Optuna:   3.x.x
SHAP:     0.x.x
Verifying GPU...
✓ XGBoost GPU (cuda) OK
```
If GPU fails, set `DEVICE = 'cpu'` in Cell 2 before proceeding.

---

## Task 2: Cell 2 — Configuration

**Files:**
- Modify: `notebooks/training.ipynb` (cell 2)

- [ ] **Step 2.1: Write Cell 2 — single CONFIG dict**

```python
CONFIG = {
    # Paths
    "data_path": "/home/linkezio/Projects/IA-SI/data/features_engineered.csv",
    "original_data_path": "/home/linkezio/Datasets/prompt_injection_dataset.csv",
    "models_dir": "/home/linkezio/Projects/IA-SI/models",

    # Split
    "test_size": 0.20,
    "val_size": 0.20,   # fraction of train used for early stopping (not the holdout)
    "cv_folds": 5,
    "seed": 42,

    # Optuna
    "n_trials": 100,
    "study_name": "xgboost_prompt_injection_v1",

    # XGBoost fixed
    "device": "cuda",          # change to "cpu" if no GPU
    "tree_method": "hist",
    "eval_metric": "auc",
    "early_stopping_rounds": 50,

    # Hyperparameter search bounds (min, max)
    "hp": {
        "n_estimators":      (300,  3000),
        "max_depth":         (3,    10),
        "learning_rate":     (0.005, 0.3),
        "subsample":         (0.5,  1.0),
        "colsample_bytree":  (0.5,  1.0),
        "min_child_weight":  (1,    15),
        "gamma":             (1e-8, 5.0),
        "reg_alpha":         (1e-8, 10.0),
        "reg_lambda":        (1e-8, 10.0),
    },
}

os.makedirs(CONFIG["models_dir"], exist_ok=True)
print(f"✓ Models directory ready: {CONFIG['models_dir']}")
print(f"✓ Config loaded — device: {CONFIG['device']}, trials: {CONFIG['n_trials']}")
```

- [ ] **Step 2.2: Run Cell 2 and verify output**

Expected:
```
✓ Models directory ready: /home/linkezio/Projects/IA-SI/models
✓ Config loaded — device: cuda, trials: 100
```

---

## Task 3: Cell 3 — Data Loading & Preprocessing

**Files:**
- Modify: `notebooks/training.ipynb` (cell 3)

- [ ] **Step 3.1: Write Cell 3 — load CSV and preprocess**

```python
df = pd.read_csv(CONFIG["data_path"])
print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Sanity checks
assert df.shape[1] == 36, f"Expected 36 columns, got {df.shape[1]}"
assert 'label' in df.columns, "Missing 'label' column"
assert df['label'].nunique() == 2, "Expected binary label"

# Null report
nulls = df.isnull().sum()
print(f"\nNulls before preprocessing:")
print(nulls[nulls > 0].to_string())

# Fill lexical_diversity nulls with median
median_ld = df['lexical_diversity'].median()
df['lexical_diversity'] = df['lexical_diversity'].fillna(median_ld)
print(f"\n✓ Filled lexical_diversity nulls with median = {median_ld:.6f}")

# Clip comma_to_period_ratio at 99th percentile (removes extreme outliers)
p99_cpr = df['comma_to_period_ratio'].quantile(0.99)
df['comma_to_period_ratio'] = df['comma_to_period_ratio'].clip(upper=p99_cpr)
print(f"✓ Clipped comma_to_period_ratio at 99th pct = {p99_cpr:.4f}")

# Final null check
assert df.isnull().sum().sum() == 0, "Unexpected nulls after preprocessing"
print(f"✓ Zero nulls after preprocessing")

# Class distribution
print(f"\nClass distribution:")
print(df['label'].value_counts().to_string())
print(df['label'].value_counts(normalize=True).round(4).to_string())
```

- [ ] **Step 3.2: Run Cell 3 and verify output**

Expected:
```
Loaded: 399,741 rows × 36 columns
Nulls before preprocessing:
lexical_diversity    124
✓ Filled lexical_diversity nulls with median = x.xxxxxx
✓ Clipped comma_to_period_ratio at 99th pct = x.xxxx
✓ Zero nulls after preprocessing
Class distribution:
label
0    203067
1    196674
```

---

## Task 4: Cell 4 — Train/Test Split

**Files:**
- Modify: `notebooks/training.ipynb` (cell 4)

- [ ] **Step 4.1: Write Cell 4 — stratified split**

```python
FEATURE_COLS = [c for c in df.columns if c != 'label']
X = df[FEATURE_COLS].reset_index(drop=True)
y = df['label'].reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=CONFIG["test_size"],
    stratify=y,
    random_state=CONFIG["seed"],
)

# Reset indices for clean indexing later
X_train = X_train.reset_index(drop=True)
X_test  = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test  = y_test.reset_index(drop=True)

print(f"Train set: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Test set:  {X_test.shape[0]:,} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"\nTrain class distribution: {y_train.value_counts(normalize=True).round(4).to_dict()}")
print(f"Test  class distribution: {y_test.value_counts(normalize=True).round(4).to_dict()}")

# Stratification check: class ratios should be within 1% of each other
assert abs(y_train.mean() - y_test.mean()) < 0.01, "Stratification failed"
print(f"\n✓ Stratification check passed (|train_mean - test_mean| < 0.01)")
print(f"  Features: {len(FEATURE_COLS)}")
```

- [ ] **Step 4.2: Run Cell 4 and verify output**

Expected:
```
Train set: 319,792 samples (80.0%)
Test set:   79,949 samples (20.0%)
Train class distribution: {0: 0.508, 1: 0.492}
Test  class distribution: {0: 0.508, 1: 0.492}
✓ Stratification check passed
  Features: 35
```

---

## Task 5: Cell 5 — Optuna Objective Function

**Files:**
- Modify: `notebooks/training.ipynb` (cell 5)

- [ ] **Step 5.1: Write Cell 5 — define objective**

```python
def objective(trial: optuna.Trial) -> float:
    """5-fold stratified CV, returns mean AUC-ROC. GPU-accelerated."""
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", *CONFIG["hp"]["n_estimators"]),
        "max_depth":        trial.suggest_int("max_depth",    *CONFIG["hp"]["max_depth"]),
        "learning_rate":    trial.suggest_float("learning_rate",   *CONFIG["hp"]["learning_rate"],   log=True),
        "subsample":        trial.suggest_float("subsample",       *CONFIG["hp"]["subsample"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree",*CONFIG["hp"]["colsample_bytree"]),
        "min_child_weight": trial.suggest_int("min_child_weight",  *CONFIG["hp"]["min_child_weight"]),
        "gamma":            trial.suggest_float("gamma",           *CONFIG["hp"]["gamma"],       log=True),
        "reg_alpha":        trial.suggest_float("reg_alpha",       *CONFIG["hp"]["reg_alpha"],   log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda",      *CONFIG["hp"]["reg_lambda"],  log=True),
        # Fixed
        "device":          CONFIG["device"],
        "tree_method":     CONFIG["tree_method"],
        "eval_metric":     CONFIG["eval_metric"],
        "random_state":    CONFIG["seed"],
        "verbosity":       0,
    }

    cv = StratifiedKFold(n_splits=CONFIG["cv_folds"], shuffle=True, random_state=CONFIG["seed"])
    fold_aucs = []

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_f_tr, y_f_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_f_val, y_f_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_f_tr, y_f_tr,
            eval_set=[(X_f_val, y_f_val)],
            early_stopping_rounds=CONFIG["early_stopping_rounds"],
            verbose=False,
        )

        proba = model.predict_proba(X_f_val)[:, 1]
        fold_aucs.append(roc_auc_score(y_f_val, proba))

        # Report to pruner after each fold
        trial.report(np.mean(fold_aucs), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_aucs))


print("✓ Optuna objective defined")
print(f"  CV folds: {CONFIG['cv_folds']}")
print(f"  Early stopping rounds: {CONFIG['early_stopping_rounds']}")
print(f"  Primary metric: AUC-ROC (maximize)")

# Quick smoke test: run 1 trial to verify no crashes
print("\nSmoke test (1 trial)...")
_study_test = optuna.create_study(direction="maximize")
_study_test.optimize(objective, n_trials=1, show_progress_bar=False)
print(f"✓ Smoke test passed — trial AUC: {_study_test.best_value:.4f}")
```

- [ ] **Step 5.2: Run Cell 5 and verify smoke test passes**

Expected:
```
✓ Optuna objective defined
  CV folds: 5
  Early stopping rounds: 50
  Primary metric: AUC-ROC (maximize)
Smoke test (1 trial)...
✓ Smoke test passed — trial AUC: 0.9xxx
```
If the smoke test AUC is below 0.70, something is wrong — re-check data loading.

---

## Task 6: Cell 6 — Optuna Hyperparameter Search

**Files:**
- Modify: `notebooks/training.ipynb` (cell 6)

- [ ] **Step 6.1: Write Cell 6 — run full search**

```python
print(f"Starting Optuna search: {CONFIG['n_trials']} trials, {CONFIG['cv_folds']}-fold CV...")
print(f"GPU: device={CONFIG['device']}, tree_method={CONFIG['tree_method']}\n")

sampler = optuna.samplers.TPESampler(seed=CONFIG["seed"])
pruner  = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2)

study = optuna.create_study(
    direction="maximize",
    study_name=CONFIG["study_name"],
    sampler=sampler,
    pruner=pruner,
)

t0 = time.time()
study.optimize(objective, n_trials=CONFIG["n_trials"], show_progress_bar=True)
elapsed_min = (time.time() - t0) / 60

n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
n_pruned   = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])

print(f"\n{'='*60}")
print(f"Optuna Search Complete — {elapsed_min:.1f} minutes")
print(f"{'='*60}")
print(f"Best 5-fold AUC-ROC: {study.best_value:.6f}")
print(f"Best trial #:        {study.best_trial.number}")
print(f"Completed trials:    {n_complete}")
print(f"Pruned trials:       {n_pruned}")
print(f"\nBest hyperparameters:")
for k, v in sorted(study.best_params.items()):
    print(f"  {k:<25} {v}")

# Plot: optimization history
fig1, ax1 = plt.subplots(figsize=(10, 4))
values = [t.value for t in study.trials if t.value is not None]
ax1.plot(values, alpha=0.4, color='steelblue', label='Trial AUC')
running_best = pd.Series(values).cummax()
ax1.plot(running_best.values, color='darkorange', lw=2, label='Best so far')
ax1.set_xlabel('Trial')
ax1.set_ylabel('AUC-ROC (5-fold CV)')
ax1.set_title('Optuna Optimization History')
ax1.legend()
ax1.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Plot: parameter importances
try:
    fig2 = optuna.visualization.matplotlib.plot_param_importances(study)
    plt.title('Hyperparameter Importances (FAnova)')
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"(Param importance plot skipped: {e})")
```

- [ ] **Step 6.2: Run Cell 6 — this is the long step (~30-40 min)**

Expected final output:
```
Optuna Search Complete — xx.x minutes
============================================================
Best 5-fold AUC-ROC: 0.9xxxxx
Best trial #:        xx
Completed trials:    xx
Pruned trials:       xx
Best hyperparameters:
  colsample_bytree         x.xxxx
  gamma                    x.xxxx
  learning_rate            x.xxxx
  ...
```
If fewer than 50 trials complete (rest pruned), set `pruner = optuna.pruners.NopPruner()` and re-run.

---

## Task 7: Cell 7 — Final Model Training

**Files:**
- Modify: `notebooks/training.ipynb` (cell 7)

- [ ] **Step 7.1: Write Cell 7 — train final model on full train set**

```python
# Merge best params with fixed GPU settings
best_params = {**study.best_params}
best_params.update({
    "device":       CONFIG["device"],
    "tree_method":  CONFIG["tree_method"],
    "eval_metric":  CONFIG["eval_metric"],
    "random_state": CONFIG["seed"],
    "verbosity":    0,
})

# Internal early-stopping split from train set (NOT the holdout)
X_tr, X_es_val, y_tr, y_es_val = train_test_split(
    X_train, y_train,
    test_size=CONFIG["val_size"],
    stratify=y_train,
    random_state=CONFIG["seed"],
)

print(f"Final model training")
print(f"  Train:          {X_tr.shape[0]:,} samples")
print(f"  Early-stop val: {X_es_val.shape[0]:,} samples  (internal only, not the holdout)")
print(f"  Max trees:      {best_params['n_estimators']}")
print(f"  Early stopping: {CONFIG['early_stopping_rounds']} rounds\n")

final_model = xgb.XGBClassifier(**best_params)
final_model.fit(
    X_tr, y_tr,
    eval_set=[(X_es_val, y_es_val)],
    early_stopping_rounds=CONFIG["early_stopping_rounds"],
    verbose=100,
)

print(f"\n✓ Training complete")
print(f"  Best iteration (trees used): {final_model.best_iteration}")
print(f"  Early-stop val AUC:          {final_model.best_score:.6f}")
```

- [ ] **Step 7.2: Run Cell 7 and verify training log**

Expected: training log showing AUC increasing each 100 rounds, then stopping. Best iteration should be between 100 and `n_estimators`. Best val AUC should be ≥ 0.90.

---

## Task 8: Cell 8 — Full Evaluation on Holdout Test Set

**Files:**
- Modify: `notebooks/training.ipynb` (cell 8)

- [ ] **Step 8.1: Write Cell 8 — metrics + 6-panel plot**

```python
y_pred_proba = final_model.predict_proba(X_test)[:, 1]
y_pred       = final_model.predict(X_test)

# Compute metrics
auc_roc = roc_auc_score(y_test, y_pred_proba)
auc_pr  = average_precision_score(y_test, y_pred_proba)
acc     = accuracy_score(y_test, y_pred)
prec    = precision_score(y_test, y_pred)
rec     = recall_score(y_test, y_pred)
f1      = f1_score(y_test, y_pred)
cm      = confusion_matrix(y_test, y_pred)

print("=" * 56)
print("HOLDOUT TEST SET EVALUATION")
print("=" * 56)
print(f"  AUC-ROC  (primary) : {auc_roc:.6f}")
print(f"  AUC-PR             : {auc_pr:.6f}")
print(f"  Accuracy           : {acc:.6f}")
print(f"  Precision          : {prec:.6f}")
print(f"  Recall             : {rec:.6f}")
print(f"  F1 Score           : {f1:.6f}")
print(f"\n  Confusion Matrix (raw):")
print(f"    TN={cm[0,0]:,}  FP={cm[0,1]:,}")
print(f"    FN={cm[1,0]:,}  TP={cm[1,1]:,}")
print()
print(classification_report(y_test, y_pred, target_names=['Benign (0)', 'Injection (1)']))

# 6-panel evaluation figure
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("XGBoost Evaluation — Prompt Injection Detection", fontsize=15, fontweight='bold', y=1.01)

# Panel 1: ROC Curve
ax = axes[0, 0]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC-ROC = {auc_roc:.4f}')
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)

# Panel 2: Precision-Recall Curve
ax = axes[0, 1]
prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_pred_proba)
ax.plot(rec_vals, prec_vals, color='steelblue', lw=2, label=f'AUC-PR = {auc_pr:.4f}')
ax.axhline(y=y_test.mean(), color='k', linestyle='--', lw=1, label=f'Baseline = {y_test.mean():.3f}')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve')
ax.legend()
ax.grid(alpha=0.3)

# Panel 3: Confusion Matrix (normalized)
ax = axes[0, 2]
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues', ax=ax,
            xticklabels=['Benign', 'Injection'],
            yticklabels=['Benign', 'Injection'],
            annot_kws={'size': 13})
ax.set_ylabel('Actual', fontweight='bold')
ax.set_xlabel('Predicted', fontweight='bold')
ax.set_title(f'Confusion Matrix (normalized)\nTN={cm[0,0]:,} | FP={cm[0,1]:,} | FN={cm[1,0]:,} | TP={cm[1,1]:,}')

# Panel 4: Score distribution by class
ax = axes[1, 0]
ax.hist(y_pred_proba[y_test == 0], bins=60, alpha=0.6, color='green', label='Benign (0)', density=True)
ax.hist(y_pred_proba[y_test == 1], bins=60, alpha=0.6, color='red',   label='Injection (1)', density=True)
ax.axvline(x=0.5, color='black', linestyle='--', lw=1.5, label='Threshold=0.5')
ax.set_xlabel('Predicted Probability (Injection)')
ax.set_ylabel('Density')
ax.set_title('Score Distribution by Class')
ax.legend()
ax.grid(alpha=0.3)

# Panel 5: Calibration curve
ax = axes[1, 1]
frac_pos, mean_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
ax.plot(mean_pred, frac_pos, 's-', color='darkorange', lw=2, label='XGBoost')
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect calibration')
ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Fraction of Positives')
ax.set_title('Calibration Curve (Reliability Diagram)')
ax.legend()
ax.grid(alpha=0.3)

# Panel 6: Metrics vs. threshold
ax = axes[1, 2]
thresholds_arr = np.linspace(0.05, 0.95, 100)
t_prec, t_rec, t_f1 = [], [], []
for t in thresholds_arr:
    y_t = (y_pred_proba >= t).astype(int)
    t_prec.append(precision_score(y_test, y_t, zero_division=0))
    t_rec.append(recall_score(y_test, y_t, zero_division=0))
    t_f1.append(f1_score(y_test, y_t, zero_division=0))
ax.plot(thresholds_arr, t_prec, label='Precision', color='blue', lw=1.5)
ax.plot(thresholds_arr, t_rec,  label='Recall',    color='red',  lw=1.5)
ax.plot(thresholds_arr, t_f1,   label='F1',        color='green', lw=2)
ax.axvline(x=0.5, color='black', linestyle='--', lw=1, label='0.5')
best_f1_thresh = thresholds_arr[np.argmax(t_f1)]
ax.axvline(x=best_f1_thresh, color='green', linestyle=':', lw=1.5, label=f'Best F1 thresh={best_f1_thresh:.2f}')
ax.set_xlabel('Classification Threshold')
ax.set_ylabel('Score')
ax.set_title('Precision / Recall / F1 vs. Threshold')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(CONFIG["models_dir"], "evaluation_plots.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"✓ Evaluation plot saved: {plot_path}")
```

- [ ] **Step 8.2: Run Cell 8 and verify**

Expected: AUC-ROC ≥ 0.93. All 6 panels render. Plot saved to `models/evaluation_plots.png`.

---

## Task 9: Cell 9 — Error Analysis

**Files:**
- Modify: `notebooks/training.ipynb` (cell 9)

- [ ] **Step 9.1: Write Cell 9 — error analysis with original texts**

```python
print("Loading original texts for error analysis...")
df_orig = pd.read_csv(CONFIG["original_data_path"])
if 'prompt' in df_orig.columns and 'text' not in df_orig.columns:
    df_orig = df_orig.rename(columns={'prompt': 'text'})
df_orig = df_orig[['text', 'label']].reset_index(drop=True)

# Reproduce the same split to get test-set texts (same seed + same label order)
_, df_orig_test = train_test_split(
    df_orig,
    test_size=CONFIG["test_size"],
    stratify=df_orig['label'],
    random_state=CONFIG["seed"],
)
df_orig_test = df_orig_test.reset_index(drop=True)

# Sanity check: labels must match y_test
assert (df_orig_test['label'].values == y_test.values).all(), \
    "Label mismatch between feature CSV and original dataset — check split parameters"
print(f"✓ Label alignment verified ({len(df_orig_test):,} test samples)")

# Build error dataframe
df_errors = df_orig_test.copy()
df_errors['y_pred_proba'] = y_pred_proba
df_errors['y_pred']       = y_pred
df_errors['correct']      = (df_errors['label'] == df_errors['y_pred'])

fps = df_errors[(df_errors['label'] == 0) & (df_errors['y_pred'] == 1)].sort_values('y_pred_proba', ascending=False)
fns = df_errors[(df_errors['label'] == 1) & (df_errors['y_pred'] == 0)].sort_values('y_pred_proba', ascending=True)
correct_benign     = df_errors[(df_errors['label'] == 0) & (df_errors['y_pred'] == 0)]
correct_injection  = df_errors[(df_errors['label'] == 1) & (df_errors['y_pred'] == 1)]

n_test = len(df_errors)
print(f"\n{'='*60}")
print(f"ERROR ANALYSIS — {n_test:,} test samples")
print(f"{'='*60}")
print(f"  False Positives (benign→injection): {len(fps):,}  ({len(fps)/n_test*100:.2f}%)")
print(f"  False Negatives (injection→benign): {len(fns):,}  ({len(fns)/n_test*100:.2f}%)")
print(f"  Correct (benign):                   {len(correct_benign):,}")
print(f"  Correct (injection):                {len(correct_injection):,}")

# Top 5 False Positives
print(f"\n{'─'*60}")
print("TOP 5 FALSE POSITIVES (benign classified as injection, highest confidence)")
print(f"{'─'*60}")
for i, (_, row) in enumerate(fps.head(5).iterrows(), 1):
    print(f"\n[FP #{i}] Score = {row['y_pred_proba']:.4f}")
    print(f"  {str(row['text'])[:250]}")

# Top 5 False Negatives
print(f"\n{'─'*60}")
print("TOP 5 FALSE NEGATIVES (injection the model missed, lowest score)")
print(f"{'─'*60}")
for i, (_, row) in enumerate(fns.head(5).iterrows(), 1):
    print(f"\n[FN #{i}] Score = {row['y_pred_proba']:.4f}")
    print(f"  {str(row['text'])[:250]}")

# Feature profile comparison
print(f"\n{'─'*60}")
print("FEATURE PROFILE: error groups vs. correctly classified")
print(f"{'─'*60}")
diagnostic_feats = [
    'total_injection_keywords', 'keyword_density', 'entropy',
    'uppercase_ratio', 'type_token_ratio', 'text_length', 'has_override_keyword',
]
print(f"\n{'Feature':<32} {'FP':>8} {'Correct-B':>10} {'FN':>8} {'Correct-I':>10}")
print("─" * 72)
for feat in diagnostic_feats:
    fp_m  = X_test.loc[fps.index, feat].mean()            if len(fps) > 0 else float('nan')
    cb_m  = X_test.loc[correct_benign.index, feat].mean() if len(correct_benign) > 0 else float('nan')
    fn_m  = X_test.loc[fns.index, feat].mean()            if len(fns) > 0 else float('nan')
    ci_m  = X_test.loc[correct_injection.index, feat].mean() if len(correct_injection) > 0 else float('nan')
    print(f"  {feat:<30} {fp_m:>8.4f} {cb_m:>10.4f} {fn_m:>8.4f} {ci_m:>10.4f}")

# Score distributions for error groups
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Error Analysis — Score Distributions", fontsize=13, fontweight='bold')

ax = axes[0]
ax.hist(fps['y_pred_proba'], bins=30, color='salmon', edgecolor='white', label=f'False Positives (n={len(fps)})')
ax.set_title('False Positives: Score Distribution')
ax.set_xlabel('Injection Probability Score')
ax.set_ylabel('Count')
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1]
ax.hist(fns['y_pred_proba'], bins=30, color='lightcoral', edgecolor='white', label=f'False Negatives (n={len(fns)})')
ax.set_title('False Negatives: Score Distribution')
ax.set_xlabel('Injection Probability Score')
ax.set_ylabel('Count')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

- [ ] **Step 9.2: Run Cell 9 and verify**

Expected: label alignment check passes, FP/FN texts print, feature profile table prints. If assertion fails, the original dataset and features CSV have different ordering — add `df_orig = df_orig.iloc[:len(df)]` before the split.

---

## Task 10: Cell 10 — Feature Importance & SHAP

**Files:**
- Modify: `notebooks/training.ipynb` (cell 10)

- [ ] **Step 10.1: Write Cell 10 — importance plots + SHAP**

```python
# XGBoost built-in importance (3 metrics)
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle("XGBoost Feature Importance", fontsize=14, fontweight='bold')

for ax, imp_type in zip(axes, ['gain', 'weight', 'cover']):
    raw_imp = final_model.get_booster().get_score(importance_type=imp_type)
    imp_df = pd.DataFrame({
        'feature':    list(raw_imp.keys()),
        'importance': list(raw_imp.values()),
    }).sort_values('importance', ascending=True).tail(20)
    ax.barh(range(len(imp_df)), imp_df['importance'], color='steelblue', edgecolor='white')
    ax.set_yticks(range(len(imp_df)))
    ax.set_yticklabels(imp_df['feature'], fontsize=9)
    ax.set_title(f'By {imp_type.capitalize()}')
    ax.set_xlabel('Importance')
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
imp_path = os.path.join(CONFIG["models_dir"], "feature_importance.png")
plt.savefig(imp_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"✓ Feature importance plot saved: {imp_path}")

# SHAP values
print("\nComputing SHAP values on 2,000-sample subset...")
shap_sample = X_test.sample(n=min(2_000, len(X_test)), random_state=CONFIG["seed"])
explainer   = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(shap_sample)
shap_exp    = explainer(shap_sample)

# Beeswarm (global importance + direction)
print("Plotting SHAP beeswarm...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, shap_sample, plot_type="dot", max_display=20, show=False)
plt.title("SHAP Beeswarm — Global Feature Impact on Injection Probability", fontsize=12)
plt.tight_layout()
shap_bees_path = os.path.join(CONFIG["models_dir"], "shap_beeswarm.png")
plt.savefig(shap_bees_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"✓ SHAP beeswarm saved: {shap_bees_path}")

# Waterfall plots for 3 representative samples
sample_proba = final_model.predict_proba(shap_sample)[:, 1]
idx_high_inj  = int(np.argmax(sample_proba))
idx_high_ben  = int(np.argmin(sample_proba))
idx_uncertain = int(np.argmin(np.abs(sample_proba - 0.5)))

cases = [
    (idx_high_inj,  f"Highest-confidence INJECTION  (score={sample_proba[idx_high_inj]:.4f})"),
    (idx_high_ben,  f"Highest-confidence BENIGN      (score={sample_proba[idx_high_ben]:.4f})"),
    (idx_uncertain, f"Most UNCERTAIN                 (score={sample_proba[idx_uncertain]:.4f})"),
]
for idx, title in cases:
    print(f"\n--- SHAP Waterfall: {title} ---")
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_exp[idx], max_display=15, show=False)
    plt.title(title, fontsize=11)
    plt.tight_layout()
    plt.show()
```

- [ ] **Step 10.2: Run Cell 10 and verify**

Expected: 3 feature importance panels, beeswarm plot, 3 waterfall plots. `feature_importance.png` and `shap_beeswarm.png` saved to `models/`.

---

## Task 11: Cell 11 — Save All Artifacts

**Files:**
- Modify: `notebooks/training.ipynb` (cell 11)
- Create: `models/xgboost_model.json`
- Create: `models/best_params.json`
- Create: `models/evaluation_metrics.json`
- Create: `models/optuna_study.pkl`

- [ ] **Step 11.1: Write Cell 11 — serialize all outputs**

```python
os.makedirs(CONFIG["models_dir"], exist_ok=True)

# 1. Model
model_path = os.path.join(CONFIG["models_dir"], "xgboost_model.json")
final_model.save_model(model_path)
print(f"✓ Model saved:   {model_path}")

# 2. Best hyperparameters
params_path = os.path.join(CONFIG["models_dir"], "best_params.json")
with open(params_path, 'w') as f:
    json.dump(study.best_params, f, indent=2)
print(f"✓ Params saved:  {params_path}")

# 3. Evaluation metrics
metrics_dict = {
    "auc_roc":              float(auc_roc),
    "auc_pr":               float(auc_pr),
    "accuracy":             float(acc),
    "precision":            float(prec),
    "recall":               float(rec),
    "f1":                   float(f1),
    "n_test_samples":       int(len(y_test)),
    "n_estimators_used":    int(final_model.best_iteration),
    "optuna_best_cv_auc":   float(study.best_value),
    "n_trials_completed":   int(len([t for t in study.trials if t.value is not None])),
    "confusion_matrix": {
        "TN": int(cm[0, 0]), "FP": int(cm[0, 1]),
        "FN": int(cm[1, 0]), "TP": int(cm[1, 1]),
    },
}
metrics_path = os.path.join(CONFIG["models_dir"], "evaluation_metrics.json")
with open(metrics_path, 'w') as f:
    json.dump(metrics_dict, f, indent=2)
print(f"✓ Metrics saved: {metrics_path}")

# 4. Optuna study
study_path = os.path.join(CONFIG["models_dir"], "optuna_study.pkl")
with open(study_path, 'wb') as f:
    pickle.dump(study, f)
print(f"✓ Study saved:   {study_path}")

# Final summary
print(f"\n{'='*56}")
print("TRAINING COMPLETE")
print(f"{'='*56}")
print(f"  AUC-ROC  (primary):  {auc_roc:.6f}")
print(f"  AUC-PR:              {auc_pr:.6f}")
print(f"  F1 Score:            {f1:.6f}")
print(f"  Accuracy:            {acc:.6f}")
print(f"\n  Trees used:          {final_model.best_iteration}")
print(f"  Optuna CV AUC:       {study.best_value:.6f}")
print(f"\n  Artifacts: {CONFIG['models_dir']}/")
for fname in ['xgboost_model.json', 'best_params.json', 'evaluation_metrics.json', 'optuna_study.pkl',
              'evaluation_plots.png', 'feature_importance.png', 'shap_beeswarm.png']:
    fpath = os.path.join(CONFIG["models_dir"], fname)
    size_kb = os.path.getsize(fpath) / 1024 if os.path.exists(fpath) else 0
    status = f"{size_kb:.1f} KB" if size_kb > 0 else "missing!"
    print(f"    {fname:<35} {status}")
```

- [ ] **Step 11.2: Run Cell 11 and verify all files exist**

Expected: all 4 JSON/pkl files created, all 3 PNG files present, summary table shows non-zero sizes.

- [ ] **Step 11.3: Verify model loads correctly**

Run this in a separate cell as a smoke test:

```python
_loaded = xgb.XGBClassifier()
_loaded.load_model(os.path.join(CONFIG["models_dir"], "xgboost_model.json"))
_check_proba = _loaded.predict_proba(X_test[:5])[:, 1]
assert len(_check_proba) == 5
print("✓ Model round-trip load OK")
print("  First 5 probabilities:", _check_proba.round(4))
```

- [ ] **Step 11.4: Commit the completed notebook**

```bash
git add notebooks/training.ipynb models/
git commit -m "feat: add XGBoost GPU training pipeline for prompt injection detection"
```

---

## Self-Review

### Spec Coverage

| Spec Section | Task |
|---|---|
| Cell 1: Imports & Setup (install xgboost/optuna/shap) | Task 1 |
| Cell 2: CONFIG dict with all paths and bounds | Task 2 |
| Cell 3: Load CSV, fill 124 nulls, clip outlier | Task 3 |
| Cell 4: 80/20 stratified split | Task 4 |
| Cell 5: Optuna objective with 5-fold CV | Task 5 |
| Cell 6: 100-trial Optuna search + plots | Task 6 |
| Cell 7: Final model training with early stopping | Task 7 |
| Cell 8: AUC-ROC, AUC-PR, confusion matrix, score dist, calibration, threshold analysis | Task 8 |
| Cell 9: Error analysis — FP/FN texts + feature profile | Task 9 |
| Cell 10: Feature importance (gain/weight/cover) + SHAP beeswarm + 3 waterfall plots | Task 10 |
| Cell 11: Save model, params, metrics, Optuna study | Task 11 |
| GPU: device='cuda', tree_method='hist' | Tasks 1, 2, 5, 7 |
| Artifacts in models/ | Task 11 |

### Placeholder Scan
No TBD, TODO, or vague steps found.

### Type Consistency
- `study` created in Task 6, used in Tasks 7, 11 ✓
- `final_model` created in Task 7, used in Tasks 8, 9, 10, 11 ✓
- `y_pred_proba`, `y_pred`, `cm` created in Task 8, used in Tasks 9, 11 ✓
- `X_test`, `y_test` created in Task 4, used in Tasks 8, 9, 10 ✓
- `FEATURE_COLS` created in Task 4 ✓
- `CONFIG` created in Task 2, used throughout ✓
