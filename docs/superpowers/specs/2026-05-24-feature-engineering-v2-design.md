# Feature Engineering V2 — Prompt Injection Detection

**Date:** 2026-05-24
**Status:** Approved
**Primary Goal:** Reduce false negatives (missed injections) while maintaining interpretability

---

## Context

The V1 model (XGBoost, 35 hand-crafted features, AUC-ROC 0.918) has two structural problems:

1. **Keyword over-reliance:** 14 of 35 features (40%) are regex-based keyword matches. These produce many false positives (benign texts with words like "forget", "become", "system") and fail to detect injections that paraphrase or use creative language the regexes don't cover.
2. **Poor generalization:** The model can't detect semantic similarity — "disregard prior instructions" and "ignore previous commands" mean the same thing but look completely different to regex. This explains the 22.8% false negative rate.

Dataset analysis reveals ~98% of prompts are in English, with <2% spread across German, Dutch, Danish, and others. Regex multilíngue approaches would cover a tiny fraction of data; semantic embeddings handle this naturally.

---

## Architecture: V1 and V2 as Separate Applications

V1 and V2 coexist independently. No V1 code is modified or removed.

```
src/                          # V1 — untouched
├── features.py               # V1 feature extraction (35 features)
├── model.py                  # V1 model loading and prediction
├── __init__.py
app.py                        # V1 FastAPI app
ui.py                         # V1 Gradio app

src_v2/                       # V2 — new
├── features.py               # V2 feature extraction (19 hand-crafted)
├── embeddings.py             # V2 embedding generation (384-dim)
├── model.py                  # V2 model loading and prediction
├── __init__.py
app_v2.py                     # V2 FastAPI app
ui_v2.py                      # V2 Gradio app

models/
├── xgboost_model.json        # V1 model
└── xgboost_model_v2.json     # V2 model (after training)

notebooks/
├── feature_engineering.ipynb  # V1 notebook
├── feature_engineering_v2.ipynb # V2 notebook (new)
├── training.ipynb             # V1 training
└── training_v2.ipynb          # V2 training (new)

data/
├── features_engineered.csv           # V1 features
└── features_engineered_v2.csv        # V2 features (new)
```

V2 has its own modules, apps, models, notebooks, and data files. V1 remains fully operational.

---

## Feature Set V2 (403 features total)

### Group A — Hand-crafted Refined Features (19 features)

| # | Feature | Source | Change from V1 | Rationale |
|---|---------|--------|---------------|-----------|
| 1 | `text_length` | Morphological | Kept | SHAP: contributes to injection detection |
| 2 | `word_count` | Morphological | Kept | Correlated with text structure |
| 3 | `unique_words` | Morphological | Kept | Low unique_words → injection (repetitive prompts) |
| 4 | `sentence_count` | Morphological | Kept | SHAP: low sentence_count → injection (direct commands) |
| 5 | `avg_word_length` | Morphological | Kept | Moderate SHAP importance |
| 6 | `char_count_no_space` | Character | Kept | SHAP top-3 by gain |
| 7 | `uppercase_ratio` | Character | Kept | SHAP: strongest positive signal for injection |
| 8 | `lowercase_ratio` | Character | Kept | SHAP: negative correlation with injection |
| 9 | `special_char_ratio` | Character | Kept | Moderate correlation (r=0.058) |
| 10 | `punctuation_ratio` | Character | Kept | Non-negligible (r=0.071) |
| 11 | `space_ratio` | Character | Kept | Top-10 correlation (r=0.29) |
| 12 | `newline_count` | Character | Kept | SHAP: moderate cover importance |
| 13 | `type_token_ratio` | Linguistic | Kept | Linguistic diversity indicator |
| 14 | `lexical_diversity` | Linguistic | Kept | Already clipped for NaN/inf |
| 15 | `avg_word_frequency` | Linguistic | Kept | Word repetition indicator |
| 16 | `word_length_variance` | Linguistic | Kept | Top by weight importance |
| 17 | `entropy` | Linguistic | Kept | Top-10 correlation (r=0.33) |
| 18 | `colon_count` | Structural | Kept | Common in instruction-style prompts |
| 19 | `keyword_diversity` | Attack pattern | **New** | Replaces all V1 keyword features |

**Removed from V1 (16 features):**
- `has_ignore_keyword`, `count_ignore_keyword` — redundant with `keyword_diversity`
- `has_act_as_keyword`, `count_act_as_keyword` — redundant
- `has_system_keyword`, `count_system_keyword` — redundant
- `has_override_keyword`, `count_override_keyword` — redundant
- `has_execute_keyword`, `count_execute_keyword` — redundant
- `total_injection_keywords` — redundant with `keyword_diversity`
- `keyword_density` — redundant with `keyword_diversity`
- `digit_ratio` — near-zero correlation (r=0.034)
- `bracket_count` — low importance (r=0.054)
- `parenthesis_count` — not in top-20
- `quote_count` — not in top-20
- `comma_to_period_ratio` — low correlation (r=0.028), already clipped

### Group B — Refined Regex Patterns (internal logic for `keyword_diversity`)

`keyword_diversity` counts how many of the 7 categories below have at least one match (range 0-7). This replaces the per-category `has_*` and `count_*` features with a single feature that captures structural diversity of attack patterns.

**7 categories with refined patterns (English only):**

| Category | Patterns | Change from V1 |
|----------|----------|---------------|
| `ignore` | `ignore\s+(previous|past|prior|all|everything|above)`, `forget\s+(all|previous|past|prior|everything|about)`, `disregard\s+(all|any|previous|prior|above|instructions?|rules?)` | `forget` and `disregard` now require context; removed bare `forget` |
| `act_as` | `act\s+as`, `pretend\s+(to\s+)?be`, `you\s+are\s+now\s+(a|an|the|my)` | Removed `become` (too common in benign text); `you are now` requires role |
| `system` | `system\s*:\s*[{("\[]`, `admin\s*:\s*[{("\[]`, `root\s*:\s*[{("\[]` | Requires structural indicator (`{`, `(`, `"`, `[`) after colon to reduce false positives |
| `override` | `bypass`, `override`, `ignore\s+(restrictions|rules)` | Unchanged (already specific) |
| `execute` | `(execute|run|perform|do)\s+(this|the\s+following)`, `instead\s*,?\s*(do|execute|output|respond|answer|follow)`, `from\s+now\s+on` | `instead` now requires imperative verb; removed bare `instead` |
| `instruction_verb` | `(tell|make|force|order|command|demand|instruct)\s+(me|you|them|us|the\s+AI|the\s+model|the\s+assistant)\s+to` | **New category** — captures imperative instructions directed at AI |
| `role_switch` | `(you\s+are\s+no\s+longer|stop\s+being|cease\s+to\s+be)` | **New category** — captures explicit role abandonment |

### Group C — Semantic Embeddings (384 features)

| Property | Value |
|----------|-------|
| Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Dimensions | 384 |
| Multilingual | Yes (50+ languages via MEAN pooling) |
| Inference speed (CPU) | ~5-8ms per text |
| Model size | ~80MB |
| Feature names | `embedding_0` through `embedding_383` |

Embeddings are pre-computed for the entire training dataset and stored as columns in `features_engineered_v2.csv`. At inference time, the embedding is generated on-the-fly and concatenated with hand-crafted features.

No language detection feature is added — embeddings naturally capture cross-lingual semantic similarity, and `is_non_english` would be too sparse (<2% of data) and add noise.

---

## Feature Engineering V2 Module (`src_v2/features.py`)

```python
ATTACK_PATTERNS_V2 = {
    "ignore": [
        r"\bignore\s+(previous|past|prior|all|everything|above)\b",
        r"\bforget\s+(all|previous|past|prior|everything|about)\b",
        r"\bdisregard\s+(all|any|previous|prior|above|instructions?|rules?)\b",
    ],
    "act_as": [
        r"\bact\s+as\b",
        r"\bpretend\s+(to\s+)?be\b",
        r"\byou\s+are\s+now\s+(a|an|the|my)\b",
    ],
    "system": [
        r"\bsystem\s*:\s*[{(\"'\[]",
        r"\badmin\s*:\s*[{(\"'\[]",
        r"\broot\s*:\s*[{(\"'\[]",
    ],
    "override": [
        r"\bbypass\b",
        r"\boverride\b",
        r"\bignore\s+(restrictions|rules)\b",
    ],
    "execute": [
        r"\b(execute|run|perform|do)\s+(this|the\s+following)\b",
        r"\binstead\s*,?\s*(do|execute|output|respond|answer|follow)\b",
        r"\bfrom\s+now\s+on\b",
    ],
    "instruction_verb": [
        r"\b(tell|make|force|order|command|demand|instruct)\s+(me|you|them|us|the\s+AI|the\s+model|the\s+assistant)\s+to\b",
    ],
    "role_switch": [
        r"\b(you\s+are\s+no\s+longer|stop\s+being|cease\s+to\s+be)\b",
    ],
}

FEATURE_NAMES_V2 = [
    # Group A — Hand-crafted (19)
    "text_length", "word_count", "unique_words", "sentence_count",
    "avg_word_length", "char_count_no_space",
    "uppercase_ratio", "lowercase_ratio", "special_char_ratio",
    "punctuation_ratio", "space_ratio", "newline_count",
    "type_token_ratio", "lexical_diversity", "avg_word_frequency",
    "word_length_variance", "entropy", "colon_count",
    "keyword_diversity",
    # Group C — Embeddings (384)
    # + f"embedding_{i}" for i in range(384)
]
```

The `PromptInjectionFeatureEngineerV2` class mirrors the V1 interface (`extract_all()`, `extract_as_dataframe()`) but produces 19 features instead of 35. Embeddings are handled separately via `src_v2/embeddings.py`.

---

## Embedding Module (`src_v2/embeddings.py`)

```python
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

class EmbeddingGenerator:
    def __init__(self, model_name=MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def encode(self, text: str) -> list[float]:
        return self.model.encode(text, show_progress_bar=False).tolist()

    def encode_batch(self, texts: list[str], batch_size=64, show_progress_bar=True) -> list[list[float]]:
        return self.model.encode(texts, batch_size=batch_size,
                                 show_progress_bar=show_progress_bar).tolist()
```

The model is downloaded on first use (~80MB). For batch generation over the full dataset, GPU acceleration is used if available; inference in production is CPU-only at ~5-8ms per text.

---

## Model Module (`src_v2/model.py`)

Same interface as V1, but loads `models/xgboost_model_v2.json` and expects 403 features (19 hand-crafted + 384 embeddings).

```python
import xgboost as xgb
from src_v2.features import PromptInjectionFeatureEngineerV2
from src_v2.embeddings import EmbeddingGenerator

def load_model(path="models/xgboost_model_v2.json") -> xgb.Booster:
    booster = xgb.Booster()
    booster.load_model(path)
    return booster

def predict(text, booster,
            feature_engineer=None,
            embedding_generator=None) -> dict:
    if feature_engineer is None:
        feature_engineer = PromptInjectionFeatureEngineerV2()
    if embedding_generator is None:
        embedding_generator = EmbeddingGenerator()

    handcrafted = feature_engineer.extract_all(text)
    embedding = embedding_generator.encode(text)

    import pandas as pd
    features = {**handcrafted}
    for i, val in enumerate(embedding):
        features[f"embedding_{i}"] = val

    df = pd.DataFrame([features])
    dm = xgb.DMatrix(df)
    best_iter = getattr(booster, "best_iteration", 0)
    proba = float(booster.predict(dm, iteration_range=(0, best_iter + 1))[0])
    label = int(proba > 0.5)
    return {"label": label, "probability": round(proba, 4), "is_injection": label == 1}
```

---

## Training Pipeline V2

### Notebook: `notebooks/feature_engineering_v2.ipynb`

1. Load `prompt_injection_dataset.csv`
2. Extract V2 hand-crafted features (19 features) using `PromptInjectionFeatureEngineerV2`
3. Generate embeddings using `EmbeddingGenerator.encode_batch()` over all 534k texts
4. Concatenate into single DataFrame: 19 + 384 = 403 features + `label`
5. Preprocessing: fill `lexical_diversity` nulls with median (same as V1)
6. Save to `data/features_engineered_v2.csv`

### Notebook: `notebooks/training_v2.ipynb`

Same Optuna pipeline as V1 (100 trials, 5-fold CV, AUC-ROC optimization) but:
- Input: `data/features_engineered_v2.csv` (403 features)
- Output model: `models/xgboost_model_v2.json`
- Same seed (42), same train/test split for comparability
- Evaluation on same holdout test set
- Target: beat V1 AUC-ROC of 0.918 and reduce false negatives below 22.8%

---

## Expected Improvements

| Metric | V1 (current) | V2 (expected) | Reason |
|--------|--------------|----------------|--------|
| AUC-ROC | 0.918 | 0.94+ | Embeddings capture semantic patterns regex can't |
| False negatives | 22.8% | <15% | Primary goal — embeddings detect paraphrased injections |
| False positives | 12.1% | ~10-12% | Refined regex reduces FPs; embeddings may tighten decision boundary |
| Feature count | 35 | 403 | More features but less redundant; XGBoost handles this well |
| Inference time (CPU) | ~0.2ms | ~5-10ms | Dominated by embedding generation; still acceptable |

---

## Dependencies

```
sentence-transformers>=2.2
torch>=2.0  # Required by sentence-transformers, CPU-only is fine
```

Added to `requirements.txt` alongside existing dependencies.

---

## Risks

| Risk | Mitigation |
|------|-----------|
| Embedding model too slow on CPU | Benchmark before training; fall back to smaller model (`all-MiniLM-L4-v2`, 256-dim) if needed |
| V2 doesn't beat V1 on AUC-ROC | Keep V1 deployed; V2 is separate application |
| Embedding download size (~80MB) | Cache locally after first use; shipped with deployment |
| XGBoost overfits on 403 features | Optuna tunes `colsample_bytree`, `max_depth`, `min_child_weight`; early stopping prevents overfitting |