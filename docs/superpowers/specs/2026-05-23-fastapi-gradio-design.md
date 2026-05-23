# FastAPI + Gradio para Detecção de Prompt Injection

## Contexto

Projeto de detecção binária de prompt injection. O modelo XGBoost já está treinado (AUC-ROC: 0.918, F1: 0.814) e salvo em `models/xgboost_model.json`. As funções de feature extraction (35 features em 7 layers) estão nos notebooks. Faltam as etapas 3 (endpoint) e 4 (frontend) do projeto.

## Estrutura de Arquivos

```
src/
  __init__.py
  features.py       # Feature extraction portado do notebook
  model.py           # Carregamento xgb.Booster + predição
app.py              # FastAPI com /predict e /health
ui.py               # Gradio Interface
requirements.txt    # Dependências atualizadas
```

## Componentes

### src/features.py

Porta as 5 funções de extração do notebook de feature engineering:

- `extract_morfologicas(text)` — text_length, word_count, unique_words, sentence_count, avg_word_length, char_count_no_space
- `extract_caracteres(text)` — uppercase_ratio, lowercase_ratio, digit_ratio, special_char_ratio, punctuation_ratio, space_ratio, newline_count
- `extract_linguisticas(text)` — type_token_ratio, lexical_diversity, avg_word_frequency, word_length_variance, entropy
- `extract_padroes_ataque(text)` — has_*_keyword, count_*_keyword (5 categorias), total_injection_keywords, keyword_density
- `extract_estruturais(text)` — colon_count, bracket_count, parenthesis_count, quote_count, comma_to_period_ratio

Classe `PromptInjectionFeatureEngineer` com:
- `extract_all(text)` → dict com 35 features na ordem esperada pelo modelo
- Pós-processamento: fill `lexical_diversity` nulls com mediana 1.041393, clip `comma_to_period_ratio` em 3.5

Ordem das features (igual ao treinamento):
text_length, word_count, unique_words, sentence_count, avg_word_length, char_count_no_space, uppercase_ratio, lowercase_ratio, digit_ratio, special_char_ratio, punctuation_ratio, space_ratio, newline_count, type_token_ratio, lexical_diversity, avg_word_frequency, word_length_variance, entropy, has_ignore_keyword, count_ignore_keyword, has_act_as_keyword, count_act_as_keyword, has_system_keyword, count_system_keyword, has_override_keyword, count_override_keyword, has_execute_keyword, count_execute_keyword, total_injection_keywords, keyword_density, colon_count, bracket_count, parenthesis_count, quote_count, comma_to_period_ratio

### src/model.py

- `load_model(path: str) -> xgb.Booster` — carrega `xgboost_model.json`
- `predict(text: str, booster: xgb.Booster, feature_engineer: PromptInjectionFeatureEngineer) -> dict` — extrai features, cria DMatrix, retorna `{label: int, probability: float, is_injection: bool}`
- Usa `iteration_range=(0, best_iteration+1)` para respeitar early stopping (best_iteration=2397)
- Threshold padrão: 0.5 (probabilidade > 0.5 → injection)

### app.py (FastAPI)

Endpoints:
- `POST /predict` — Request: `{"text": "string"}` → Response: `{"label": 0, "probability": 0.12, "is_injection": false}`
- `GET /health` — Response: `{"status": "ok", "model_loaded": true}`

Modelo e feature engineer carregados no startup. Validação com Pydantic (`PredictRequest`, `PredictResponse`). Roda na porta 8000.

### ui.py (Gradio)

Interface com:
- Input: `gr.Textbox` multiline para o prompt
- Output: `gr.Label` (classificação: Benigno/Injection) + `gr.Number` (probabilidade)
- Imports diretos de `src/features.py` e `src/model.py`, sem depender de FastAPI
- `launch(server_port=7860)`

### requirements.txt

Acrescentar ao existente:
- fastapi
- uvicorn
- gradio
- xgboost>=2.0
- pydantic

## Decisões de Design

1. **Gradio via imports diretos** — não passa por HTTP, mais simples e rápido para demo
2. **xgb.Booster nativo** — formato JSON leve, sem dependência de scikit-learn
3. **Feature extraction reutilizável** — mesmo módulo usado por FastAPI e Gradio
4. **Servidores separados** — FastAPI (porta 8000) e Gradio (porta 7860) rodam independentemente

## Fluxo de Predição

```
Texto → PromptInjectionFeatureEngineer.extract_all() → dict[35 features]
     → pandas DataFrame (1 linha) → xgb.DMatrix → booster.predict()
     → {label, probability, is_injection}
```