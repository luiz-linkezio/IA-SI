# Design: Análise Exploratória — Prompt Injection Detection Dataset

**Data:** 2026-05-16
**Autor:** Luiz Henrique
**Status:** Aprovado

---

## Contexto

Notebook Jupyter de Análise Exploratória de Dados (EDA) completo, em português, para entrega acadêmica em cadeira de graduação. O dataset alvo é o `hlyn-labs/prompt-injection-judge-deberta-dataset` (HuggingFace), com ~400K amostras binárias de detecção de prompt injection / jailbreak.

---

## Objetivo

Produzir um notebook acadêmico, narrativo e completo que explore estatisticamente e linguisticamente o dataset, incluindo análises NLP profundas (n-gramas, nuvem de palavras, padrões de ataque), com interpretações em português e referências formatadas.

---

## Estrutura do Notebook (`notebooks/exploratory_analysis.ipynb`)

### 1. Capa e Introdução
- Título, autor, data
- Descrição do dataset (origem, propósito, schema)
- Objetivos da análise

### 2. Configuração do Ambiente
- Instalação de dependências via `!pip install` no venv do projeto
- Imports organizados por categoria (data, viz, NLP)
- Configurações globais (estilo matplotlib, seed, paths)

### 3. Coleta dos Dados
- Download via `datasets` (HuggingFace)
- Salvamento em `/home/linkezio/Datasets/prompt_injection_dataset.csv`
- Leitura do CSV em execuções subsequentes (evita re-download)
- Verificação de integridade (shape, contagem de linhas)

### 4. Visão Geral do Dataset
- `.shape`, `.dtypes`, `.head()`, `.tail()`
- `.info()` e `.describe()`
- Exemplos de textos benignos e maliciosos

### 5. Qualidade dos Dados
- Contagem de valores nulos por coluna
- Detecção de duplicatas exatas
- Textos vazios ou só-whitespace
- Comprimento mínimo suspeito (< 5 caracteres)
- Distribuição de labels (sanidade)

### 6. Distribuição de Classes
- Contagem absoluta e relativa (benigno vs. malicioso)
- Gráfico de barras + gráfico de pizza
- Discussão sobre balanceamento natural (~1:1)

### 7. Análise de Comprimento de Texto
- Comprimento em caracteres por classe (histograma + boxplot)
- Comprimento em palavras por classe (histograma + boxplot)
- Percentis (P25, P50, P75, P95, P99)
- Teste estatístico de diferença entre classes (Mann-Whitney U)
- Interpretação: prompts maliciosos tendem a ser mais longos/curtos?

### 8. Análise de Vocabulário
- Vocabulário total, por classe benigna e maliciosa
- Hapax legomena (palavras que aparecem apenas 1 vez)
- Densidade lexical (type-token ratio)
- Palavras exclusivas de cada classe (top-N)

### 9. Análise de N-gramas
- Unigramas mais frequentes (top-20) por classe — com e sem stopwords
- Bigramas mais frequentes (top-15) por classe
- Trigramas mais frequentes (top-10) por classe
- Visualizações em barras horizontais lado a lado

### 10. Nuvem de Palavras
- WordCloud para classe benigna
- WordCloud para classe maliciosa
- Stopwords em inglês (NLTK) removidas

### 11. Análise de Padrões de Ataque
- Busca por frases-gatilho conhecidas em prompts maliciosos:
  - "ignore previous instructions", "ignore all previous", "jailbreak", "DAN", "act as", "pretend you are", "disregard", "override", "sudo", "bypass"
- Frequência absoluta e relativa de cada padrão
- Gráfico de barras dos padrões mais comuns
- Exemplos de textos que contêm cada padrão

### 12. Análise de Clusters por Comprimento
- Já que o dataset não possui coluna de fonte, agrupar amostras por faixas de comprimento (curto/médio/longo/muito longo)
- Distribuição de labels por faixa
- Discussão sobre como diferentes fontes (ex: Gandalf CTF = curtos; WildJailbreak = longos) se manifestam

### 13. Correlações e Insights Finais
- Resumo dos principais achados
- Tabela comparativa benigno vs. malicioso nas métricas-chave
- Discussão sobre implicações para treinamento de classificadores
- Limitações da análise

### 14. Referências e Citações
- Citação do dataset agregado (BibTeX + formato ABNT)
- Citações dos 12 datasets de origem
- Citação do modelo treinado (DeBERTa-v3-xsmall)

---

## Decisões Técnicas

| Decisão | Escolha |
|---|---|
| Linguagem do notebook | Português |
| Abordagem narrativa | Linear (Opção A) — documento acadêmico contínuo |
| Download do dataset | `datasets` (HuggingFace) |
| Armazenamento | `/home/linkezio/Datasets/prompt_injection_dataset.csv` |
| Kernel | venv do projeto (`/home/linkezio/Projects/IA-SI/venv`) |
| Visualizações | matplotlib + seaborn (estáticas) + plotly (interativas) |
| NLP | nltk (tokenização, stopwords, n-gramas) |
| Nuvem de palavras | wordcloud |
| Estatística | scipy (Mann-Whitney U) |

## Dependências a Instalar no Venv

```
datasets
pandas
numpy
matplotlib
seaborn
plotly
wordcloud
nltk
scikit-learn
scipy
ipykernel
nbformat
```

---

## Arquivos Afetados

- `notebooks/exploratory_analysis.ipynb` — notebook principal (sobrescrever)
- `/home/linkezio/Datasets/prompt_injection_dataset.csv` — dataset salvo localmente

---

## Critérios de Sucesso

- Notebook executável do início ao fim sem erros
- Todas as seções com célula markdown explicativa em português + código + interpretação dos resultados
- Referências completas e formatadas
- Visualizações legíveis e com títulos/labels em português
