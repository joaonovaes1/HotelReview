# 2VA — Guia de Apresentação
## Estudo Comparativo: Ensemble vs Modelo Único

---

# PARTE 1 — O Problema e as Técnicas

> **Objetivo desta parte:** apresentar o caso de uso, as 3 técnicas escolhidas e justificar por que faz sentido usá-las juntas.

---

### O que o sistema faz

Classifica reviews de hotéis automaticamente em 4 dimensões simultâneas:

| Tarefa | Saída |
|--------|-------|
| Sentimento | negativo / neutro / positivo |
| Categoria | limpeza, atendimento, localização... (8 categorias) |
| Rating previsto | escala 1–5 |
| Prioridade | alta / normal |

O objetivo é dar ao gestor do hotel uma visão rápida do que os hóspedes estão dizendo — sem ler cada review manualmente.

---

### As 3 técnicas e por que foram escolhidas

**Técnica A — Transformers + Mecanismo de Atenção**
- Define *a arquitetura* do modelo
- O XLM-RoBERTa é um Transformer: processa a frase inteira de uma vez (não palavra por palavra como RNNs), usando atenção para pesar quais palavras importam mais para cada decisão
- Exemplo: na frase *"o quarto era sujo mas o atendimento foi excelente"*, a atenção aprende que "sujo" é relevante para limpeza e "excelente" para atendimento

> `src/models/classifier.py` — backbone XLM-RoBERTa + 4 cabeças de classificação

**Técnica B — Transfer Learning**
- Define *a estratégia de treinamento*
- Em vez de treinar do zero (precisaria de milhões de exemplos), partimos de um modelo já pré-treinado em 100+ idiomas e *fine-tunamos* para a tarefa de reviews de hotel
- É como contratar alguém que já sabe ler e escrever, e ensinar só o vocabulário específico do negócio

> `src/models/trainer.py` — AdamW, scheduler com warmup, MultiTaskLoss

**Por que A e B não são a mesma coisa:**
- A = o motor (arquitetura)
- B = a estratégia de uso (partir de pesos pré-treinados)
- Poderia existir Transfer Learning com uma RNN (ELMo), ou um Transformer treinado do zero. Neste projeto os dois andam juntos porque se complementam — e isso é declarado explicitamente no relatório.

**Técnica C — Ensemble Learning (o que está sendo testado)**
- Treina N modelos independentes (mesma arquitetura, seeds diferentes) e agrega as predições pela média das probabilidades
- Hipótese: 3 modelos errando em direções diferentes se corrigem mutuamente

> `2VA/src/ensemble.py` — `HotelReviewEnsemble`: carrega N checkpoints, `_aggregate` faz a média

---

# PARTE 2 — O Experimento

> **Objetivo desta parte:** explicar como as 30 simulações funcionam, por que esse design é válido estatisticamente, e o que cada arquivo faz.

---

### Os 2 cenários

```
Cenário H1 (Alternativa):  A + B + C      → ensemble de 3 modelos
Cenário H0 (Nula):         A + B + C_alt  → modelo único (baseline)
```

A e B são fixos nos dois cenários. O único elemento que muda é C — ensemble ou modelo único. Isso isola o efeito do ensemble.

---

### Por que 30 simulações

Um único treino não é confiável — o resultado varia com a seed (inicialização aleatória, ordem dos batches, dropout). Com 30 seeds distintas, obtemos uma **distribuição** de resultados para cada cenário:

```
seed 0  → par (F1_single_0,  F1_ensemble_0)
seed 1  → par (F1_single_1,  F1_ensemble_1)
...
seed 29 → par (F1_single_29, F1_ensemble_29)
```

**Pareamento:** a mesma seed gera o mesmo split treino/validação nos dois cenários. Ensemble e modelo único são avaliados nos *mesmos exemplos* — comparação justa.

> `2VA/run_single.py` — 30 seeds, 1 modelo por seed, salva métricas em CSV
> `2VA/run_ensemble.py` — 30 seeds, 3 modelos por seed, agrega, salva métricas em CSV

---

### Como o ensemble agrega (o detalhe técnico)

Para cada batch na avaliação:
1. Passa o input pelos 3 modelos → 3 dicts de logits
2. Aplica softmax/sigmoid em cada → 3 dicts de probabilidades
3. Faz a média das probabilidades → 1 dict agregado
4. Decide: argmax para sentimento/prioridade, threshold 0.5 para categorias, média direta para rating

```
modelo 1 → softmax → [0.10, 0.20, 0.70]
modelo 2 → softmax → [0.20, 0.10, 0.70]
modelo 3 → softmax → [0.15, 0.30, 0.55]
                      ─────────────────
média               → [0.15, 0.20, 0.65] → positivo
```

> `2VA/src/ensemble.py` — método `_aggregate`

---

### Teste estatístico — Wilcoxon Signed-Rank

**Por que não o t-test?**
O t-test assume distribuição normal. Métricas de deep learning com 30 amostras não garantem isso. O Wilcoxon é não-paramétrico: só assume que as diferenças têm uma direção consistente.

**O que ele testa:**
- `alternative='greater'` para F1 → H1: ensemble_F1 > single_F1
- `alternative='less'` para MAE → H1: ensemble_MAE < single_MAE
- p < 0.05 → rejeita H0

> `2VA/notebooks/03_statistical_analysis.ipynb` — Wilcoxon + boxplots + histogramas de diferença

---

# PARTE 3 — Resultados e Conclusão

> **Objetivo desta parte:** apresentar os números, interpretar o veredicto e discutir o que ele significa para o projeto.

---

### C_alt — Modelo Único (30 seeds)

| Métrica | Média | Desvio padrão | Mín | Máx |
|---------|-------|---------------|-----|-----|
| F1 macro | **0.6353** | 0.0491 | 0.5033 | 0.7091 |
| MAE | **0.1568** | 0.0161 | 0.1221 | 0.1855 |

**O que o desvio de 0.049 no F1 significa:** uma mesma arquitetura, treinada com configurações idênticas, produz resultados que variam em ~5 pontos percentuais dependendo da seed. Esse é o espaço onde o ensemble atua.

---

### C — Ensemble (30 seeds)

| Métrica | Média | Desvio padrão | Mín | Máx |
|---------|-------|---------------|-----|-----|
| F1 macro | **0.6932** | 0.0563 | 0.5865 | 0.7958 |
| MAE | **0.1425** | 0.0118 | 0.1267 | 0.1670 |

**O ganho do ensemble:** +0.058 no F1 (↑9.1%) e −0.014 no MAE (↓9.1%).

---

### Wilcoxon — Veredicto

| Métrica | W | p-value | Decisão |
|---------|---|---------|---------|
| F1 macro | 424.0 | **0.000009** | REJEITA H0 |
| MAE | 53.5 | **0.000116** | REJEITA H0 |

**Veredicto final: H1 ACEITA — o ensemble é estatisticamente superior em ambas as métricas (α = 0.05).**

---

### Referência rápida dos arquivos

| Arquivo | O que é |
|---------|---------|
| `src/models/classifier.py` | Técnica A — arquitetura Transformer |
| `src/models/trainer.py` | Técnica B — fine-tuning, losses, métricas |
| `2VA/src/ensemble.py` | Técnica C — HotelReviewEnsemble |
| `2VA/run_single.py` | Simulação C_alt (30 seeds) |
| `2VA/run_ensemble.py` | Simulação C (30 seeds × 3 modelos) |
| `2VA/notebooks/03_statistical_analysis.ipynb` | Wilcoxon + gráficos |
| `2VA/results/single_model_results.csv` | Dados do modelo único |
| `2VA/results/ensemble_results.csv` | Dados do ensemble |
| `2VA/results/comparativo.png` | Gráfico para o relatório |
