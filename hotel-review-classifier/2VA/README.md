# 2VA — Estudo Comparativo: Ensemble vs Modelo Único

## Técnicas selecionadas

| Slot  | Técnica                                  | Papel nos cenários |
|-------|------------------------------------------|--------------------|
| A     | Transformers + Mecanismo de Atenção      | Base fixa (arquitetura) |
| B     | Transfer Learning                        | Base fixa (estratégia de treino) |
| C     | Ensemble Learning                        | Cenário 1 — H1 |
| C_alt | Modelo único (baseline)                  | Cenário 2 — H0 |

## Hipóteses

- **H0:** Modelo único (A + B) tem performance equivalente ao ensemble
- **H1:** Ensemble (A + B + C) tem performance superior

## Cenários

| Cenário | Composição | Descrição |
|---------|-----------|-----------|
| Cenário 1 | A + B + **C** | XLM-RoBERTa fine-tuned + Ensemble (voting/averaging) |
| Cenário 2 | A + B + **C_alt** | XLM-RoBERTa fine-tuned + Modelo único |

## Estrutura

```
2VA/
├── notebooks/
│   ├── 01_single_model_simulation.ipynb  # 30 seeds — C_alt (baseline)
│   ├── 02_ensemble_simulation.ipynb      # 30 seeds — C (ensemble)
│   └── 03_statistical_analysis.ipynb     # Wilcoxon, t-test, gráficos
├── src/
│   └── ensemble.py                       # Wrapper de N modelos
├── results/                              # CSVs com métricas por seed
└── report/                               # Relatório IEEE
```

## Métricas

- F1 macro (sentimento)
- MAE (rating)

## Teste estatístico

- Wilcoxon signed-rank (dados pareados)
- Mesmas 30 seeds nos dois cenários para pareamento
