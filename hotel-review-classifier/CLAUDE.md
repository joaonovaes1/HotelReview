# Hotel Review Classifier

Sistema de classificação de reviews de hotéis usando Deep Learning (NLP) com XLM-RoBERTa e dashboard Streamlit.

## Objetivo
Classificar reviews de hotéis em múltiplas dimensões (sentimento, categoria, rating, prioridade) e apresentar os resultados em um dashboard visual para gestores.

## Dataset
- Fonte: [harrachimustapha/hotel-reviews-dataset](https://www.kaggle.com/datasets/harrachimustapha/hotel-reviews-dataset) no Kaggle
- Colunas principais: `Hotel_Name`, `Reviewer_Nationality`, `Positive_Review`, `Negative_Review`, `Reviewer_Score`, `Tags`, `Review_Date`
- Idiomas suportados: PT-BR e EN

## Modelo
- Backbone: `xlm-roberta-base` (HuggingFace)
- Suporta PT-BR e EN nativamente
- Arquitetura multi-task: 4 cabeças de classificação

## Tarefas de Classificação
| Tarefa | Tipo | Classes/Saída |
|---|---|---|
| Sentimento | Classificação (3 classes) | negativo / neutro / positivo |
| Categoria | Multi-label (8 categorias) | limpeza, atendimento, localização, alimentação, preço, conforto, wifi, instalações |
| Rating Previsto | Regressão | 0–10 (normalizado 0–1 no treino) |
| Prioridade | Binário | alta / normal |

## Label Engineering
- **Sentimento**: `Reviewer_Score` ≤ 5 → negativo, 5–7 → neutro, > 7 → positivo
- **Categoria**: zero-shot com `facebook/bart-large-mnli` ou keyword matching
- **Prioridade**: regra — sentimento=negativo AND score < 6

## Estrutura
```
hotel-review-classifier/
├── data/raw/           # CSV original do Kaggle
├── data/processed/     # Dados limpos com labels
├── notebooks/          # EDA, preprocessamento, treinamento
├── src/data/           # loader.py, preprocessor.py
├── src/models/         # classifier.py, trainer.py
├── src/utils/          # metrics.py
├── app/                # Streamlit (main.py + pages/)
└── models/saved/       # Checkpoints treinados
```

## Como Rodar

### Setup
```bash
pip install -r requirements.txt
```

### Download do Dataset (Kaggle)
```bash
# Configure ~/.kaggle/kaggle.json com suas credenciais
kaggle datasets download -d harrachimustapha/hotel-reviews-dataset -p data/raw --unzip
```

### EDA
```bash
jupyter notebook notebooks/01_eda.ipynb
```

### Preprocessamento
```bash
jupyter notebook notebooks/02_preprocessing.ipynb
```

### Treinamento
```bash
jupyter notebook notebooks/03_training.ipynb
```

### Dashboard
```bash
streamlit run app/main.py
```

### Teste rápido do preprocessador
```bash
python -c "from src.data.preprocessor import clean_text; print(clean_text('Ótimo hotel!! <br> Muito limpo.'))"
```

## Métricas Alvo
- Sentimento: F1 macro ≥ 0.80
- Regressão de rating: MAE ≤ 0.8

## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- For cross-module "how does X relate to Y" questions, prefer `graphify query "<question>"`, `graphify path "<A>" "<B>"`, or `graphify explain "<concept>"` over grep — these traverse the graph's EXTRACTED + INFERRED edges instead of scanning files
- After modifying code files in this session, run `graphify update .` to keep the graph current (AST-only, no API cost)
