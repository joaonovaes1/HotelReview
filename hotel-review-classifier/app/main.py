import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from src.models.inference import ReviewInference

st.set_page_config(
    page_title="Hotel Review Classifier",
    page_icon="🏨",
    layout="wide",
)

CHECKPOINT = os.path.join(os.path.dirname(__file__), "..", "models", "saved", "best_model.pt")

@st.cache_resource
def load_model():
    return ReviewInference(CHECKPOINT)


with st.sidebar:
    st.header("Hotel Review Classifier")
    st.divider()
    with st.spinner("Carregando modelo..."):
        model = load_model()
    st.success("Modelo carregado")
    st.caption("XLM-RoBERTa · F1=0.74 · MAE=0.44")
    st.divider()
    st.markdown("**Navegação**")
    st.markdown("- 📤 Upload — carregar CSV")
    st.markdown("- 📊 Dashboard — visão geral")
    st.markdown("- 🔍 Detalhe — review individual")

st.title("Hotel Review Classifier")
st.markdown("""
Plataforma de análise automática de reviews de hotéis usando Deep Learning (XLM-RoBERTa).

### Como usar
1. Acesse **Upload** no menu lateral e carregue um CSV com a coluna `review_text`
2. O modelo classifica cada review em sentimento, categorias, rating previsto e prioridade
3. Explore os resultados no **Dashboard** ou analise reviews individuais em **Detalhe**

### O que o modelo classifica
| Dimensão | Saída |
|---|---|
| Sentimento | negativo / neutro / positivo |
| Categorias | limpeza, atendimento, localização, alimentação, preço, conforto, wifi, instalações |
| Rating previsto | 1.0 – 5.0 |
| Prioridade | normal / alta |

> **Prioridade alta:** reviews negativos com rating ≤ 2 — requerem atenção imediata da gestão.
""")

st.info("Use o menu lateral para navegar entre as páginas.")

