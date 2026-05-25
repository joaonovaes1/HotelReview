import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
import pandas as pd

st.header("Dashboard")

if "results" not in st.session_state:
    st.warning("Nenhum dado carregado. Vá para Upload e classifique um CSV primeiro.")
    st.stop()

df = st.session_state["results"]
total       = len(df)
n_alta      = (df["priority"] == "alta").sum()
n_negativo  = (df["sentiment"] == "negativo").sum()
n_positivo  = (df["sentiment"] == "positivo").sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total de reviews", f"{total:,}")
col2.metric("Positivos",        f"{n_positivo:,}", f"{n_positivo/total:.0%}")
col3.metric("Negativos",        f"{n_negativo:,}", f"{n_negativo/total:.0%}", delta_color="inverse")
col4.metric("Alta prioridade",  f"{n_alta:,}",     f"{n_alta/total:.0%}",     delta_color="inverse")

st.divider()
st.subheader("Distribuição de Sentimento")

sentiment_counts = (
    df["sentiment"]
    .value_counts()
    .reindex(["positivo", "neutro", "negativo"])
    .reset_index()
)
sentiment_counts.columns = ["sentimento", "quantidade"]

st.bar_chart(sentiment_counts, x="sentimento", y="quantidade", color="sentimento")

st.divider()
st.subheader("Categorias Mais Mencionadas")

CATEGORIAS = ["limpeza", "atendimento", "localização", "alimentação",
              "preço", "conforto", "wifi", "instalações"]

cat_counts = {
    cat: df["categories"].str.contains(cat, na=False).sum()
    for cat in CATEGORIAS
}
cat_df = (
    pd.DataFrame(list(cat_counts.items()), columns=["categoria", "menções"])
    .sort_values("menções", ascending=False)
)

st.bar_chart(cat_df, x="categoria", y="menções")
