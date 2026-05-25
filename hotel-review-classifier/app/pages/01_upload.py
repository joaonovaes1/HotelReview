import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
import pandas as pd
from app.main import load_model

st.header("Upload de Reviews")

uploaded = st.file_uploader("Carregue um CSV com a coluna `review_text`", type="csv")

if uploaded:
    df = pd.read_csv(uploaded, sep=None, engine="python", encoding_errors="replace")
    df.columns = df.columns.str.strip().str.replace("﻿", "", regex=False)

    if "review_text" not in df.columns:
        st.error(f"Coluna `review_text` não encontrada. Colunas detectadas: {list(df.columns)}")
        st.stop()

    st.success(f"{len(df):,} reviews carregados.")
    st.dataframe(df[["review_text"]].head(5), use_container_width=True)

    if st.button("Classificar reviews"):
        model = load_model()
        results = []

        progress = st.progress(0, text="Classificando...")
        total = len(df)

        for i, text in enumerate(df["review_text"]):
            pred = model.predict(str(text))
            results.append(pred)
            progress.progress((i + 1) / total, text=f"Classificando {i+1}/{total}...")

        progress.empty()

        results_df = pd.DataFrame(results)
        results_df.insert(0, "review_text", df["review_text"].values)
        results_df["categories"] = results_df["categories"].apply(lambda x: ", ".join(x) if x else "—")

        st.session_state["results"] = results_df
        st.success("Classificação concluída!")

if "results" in st.session_state:
    st.divider()
    st.subheader("Preview dos resultados")
    st.dataframe(st.session_state["results"], use_container_width=True)
    st.caption("Acesse o Dashboard ou Detalhe no menu lateral para explorar os resultados.")
