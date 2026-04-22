import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.classifier import CATEGORIES

st.set_page_config(page_title="Tabela Detalhada", page_icon="📋", layout="wide")
st.title("📋 Reviews Detalhados")

if "classified_df" not in st.session_state:
    st.warning("Nenhum dado classificado. Vá para a página **Upload** primeiro.")
    st.stop()

df = st.session_state["classified_df"].copy()

# --- Filtros ---
st.sidebar.header("Filtros")

if "Hotel_Name" in df.columns:
    hotels = ["Todos"] + sorted(df["Hotel_Name"].dropna().unique().tolist())
    sel_hotel = st.sidebar.selectbox("Hotel", hotels)
    if sel_hotel != "Todos":
        df = df[df["Hotel_Name"] == sel_hotel]

sel_sent = st.sidebar.multiselect("Sentimento", ["negativo", "neutro", "positivo"], default=["negativo", "neutro", "positivo"])
if sel_sent:
    df = df[df["sentiment"].isin(sel_sent)]

sel_prio = st.sidebar.multiselect("Prioridade", ["alta", "normal"], default=["alta", "normal"])
if sel_prio:
    df = df[df["priority"].isin(sel_prio)]

sel_cats = st.sidebar.multiselect("Categorias", CATEGORIES)

if sel_cats:
    mask = pd.Series([False] * len(df), index=df.index)
    for cat in sel_cats:
        if f"cat_{cat}" in df.columns:
            mask = mask | (df[f"cat_{cat}"] == 1)
    df = df[mask]

st.caption(f"Exibindo **{len(df):,}** reviews após filtros.")

# --- Tabela ---
PRIORITY_BADGE = {"alta": "🔴 Alta", "normal": "🟢 Normal"}
SENTIMENT_BADGE = {"negativo": "😠 Negativo", "neutro": "😐 Neutro", "positivo": "😊 Positivo"}

display_cols = []
if "Hotel_Name" in df.columns:
    display_cols.append("Hotel_Name")
if "Review_Date" in df.columns:
    display_cols.append("Review_Date")
display_cols += ["text", "sentiment", "priority", "Reviewer_Score", "categories"]

available = [c for c in display_cols if c in df.columns]
view = df[available].copy()
view["sentiment"] = view["sentiment"].map(SENTIMENT_BADGE).fillna(view["sentiment"])
view["priority"] = view["priority"].map(PRIORITY_BADGE).fillna(view["priority"])

if "categories" in view.columns:
    view["categories"] = view["categories"].apply(
        lambda x: ", ".join(x) if isinstance(x, list) else str(x)
    )

rename_map = {
    "Hotel_Name": "Hotel",
    "Review_Date": "Data",
    "text": "Review",
    "sentiment": "Sentimento",
    "priority": "Prioridade",
    "Reviewer_Score": "Score",
    "categories": "Categorias",
}
view = view.rename(columns={k: v for k, v in rename_map.items() if k in view.columns})

st.dataframe(
    view,
    use_container_width=True,
    height=500,
    column_config={
        "Review": st.column_config.TextColumn(width="large"),
        "Score": st.column_config.NumberColumn(format="%.1f"),
    },
)

# --- Reviews de alta prioridade ---
st.divider()
st.subheader("🔴 Reviews de Alta Prioridade")
high_prio = df[df["priority"] == "alta"].sort_values("Reviewer_Score") if "Reviewer_Score" in df.columns else df[df["priority"] == "alta"]

if len(high_prio) == 0:
    st.success("Nenhum review de alta prioridade nos filtros atuais.")
else:
    st.caption(f"{len(high_prio)} reviews de alta prioridade")
    for _, row in high_prio.head(20).iterrows():
        with st.expander(
            f"🔴 Score {row.get('Reviewer_Score', 'N/A')} — {row.get('Hotel_Name', '')} — {', '.join(row['categories']) if isinstance(row.get('categories'), list) else row.get('categories', '')}"
        ):
            st.write(row.get("text", ""))
            cols = st.columns(3)
            cols[0].metric("Sentimento", SENTIMENT_BADGE.get(row.get("sentiment"), row.get("sentiment", "")))
            cols[1].metric("Score", f"{row.get('Reviewer_Score', 'N/A')}")
            cols[2].metric("Idioma", row.get("language", "N/A"))

# --- Exportar ---
st.divider()
st.subheader("Exportar Resultados")
csv_data = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download CSV completo",
    data=csv_data,
    file_name="reviews_classificados.csv",
    mime="text/csv",
    use_container_width=True,
)
