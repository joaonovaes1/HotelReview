import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.classifier import CATEGORIES

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")
st.title("📊 Dashboard de Reviews")

if "classified_df" not in st.session_state:
    st.warning("Nenhum dado classificado. Vá para a página **Upload** primeiro.")
    st.stop()

df = st.session_state["classified_df"].copy()

# --- Filtros ---
st.sidebar.header("Filtros")

hotels = ["Todos"] + sorted(df["Hotel_Name"].dropna().unique().tolist()) if "Hotel_Name" in df.columns else ["Todos"]
selected_hotel = st.sidebar.selectbox("Hotel", hotels)
if selected_hotel != "Todos":
    df = df[df["Hotel_Name"] == selected_hotel]

sentiments = ["Todos"] + ["negativo", "neutro", "positivo"]
selected_sentiment = st.sidebar.selectbox("Sentimento", sentiments)
if selected_sentiment != "Todos":
    df = df[df["sentiment"] == selected_sentiment]

priorities = ["Todos", "alta", "normal"]
selected_priority = st.sidebar.selectbox("Prioridade", priorities)
if selected_priority != "Todos":
    df = df[df["priority"] == selected_priority]

if "language" in df.columns:
    langs = ["Todos"] + sorted(df["language"].dropna().unique().tolist())
    selected_lang = st.sidebar.selectbox("Idioma", langs)
    if selected_lang != "Todos":
        df = df[df["language"] == selected_lang]

if "Review_Date" in df.columns:
    df["Review_Date"] = pd.to_datetime(df["Review_Date"], errors="coerce")
    min_date = df["Review_Date"].min()
    max_date = df["Review_Date"].max()
    if pd.notna(min_date) and pd.notna(max_date):
        date_range = st.sidebar.date_input("Período", value=(min_date.date(), max_date.date()))
        if len(date_range) == 2:
            df = df[(df["Review_Date"].dt.date >= date_range[0]) & (df["Review_Date"].dt.date <= date_range[1])]

st.caption(f"Exibindo **{len(df):,}** reviews após filtros.")

# --- KPIs ---
st.subheader("Indicadores Gerais")
k1, k2, k3, k4 = st.columns(4)
with k1:
    avg = df["Reviewer_Score"].mean() if "Reviewer_Score" in df.columns else 0
    st.metric("Score Médio", f"{avg:.2f} / 10")
with k2:
    pct_neg = (df["sentiment"] == "negativo").mean() * 100
    st.metric("Reviews Negativos", f"{pct_neg:.1f}%")
with k3:
    pct_alta = (df["priority"] == "alta").mean() * 100
    st.metric("Alta Prioridade", f"{pct_alta:.1f}%")
with k4:
    n_hotels = df["Hotel_Name"].nunique() if "Hotel_Name" in df.columns else 1
    st.metric("Hotéis", n_hotels)

st.divider()

# --- Gráficos linha 1 ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribuição de Sentimento")
    sent_counts = df["sentiment"].value_counts().reset_index()
    sent_counts.columns = ["Sentimento", "Quantidade"]
    color_map = {"negativo": "#EF4444", "neutro": "#F59E0B", "positivo": "#10B981"}
    fig_pie = px.pie(
        sent_counts, values="Quantidade", names="Sentimento",
        color="Sentimento", color_discrete_map=color_map,
        hole=0.4,
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader("Categorias Mais Mencionadas")
    cat_counts = {cat: df[f"cat_{cat}"].sum() for cat in CATEGORIES if f"cat_{cat}" in df.columns}
    cat_df = pd.DataFrame({"Categoria": list(cat_counts.keys()), "Menções": list(cat_counts.values())})
    cat_df = cat_df.sort_values("Menções", ascending=True)
    fig_bar = px.bar(cat_df, x="Menções", y="Categoria", orientation="h", color="Menções",
                     color_continuous_scale="Blues")
    fig_bar.update_layout(showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig_bar, use_container_width=True)

# --- Gráficos linha 2 ---
col3, col4 = st.columns(2)

with col3:
    st.subheader("Distribuição de Scores")
    if "Reviewer_Score" in df.columns:
        fig_hist = px.histogram(df, x="Reviewer_Score", nbins=20, color_discrete_sequence=["#6366F1"])
        fig_hist.update_layout(xaxis_title="Score", yaxis_title="Quantidade")
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Coluna Reviewer_Score não disponível.")

with col4:
    st.subheader("Sentimento por Categoria")
    rows = []
    for cat in CATEGORIES:
        if f"cat_{cat}" in df.columns:
            sub = df[df[f"cat_{cat}"] == 1]
            for sent in ["negativo", "neutro", "positivo"]:
                rows.append({"Categoria": cat, "Sentimento": sent, "Quantidade": (sub["sentiment"] == sent).sum()})
    if rows:
        heatmap_df = pd.DataFrame(rows)
        pivot = heatmap_df.pivot(index="Categoria", columns="Sentimento", values="Quantidade").fillna(0)
        fig_heat = px.imshow(pivot, color_continuous_scale="RdYlGn", text_auto=True, aspect="auto")
        st.plotly_chart(fig_heat, use_container_width=True)

# --- Word Cloud ---
st.subheader("Nuvem de Palavras")
wc_tab1, wc_tab2, wc_tab3 = st.tabs(["Todos", "Negativos", "Positivos"])

def render_wordcloud(text_series, title=""):
    text = " ".join(text_series.dropna().astype(str).tolist())
    if not text.strip():
        st.info("Sem texto suficiente para gerar nuvem.")
        return
    wc = WordCloud(width=800, height=400, background_color="white", max_words=100).generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
    plt.close()

with wc_tab1:
    render_wordcloud(df["text"])
with wc_tab2:
    render_wordcloud(df[df["sentiment"] == "negativo"]["text"])
with wc_tab3:
    render_wordcloud(df[df["sentiment"] == "positivo"]["text"])

# --- Evolução temporal ---
if "Review_Date" in df.columns and df["Review_Date"].notna().any():
    st.subheader("Evolução Temporal do Sentimento")
    temp_df = df[df["Review_Date"].notna()].copy()
    temp_df["mes"] = temp_df["Review_Date"].dt.to_period("M").astype(str)
    timeline = temp_df.groupby(["mes", "sentiment"]).size().reset_index(name="count")
    color_map2 = {"negativo": "#EF4444", "neutro": "#F59E0B", "positivo": "#10B981"}
    fig_line = px.line(timeline, x="mes", y="count", color="sentiment",
                       color_discrete_map=color_map2, markers=True)
    fig_line.update_layout(xaxis_title="Mês", yaxis_title="Quantidade")
    st.plotly_chart(fig_line, use_container_width=True)
