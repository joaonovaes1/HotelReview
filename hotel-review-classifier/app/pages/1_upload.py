import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.loader import load_uploaded_file
from src.data.preprocessor import derive_labels, add_category_labels, clean_text

st.set_page_config(page_title="Upload de Reviews", page_icon="📂", layout="wide")
st.title("📂 Upload de Reviews")

st.markdown(
    """
    Faça upload do seu arquivo de reviews. O arquivo deve conter pelo menos uma coluna de texto com os comentários.

    **Formatos aceitos:** CSV, Excel (.xlsx, .xls)

    **Colunas esperadas (Booking/Kaggle):**
    `Hotel_Name`, `Positive_Review`, `Negative_Review`, `Reviewer_Score`, `Review_Date`, `Tags`
    """
)

uploaded = st.file_uploader("Selecione o arquivo de reviews", type=["csv", "xlsx", "xls"])

if uploaded:
    with st.spinner("Carregando arquivo..."):
        try:
            df = load_uploaded_file(uploaded)
        except Exception as e:
            st.error(f"Erro ao carregar arquivo: {e}")
            st.stop()

    st.success(f"Arquivo carregado: **{len(df):,} linhas** e **{len(df.columns)} colunas**")
    st.subheader("Prévia dos dados")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Configurações de classificação")
    col1, col2 = st.columns(2)

    with col1:
        has_booking_cols = "Positive_Review" in df.columns and "Negative_Review" in df.columns
        if has_booking_cols:
            st.info("Formato Booking/Kaggle detectado — colunas Positive_Review e Negative_Review serão combinadas.")
            text_col = None
        else:
            text_col = st.selectbox("Coluna de texto dos reviews", df.columns.tolist())

    with col2:
        has_score = "Reviewer_Score" in df.columns
        if has_score:
            st.info("Coluna Reviewer_Score detectada — labels de sentimento e prioridade serão derivadas automaticamente.")
            score_col = "Reviewer_Score"
        else:
            score_col = st.selectbox("Coluna de nota/score (opcional)", ["(nenhuma)"] + df.columns.tolist())
            score_col = None if score_col == "(nenhuma)" else score_col

    if st.button("Classificar Reviews", type="primary", use_container_width=True):
        with st.spinner("Pré-processando e classificando... Isso pode levar alguns minutos."):
            try:
                if has_booking_cols:
                    processed = derive_labels(df)
                else:
                    processed = df.copy()
                    processed["text"] = processed[text_col].apply(clean_text)
                    if score_col:
                        processed["Reviewer_Score"] = pd.to_numeric(processed[score_col], errors="coerce").fillna(5.0)
                        from src.data.preprocessor import derive_sentiment, derive_priority
                        processed["sentiment_label"] = processed["Reviewer_Score"].apply(derive_sentiment)
                        processed["rating_normalized"] = processed["Reviewer_Score"] / 10.0
                        processed["priority_label"] = processed.apply(
                            lambda r: derive_priority(r["sentiment_label"], r["Reviewer_Score"]), axis=1
                        )
                    else:
                        processed["Reviewer_Score"] = 5.0
                        processed["sentiment_label"] = 1
                        processed["rating_normalized"] = 0.5
                        processed["priority_label"] = 0

                processed = add_category_labels(processed)

                SENTIMENT_MAP = {0: "negativo", 1: "neutro", 2: "positivo"}
                PRIORITY_MAP = {0: "normal", 1: "alta"}
                processed["sentiment"] = processed["sentiment_label"].map(SENTIMENT_MAP)
                processed["priority"] = processed["priority_label"].map(PRIORITY_MAP)
                processed["rating_predicted"] = processed["Reviewer_Score"]

                from src.models.classifier import CATEGORIES
                processed["categories"] = processed.apply(
                    lambda r: [c for c in CATEGORIES if r.get(f"cat_{c}", 0) == 1] or ["geral"],
                    axis=1,
                )

                st.session_state["classified_df"] = processed
                st.session_state["source_name"] = uploaded.name

                st.success(f"Classificação concluída! {len(processed):,} reviews processados.")

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    n_neg = (processed["sentiment"] == "negativo").sum()
                    st.metric("Reviews negativos", f"{n_neg:,}", f"{n_neg/len(processed)*100:.1f}%")
                with col_b:
                    n_alta = (processed["priority"] == "alta").sum()
                    st.metric("Alta prioridade", f"{n_alta:,}", f"{n_alta/len(processed)*100:.1f}%")
                with col_c:
                    avg_score = processed["Reviewer_Score"].mean()
                    st.metric("Score médio", f"{avg_score:.2f}")

                st.info("Acesse as páginas **Dashboard** e **Tabela Detalhada** no menu lateral para explorar os resultados.")

            except Exception as e:
                st.error(f"Erro durante a classificação: {e}")
                raise

elif "classified_df" in st.session_state:
    st.info(f"Dados já classificados em memória: **{len(st.session_state['classified_df']):,} reviews** de `{st.session_state.get('source_name', 'arquivo')}`.")
    st.markdown("Navegue para o **Dashboard** ou **Tabela Detalhada** para explorar.")
