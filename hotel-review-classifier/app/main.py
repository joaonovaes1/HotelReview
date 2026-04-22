import streamlit as st

st.set_page_config(
    page_title="Hotel Review Classifier",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏨 Hotel Review Classifier")
st.markdown(
    """
    Sistema de classificação de reviews de hotéis com Deep Learning.

    **Como usar:**
    1. Acesse **Upload** para enviar seu arquivo de reviews (CSV ou Excel)
    2. Clique em **Classificar Reviews** para rodar o modelo
    3. Explore os resultados no **Dashboard** e na **Tabela Detalhada**

    ---
    """
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Sentimentos", "3 classes", "negativo / neutro / positivo")
with col2:
    st.metric("Categorias", "8 temas", "limpeza, atendimento...")
with col3:
    st.metric("Rating", "Regressão", "0–10")
with col4:
    st.metric("Prioridade", "Binário", "alta / normal")

st.info("Navegue pelas páginas no menu lateral para começar.")
