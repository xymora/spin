import streamlit as st
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
import json

firestore_active = False
try:
    if not firebase_admin._apps:
        cred = credentials.Certificate(dict(st.secrets["firebase"]))
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    collection_name = "clients"
    firestore_active = True
except Exception as e:
    st.warning(f"Firebase error: {e}")
    firestore_active = False

@st.cache_data
def load_data():
    if firestore_active:
        try:
            docs = db.collection(collection_name).stream()
            data = [doc.to_dict() for doc in docs]
            return pd.DataFrame(data)
        except:
            pass
    try:
        return pd.read_csv("notebooks/segmentation_data_recruitment.csv")
    except:
        st.error("No se pudo cargar los datos.")
        return pd.DataFrame()

df = load_data()
st.title("🏦 Dashboard de Clientes Bancarios")

if df.empty:
    st.warning("No hay datos disponibles.")
else:
    with st.sidebar:
        st.header("🔍 Filtros")
        filters = {}
        for col in df.columns:
            if df[col].dtype == 'object' and df[col].nunique() < 50:
                selected = st.multiselect(col, sorted(df[col].dropna().unique()))
                if selected:
                    filters[col] = selected

    filtered_df = df.copy()
    for col, selected in filters.items():
        filtered_df = filtered_df[filtered_df[col].isin(selected)]

    st.subheader("📋 Clientes filtrados")
    st.dataframe(filtered_df)
    st.markdown(f"🔎 Total encontrados: **{len(filtered_df)}**")
