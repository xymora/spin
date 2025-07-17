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
        url = "https://covenantaegis.com/segmentation_data_recruitment.csv"
        return pd.read_csv(url)
    except:
        st.error("No se pudo cargar los datos.")
        return pd.DataFrame()

def clasificar_riesgo(ingreso, pagos):
    try:
        ingreso = float(ingreso)
        pagos = int(pagos)
    except:
        return '❓ No evaluado'

    if ingreso > 50000 and pagos == 0:
        return '🔵 Crédito Premium'
    elif ingreso > 20000 and pagos <= 1:
        return '🟢 Crédito Básico'
    elif ingreso > 10000:
        return '🟡 Riesgo Moderado'
    else:
        return '🔴 Alto Riesgo'

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

    if 'ingreso_mensual' in filtered_df.columns and 'pagos_atrasados' in filtered_df.columns:
        filtered_df['riesgo_crediticio'] = filtered_df.apply(
            lambda row: clasificar_riesgo(row['ingreso_mensual'], row['pagos_atrasados']), axis=1
        )

    st.subheader("📋 Clientes filtrados")
    st.dataframe(filtered_df)
    st.markdown(f"🔎 Total encontrados: **{len(filtered_df)}**")
