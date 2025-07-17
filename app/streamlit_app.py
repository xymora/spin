import streamlit as st
import pandas as pd
import requests

# URL del CSV
DATA_URL = "https://covenantaegis.com/segmentation_data_recruitment.csv"

@st.cache_data
def load_data():
    try:
        return pd.read_csv(DATA_URL)
    except Exception as e:
        st.error(f"No se pudo cargar el archivo desde la URL: {e}")
        return pd.DataFrame()

def clasificar_riesgo(ingreso, pagos):
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
    # Cálculo de riesgo
    if 'ingreso_mensual' in df.columns and 'pagos_mensuales' in df.columns:
        df['Riesgo Financiero'] = df.apply(
            lambda row: clasificar_riesgo(row['ingreso_mensual'], row['pagos_mensuales']), axis=1
        )

    with st.sidebar:
        st.header("🔍 Filtros")
        filtros = {}
        for col in df.columns:
            if df[col].dtype == 'object' and df[col].nunique() < 50:
                seleccion = st.multiselect(col, sorted(df[col].dropna().unique()))
                if seleccion:
                    filtros[col] = seleccion

    df_filtrado = df.copy()
    for col, valores in filtros.items():
        df_filtrado = df_filtrado[df_filtrado[col].isin(valores)]

    st.subheader("📋 Clientes Filtrados")
    st.dataframe(df_filtrado)
    st.markdown(f"🔎 Total encontrados: **{len(df_filtrado)}**")
