import streamlit as st
import pandas as pd
import numpy as np
import requests

st.set_page_config(page_title="Dashboard de Clientes Bancarios", layout="wide")
st.title("🏦 Dashboard de Clientes Bancarios")

# =====================
# Cargar datos desde URL
# =====================
DATA_URL = "https://covenantaegis.com/segmentation_data_recruitment.csv"

@st.cache_data
def load_data():
    try:
        return pd.read_csv(DATA_URL)
    except Exception as e:
        st.error(f"No se pudo cargar la base de datos: {e}")
        return pd.DataFrame()

df = load_data()

# =====================
# Clasificación de riesgo
# =====================
def clasificar_riesgo(retiros, compras):
    if retiros > 50000 and compras == 0:
        return '🔵 Crédito Premium'
    elif retiros > 20000 and compras <= 1:
        return '🟢 Crédito Básico'
    elif retiros > 10000:
        return '🟡 Riesgo Moderado'
    else:
        return '🔴 Riesgo Alto'

# Validar columnas requeridas
if not df.empty and {'avg_amount_withdrawals', 'avg_purchases_per_week'}.issubset(df.columns):
    df['Clasificación Automática'] = df.apply(
        lambda row: clasificar_riesgo(row['avg_amount_withdrawals'], row['avg_purchases_per_week']),
        axis=1
    )
else:
    st.error("No se encontraron las columnas necesarias para clasificar el riesgo.")
    st.stop()

# =====================
# Filtros en barra lateral
# =====================
with st.sidebar:
    st.header("🔍 Filtros")
    
    edad_min, edad_max = int(df['age'].min()), int(df['age'].max())
    retiro_min, retiro_max = float(df['avg_amount_withdrawals'].min()), float(df['avg_amount_withdrawals'].max())
    compra_min, compra_max = float(df['avg_purchases_per_week'].min()), float(df['avg_purchases_per_week'].max())

    edad = st.slider("Edad", edad_min, edad_max, (edad_min, edad_max))
    retiros = st.slider("Monto promedio de retiros", retiro_min, retiro_max, (retiro_min, retiro_max))
    compras = st.slider("Compras promedio por semana", compra_min, compra_max, (compra_min, compra_max))

# =====================
# Aplicar filtros
# =====================
df_filtrado = df[
    (df['age'].between(*edad)) &
    (df['avg_amount_withdrawals'].between(*retiros)) &
    (df['avg_purchases_per_week'].between(*compras))
]

# =====================
# Mostrar resultados
# =====================
st.subheader("📋 Clientes Filtrados")
columnas_mostrar = ['user', 'age', 'avg_amount_withdrawals', 'avg_purchases_per_week', 'Clasificación Automática']
columnas_mostrar = [col for col in columnas_mostrar if col in df_filtrado.columns]
st.dataframe(df_filtrado[columnas_mostrar])
st.markdown(f"🔎 Total encontrados: **{len(df_filtrado):,}**")
