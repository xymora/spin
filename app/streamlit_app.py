import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Panel de Clientes Bancarios", layout="wide")
st.title("🏦 Panel de Clientes Bancarios")

# =====================
# Cargar datos
# =====================
DATA_URL = "https://covenantaegis.com/segmentation_data_recruitment.csv"

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_URL)
        df['avg_amount_withdrawals'] = pd.to_numeric(df['avg_amount_withdrawals'], errors='coerce').fillna(0)
        df['avg_purchases_per_week'] = pd.to_numeric(df['avg_purchases_per_week'], errors='coerce').fillna(0)
        return df
    except Exception as e:
        st.error(f"No se pudo cargar la base de datos: {e}")
        return pd.DataFrame()

df = load_data()
if df.empty:
    st.stop()

# =====================
# Credit Score Classification
# =====================
def clasificar_credito(retiros, compras):
    if retiros > 50000 and compras == 0:
        return '🔵 Crédito Premium'
    elif retiros > 20000 and compras <= 1:
        return '🟢 Crédito Básico'
    elif retiros > 10000:
        return '🟡 Riesgo Moderado'
    else:
        return '🔴 Riesgo Alto'

df['credit_score'] = df.apply(
    lambda row: clasificar_credito(row['avg_amount_withdrawals'], row['avg_purchases_per_week']),
    axis=1
)

# =====================
# Filtros laterales
# =====================
with st.sidebar:
    st.header("🔍 Filtros Opcionales")
    aplicar_filtros = st.checkbox("Aplicar filtros", value=False)

    if aplicar_filtros:
        edad_min, edad_max = int(df['age'].min()), int(df['age'].max())
        retiros_min, retiros_max = df['avg_amount_withdrawals'].min(), df['avg_amount_withdrawals'].max()
        compras_min, compras_max = df['avg_purchases_per_week'].min(), df['avg_purchases_per_week'].max()

        rango_edad = st.slider("Edad", edad_min, edad_max, (edad_min, edad_max))
        rango_retiros = st.slider("Retiros Promedio", float(retiros_min), float(retiros_max), (float(retiros_min), float(retiros_max)))
        rango_compras = st.slider("Compras por Semana", float(compras_min), float(compras_max), (float(compras_min), float(compras_max)))

        tipos_credito = df['credit_score'].unique().tolist()
        tipos_seleccionados = st.multiselect("credit_score", sorted(tipos_credito), default=tipos_credito)

# =====================
# Aplicar filtros
# =====================
if aplicar_filtros:
    df_filtrado = df[
        (df['age'].between(*rango_edad)) &
        (df['avg_amount_withdrawals'].between(*rango_retiros)) &
        (df['avg_purchases_per_week'].between(*rango_compras)) &
        (df['credit_score'].isin(tipos_seleccionados))
    ]
else:
    df_filtrado = df.copy()

# =====================
# Reordenar columnas
# =====================
primeras_columnas = [
    'user',
    'age',
    'index',
    'credit_score',
    'user_type',
    'registration_channel',
    'creation_flow',
    'creation_date',
    'avg_amount_withdrawals'
]
otras_columnas = sorted([col for col in df_filtrado.columns if col not in primeras_columnas])
columnas_finales = primeras_columnas + otras_columnas
df_vista = df_filtrado[columnas_finales]

# =====================
# Mostrar resultados
# =====================
st.subheader("📋 Clientes Visualizados")
st.dataframe(df_vista, use_container_width=True)
st.markdown(f"🔎 Total mostrados: **{len(df_vista):,}** / 100,000")
