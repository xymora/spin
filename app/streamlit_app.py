import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Dashboard de Clientes Bancarios", layout="wide")
st.title("🏦 Dashboard de Clientes Bancarios")

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
# Calificación crediticia
# =====================
def clasificar_credito(retiros, compras):
    if retiros > 50000 and compras == 0:
        return '🔵 Premium Credit'
    elif retiros > 20000 and compras <= 1:
        return '🟢 Basic Credit'
    elif retiros > 10000:
        return '🟡 Moderate Risk'
    else:
        return '🔴 High Risk'

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
        retiro_min, retiro_max = df['avg_amount_withdrawals'].min(), df['avg_amount_withdrawals'].max()
        compra_min, compra_max = df['avg_purchases_per_week'].min(), df['avg_purchases_per_week'].max()

        edad = st.slider("Edad", edad_min, edad_max, (edad_min, edad_max))
        retiros = st.slider("Retiros promedio", float(retiro_min), float(retiro_max), (float(retiro_min), float(retiro_max)))
        compras = st.slider("Compras por semana", float(compra_min), float(compra_max), (float(compra_min), float(compra_max)))

        tipos_credito = df['credit_score'].unique().tolist()
        tipos_seleccionados = st.multiselect("Credit Score", sorted(tipos_credito), default=tipos_credito)

# =====================
# Aplicar filtros
# =====================
if aplicar_filtros:
    df_filtrado = df[
        (df['age'].between(*edad)) &
        (df['avg_amount_withdrawals'].between(*retiros)) &
        (df['avg_purchases_per_week'].between(*compras)) &
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
df_mostrar = df_filtrado[columnas_finales]

# =====================
# Mostrar datos
# =====================
st.subheader("📋 Clientes Visualizados")
st.dataframe(df_mostrar, use_container_width=True)
st.markdown(f"🔎 Total mostrados: **{len(df_mostrar):,}** / 100,000")
