import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Dashboard de Clientes Bancarios", layout="wide")
st.title("🏦 Dashboard de Clientes Bancarios")

# =====================
# Cargar datos desde URL
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
# Clasificación automática
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

df['credit_score'] = df.apply(
    lambda row: clasificar_riesgo(row['avg_amount_withdrawals'], row['avg_purchases_per_week']),
    axis=1
)

# =====================
# Filtros laterales
# =====================
with st.sidebar:
    st.header("🔍 Filtros (opcional)")
    
    aplicar_filtros = st.checkbox("Aplicar filtros", value=False)

    if aplicar_filtros:
        edad_min, edad_max = int(df['age'].min()), int(df['age'].max())
        retiro_min, retiro_max = df['avg_amount_withdrawals'].min(), df['avg_amount_withdrawals'].max()
        compra_min, compra_max = df['avg_purchases_per_week'].min(), df['avg_purchases_per_week'].max()

        edad = st.slider("Edad", edad_min, edad_max, (edad_min, edad_max))
        retiros = st.slider("Promedio de Retiros", float(retiro_min), float(retiro_max), (float(retiro_min), float(retiro_max)))
        compras = st.slider("Compras por semana", float(compra_min), float(compra_max), (float(compra_min), float(compra_max)))

        tipos_credito = df['credit_score'].unique().tolist()
        tipos_seleccionados = st.multiselect("Tipo de crédito", sorted(tipos_credito), default=tipos_credito)

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
columnas_inicio = ['user', 'age', 'avg_amount_withdrawals', 'index', 'credit_score']
columnas_final = ['registration_channel', 'creation_date', 'creation_flow']
columnas_intermedias = sorted([col for col in df_filtrado.columns if col not in columnas_inicio + columnas_final])

orden_final = columnas_inicio + columnas_intermedias + columnas_final
df_final = df_filtrado[orden_final]

# =====================
# Mostrar resultados
# =====================
st.subheader("📋 Clientes Visualizados")
st.dataframe(df_final, use_container_width=True)
st.markdown(f"🔎 Total mostrados: **{len(df_final):,}** / 100,000")
