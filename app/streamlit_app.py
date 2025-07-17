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

df['Clasificación Automática'] = df.apply(
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

        tipos_credito = df['Clasificación Automática'].unique().tolist()
        tipos_seleccionados = st.multiselect("Tipo de crédito", sorted(tipos_credito), default=tipos_credito)

# =====================
# Aplicar filtros
# =====================
if aplicar_filtros:
    df_filtrado = df[
        (df['age'].between(*edad)) &
        (df['avg_amount_withdrawals'].between(*retiros)) &
        (df['avg_purchases_per_week'].between(*compras)) &
        (df['Clasificación Automática'].isin(tipos_seleccionados))
    ]
else:
    df_filtrado = df.copy()

# =====================
# Diccionario de nombres en español
# =====================
nombres_columnas_es = {
    "user": "ID",
    "age": "Edad",
    "gender": "Género",
    "marital_status": "Estado civil",
    "education_level": "Nivel educativo",
    "employment_status": "Ocupación",
    "account_balance": "Capital",
    "avg_amount_withdrawals": "Promedio de retiros",
    "avg_purchases_per_week": "Compras por semana",
    "is_homeowner": "Es propietario",
    "has_credit_card": "Tiene tarjeta de crédito",
    "num_products_owned": "Productos contratados",
    "days_active_per_month": "Días activo por mes",
    "device_type": "Tipo de dispositivo",
    "region": "Región",
    "Clasificación Automática": "Clasificación automática"
}

# =====================
# Reordenar columnas para mostrar primero la "Capital" como índice visual
# =====================
df_mostrar = df_filtrado.copy()
df_mostrar.index.name = "Capital"
df_mostrar = df_mostrar.rename(columns=nombres_columnas_es)

# Reordenar si quieres que "Nombre" esté al inicio
if "ID" in df_mostrar.columns:
    cols = df_mostrar.columns.tolist()
    if "ID" in cols:
        cols.remove("ID")
        df_mostrar = df_mostrar[["ID"] + cols]

st.subheader("📋 Clientes Visualizados")

st.dataframe(df_mostrar, use_container_width=True)
st.markdown(f"🔎 Total mostrados: **{len(df_mostrar):,}** / 100,000")
