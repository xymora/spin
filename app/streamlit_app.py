import pandas as pd
import streamlit as st

# URL del dataset
DATA_URL = "https://covenantaegis.com/segmentation_data_recruitment.csv"

# Clasificación automática
def clasificar_riesgo(retiros, compras):
    if retiros > 50000 and compras == 0:
        return '🔵 Crédito Premium'
    elif retiros > 20000 and compras <= 1:
        return '🟢 Crédito Básico'
    elif retiros > 10000:
        return '🟡 Riesgo Moderado'
    else:
        return '🔴 Riesgo Alto'

# Cargar datos desde la URL
@st.cache_data
def load_data():
    try:
        return pd.read_csv(DATA_URL)
    except Exception as e:
        st.error(f"No se pudo cargar el archivo desde la URL: {e}")
        return pd.DataFrame()

# Título
st.title("🏦 Dashboard de Clientes Bancarios")

# Cargar datos
df = load_data()

# Verificar que no esté vacío
if df.empty:
    st.warning("No hay datos disponibles.")
    st.stop()

# Clasificar automáticamente
if 'avg_amount_withdrawals' in df.columns and 'avg_purchases_per_week' in df.columns:
    df['Clasificación Automática'] = df.apply(
        lambda row: clasificar_riesgo(row['avg_amount_withdrawals'], row['avg_purchases_per_week']),
        axis=1
    )

# Filtros en la barra lateral
with st.sidebar:
    st.header("🔍 Filtros")
    edad = st.slider("Edad", int(df['age'].min()), int(df['age'].max()), (int(df['age'].min()), int(df['age'].max())))
    retiros = st.slider("Monto promedio de retiros", float(df['avg_amount_withdrawals'].min()), float(df['avg_amount_withdrawals'].max()), (float(df['avg_amount_withdrawals'].min()), float(df['avg_amount_withdrawals'].max())))
    compras = st.slider("Compras promedio por semana", float(df['avg_purchases_per_week'].min()), float(df['avg_purchases_per_week'].max()), (float(df['avg_purchases_per_week'].min()), float(df['avg_purchases_per_week'].max())))

# Aplicar filtros
df_filtrado = df[
    (df['age'].between(*edad)) &
    (df['avg_amount_withdrawals'].between(*retiros)) &
    (df['avg_purchases_per_week'].between(*compras))
]

# Mostrar resultados filtrados
st.subheader("📋 Clientes Filtrados")
columnas_mostrar = ['user', 'age', 'avg_amount_withdrawals', 'avg_purchases_per_week', 'Clasificación Automática']
columnas_mostrar = [col for col in columnas_mostrar if col in df_filtrado.columns]
st.dataframe(df_filtrado[columnas_mostrar])
st.markdown(f"🔎 Total encontrados: **{len(df_filtrado)}**")
