import streamlit as st
import pandas as pd

DATA_URL = "https://covenantaegis.com/segmentation_data_recruitment.csv"

@st.cache_data
def load_data():
    try:
        return pd.read_csv(DATA_URL)
    except Exception as e:
        st.error(f"No se pudo cargar el archivo desde la URL: {e}")
        return pd.DataFrame()

# Función para clasificar el riesgo
def clasificar_riesgo(retiros, compras):
    if retiros > 50000 and compras == 0:
        return '🔵 Crédito Premium'
    elif retiros > 20000 and compras <= 1:
        return '🟢 Crédito Básico'
    elif retiros > 10000:
        return '🟡 Riesgo Moderado'
    else:
        return '🔴 Riesgo Alto'

# Cargar datos
df = load_data()
st.title("🏦 Dashboard de Clientes Bancarios")

if df.empty:
    st.warning("No hay datos disponibles.")
else:
    # Aplicar la clasificación de riesgo a cada cliente
    if 'avg_amount_withdrawals' in df.columns and 'avg_purchase_per_week' in df.columns:
        df['Clasificación Automática'] = df.apply(
            lambda row: clasificar_riesgo(row['avg_amount_withdrawals'], row['avg_purchase_per_week']),
            axis=1
        )

    # Sidebar de filtros
    with st.sidebar:
        st.header("🔍 Filtros")
        
        # Filtro de Edad (asumido como 'age')
        if 'age' in df.columns:
            filtro_edad = st.slider('Selecciona rango de edad', min_value=int(df['age'].min()), max_value=int(df['age'].max()), value=(int(df['age'].min()), int(df['age'].max())))
            df = df[(df['age'] >= filtro_edad[0]) & (df['age'] <= filtro_edad[1])]

        # Filtro de Ingreso (index, asumido como 'total_amount')
        if 'total_amount' in df.columns:
            filtro_ingreso = st.slider('Selecciona rango de ingreso', min_value=int(df['total_amount'].min()), max_value=int(df['total_amount'].max()), value=(int(df['total_amount'].min()), int(df['total_amount'].max())))
            df = df[(df['total_amount'] >= filtro_ingreso[0]) & (df['total_amount'] <= filtro_ingreso[1])]

        # Filtros adicionales, si se necesitan
        filtros_adicionales = {}
        for col in df.columns:
            if df[col].dtype == 'object' and df[col].nunique() < 50:
                seleccion = st.multiselect(col, sorted(df[col].dropna().unique()), key=col)
                if seleccion:
                    filtros_adicionales[col] = seleccion

        for col, valores in filtros_adicionales.items():
            if col in df.columns:
                df = df[df[col].isin(valores)]

    # Mostrar resultados filtrados
    st.subheader("📋 Clientes Filtrados")
    st.dataframe(df)
    st.markdown(f"🔎 Total encontrados: **{len(df)}**")
