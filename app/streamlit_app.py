import streamlit as st
import pandas as pd

# URL directa del CSV (sin descarga previa)
DATA_URL = "https://covenantaegis.com/segmentation_data_recruitment.csv"

@st.cache_data
def load_data():
    try:
        return pd.read_csv(DATA_URL)
    except Exception as e:
        st.error(f"❌ Error al cargar los datos desde la URL: {e}")
        return pd.DataFrame()

# Función para clasificar riesgo financiero
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

# Carga de datos
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
