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

def clasificar_riesgo(ingreso, pagos):
    if ingreso > 50000 and pagos == 0:
        return '🔵 Crédito Premium'
    elif ingreso > 20000 and pagos <= 1:
        return '🟢 Crédito Básico'
    elif ingreso > 10000:
        return '🟡 Riesgo Moderado'
    else:
        return '🔴 Riesgo Alto'

df = load_data()
st.title("🏦 Dashboard de Clientes Bancarios")

if df.empty:
    st.warning("No hay datos disponibles.")
else:
    st.write("🧾 Columnas del DataFrame:", df.columns.tolist())

    ingreso_col = 'ingreso_mensual' if 'ingreso_mensual' in df.columns else None
    pagos_col = 'pagos_mensuales' if 'pagos_mensuales' in df.columns else None

    if ingreso_col and pagos_col:
        df['Clasificación Automática'] = df.apply(
            lambda row: clasificar_riesgo(row[ingreso_col], row[pagos_col]), axis=1
        )

    with st.sidebar:
        st.header("🔍 Filtros")
        filtros = {}
        for col in df.columns:
            if df[col].dtype == 'object' and df[col].nunique() < 50:
                seleccion = st.multiselect(col, sorted(df[col].dropna().unique()), key=col)
                if seleccion:
                    filtros[col] = seleccion

        if 'Clasificación Automática' in df.columns:
            clasificaciones = st.multiselect(
                "Clasificación Automática",
                ['🔵 Crédito Premium', '🟢 Crédito Básico', '🟡 Riesgo Moderado', '🔴 Riesgo Alto'],
                key='clasificacion'
            )
            if clasificaciones:
                filtros['Clasificación Automática'] = clasificaciones

    df_filtrado = df.copy()
    for col, valores in filtros.items():
        if col in df_filtrado.columns:
            df_filtrado = df_filtrado[df_filtrado[col].isin(valores)]

    st.subheader("📋 Clientes Filtrados")
    st.dataframe(df_filtrado)
    st.markdown(f"🔎 Total encontrados: **{len(df_filtrado)}**")
