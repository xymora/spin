import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

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
# Clasificación crediticia
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

orden_credit = [
    '🔵 Premium Credit',
    '🟢 Basic Credit',
    '🟡 Moderate Risk',
    '🔴 High Risk'
]

df['credit_score'] = df.apply(
    lambda row: clasificar_credito(row['avg_amount_withdrawals'], row['avg_purchases_per_week']),
    axis=1
)

# =====================
# Filtros
# =====================
with st.sidebar:
    st.header("🔍 Filtros opcionales")
    aplicar_filtros = st.checkbox("Aplicar filtros", value=False)
    if aplicar_filtros:
        tipos = [t for t in orden_credit if t in df['credit_score'].unique()]
        seleccionados = st.multiselect("Credit Score", tipos, default=tipos)
    else:
        seleccionados = orden_credit

df_filtrado = df[df['credit_score'].isin(seleccionados)]

# =====================
# Reordenar columnas
# =====================
primeras_columnas = [
    'user', 'age', 'index', 'credit_score', 'user_type',
    'registration_channel', 'creation_flow', 'creation_date',
    'avg_amount_withdrawals'
]
otras_columnas = sorted([col for col in df_filtrado.columns if col not in primeras_columnas])
columnas_finales = primeras_columnas + otras_columnas
df_mostrar = df_filtrado[columnas_finales]

# =====================
# Mostrar tabla
# =====================
st.subheader("📋 Clientes Visualizados")
st.dataframe(df_mostrar, use_container_width=True)
st.markdown(f"🔎 Total mostrados: **{len(df_mostrar):,}** / 100,000")

# =====================
# Gráfica de barras Credit Score
# =====================
conteo = df_filtrado['credit_score'].value_counts().reindex(orden_credit).dropna().reset_index()
conteo.columns = ['credit_score', 'count']

fig = px.bar(
    conteo, x='credit_score', y='count', color='credit_score', text='count',
    title="Distribución de clientes por tipo de Credit Score",
    color_discrete_sequence=["blue", "green", "gold", "red"]
)
fig.update_layout(showlegend=False, height=400)
fig.update_traces(textposition='outside')
st.plotly_chart(fig, use_container_width=True)

# =====================
# Gráficas financieras por grupo
# =====================
st.subheader("📊 Análisis Financiero por Credit Score")
for score in seleccionados:
    st.markdown(f"### {score}")
    sub_df = df_filtrado[df_filtrado['credit_score'] == score]
    col1, col2, col3 = st.columns(3)

    with col1:
        fig1 = px.violin(sub_df, y='avg_amount_withdrawals', box=True, points='all',
                         title="Distribución de Retiros Promedio")
        fig1.update_layout(height=250)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.box(sub_df, y='avg_purchases_per_week',
                      title="Compras por Semana - Box Plot")
        fig2.update_layout(height=250)
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        fig3 = px.histogram(sub_df, x='age', nbins=30, marginal='rug',
                            title="Distribución de Edad")
        fig3.update_layout(height=250)
        st.plotly_chart(fig3, use_container_width=True)
