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
def classify_credit(withdrawals, purchases):
    if withdrawals > 50000 and purchases == 0:
        return '🔵 Premium Credit'
    elif withdrawals > 20000 and purchases <= 1:
        return '🟢 Basic Credit'
    elif withdrawals > 10000:
        return '🟡 Moderate Risk'
    else:
        return '🔴 High Risk'

df['credit_score'] = df.apply(
    lambda row: classify_credit(row['avg_amount_withdrawals'], row['avg_purchases_per_week']),
    axis=1
)

# =====================
# Filtro por credit_score
# =====================
with st.sidebar:
    st.header("🔍 Filtros opcionales")
    orden_credit = [
        '🔵 Premium Credit',
        '🟢 Basic Credit',
        '🟡 Moderate Risk',
        '🔴 High Risk'
    ]
    tipos_credito = [c for c in orden_credit if c in df['credit_score'].unique()]
    seleccionados = st.multiselect("Credit Score", tipos_credito, default=tipos_credito)

df_filtrado = df[df['credit_score'].isin(seleccionados)]

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
st.subheader("📋 Clientes mostrados")
st.dataframe(df_mostrar, use_container_width=True)
st.markdown(f"🔎 Total mostrados: **{len(df_mostrar):,}** / 100,000")

# =====================
# Gráfica principal
# =====================
if seleccionados:
    conteo = df_filtrado['credit_score'].value_counts().reindex(orden_credit).dropna().reset_index()
    conteo.columns = ['credit_score', 'count']
    fig = px.bar(
        conteo,
        x='credit_score', y='count',
        color='credit_score', text='count',
        title="Distribución de clientes por tipo de Credit Score",
        color_discrete_sequence=["blue", "green", "gold", "red"]
    )
    fig.update_layout(showlegend=False, height=400)
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Análisis Financiero por Credit Score")
    for score in seleccionados:
        sub_df = df_filtrado[df_filtrado['credit_score'] == score]
        st.markdown(f"### {score}")

        col1, col2, col3 = st.columns(3)

        with col1:
            sub_df_sorted = sub_df.sort_values('avg_amount_withdrawals').reset_index(drop=True)
            sub_df_sorted['cliente'] = sub_df_sorted.index
            fig1 = px.line(
                sub_df_sorted,
                x='cliente',
                y='avg_amount_withdrawals',
                markers=True,
                title="Retiros Promedio (Línea)"
            )
            fig1.update_traces(line=dict(color='brown'), marker=dict(color='green', size=6))
            fig1.update_layout(height=250, xaxis_title='Clientes ordenados', yaxis_title='Retiros')
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = px.histogram(sub_df, x='avg_purchases_per_week', nbins=10, title="Compras por Semana")
            fig2.update_layout(height=250)
            st.plotly_chart(fig2, use_container_width=True)

        with col3:
            fig3 = px.histogram(sub_df, x='age', nbins=20, title="Distribución de Edad")
            fig3.update_layout(height=250)
            st.plotly_chart(fig3, use_container_width=True)
