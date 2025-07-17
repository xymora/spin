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
            # Línea con marcadores
            bins = np.histogram_bin_edges(sub_df['avg_amount_withdrawals'], bins=15)
            counts, edges = np.histogram(sub_df['avg_amount_withdrawals'], bins=bins)
            midpoints = 0.5 * (edges[1:] + edges[:-1])

            fig1 = px.line(
                x=midpoints,
                y=counts,
                title="Retiros Promedio (Distribución)",
                markers=True
            )
            fig1.update_traces(line=dict(color='brown', width=3), marker=dict(color='green', size=8))
            fig1.update_layout(xaxis_title="Promedio de Retiros", yaxis_title="Frecuencia", height=250)
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            # Barras separadas para compras por semana
            compras_df = sub_df['avg_purchases_per_week'].round(0).astype(int)
            compras_count = compras_df.value_counts().sort_index().reset_index()
            compras_count.columns = ['compras', 'frecuencia']
            fig2 = px.bar(compras_count, x='compras', y='frecuencia', title="Compras por Semana (Barras Separadas)")
            fig2.update_layout(xaxis_title="Compras por semana", yaxis_title="Frecuencia", height=250)
            st.plotly_chart(fig2, use_container_width=True)

        with col3:
            fig3 = px.histogram(sub_df, x='age', nbins=20, title="Distribución de Edad")
            fig3.update_layout(height=250)
            st.plotly_chart(fig3, use_container_width=True)
