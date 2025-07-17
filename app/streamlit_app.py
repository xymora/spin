import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import gaussian_kde

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
    'user', 'age', 'index', 'credit_score',
    'user_type', 'registration_channel', 'creation_flow',
    'creation_date', 'avg_amount_withdrawals'
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

        # Retiros promedio: línea
        with col1:
            ordenados = sub_df.sort_values('creation_date')
            fig1 = px.line(
                ordenados,
                x='creation_date',
                y='avg_amount_withdrawals',
                title="Retiros promedio a lo largo del tiempo"
            )
            fig1.update_layout(height=250, xaxis_title="Fecha", yaxis_title="Monto")
            st.plotly_chart(fig1, use_container_width=True)

        # Compras por semana: pastel
        with col2:
            bins = [0, 1, 2, 3, 5, 10]
            labels = ['0', '1', '2', '3-4', '5-9', '10+']
            compras = pd.cut(
                sub_df['avg_purchases_per_week'],
                bins=bins + [np.inf],
                labels=labels,
                right=False
            )
            compras_counts = compras.value_counts().sort_index().reset_index()
            compras_counts.columns = ['compras', 'count']
            fig2 = px.pie(
                compras_counts,
                names='compras',
                values='count',
                title="Promedio de compras semanales"
            )
            fig2.update_layout(height=250)
            st.plotly_chart(fig2, use_container_width=True)

        # Distribución de edad: histograma + KDE
        with col3:
            edades = sub_df['age'].dropna()
            fig3 = px.histogram(
                edades,
                nbins=30,
                histnorm="probability density",
                opacity=0.6,
                title="Distribución de edad"
            )
            kde = gaussian_kde(edades)
            x_vals = np.linspace(edades.min(), edades.max(), 200)
            y_vals = kde(x_vals)

            fig3.add_scatter(x=x_vals, y=y_vals, mode='lines', name='KDE', line=dict(color='firebrick'))
            fig3.update_layout(height=250, xaxis_title="Edad", yaxis_title="Densidad")
            st.plotly_chart(fig3, use_container_width=True)
