import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Dashboard de Clientes Bancarios", layout="wide")
st.title("🏦 Dashboard de Clientes Bancarios")

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

# Filtros
with st.sidebar:
    st.header("🔍 Filtros opcionales")
    apply_filters = st.checkbox("Aplicar filtros", value=False)

    if apply_filters:
        edad_range = st.slider("Edad", int(df['age'].min()), int(df['age'].max()), (18, 80))
        retiro_range = st.slider("Retiros promedio", float(df['avg_amount_withdrawals'].min()), float(df['avg_amount_withdrawals'].max()), (0.0, 100000.0))
        compra_range = st.slider("Compras por semana", float(df['avg_purchases_per_week'].min()), float(df['avg_purchases_per_week'].max()), (0.0, 10.0))

        credit_order = [
            '🔵 Premium Credit',
            '🟢 Basic Credit',
            '🟡 Moderate Risk',
            '🔴 High Risk'
        ]
        credit_types = [c for c in credit_order if c in df['credit_score'].unique()]
        selected_types = st.multiselect("Credit Score", credit_types, default=credit_types)

if apply_filters:
    df_filtered = df[
        (df['age'].between(*edad_range)) &
        (df['avg_amount_withdrawals'].between(*retiro_range)) &
        (df['avg_purchases_per_week'].between(*compra_range)) &
        (df['credit_score'].isin(selected_types))
    ]
else:
    df_filtered = df.copy()

# Mostrar tabla
st.subheader("📋 Clientes mostrados")
st.dataframe(df_filtered, use_container_width=True)
st.markdown(f"🔎 Total mostrados: **{len(df_filtered):,}** / 100,000")

# Gráfica general de credit score
if apply_filters and selected_types:
    count_by_score = df_filtered['credit_score'].value_counts().reindex(credit_order).dropna().reset_index()
    count_by_score.columns = ['credit_score', 'count']
    fig = px.bar(
        count_by_score,
        x='credit_score', y='count',
        color='credit_score', text='count',
        title="Distribución de clientes por tipo de Credit Score",
        color_discrete_sequence=["blue", "green", "gold", "red"]
    )
    fig.update_layout(showlegend=False, height=400)
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

    # Gráficas financieras individuales
    st.subheader("📊 Análisis Financiero por Credit Score")
    for score in selected_types:
        sub_df = df_filtered[df_filtered['credit_score'] == score]
        st.markdown(f"### {score}")

        col1, col2, col3 = st.columns(3)

        with col1:
            fig1 = px.histogram(sub_df, x='avg_amount_withdrawals', nbins=20, title="Retiros Promedio")
            fig1.update_layout(height=250)
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = px.histogram(sub_df, x='avg_purchases_per_week', nbins=10, title="Compras por Semana")
            fig2.update_layout(height=250)
            st.plotly_chart(fig2, use_container_width=True)

        with col3:
            rc_counts = sub_df['registration_channel'].value_counts().reset_index()
            rc_counts.columns = ['channel', 'count']
            fig3 = px.bar(rc_counts, x='channel', y='count', title="Canal de Registro")
            fig3.update_layout(height=250)
            st.plotly_chart(fig3, use_container_width=True)
