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
# Filtros laterales
# =====================
with st.sidebar:
    st.header("🔍 Filtros opcionales")
    apply_filters = st.checkbox("Aplicar filtros", value=False)

    if apply_filters:
        edad_min, edad_max = int(df['age'].min()), int(df['age'].max())
        retiro_min, retiro_max = df['avg_amount_withdrawals'].min(), df['avg_amount_withdrawals'].max()
        compra_min, compra_max = df['avg_purchases_per_week'].min(), df['avg_purchases_per_week'].max()

        edad_range = st.slider("Edad", edad_min, edad_max, (edad_min, edad_max))
        retiro_range = st.slider("Retiros promedio", float(retiro_min), float(retiro_max), (float(retiro_min), float(retiro_max)))
        compra_range = st.slider("Compras por semana", float(compra_min), float(compra_max), (float(compra_min), float(compra_max)))

        ordered_scores = ['🔵 Premium Credit', '🟢 Basic Credit', '🟡 Moderate Risk', '🔴 High Risk']
        available_scores = [score for score in ordered_scores if score in df['credit_score'].unique()]
        selected_types = st.multiselect("Credit Score", available_scores, default=available_scores)

# =====================
# Aplicar filtros
# =====================
if apply_filters:
    df_filtered = df[
        (df['age'].between(*edad_range)) &
        (df['avg_amount_withdrawals'].between(*retiro_range)) &
        (df['avg_purchases_per_week'].between(*compra_range)) &
        (df['credit_score'].isin(selected_types))
    ]
else:
    df_filtered = df.copy()

# =====================
# Reordenar columnas
# =====================
first_cols = [
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
other_cols = sorted([col for col in df_filtered.columns if col not in first_cols])
final_cols = first_cols + other_cols
df_display = df_filtered[final_cols]

# =====================
# Mostrar datos
# =====================
st.subheader("📋 Clientes mostrados")
st.dataframe(df_display, use_container_width=True)
st.markdown(f"🔎 Total mostrados: **{len(df_display):,}** / 100,000")

# =====================
# Gráfica Credit Score
# =====================
if apply_filters and selected_types:
    count_by_score = df_filtered['credit_score'].value_counts().reindex(ordered_scores).dropna().reset_index()
    count_by_score.columns = ['credit_score', 'count']

    fig = px.bar(
        count_by_score,
        x='credit_score',
        y='count',
        color='credit_score',
        text='count',
        title='Distribución de clientes por tipo de Credit Score',
        color_discrete_sequence=["blue", "green", "gold", "red"]
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis_title="Credit Score",
        yaxis_title="Cantidad de Clientes",
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)
