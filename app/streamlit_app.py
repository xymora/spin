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
    aplicar_filtros = st.checkbox("Aplicar filtros", value=False)

    if aplicar_filtros:
        edad_rango = st.slider("Edad", int(df['age'].min()), int(df['age'].max()), (18, 80))
        retiro_rango = st.slider("Retiros promedio", float(df['avg_amount_withdrawals'].min()), float(df['avg_amount_withdrawals'].max()), (0.0, 100000.0))
        compra_rango = st.slider("Compras por semana", float(df['avg_purchases_per_week'].min()), float(df['avg_purchases_per_week'].max()), (0.0, 10.0))

        orden_credit = [
            '🔵 Premium Credit',
            '🟢 Basic Credit',
            '🟡 Moderate Risk',
            '🔴 High Risk'
        ]
        tipos_credito = [c for c in orden_credit if c in df['credit_score'].unique()]
        seleccionados = st.multiselect("Credit Score", tipos_credito, default=tipos_credito)

# Aplicar filtros
if aplicar_filtros:
    df_filtrado = df[
        (df['age'].between(*edad_rango)) &
        (df['avg_amount_withdrawals'].between(*retiro_rango)) &
        (df['avg_purchases_per_week'].between(*compra_rango)) &
        (df['credit_score'].isin(seleccionados))
    ]
else:
    df_filtrado = df.copy()

# Reordenar columnas
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

# Mostrar datos
st.subheader("📋 Clientes mostrados")
st.dataframe(df_mostrar, use_container_width=True)
st.markdown(f"🔎 Total mostrados: **{len(df_mostrar):,}** / 100,000")

# Gráfica principal de Credit Score
orden_credit = ['🔵 Premium Credit', '🟢 Basic Credit', '🟡 Moderate Risk', '🔴 High Risk']
if aplicar_filtros and seleccionados:
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

    # Análisis financiero individual por tipo
    st.subheader("📊 Análisis Financiero por Credit Score")
    for score in seleccionados:
        sub_df = df_filtrado[df_filtrado['credit_score'] == score]
        st.markdown(f"### {score}")

        col1, col2, col3 = st.columns(3)

        # Línea: Retiros
        with col1:
            fig1 = px.line(sub_df.sort_values('avg_amount_withdrawals').reset_index(),
                           y='avg_amount_withdrawals',
                           title="Retiros Promedio (Gráfica de Línea)")
            fig1.update_layout(height=250, xaxis_title='Cliente', yaxis_title='Retiros')
            st.plotly_chart(fig1, use_container_width=True)

        # Boxplot: Compras por semana
        with col2:
            fig2 = px.box(sub_df, y='avg_purchases_per_week', title="Compras por Semana (Boxplot)")
            fig2.update_layout(height=250, yaxis_title='Compras')
            st.plotly_chart(fig2, use_container_width=True)

        # Pastel: Edades
        with col3:
            bins = pd.cut(sub_df['age'], bins=[0, 20, 30, 40, 50, 60, 70, 80, 100])
            edad_counts = bins.value_counts().sort_index().reset_index()
            edad_counts.columns = ['rango_edad', 'count']
            edad_counts['rango_edad'] = edad_counts['rango_edad'].astype(str)  # ✅ CORRECCIÓN CLAVE
            fig3 = px.pie(edad_counts, values='count', names='rango_edad', title="Distribución de Edad (Pastel)")
            fig3.update_layout(height=250)
            st.plotly_chart(fig3, use_container_width=True)
