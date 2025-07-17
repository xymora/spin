import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

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
# Sidebar
# =====================
with st.sidebar:
    st.header("🔍 Filtros opcionales")

    # Filtro por usuario individual
    user_input = st.text_input("Buscar por usuario exacto", "")
    
    # Filtro por credit_score
    orden_credit = [
        '🔵 Premium Credit',
        '🟢 Basic Credit',
        '🟡 Moderate Risk',
        '🔴 High Risk'
    ]
    tipos_credito = [c for c in orden_credit if c in df['credit_score'].unique()]
    seleccionados = st.multiselect("Credit Score", tipos_credito, default=tipos_credito)

# =====================
# Filtro y visualización por usuario
# =====================
if user_input:
    user_df = df[df['user'] == user_input]
    if user_df.empty:
        st.warning("⚠️ Usuario no encontrado.")
    else:
        st.subheader(f"📌 Información detallada de {user_input}")
        st.dataframe(user_df, use_container_width=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Edad", int(user_df.iloc[0]['age']))
            fig = px.bar(x=["Edad"], y=[user_df.iloc[0]['age']])
            fig.update_layout(title="Edad del usuario", height=250)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.metric("Índice", int(user_df.iloc[0]['index']))
            fig = px.bar(x=["Índice"], y=[user_df.iloc[0]['index']])
            fig.update_layout(title="Índice del usuario", height=250)
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            st.metric("Tipo de Crédito", user_df.iloc[0]['credit_score'])
            fig = px.bar(x=["Score"], y=[1], color=[user_df.iloc[0]['credit_score']])
            fig.update_layout(title="Historial crediticio", height=250, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        col4, col5 = st.columns(2)

        with col4:
            st.metric("Retiros promedio", f"${user_df.iloc[0]['avg_amount_withdrawals']:.2f}")
            fig = px.line(x=[0, 1], y=[0, user_df.iloc[0]['avg_amount_withdrawals']])
            fig.update_layout(title="Retiros promedio", height=250)
            st.plotly_chart(fig, use_container_width=True)

        with col5:
            st.metric("Compras por semana", f"{user_df.iloc[0]['avg_purchases_per_week']:.2f}")
            fig = px.line(x=[0, 1], y=[0, user_df.iloc[0]['avg_purchases_per_week']])
            fig.update_layout(title="Compras por semana", height=250)
            st.plotly_chart(fig, use_container_width=True)

        st.stop()

# =====================
# Filtrar datos por Credit Score
# =====================
df_filtrado = df[df['credit_score'].isin(seleccionados)]

# =====================
# Mostrar tabla
# =====================
st.subheader("📋 Clientes mostrados")
st.dataframe(df_filtrado, use_container_width=True)
st.markdown(f"🔎 Total mostrados: **{len(df_filtrado):,}** / 100,000")

# =====================
# Gráfica por Credit Score
# =====================
if seleccionados:
    conteo = df_filtrado['credit_score'].value_counts().reindex(orden_credit).dropna().reset_index()
    conteo.columns = ['credit_score', 'count']
    fig = px.bar(
        conteo, x='credit_score', y='count',
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
            fig1 = px.line(sub_df.sort_values('avg_amount_withdrawals'), y='avg_amount_withdrawals')
            fig1.update_layout(title="Retiros promedio", height=250)
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            compras_labels = ["0", "1", "2", "3", "4", "5 o más"]
            compras_bins = [0, 1, 2, 3, 4, 5, np.inf]
            sub_df['compras_binned'] = pd.cut(
                sub_df['avg_purchases_per_week'], bins=compras_bins, labels=compras_labels, right=False
            )
            compras_counts = sub_df['compras_binned'].value_counts().sort_index()
            fig2 = px.bar(
                x=compras_counts.index.astype(str),
                y=compras_counts.values,
                labels={'x': 'Compras por semana', 'y': 'Cantidad'},
                text=compras_counts.values
            )
            fig2.update_layout(title="Compras promedio por semana", height=250)
            fig2.update_traces(textposition='outside')
            st.plotly_chart(fig2, use_container_width=True)

        with col3:
            kde = gaussian_kde(sub_df['age'])
            x_vals = np.linspace(sub_df['age'].min(), sub_df['age'].max(), 100)
            y_vals = kde(x_vals)
            fig3 = px.area(x=x_vals, y=y_vals)
            fig3.update_layout(title="Distribución de edad", height=250)
            st.plotly_chart(fig3, use_container_width=True)

# =====================
# Agrupamiento automático
# =====================
st.subheader("🤖 Agrupamiento Inteligente (K-Means)")

features = ['avg_amount_withdrawals', 'avg_purchases_per_week', 'age']
data_for_cluster = df[features].copy()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_for_cluster)

kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

df['cluster'] = clusters

fig_cluster = px.scatter_3d(
    df, x='avg_amount_withdrawals', y='avg_purchases_per_week', z='age',
    color='cluster', title="Agrupamiento de Clientes (K-Means)"
)
st.plotly_chart(fig_cluster, use_container_width=True)
