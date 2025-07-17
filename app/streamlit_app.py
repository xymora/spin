import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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
# Filtros en Sidebar
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
    st.divider()
    if "usuario_input" not in st.session_state:
        st.session_state.usuario_input = ""
    usuario_input = st.text_input("👤 Ingresar usuario exacto", value=st.session_state.usuario_input, key="usuario_box")
    col_buscar, col_borrar = st.columns([1, 1])
    with col_buscar:
        buscar = st.button("Buscar")
    with col_borrar:
        borrar = st.button("Borrar")

if borrar:
    st.session_state.usuario_input = ""
    st.experimental_rerun()

df_filtrado = df[df['credit_score'].isin(seleccionados)]

# =====================
# Mostrar datos
# =====================
primeras_columnas = [
    'user', 'age', 'index', 'credit_score',
    'user_type', 'registration_channel', 'creation_flow',
    'creation_date', 'avg_amount_withdrawals'
]
otras_columnas = sorted([col for col in df_filtrado.columns if col not in primeras_columnas])
df_mostrar = df_filtrado[primeras_columnas + otras_columnas]

st.subheader("📋 Clientes mostrados")
st.dataframe(df_mostrar, use_container_width=True)
st.markdown(f"🔎 Total mostrados: **{len(df_mostrar):,}** / 100,000")

# =====================
# Gráfica general por Credit Score
# =====================
if seleccionados:
    conteo = df_filtrado['credit_score'].value_counts().reindex(orden_credit).dropna().reset_index()
    conteo.columns = ['credit_score', 'count']
    fig = px.bar(conteo, x='credit_score', y='count', color='credit_score', text='count',
                 title="Distribución de clientes por tipo de Credit Score",
                 color_discrete_sequence=["blue", "green", "gold", "red"])
    fig.update_layout(showlegend=False, height=400)
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

# =====================
# Análisis Financiero por tipo de Credit Score
# =====================
st.subheader("📊 Análisis Financiero por Credit Score")
for score in seleccionados:
    sub_df = df_filtrado[df_filtrado['credit_score'] == score]
    st.markdown(f"### {score}")
    col1, col2, col3 = st.columns(3)

    with col1:
        fig1 = px.histogram(sub_df, x='avg_amount_withdrawals', nbins=20)
        fig1.update_layout(title="Distribución de retiros promedio", height=250)
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
        fig2.update_layout(title="Frecuencia de compras semanales", height=250)
        fig2.update_traces(textposition='outside')
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        kde = gaussian_kde(sub_df['age'])
        x_vals = np.linspace(sub_df['age'].min(), sub_df['age'].max(), 100)
        y_vals = kde(x_vals)
        fig3 = px.area(x=x_vals, y=y_vals)
        fig3.update_layout(title="Edad (Distribución Gaussiana)", height=250)
        st.plotly_chart(fig3, use_container_width=True)

# =====================
# Clustering automático de clientes
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
fig_cluster.update_layout(height=500)
st.plotly_chart(fig_cluster, use_container_width=True)

# =====================
# Visualización individual por usuario
# =====================
if buscar and st.session_state.usuario_input:
    user_data = df[df['user'] == st.session_state.usuario_input]
    if user_data.empty:
        st.warning("Usuario no encontrado.")
    else:
        st.markdown(f"## 📌 Detalles del usuario `{st.session_state.usuario_input}`")
        col1, col2, col3 = st.columns(3)
        col1.metric("Edad", int(user_data['age'].values[0]))
        col2.metric("Índice", int(user_data['index'].values[0]))
        col3.metric("Credit Score", user_data['credit_score'].values[0])
        col1.metric("Retiros promedio", f"${user_data['avg_amount_withdrawals'].values[0]:,.2f}")
        col2.metric("Compras x semana", user_data['avg_purchases_per_week'].values[0])

        st.markdown("### 📈 Gráficas personales")
        row1_col1, row1_col2 = st.columns(2)
        row2_col1, row2_col2 = st.columns(2)

        fig_u1 = px.bar(user_data, x="user", y="avg_amount_withdrawals", title="Retiros promedio")
        fig_u2 = px.bar(user_data, x="user", y="avg_purchases_per_week", title="Compras por semana")
        fig_u3 = px.scatter(user_data, x="age", y="avg_purchases_per_week", title="Edad vs Compras")
        fig_u4 = px.line(user_data, x="user", y="index", title="Índice del cliente")

        row1_col1.plotly_chart(fig_u1, use_container_width=True)
        row1_col2.plotly_chart(fig_u2, use_container_width=True)
        row2_col1.plotly_chart(fig_u3, use_container_width=True)
        row2_col2.plotly_chart(fig_u4, use_container_width=True)
