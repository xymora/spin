import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Configuración de la página
st.set_page_config(page_title="Dashboard de Clientes Bancarios", layout="wide")
st.title("🏦 Dashboard de Clientes Bancarios")

DATA_URL = "https://covenantaegis.com/segmentation_data_recruitment.csv"

@st.cache_data(show_spinner=False)
def load_data(url: str) -> pd.DataFrame:
    """
    Carga y formatea el DataFrame desde la URL.
    Convierte columnas clave a numérico y rellena NA con 0.
    Detecta y convierte columnas de depósito si existen.
    """
    df = pd.read_csv(url)
    # Convertir columnas de retiros y compras
    for col in ['avg_amount_withdrawals', 'avg_purchases_per_week']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    # Detectar y convertir columnas de depósito promedio
    deposit_cols = [c for c in df.columns if 'deposit' in c.lower()]
    for col in deposit_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

# Cargar datos
df = load_data(DATA_URL)
if df.empty:
    st.error("No se pudo cargar la base de datos.")
    st.stop()

# Clasificación crediticia usando vectorización
def classify_credit(df: pd.DataFrame) -> pd.DataFrame:
    conditions = [
        (df['avg_amount_withdrawals'] > 50000) & (df['avg_purchases_per_week'] == 0),
        (df['avg_amount_withdrawals'] > 20000) & (df['avg_purchases_per_week'] <= 1),
        (df['avg_amount_withdrawals'] > 10000)
    ]
    choices = ['🔵 Premium Credit', '🟢 Basic Credit', '🟡 Moderate Risk']
    df['credit_score'] = np.select(conditions, choices, default='🔴 High Risk')
    return df

df = classify_credit(df)

# Preparar categorías de compras antes de filtrar
df['compras_binned'] = pd.cut(
    df['avg_purchases_per_week'],
    bins=[0, 1, 2, 3, 4, 5, np.inf],
    labels=["0", "1", "2", "3", "4", "5 o más"],
    right=False
)

# Sidebar de filtros
st.sidebar.header("🔍 Filtros opcionales")
credit_order = ['🔵 Premium Credit', '🟢 Basic Credit', '🟡 Moderate Risk', '🔴 High Risk']
available_scores = [c for c in credit_order if c in df['credit_score'].unique()]
selected_scores = st.sidebar.multiselect("Credit Score", available_scores, default=available_scores)

# Input de usuario y botones
# Text input para el nombre de usuario con session_state
if 'user_input' not in st.session_state:
    st.session_state.user_input = ''
st.sidebar.text_input(
    "👤 Ingresar usuario exacto",
    key="user_input"
)

# Botón Borrar: solo limpia el campo de texto
def clear_text():
    st.session_state.user_input = ''
st.sidebar.button("Borrar", on_click=clear_text)

# Botón Buscar: dispara la visualización de detalles
search = st.sidebar.button("Buscar")

# Filtrar DataFrame
filtered_df = df[df['credit_score'].isin(selected_scores)].copy()

# Mostrar tabla de clientes
# Incluir columna de depósitos si existe
table_cols = [
    'user', 'age', 'index', 'credit_score',
    'user_type', 'registration_channel', 'creation_flow',
    'creation_date', 'avg_amount_withdrawals'
]
deposit_cols = [c for c in filtered_df.columns if 'deposit' in c.lower()]
for dcol in deposit_cols:
    if dcol not in table_cols:
        table_cols.append(dcol)
other_cols = sorted([col for col in filtered_df.columns if col not in table_cols])
df_display = filtered_df[table_cols + other_cols]

st.subheader("📋 Clientes mostrados")
st.dataframe(df_display, use_container_width=True)
st.markdown(f"🔎 Total mostrados: **{len(df_display):,}** / {len(df):,}")

# Distribución general por Credit Score
if selected_scores:
    count_df = (
        filtered_df['credit_score']
        .value_counts()
        .reindex(credit_order, fill_value=0)
        .reset_index()
    )
    count_df.columns = ['credit_score', 'count']
    fig = px.bar(
        count_df, x='credit_score', y='count', color='credit_score', text='count',
        title="Distribución de clientes por tipo de Credit Score"
    )
    fig.update_layout(showlegend=False, height=400)
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

# Análisis financiero por tipo de Credit Score
st.subheader("📊 Análisis Financiero por Credit Score")
for score in selected_scores:
    st.markdown(f"### {score}")
    sub = filtered_df[filtered_df['credit_score'] == score]
    col1, col2, col3 = st.columns(3)

    with col1:
        fig1 = px.line(
            sub.sort_values('avg_amount_withdrawals'),
            y='avg_amount_withdrawals',
            title="Retiros promedio",
            height=250
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        counts = sub['compras_binned'].value_counts().sort_index()
        fig2 = px.bar(
            x=counts.index.astype(str),
            y=counts.values,
            labels={'x': 'Compras/semana', 'y': 'Cantidad'},
            text=counts.values,
            title="Compras promedio por semana",
            height=250
        )
        fig2.update_traces(textposition='outside')
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        ages = sub['age']
        kde = gaussian_kde(ages)
        x_vals = np.linspace(ages.min(), ages.max(), 100)
        y_vals = kde(x_vals)
        fig3 = px.area(x=x_vals, y=y_vals, title="Distribución de edad", height=250)
        st.plotly_chart(fig3, use_container_width=True)

# Agrupamiento K-Means
st.subheader("🤖 Agrupamiento Inteligente (K-Means)")
features = ['avg_amount_withdrawals', 'avg_purchases_per_week', 'age']
scaled = StandardScaler().fit_transform(df[features])
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(scaled)
df['cluster'] = labels

fig_cluster = px.scatter_3d(
    df,
    x='avg_amount_withdrawals',
    y='avg_purchases_per_week',
    z='age',
    color='cluster',
    title="Agrupamiento de Clientes (K-Means)",
    height=500
)
st.plotly_chart(fig_cluster, use_container_width=True)

# Visualización individual por usuario
if search and st.session_state.user_input:
    user_df = df[df['user'] == st.session_state.user_input]
    if user_df.empty:
        st.warning("Usuario no encontrado.")
    else:
        st.markdown(f"## 📌 Detalles del usuario `{st.session_state.user_input}`")
        # Primera fila de métricas
        m1, m2, m3 = st.columns(3)
        m1.metric("Edad", int(user_df['age'].iloc[0]))
        m2.metric("Índice", int(user_df['index'].iloc[0]))
        m3.metric("Credit Score", user_df['credit_score'].iloc[0])

        # Segunda fila de métricas
        m4, m5, m6 = st.columns(3)
        m4.metric("Retiros promedio", f"${user_df['avg_amount_withdrawals'].iloc[0]:,.2f}")
        m5.metric("Compras x semana", round(user_df['avg_purchases_per_week'].iloc[0], 2))
        m6.metric("Tipo de usuario", user_df['user_type'].iloc[0])

        st.markdown("### 📈 Gráficas personales")
        r1c1, r1c2 = st.columns(2)
        r2c1, r2c2 = st.columns(2)
        user_figs = [
            px.bar(user_df, x='user', y='avg_amount_withdrawals', title="Retiros promedio"),
            px.pie(user_df, names='credit_score', title="Tipo de Score"),
            px.scatter(user_df, x='age', y='avg_purchases_per_week', title="Edad vs Compras"),
            px.line(user_df, x='user', y='index', title="Índice del cliente")
        ]
        for col, fig in zip([r1c1, r1c2, r2c1, r2c2], user_figs):
            col.plotly_chart(fig, use_container_width=True)
