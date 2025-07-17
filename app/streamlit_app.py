import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
    for col in ['avg_amount_withdrawals', 'avg_purchases_per_week']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

# Cargar datos
try:
    df = load_data(DATA_URL)
except Exception as e:
    st.error(f"Error al cargar datos: {e}")
    st.stop()

# Clasificación crediticia
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

# Sidebar - filtros y búsqueda de usuario
st.sidebar.header("🔍 Filtros y Búsqueda")

# Filtro por tipo de crédito
credit_order = ['🔵 Premium Credit', '🟢 Basic Credit', '🟡 Moderate Risk', '🔴 High Risk']
available_scores = [c for c in credit_order if c in df['credit_score'].unique()]
selected_scores = st.sidebar.multiselect(
    "Credit Score", available_scores, default=available_scores
)

# Búsqueda de usuario
if 'user_search' not in st.session_state:
    st.session_state['user_search'] = ''
if 'search_active' not in st.session_state:
    st.session_state['search_active'] = False

# Input de usuario exacto
given_user = st.sidebar.text_input(
    "👤 Usuario exacto:",
    value=st.session_state['user_search'],
    key='user_search'
)

# Callbacks y botones
def clear_search():
    st.session_state['user_search'] = ''
    st.session_state['search_active'] = False

search_clicked = st.sidebar.button("Buscar Usuario")
clear_clicked = st.sidebar.button("Borrar Búsqueda", on_click=clear_search)

if search_clicked:
    st.session_state['search_active'] = True

# Filtrar DataFrame por credit score
df_filtered = df[df['credit_score'].isin(selected_scores)].copy()
# Si hay búsqueda de usuario activa, filtrar por ese usuario
if st.session_state.search_active and st.session_state.user_search:
    df_filtered = df_filtered[df_filtered['user'] == st.session_state.user_search]

# Mostrar conteo de filtrados
total_clients = len(df)
st.markdown(f"Filtrados: **{len(df_filtered):,}** de **{total_clients:,}** clientes")

# --------------------------------------
# Vista de tabla de clientes
# --------------------------------------
st.subheader("📋 Clientes mostrados")
if not df_filtered.empty:
    base_cols = [
        'user', 'age', 'index', 'credit_score',
        'user_type', 'registration_channel', 'creation_flow', 'creation_date',
        'avg_amount_withdrawals'
    ]
    deposit_cols = [c for c in df_filtered.columns if 'deposit' in c.lower()]
    for d in deposit_cols:
        if d not in base_cols:
            base_cols.append(d)
    other_cols = sorted([c for c in df_filtered.columns if c not in base_cols])
    display_cols = base_cols + other_cols
    st.dataframe(df_filtered[display_cols], use_container_width=True)
else:
    st.warning("No hay clientes para mostrar con los filtros actuales.")

# --------------------------------------
# Gráfica de distribución por Credit Score
# --------------------------------------
if selected_scores:
    count_df = (
        df_filtered['credit_score']
        .value_counts()
        .reindex(credit_order, fill_value=0)
        .reset_index()
    )
    count_df.columns = ['credit_score', 'count']
    fig = px.bar(
        count_df, x='credit_score', y='count', color='credit_score', text='count',
        title="Distribución de clientes por Credit Score"
    )
    fig.update_layout(showlegend=False, height=400)
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------
# Análisis financiero por tipo de Credit Score
# --------------------------------------
st.subheader("📊 Análisis Financiero por Credit Score")
for score in selected_scores:
    sub = df_filtered[df_filtered['credit_score'] == score]
    if not sub.empty:
        st.markdown(f"### {score}")
        c1, c2, c3 = st.columns(3)
        with c1:
            fig1 = px.line(
                sub.sort_values('avg_amount_withdrawals'),
                y='avg_amount_withdrawals', title="Retiros promedio", height=250
            )
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            counts = sub['compras_binned'].value_counts().sort_index()
            fig2 = px.bar(
                x=counts.index.astype(str), y=counts.values,
                labels={'x':'Compras/semana','y':'Cantidad'}, text=counts.values,
                title="Compras promedio por semana", height=250
            )
            fig2.update_traces(textposition='outside')
            st.plotly_chart(fig2, use_container_width=True)

        with c3:
            # Distribución de edad (KDE) - requiere al menos 2 puntos
            if sub['age'].size > 1:
                kde = gaussian_kde(sub['age'])
                x_vals = np.linspace(sub['age'].min(), sub['age'].max(), 100)
                y_vals = kde(x_vals)
                fig3 = px.area(x=x_vals, y=y_vals, title="Distribución de edad", height=250)
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.write("Distribución de edad no disponible: solo un dato")

# --------------------------------------
# K-Means (constante K=4)
# --------------------------------------
st.subheader("🤖 Agrupamiento Inteligente (K-Means)")
K = 4
features = ['avg_amount_withdrawals', 'avg_purchases_per_week', 'age']
scaled = StandardScaler().fit_transform(df[features])
kmeans = KMeans(n_clusters=K, random_state=42)
clusters = kmeans.fit_predict(scaled)
df['cluster'] = clusters
fig_cluster = px.scatter_3d(
    df, x='avg_amount_withdrawals', y='avg_purchases_per_week', z='age',
    color='cluster', title=f"Agrupamiento K-Means (K={K})", height=500
)
st.plotly_chart(fig_cluster, use_container_width=True)
