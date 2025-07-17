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
    for col in ['avg_amount_withdrawals', 'avg_purchases_per_week']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    deposit_cols = [c for c in df.columns if 'deposit' in c.lower()]
    for col in deposit_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

# Cargar datos
df = load_data(DATA_URL)
if df.empty:
    st.error("No se pudo cargar la base de datos.")
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

# Sidebar de filtros
st.sidebar.header("🔍 Filtros opcionales")
# Filtro por tipo de crédito
credit_order = ['🔵 Premium Credit', '🟢 Basic Credit', '🟡 Moderate Risk', '🔴 High Risk']
available_scores = [c for c in credit_order if c in df['credit_score'].unique()]
selected_scores = st.sidebar.multiselect("Credit Score", available_scores, default=available_scores)

# Input de usuario y botones alineados
# Inicializar estados
if 'user_input' not in st.session_state:
    st.session_state.user_input = ''
if 'search' not in st.session_state:
    st.session_state.search = False

# Caja de texto para usuario
st.sidebar.text_input(
    "👤 Ingresar usuario exacto", key="user_input"
)

# Botones en la misma fila: Buscar (izquierda) y Borrar (derecha)
btn_buscar, btn_borrar = st.sidebar.columns(2)
if btn_buscar.button("Buscar"):
    st.session_state.search = True
if btn_borrar.button("Borrar"):
    st.session_state.user_input = ''
    st.session_state.search = False

# Control de búsqueda
search_active = st.session_state.get('search', False)

# Filtrar DataFrame
filtered_df = df[df['credit_score'].isin(selected_scores)].copy()

# Tabla de clientes
table_cols = ['user', 'age', 'index', 'credit_score', 'user_type',
              'registration_channel', 'creation_flow', 'creation_date', 'avg_amount_withdrawals']
deposit_cols = [c for c in filtered_df.columns if 'deposit' in c.lower()]
for d in deposit_cols:
    if d not in table_cols:
        table_cols.append(d)
other_cols = sorted([c for c in filtered_df.columns if c not in table_cols])
df_display = filtered_df[table_cols + other_cols]

st.subheader("📋 Clientes mostrados")
st.dataframe(df_display, use_container_width=True)
st.markdown(f"🔎 Total mostrados: **{len(df_display):,}** / {len(df):,}")

# Gráfica distribución
if selected_scores:
    count_df = (filtered_df['credit_score'].value_counts()
                .reindex(credit_order, fill_value=0).reset_index())
    count_df.columns = ['credit_score', 'count']
    fig = px.bar(count_df, x='credit_score', y='count', color='credit_score', text='count',
                 title="Distribución de clientes por Credit Score")
    fig.update_layout(showlegend=False, height=400)
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

# Análisis financiero\st.subheader("📊 Análisis Financiero por Credit Score")
for score in selected_scores:
    st.markdown(f"### {score}")
    sub = filtered_df[filtered_df['credit_score'] == score]
    c1, c2, c3 = st.columns(3)
    with c1:
        fig1 = px.line(sub.sort_values('avg_amount_withdrawals'), y='avg_amount_withdrawals',
                       title="Retiros promedio", height=250)
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        counts = sub['compras_binned'].value_counts().sort_index()
        fig2 = px.bar(x=counts.index.astype(str), y=counts.values,
                      labels={'x':'Compras/semana','y':'Cantidad'}, text=counts.values,
                      title="Compras promedio por semana", height=250)
        fig2.update_traces(textposition='outside')
        st.plotly_chart(fig2, use_container_width=True)
    with c3:
        kde = gaussian_kde(sub['age'])
        x_vals = np.linspace(sub['age'].min(), sub['age'].max(), 100)
        y_vals = kde(x_vals)
        fig3 = px.area(x=x_vals, y=y_vals, title="Distribución de edad", height=250)
        st.plotly_chart(fig3, use_container_width=True)

# K-Means\st.subheader("🤖 Agrupamiento (K-Means)")
features = ['avg_amount_withdrawals','avg_purchases_per_week','age']
scaled = StandardScaler().fit_transform(df[features])
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(scaled)
df['cluster'] = labels
fig_cluster = px.scatter_3d(df, x='avg_amount_withdrawals', y='avg_purchases_per_week',
                            z='age', color='cluster', title="Agrupamiento Clientes", height=500)
st.plotly_chart(fig_cluster, use_container_width=True)

# Detalle por usuario
if search_active and st.session_state.user_input:
    user_df = df[df['user'] == st.session_state.user_input]
    if user_df.empty:
        st.warning("Usuario no encontrado.")
    else:
        st.markdown(f"## 📌 Detalles del usuario `{st.session_state.user_input}`")
        m1, m2, m3 = st.columns(3)
        m1.metric("Edad", int(user_df['age'].iat[0]))
        m2.metric("Índice", int(user_df['index'].iat[0]))
        m3.metric("Credit Score", user_df['credit_score'].iat[0])
        # Variables para gráficos
        user_withdrawals = user_df['avg_amount_withdrawals'].iat[0]
        user_purchases_count = user_df['avg_purchases_per_week'].iat[0]
        user_purchases_amount = user_df['avg_amount_purchases'].iat[0]
        user_age = user_df['age'].iat[0]

        st.markdown("### 📈 Perfil Gráfico")
        r1c1, r1c2 = st.columns(2)
        r2c1, r2c2 = st.columns(2)

        # Histograma de retiros con posición del usuario
        fig1 = px.histogram(df, x='avg_amount_withdrawals', nbins=20,
                            title="Histograma Retiros (tu posición)")
        fig1.add_vline(x=user_withdrawals, line_dash="dash",
                       annotation_text="Tú",
                       annotation_position="top right")
        r1c1.plotly_chart(fig1, use_container_width=True)

        # Histograma de compras por semana con posición del usuario
        fig2 = px.histogram(df, x='avg_purchases_per_week', nbins=20,
                            title="Histograma Compras/Semana (tu posición)")
        fig2.add_vline(x=user_purchases_count, line_dash="dash",
                       annotation_text="Tú",
                       annotation_position="top right")
        r1c2.plotly_chart(fig2, use_container_width=True)

        # Comparativa Retiro vs Compra (monto)
        comp_df = pd.DataFrame({
            'Tipo': ['Retiros', 'Compras'],
            'Promedio': [user_withdrawals, user_purchases_amount]
        })
        fig3 = px.bar(comp_df, x='Tipo', y='Promedio',
                      title="Comparativa Retiro vs Compra")
        r2c1.plotly_chart(fig3, use_container_width=True)

        # Distribución de edad con marcador del usuario
        fig4 = px.histogram(df, x='age', nbins=20,
                            title="Distribución de Edad (tu posición)")
        fig4.add_vline(x=user_age, line_dash="dash",
                       annotation_text="Tú",
                       annotation_position="top right")
        r2c2.plotly_chart(fig4, use_container_width=True)
