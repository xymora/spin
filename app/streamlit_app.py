import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans import KMeans
from sklearn.preprocessing import StandardScaler
import logging

# -----------------------------------
# Configuración de logging
# -----------------------------------
logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# -----------------------------------
# Configurar Kaleido para exportar imágenes
# -----------------------------------
import plotly.io as pio
try:
    pio.kaleido.scope.default_format = "png"
except Exception as e:
    logger.warning(f"No se pudo configurar Kaleido: {e}")

# -----------------------------------
# Configuración de la página
# -----------------------------------
st.set_page_config(
    page_title="Dashboard de Clientes Bancarios",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🏦 Dashboard de Clientes Bancarios")

# ====================================
# Función de carga de datos con cache
# ====================================
DATA_URL = "https://covenantaegis.com/segmentation_data_recruitment.csv"

@st.cache_data(show_spinner=False)
def load_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    # Asegurar numérico
    for col in ['avg_amount_withdrawals', 'avg_purchases_per_week']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

# -----------------------------------
# Cargar datos y mostrar KPI iniciales
# -----------------------------------
try:
    df = load_data(DATA_URL)
    total_clients = len(df)
    avg_withdrawals = df['avg_amount_withdrawals'].mean()
    avg_purchases = df['avg_purchases_per_week'].mean()
    avg_age = df['age'].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Clientes", f"{total_clients:,}")
    col2.metric("Retiro Promedio", f"${avg_withdrawals:,.2f}")
    col3.metric("Compras/Semana Prom.", f"{avg_purchases:.2f}")
    col4.metric("Edad Promedio", f"{avg_age:.1f} años")

    logger.info(f"Datos cargados: {total_clients} registros")
except Exception as e:
    logger.error(f"Error al cargar datos: {e}")
    st.error(f"Error al cargar datos: {e}")
    st.stop()

# ====================================
# Clasificación crediticia y binning
# ====================================
def classify_credit(df: pd.DataFrame) -> pd.DataFrame:
    conditions = [
        (df['avg_amount_withdrawals'] > 50000) & (df['avg_purchases_per_week'] == 0),
        (df['avg_amount_withdrawals'] > 20000) & (df['avg_purchases_per_week'] <= 1),
        (df['avg_amount_withdrawals'] > 10000)
    ]
    choices = ['🔵 Premium Credit', '🟢 Basic Credit', '🟡 Moderate Risk']
    df['credit_score'] = np.select(conditions, choices, default='🔴 High Risk')
    df['compras_binned'] = pd.cut(
        df['avg_purchases_per_week'], bins=[0,1,2,3,4,5,np.inf],
        labels=["0","1","2","3","4","5+"], right=False
    )
    return df

df = classify_credit(df)

# ====================================
# Sidebar: filtros y búsqueda de usuario
# ====================================
st.sidebar.header("🔍 Filtros y Búsqueda")

# Filtro Credit Score
overall_order = ['🔵 Premium Credit','🟢 Basic Credit','🟡 Moderate Risk','🔴 High Risk']
avail_scores = [c for c in overall_order if c in df['credit_score'].unique()]
selected_scores = st.sidebar.multiselect("Credit Score", avail_scores, default=avail_scores)

# Búsqueda de usuario exacto
if 'user_search' not in st.session_state:
    st.session_state['user_search'] = ''
if 'search_active' not in st.session_state:
    st.session_state['search_active'] = False
user_input = st.sidebar.text_input("👤 Usuario exacto", key='user_search')
btn1, btn2 = st.sidebar.columns(2)

# Botones de búsqueda con keys únicas
def clear_search():
    st.session_state['user_search'] = ''
    st.session_state['search_active'] = False

search_clicked = btn1.button("Buscar", key='search_btn')
clear_clicked = btn2.button("Borrar", key='clear_btn', on_click=clear_search)

if search_clicked:
    st.session_state['search_active'] = True

# ====================================
# Filtrar DataFrame
# ====================================
df_filtered = df[df['credit_score'].isin(selected_scores)].copy()
if st.session_state['search_active'] and st.session_state['user_search']:
    df_filtered = df_filtered[df_filtered['user'] == st.session_state['user_search']]

st.markdown(f"Filtrados: **{len(df_filtered):,}** de **{len(df):,}** clientes")

# ====================================
# Vista de tabla de clientes
# ====================================
st.subheader("📋 Clientes mostrados")
if not df_filtered.empty:
    base_cols = ['user','age','index','credit_score','user_type','registration_channel','creation_flow','creation_date','avg_amount_withdrawals']
    dep_cols = [c for c in df_filtered.columns if 'deposit' in c.lower()]
    for dc in dep_cols:
        if dc not in base_cols:
            base_cols.append(dc)
    other_cols = sorted([c for c in df_filtered.columns if c not in base_cols])
    st.dataframe(df_filtered[base_cols+other_cols], use_container_width=True)
else:
    st.warning("No hay clientes para mostrar con los filtros actuales.")("💾 Descargar CSV", data=csv, file_name='clientes_filtrados.csv')

# ====================================
# Gráficas
# ====================================
# Distribución por Credit Score
if selected_scores:
    cnt = df_filtered['credit_score'].value_counts().reindex(overall_order, fill_value=0)
    fig = px.bar(x=cnt.index, y=cnt.values, color=cnt.index, text=cnt.values,
                 title='Distribución por Credit Score')
    fig.update_layout(showlegend=False)
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

# Análisis Financiero
st.subheader("📊 Análisis Financiero por Credit Score")
for score in selected_scores:
    sub = df_filtered[df_filtered['credit_score']==score]
    if len(sub)>0:
        st.markdown(f"### {score}")
        a,b,c = st.columns(3)
        with a:
            fig1 = px.line(sub.sort_values('avg_amount_withdrawals'), y='avg_amount_withdrawals', title='Retiros promedio', height=250)
            st.plotly_chart(fig1, use_container_width=True)
        with b:
            vc = sub['compras_binned'].value_counts().sort_index()
            fig2 = px.bar(x=vc.index.astype(str), y=vc.values, text=vc.values,
                          title='Compras/semana', labels={'x':'Compras','y':'Cant.'}, height=250)
            fig2.update_traces(textposition='outside')
            st.plotly_chart(fig2, use_container_width=True)
        with c:
            if len(sub['age'])>1:
                kde = gaussian_kde(sub['age'])
                xs = np.linspace(sub['age'].min(), sub['age'].max(),100)
                ys = kde(xs)
                fig3 = px.area(x=xs, y=ys, title='Distribución de edad', height=250)
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.write('Distribución de edad: solo un valor')

# ====================================
# Clustering (K=4)
# ====================================
st.subheader("🤖 Clustering K-Means (K=4)")
features = ['avg_amount_withdrawals','avg_purchases_per_week','age']
scaled = StandardScaler().fit_transform(df[features])
km = KMeans(n_clusters=4, random_state=42).fit(scaled)
df['cluster'] = km.labels_
fig4 = px.scatter_3d(df, x='avg_amount_withdrawals', y='avg_purchases_per_week', z='age', color='cluster', title='Clustering 3D')
st.plotly_chart(fig4, use_container_width=True)

logger.info("Dashboard renderizado exitosamente.")
# ====================================
# Fin del Dashboard de Data Science
# ====================================
