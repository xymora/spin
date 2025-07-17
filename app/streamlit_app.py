import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging
# -----------------------------------
# Configuración de logging
logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuración de la página
st.set_page_config(
    page_title="Dashboard de Clientes Bancarios",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🏦 Dashboard de Clientes Bancarios")("🏦 Dashboard de Clientes Bancarios")

# ======================================
# Carga y preprocesamiento de los datos
# ======================================
DATA_URL = "https://covenantaegis.com/segmentation_data_recruitment.csv"

@st.cache_data(show_spinner=False)
def load_data(url: str) -> pd.DataFrame:
    """
    Carga el dataset, convierte tipos y rellena NA con 0.
    """
    df = pd.read_csv(url)
    # Asegurar numérico
    for col in ['avg_amount_withdrawals', 'avg_purchases_per_week']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

# Intentar cargar datos
try:
    df = load_data(DATA_URL)
    logger.info(f"Datos cargados correctamente: {len(df)} registros")
except Exception as e:
    logger.error(f"Error al cargar datos: {e}")
    st.error(f"Error al cargar datos: {e}")
    st.stop()

# ======================================
# Informe General (KPI iniciales)
# ======================================
total_clients = len(df)
avg_withdrawals = df['avg_amount_withdrawals'].mean()
avg_purchases = df['avg_purchases_per_week'].mean()
avg_age = df['age'].mean()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Clientes", f"{total_clients:,}")
col2.metric("Retiro Promedio", f"${avg_withdrawals:,.2f}")
col3.metric("Compras/Semana Prom.", f"{avg_purchases:.2f}")
col4.metric("Edad Promedio", f"{avg_age:.1f}")

# ======================================
# Análisis Exploratorio Rápido
# ======================================
st.subheader("🔎 Análisis Exploratorio")
# Mostrar primeras filas
st.write(df.head())
# Tipos y valores faltantes
dtypes = pd.DataFrame({
    'dtype': df.dtypes,
    'missing': df.isna().sum()
})
st.write(dtypes)
# Matriz de correlación
corr = df.select_dtypes(include=[np.number]).corr()
fig_heat = px.imshow(
    corr, text_auto=True, title="Mapa de Correlación Numérica"
)
st.plotly_chart(fig_heat, use_container_width=True)

# ======================================
# Feature Engineering: Credit Score
# ======================================
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

# Binning de compras semanales
df['compras_binned'] = pd.cut(
    df['avg_purchases_per_week'], bins=[0,1,2,3,4,5,np.inf],
    labels=["0","1","2","3","4","5+"], right=False
)

# ======================================
# Sidebar: Filtros y Búsqueda
# ======================================
st.sidebar.header("🔍 Filtros y Búsqueda")
# Filtro por Credit Score
credit_order = ['🔵 Premium Credit','🟢 Basic Credit','🟡 Moderate Risk','🔴 High Risk']
avail = [c for c in credit_order if c in df['credit_score'].unique()]
selected_scores = st.sidebar.multiselect(
    "Credit Score", avail, default=avail
)
# Búsqueda de usuario exacto
if 'user_search' not in st.session_state:
    st.session_state['user_search'] = ''
if 'search_active' not in st.session_state:
    st.session_state['search_active'] = False
st.sidebar.text_input(
    "👤 Usuario exacto:", key='user_search'
)
colA, colB = st.sidebar.columns(2)
if colA.button("Buscar"):
    st.session_state['search_active'] = True
if colB.button("Borrar"):
    st.session_state['user_search'] = ''
    st.session_state['search_active'] = False

# ======================================
# Filtrado de datos
# ======================================
df_filtered = df[df['credit_score'].isin(selected_scores)].copy()
if st.session_state.search_active and st.session_state.user_search:
    df_filtered = df_filtered[df_filtered['user']==st.session_state.user_search]
st.markdown(f"Filtrados: **{len(df_filtered):,}** de **{total_clients:,}** clientes")

# ======================================
# Vista de datos filtrados
# ======================================
st.subheader("📋 Clientes mostrados")
base_cols = ['user','age','index','credit_score','user_type','registration_channel','creation_flow','creation_date','avg_amount_withdrawals']
dep_cols = [c for c in df_filtered.columns if 'deposit' in c.lower()]
for c in dep_cols:
    if c not in base_cols: base_cols.append(c)
other = sorted([c for c in df_filtered.columns if c not in base_cols])
st.dataframe(df_filtered[base_cols+other], use_container_width=True)

# Botón para exportar datos filtrados
csv_data = df_filtered.to_csv(index=False).encode('utf-8')
st.download_button(
    "💾 Descargar CSV", data=csv_data, file_name='clientes_filtrados.csv'
)

# ======================================
# Gráficas principales
# ======================================
# Distribución por Credit Score
if selected_scores:
    cnt = df_filtered['credit_score'].value_counts().reindex(credit_order, fill_value=0)
    fig1 = px.bar(x=cnt.index, y=cnt.values, color=cnt.index, text=cnt.values,
                  title='Distribución por Credit Score')
    fig1.update_traces(textposition='outside')
    fig1.update_layout(showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

# Análisis Financiero por Score
st.subheader("📊 Análisis Financiero por Credit Score")
for score in selected_scores:
    sub = df_filtered[df_filtered['credit_score']==score]
    if sub.shape[0]>0:
        st.markdown(f"### {score}")
        c1,c2,c3 = st.columns(3)
        with c1:
            fig2 = px.line(sub.sort_values('avg_amount_withdrawals'), y='avg_amount_withdrawals',
                           title='Retiros promedio', height=250)
            st.plotly_chart(fig2, use_container_width=True)
        with c2:
            v = sub['compras_binned'].value_counts().sort_index()
            fig3 = px.bar(x=v.index.astype(str), y=v.values, text=v.values,
                          title='Compras/semana', labels={'x':'Compras','y':'Cantidad'}, height=250)
            fig3.update_traces(textposition='outside')
            st.plotly_chart(fig3, use_container_width=True)
        with c3:
            if sub['age'].size>1:
                kde = gaussian_kde(sub['age'])
                xk = np.linspace(sub['age'].min(), sub['age'].max(),100)
                yk = kde(xk)
                fig4 = px.area(x=xk, y=yk, title='Distribución de edad', height=250)
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.write('Distribución de edad: solo un valor')

# ======================================
# Clustering con K-Means (K=4)
# ======================================
st.subheader("🤖 Agrupamiento K-Means (K=4)")
feat = ['avg_amount_withdrawals','avg_purchases_per_week','age']
scaled = StandardScaler().fit_transform(df[feat])
km = KMeans(n_clusters=4, random_state=42).fit(scaled)
df['cluster']=km.labels_
fig5 = px.scatter_3d(df, x='avg_amount_withdrawals', y='avg_purchases_per_week', z='age',
                     color='cluster', title='Clustering 3D')
st.plotly_chart(fig5, use_container_width=True)

logger.info("Dashboard renderizado exitosamente.")
# ======================================
# Fin del Dashboard de Data Science
# ======================================
