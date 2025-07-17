import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    silhouette_score, precision_recall_curve
)
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import time

# ======================================
# Dashboard Senior Data Science - 300+ Lines
# ======================================

# --------------------------------------
# 1. Configuración Global de Streamlit
# --------------------------------------
st.set_page_config(
    page_title="Dashboard Clientes Bancarios | DS Senior",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(
    """
    <style>
        .css-1d391kg {padding-top: 1rem;}
        .css-18ni7ap h1 {font-size:2.5rem; color: #003366;}
        .stButton>button {width: 100%;}
        .stDownloadButton>button {background-color: #004C99; color: white;}
    </style>
    """, unsafe_allow_html=True
)
st.title("🏦 Dashboard Clientes Bancarios | Data Science Senior")

# --------------------------------------
# 2. Logging para seguimiento de errores
# --------------------------------------
logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --------------------------------------
# 3. Funciones Auxiliares
# --------------------------------------
def timer(func):
    """Decorator para medir tiempo de ejecución de funciones."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(f"{func.__name__} ejecutada en {end-start:.2f}s")
        return result
    return wrapper

@timer
def load_and_preprocess(url: str) -> pd.DataFrame:
    """
    Carga el dataset, convierte tipos y genera nuevas variables.
    """
    df = pd.read_csv(url)
    # Conversión de tipos numéricos
    numeric_cols = [c for c in df.columns if c.startswith('avg_') or 'age' in c or 'index' in c]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Crear variables adicionales
    if 'avg_amount_purchases' in df.columns:
        df['withdrawal_to_purchase_ratio'] = (
            df['avg_amount_withdrawals'] /
            (df['avg_amount_purchases'] + 1)
        )
    else:
        df['withdrawal_to_purchase_ratio'] = df['avg_amount_withdrawals']

    df['activity_score'] = (
        df['avg_purchases_per_week'] * 0.6 + df['avg_amount_withdrawals'] / 10000 * 0.4
    )

    # Clasificación crediticia
    conditions = [
        (df['avg_amount_withdrawals'] > 50000) & (df['avg_purchases_per_week'] == 0),
        (df['avg_amount_withdrawals'] > 20000) & (df['avg_purchases_per_week'] <= 1),
        (df['avg_amount_withdrawals'] > 10000)
    ]
    choices = ['🔵 Premium Credit','🟢 Basic Credit','🟡 Moderate Risk']
    df['credit_score'] = np.select(conditions, choices, default='🔴 High Risk')

    # Binning de compras semanales
    df['compras_binned'] = pd.cut(
        df['avg_purchases_per_week'], bins=[0,1,2,3,4,5,np.inf],
        labels=["0","1","2","3","4","5+"], right=False
    )
    return df

# --------------------------------------
# 4. Cargar Datos
# --------------------------------------
DATA_URL = "https://covenantaegis.com/segmentation_data_recruitment.csv"
df = load_and_preprocess(DATA_URL)
if df.empty:
    st.error("❌ Error al cargar datos.")
    st.stop()
logger.info(f"Datos cargados: {len(df)} registros")

# --------------------------------------
# 5. KPIs Generales
# --------------------------------------
st.header("📊 KPIs Generales del Dataset")
total_clients = len(df)
avgs = df[['avg_amount_withdrawals','avg_amount_purchases','avg_purchases_per_week','age','withdrawal_to_purchase_ratio','activity_score']].mean()
cols = st.columns(6)
cols[0].metric("Total Clientes", f"{total_clients:,}")
cols[1].metric("Retiro Medio", f"${avgs['avg_amount_withdrawals']:,.0f}")
cols[2].metric("Compra Media", f"${avgs['avg_amount_purchases']:,.0f}")
cols[3].metric("Compras/Semana", f"{avgs['avg_purchases_per_week']:.2f}")
cols[4].metric("Ratio Retiro/Compra", f"{avgs['withdrawal_to_purchase_ratio']:.2f}")
cols[5].metric("Activity Score", f"{avgs['activity_score']:.2f}")

# --------------------------------------
# 6. Filtros Avanzados Sidebar
# --------------------------------------
st.sidebar.header("🔍 Filtros y Segmentación")
# Expander para filtros avanzados
with st.sidebar.expander("Configuración de filtros"):
    credit_order = ['🔵 Premium Credit','🟢 Basic Credit','🟡 Moderate Risk','🔴 High Risk']
    selected_scores = st.multiselect(
        "Credit Score", credit_order, default=credit_order
    )
    age_min, age_max = st.slider(
        "Rango de Edad", int(df['age'].min()), int(df['age'].max()),
        (int(df['age'].min()), int(df['age'].max()))
    )
    score_min, score_max = st.slider(
        "Rango Activity Score", float(df['activity_score'].min()), float(df['activity_score'].max()),
        (float(df['activity_score'].min()), float(df['activity_score'].max()))
    )
    n_clusters = st.selectbox("Número de Cluster (K-Means)", [2,3,4,5,6,7], index=2)

# Aplicar filtros
mask = (
    df['credit_score'].isin(selected_scores) &
    df['age'].between(age_min, age_max) &
    df['activity_score'].between(score_min, score_max)
)
df_filtered = df[mask].reset_index(drop=True)
st.markdown(f"Filtrados: **{len(df_filtered):,}** de {total_clients:,} clientes")

# --------------------------------------
# 7. Vista de Datos
# --------------------------------------
st.subheader("📋 Vista de Tabla Filtrada")
st.dataframe(
    df_filtered.style.format({
        'avg_amount_withdrawals':'${:,.0f}',
        'avg_amount_purchases':'${:,.0f}',
        'avg_purchases_per_week':'{:.2f}',
        'withdrawal_to_purchase_ratio':'{:.2f}',
        'activity_score':'{:.2f}'
    }),
    use_container_width=True
)

# --------------------------------------
# 8. Visualizaciones Avanzadas
# --------------------------------------
st.subheader("📈 Visualizaciones Avanzadas")
# Boxplot
fig_box = px.box(
    df_filtered, x='credit_score', y='avg_amount_withdrawals',
    color='credit_score', title='Boxplot Retiros por Credit Score'
)
st.plotly_chart(fig_box, use_container_width=True)

# Heatmap correlación
corr = df_filtered.select_dtypes(include=[np.number]).corr()
fig_heat = go.Figure(data=go.Heatmap(
    z=corr.values,
    x=corr.columns,
    y=corr.index,
    colorscale='Viridis'
))
fig_heat.update_layout(title='Mapa de Correlación Numérica', height=500)
st.plotly_chart(fig_heat, use_container_width=True)

# Scatter Matrix
dims = ['avg_amount_withdrawals','avg_amount_purchases','age','activity_score']
fig_scatter = px.scatter_matrix(
    df_filtered, dimensions=dims, color='credit_score', title='Scatter Matrix'
)
st.plotly_chart(fig_scatter, use_container_width=True)

# Elbow Method para K-Means
st.markdown("**Análisis de Codo (Elbow Method) para K-Means**")
inertia = []
K = range(1,10)
for k in K:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(df_filtered[dims])
    inertia.append(km.inertia_)
fig_elbow = px.line(
    x=list(K), y=inertia, markers=True,
    labels={'x':'Número de Clusters','y':'Inercia'},
    title='Elbow Method'
)
st.plotly_chart(fig_elbow, use_container_width=True)

# K-Means con elegidos clusters
st.subheader(f"🤖 Clustering K-Means (K={n_clusters})")
km = KMeans(n_clusters=n_clusters, random_state=42)
df_filtered['cluster'] = km.fit_predict(df_filtered[dims])
fig_cluster = px.scatter_3d(
    df_filtered, x='avg_amount_withdrawals', y='avg_amount_purchases', z='age',
    color='cluster', title='Clustering 3D', height=600
)
st.plotly_chart(fig_cluster, use_container_width=True)

# Silhouette Score
st.markdown("**Silhouette Score:**")
if n_clusters > 1:
    score = silhouette_score(df_filtered[dims], df_filtered['cluster'])
    st.write(f"Silhouette Score para K={n_clusters}: {score:.2f}")
else:
    st.warning("No se puede calcular Silhouette Score para un cluster (K=1)")

# --------------------------------------
# 9. Modelado Predictivo
# --------------------------------------
st.subheader("🤖 Modelado Predictivo: Credit Score")
# Preparar X, y
features = ['avg_amount_withdrawals','avg_amount_purchases','avg_purchases_per_week','age','withdrawal_to_purchase_ratio','activity_score']
X = df_filtered[features]
y = df_filtered['credit_score'].map({c:i for i,c in enumerate(credit_order)})
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Entrenar RandomForest + GradientBoosting
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42)
}

results = {}
for name, model in models.items():
    grid = GridSearchCV(
        model,
        param_grid={'n_estimators':[50,100], 'max_depth':[5,10]},
        cv=3
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    results[name] = report
    st.markdown(f"**{name} - Mejor Parámetros:** {grid.best_params_}")
    st.text(classification_report(y_test, y_pred))

# Comparar F1-score promedio
f1_scores = {name: results[name]['macro avg']['f1-score'] for name in results}
fig_comp = go.Figure(
    data=[go.Bar(x=list(f1_scores.keys()), y=list(f1_scores.values()))]
)
fig_comp.update_layout(title='Comparativa F1-score Macro')
st.plotly_chart(fig_comp, use_container_width=True)

# --------------------------------------
# 10. Export y Feedback
# --------------------------------------
st.subheader("💾 Exportar y Feedback")
csv_data = df_filtered.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Exportar Datos Filtrados CSV",
    data=csv_data,
    file_name='clientes_bancarios_filtrados.csv',
    mime='text/csv'
)
feedback = st.text_area("Comentarios / Feedback:", placeholder="Escribe tu feedback aquí...")
if st.button("Enviar Feedback"):
    st.success("¡Gracias por tu feedback!")

# ======================================
# Fin del Dashboard - Data Science Senior
# ======================================


# EOF

# ----------------------------
# Helper: muestra función adicional
# ----------------------------
def describe_dataset(df):
    """
    Imprime estadísticas descriptivas del dataset.
    """
    desc = df.describe()
    return desc

if st.sidebar.checkbox("Mostrar estadísticos descriptivos"):
    st.subheader("📑 Estadísticos descriptivos")
    st.write(describe_dataset(df))

# ----------------------------
# Línea final de control
# ----------------------------
logger.info("Dashboard renderizado satisfactoriamente.")
