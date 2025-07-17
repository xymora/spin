import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

# Configuración inicial
st.set_page_config(page_title="Dashboard de Clientes Bancarios", layout="wide")
st.title("📊 Dashboard de Clientes Bancarios")

# =====================
# Cargar datos
# =====================
DATA_URL = "https://covenantaegis.com/segmentation_data_recruitment.csv"

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_URL)
        df['avg_amount_withdrawals'] = pd.to_numeric(df['avg_amount_withdrawals'], errors='coerce')
        df['avg_purchase_per_week'] = pd.to_numeric(df['avg_purchase_per_week'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"No se pudo cargar la base de datos: {e}")
        return pd.DataFrame()

df = load_data()
if df.empty:
    st.stop()

# =====================
# Filtros
# =====================
with st.sidebar:
    st.subheader("🔍 Filtros opcionales")
    credit_filter = st.multiselect(
        "Credit Score",
        options=df["credit_score"].unique(),
        default=df["credit_score"].unique(),
        format_func=lambda x: x.replace("_", " ").title()
    )
    user_filter = st.text_input("Buscar por usuario exacto")
    search = st.button("Buscar")

# =====================
# Visualización Global
# =====================
filtered_df = df[df["credit_score"].isin(credit_filter)]

# Distribución por edad (campana de Gauss)
density = gaussian_kde(filtered_df["age"].dropna())
x_vals = np.linspace(filtered_df["age"].min(), filtered_df["age"].max(), 200)
y_vals = density(x_vals)
fig1 = px.area(x=x_vals, y=y_vals, labels={"x": "Edad", "y": "Densidad"})
fig1.update_layout(title="Distribución de edades")

# Compras por semana (gráfico de barras)
avg_by_credit = filtered_df.groupby("credit_score")["avg_purchase_per_week"].mean().reset_index()
fig2 = px.bar(avg_by_credit, x="credit_score", y="avg_purchase_per_week", text_auto='.2s')
fig2.update_layout(title="Compras promedio por semana por tipo de crédito")

# Retiros promedio (línea)
fig3 = px.line(filtered_df.sort_values("avg_amount_withdrawals"), y="avg_amount_withdrawals")
fig3.update_layout(title="Retiros promedio por cliente")

st.plotly_chart(fig1, use_container_width=True)
st.plotly_chart(fig2, use_container_width=True)
st.plotly_chart(fig3, use_container_width=True)

# =====================
# Visualización individual si se busca un usuario
# =====================
if search and user_filter:
    user_df = df[df["user"] == user_filter]
    if user_df.empty:
        st.warning("Usuario no encontrado.")
    else:
        st.markdown(f"## 📌 Datos del usuario: `{user_filter}`")
        col1, col2, col3 = st.columns(3)

        col1.metric("Edad", int(user_df["age"].values[0]))
        col2.metric("Índice", int(user_df["index"].values[0]))
        col3.metric("Tipo de Crédito", user_df["credit_score"].values[0])

        col1.metric("Retiros promedio", f"${user_df['avg_amount_withdrawals'].values[0]:,.2f}")
        col2.metric("Compras por semana", user_df["avg_purchase_per_week"].values[0])

        st.markdown("### Visualización individual")

        fig_a = px.bar(user_df, x="user", y="age", title="Edad del usuario")
        fig_b = px.bar(user_df, x="user", y="index", title="Índice del usuario")
        fig_c = px.bar(user_df, x="user", y="score", title="Historial crediticio")
        fig_d = px.scatter(user_df, x="user", y="avg_amount_withdrawals", size="avg_amount_withdrawals", title="Retiros promedio")
        fig_e = px.area(user_df, x="user", y="avg_purchase_per_week", title="Compras por semana")

        st.plotly_chart(fig_a, use_container_width=True)
        st.plotly_chart(fig_b, use_container_width=True)
        st.plotly_chart(fig_c, use_container_width=True)
        st.plotly_chart(fig_d, use_container_width=True)
        st.plotly_chart(fig_e, use_container_width=True)
