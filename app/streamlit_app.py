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
# Gráficas de perfil al filtrar un usuario
# ====================================
if st.session_state['search_active'] and st.session_state['user_search'] and len(df_filtered)==1:
    user_df = df_filtered.iloc[0]
    
            # Mostrar info básica
    st.markdown(
        f"**Usuario**: {user_df['user']}  
"
        f"**Edad**: {int(user_df['age'])} años  
"
        f"**Saldo Actual**: ${user_df['index']:,}  
"
        f"**Crédito Aprobado (Índice)**: {int(user_df['index'])}  
"
        f"**Tipo Usuario**: {user_df['user_type']}  
"
        f"**Canal Registro**: {user_df['registration_channel']}  
"
        f"**Fecha Creación**: {user_df['creation_date']}  
"
        f"**Credit Score**: {user_df['credit_score']}"
    )
    # Subtítulo perfil
    st.subheader(f"📈 Perfil del Usuario")
    # Información personal adicional
    st.markdown(
        f"**Edad**: {int(user_df['age'])} años  
"
        f"**Saldo Actual**: ${user_df['index']:,}  "
    )

    # Calcular percentiles del usuario
    pct_retiros = (df['avg_amount_withdrawals'] <= user_df['avg_amount_withdrawals']).mean() * 100
    pct_compras = (df['avg_purchases_per_week'] <= user_df['avg_purchases_per_week']).mean() * 100
    pct_edad = (df['age'] <= user_df['age']).mean() * 100

    # 1) Histograma de retiros
    fig1 = px.histogram(
        df, x='avg_amount_withdrawals', nbins=20,
        title='Distribución Retiros (tu posición)'
    )
    fig1.add_vline(
        x=user_df['avg_amount_withdrawals'],
        line_dash='dash', annotation_text='Tú', annotation_position='top right'
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown(
        f"**Explicación:** El cliente retiró un promedio de **${user_df['avg_amount_withdrawals']:,.2f}**, "
        f"ubicándose en el percentil {pct_retiros:.1f}° de todos los clientes."
    )

    # 2) Histograma de compras/semana
    fig2 = px.histogram(
        df, x='avg_purchases_per_week', nbins=20,
        title='Distribución Compras/Semana (tu posición)'
    )
    fig2.add_vline(
        x=user_df['avg_purchases_per_week'],
        line_dash='dash', annotation_text='Tú', annotation_position='top right'
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown(
        f"**Explicación:** El cliente realiza en promedio **{user_df['avg_purchases_per_week']:.2f}** compras/semana, "
        f"situándose en el percentil {pct_compras:.1f}° frente a la base."
    )

    # 3) Radar chart
    radar_df = pd.DataFrame({
        'Feature': ['Retiros','Compras/Semana','Edad'],
        'Value': [user_df['avg_amount_withdrawals'], user_df['avg_purchases_per_week'], user_df['age']]
    })
    fig3 = px.line_polar(
        radar_df, r='Value', theta='Feature', line_close=True,
        title='Perfil Radar del Usuario'
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown(
        f"**Explicación:** Muestra simultáneamente retiros (${user_df['avg_amount_withdrawals']:,.2f}), "
        f"compras/semana ({user_df['avg_purchases_per_week']:.2f}) y edad ({int(user_df['age'])} años), "
        "para evaluar desequilibrios."
    )

    # 4) Comparativa vs mediana
    medians = df[['avg_amount_withdrawals','avg_purchases_per_week','age']].median()
    comp_df = pd.DataFrame({
        'Metric': ['Retiros','Compras/Semana','Edad'],
        'Usuario': [user_df['avg_amount_withdrawals'], user_df['avg_purchases_per_week'], user_df['age']],
        'Mediana': medians.values
    })
    fig4 = px.bar(
        comp_df, x='Metric', y=['Usuario','Mediana'], barmode='group',
        title='Usuario vs Mediana del Dataset'
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown(
        f"**Explicación:** Para retiros, compras/semana y edad, el cliente versus mediana: "
        f"{user_df['avg_amount_withdrawals']:,.0f} vs {medians['avg_amount_withdrawals']:,.0f}, "
        f"{user_df['avg_purchases_per_week']:.2f} vs {medians['avg_purchases_per_week']:.2f}, "
        f"{user_df['age']} vs {int(medians['age'])}."
    )

    # 5) Pie de Credit Scores global
    score_counts = df['credit_score'].value_counts().reindex(overall_order).reset_index()
    score_counts.columns = ['credit_score','count']
    fig5 = px.pie(
        score_counts, names='credit_score', values='count',
        title='Distribución Global de Credit Scores'
    )
    st.plotly_chart(fig5, use_container_width=True)
    pct_score = (score_counts.set_index('credit_score').loc[user_df['credit_score'],'count'] / len(df)) * 100
    st.markdown(
        f"**Explicación:** El cliente pertenece al {pct_score:.1f}% de usuarios con score `{user_df['credit_score']}` en la base."
    )

# Botón de descarga CSV continua
# ====================================
csv = df_filtered.to_csv(index=False).encode('utf-8')
st.download_button("💾 Descargar CSV", data=csv, file_name='clientes_filtrados.csv')

# ====================================
# Gráficas de perfil al filtrar un usuario
# ====================================
if st.session_state['search_active'] and st.session_state['user_search'] and len(df_filtered)==1:
    user_df = df_filtered.iloc[0]
        # Mostrar info básica
    st.markdown(
        f"**Usuario**: {user_df['user']}  
"
        f"**Edad**: {int(user_df['age'])} años  
"
        f"**Índice**: {int(user_df['index'])}  
"
        f"**Tipo Usuario**: {user_df['user_type']}  
"
        f"**Canal Registro**: {user_df['registration_channel']}  
"
        f"**Fecha Creación**: {user_df['creation_date']}  
"
        f"**Credit Score**: {user_df['credit_score']}"
    )
    # Subtítulo perfil
    st.subheader(f"📈 Perfil de {user_df['user']}")(f"📈 Perfil de {user_df['user']}")

    # Percentiles
    pct_r = (df['avg_amount_withdrawals'] <= user_df['avg_amount_withdrawals']).mean()*100
    pct_c = (df['avg_purchases_per_week'] <= user_df['avg_purchases_per_week']).mean()*100
    pct_a = (df['age'] <= user_df['age']).mean()*100

    # 1) Retiros
    fig1 = px.histogram(df, x='avg_amount_withdrawals', nbins=20)
    fig1.add_vline(x=user_df['avg_amount_withdrawals'], line_dash='dash', annotation_text='Tú')
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown(f"**Explicación:** Retiro medio de ${user_df['avg_amount_withdrawals']:,.2f} (percentil {pct_r:.1f}°).")

    # 2) Compras/semana
    fig2 = px.histogram(df, x='avg_purchases_per_week', nbins=20)
    fig2.add_vline(x=user_df['avg_purchases_per_week'], line_dash='dash', annotation_text='Tú')
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown(f"**Explicación:** Compras promedio {user_df['avg_purchases_per_week']:.2f}/semana (percentil {pct_c:.1f}°).")

    # 3) Radar
    radar = pd.DataFrame({
        'Feature':['Retiros','Compras/Semana','Edad'],
        'Value':[user_df['avg_amount_withdrawals'],user_df['avg_purchases_per_week'],user_df['age']]
    })
    fig3 = px.line_polar(radar, r='Value', theta='Feature', line_close=True)
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("**Explicación:** Muestra en un vistazo retiros, compras y edad del usuario.")

    # 4) vs mediana
    med = df[['avg_amount_withdrawals','avg_purchases_per_week','age']].median()
    comp = pd.DataFrame({
        'Metric':['Retiros','Compras','Edad'],
        'Usuario':[user_df['avg_amount_withdrawals'],user_df['avg_purchases_per_week'],user_df['age']],
        'Mediana':med.values
    })
    fig4 = px.bar(comp, x='Metric', y=['Usuario','Mediana'], barmode='group')
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown("**Explicación:** Compara los valores del usuario contra la mediana del dataset cada métrica.")

    # 5) Score global
    sc = df['credit_score'].value_counts().reindex(overall_order).reset_index()
    sc.columns=['credit_score','count']
    fig5 = px.pie(sc, names='credit_score', values='count')
    st.plotly_chart(fig5, use_container_width=True)
    pct_s = (sc.set_index('credit_score').loc[user_df['credit_score'],'count']/len(df))*100
    st.markdown(f"**Explicación:** El usuario está en el {pct_s:.1f}% del segmento `{user_df['credit_score']}`.")

# ====================================
# Gráficas generales y análisis final
# ====================================
if selected_scores:
    cnt = df_filtered['credit_score'].value_counts().reindex(overall_order, fill_value=0)
    fig = px.bar(x=cnt.index, y=cnt.values, color=cnt.index, text=cnt.values)
    fig.update_traces(textposition='outside')
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

st.subheader("📊 Análisis Financiero por Credit Score")
for score in selected_scores:
    sub=df_filtered[df_filtered['credit_score']==score]
    if len(sub):
        st.markdown(f"### {score}")
        c1,c2,c3=st.columns(3)
        with c1:
            _=px.line(sub.sort_values('avg_amount_withdrawals'),y='avg_amount_withdrawals',title='Retiros').update_layout(height=250)
            st.plotly_chart(_,use_container_width=True)
        with c2:
            vc=sub['compras_binned'].value_counts().sort_index()
            _=px.bar(x=vc.index.astype(str),y=vc.values,text=vc.values,title='Compras/Semana').update_layout(height=250)
            st.plotly_chart(_,use_container_width=True)
        with c3:
            if len(sub['age'])>1:
                kde=gaussian_kde(sub['age'])
                xs=np.linspace(sub['age'].min(),sub['age'].max(),100)
                ys=kde(xs)
                _=px.area(x=xs,y=ys,title='Edad').update_layout(height=250)
                st.plotly_chart(_,use_container_width=True)
            else:
                st.write('Solo un valor de edad')

st.subheader("🤖 Clustering K-Means (K=4)")
ftrs=['avg_amount_withdrawals','avg_purchases_per_week','age']
scaler=StandardScaler().fit_transform(df[ftrs])
mdl=KMeans(n_clusters=4,random_state=42).fit(scaler)
df['cluster']=mdl.labels_
fig6=px.scatter_3d(df,x='avg_amount_withdrawals',y='avg_purchases_per_week',z='age',color='cluster')
st.plotly_chart(fig6,use_container_width=True)

logger.info("Dashboard completo renderizado.")
