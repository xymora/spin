import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Bank Clients Dashboard", layout="wide")
st.title("🏦 Bank Clients Dashboard")

# =====================
# Load data
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
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()

df = load_data()
if df.empty:
    st.stop()

# =====================
# Credit Score Classification
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
# Sidebar filters
# =====================
with st.sidebar:
    st.header("🔍 Optional Filters")
    apply_filters = st.checkbox("Apply filters", value=False)

    if apply_filters:
        age_min, age_max = int(df['age'].min()), int(df['age'].max())
        wd_min, wd_max = df['avg_amount_withdrawals'].min(), df['avg_amount_withdrawals'].max()
        pw_min, pw_max = df['avg_purchases_per_week'].min(), df['avg_purchases_per_week'].max()

        # Reordenados como en la imagen
        wd_range = st.slider("Average Withdrawals", float(wd_min), float(wd_max), (float(wd_min), float(wd_max)))
        pw_range = st.slider("Purchases per Week", float(pw_min), float(pw_max), (float(pw_min), float(pw_max)))
        age_range = st.slider("Age", age_min, age_max, (age_min, age_max))

        credit_types = df['credit_score'].unique().tolist()
        selected_types = st.multiselect("Credit Score", sorted(credit_types), default=credit_types)

# =====================
# Apply filters
# =====================
if apply_filters:
    df_filtered = df[
        (df['age'].between(*age_range)) &
        (df['avg_amount_withdrawals'].between(*wd_range)) &
        (df['avg_purchases_per_week'].between(*pw_range)) &
        (df['credit_score'].isin(selected_types))
    ]
else:
    df_filtered = df.copy()

# =====================
# Reorder columns
# =====================
first_cols = [
    'user',
    'age',
    'index',
    'credit_score',
    'user_type',
    'registration_channel',
    'creation_flow',
    'creation_date',
    'avg_amount_withdrawals'
]
other_cols = sorted([col for col in df_filtered.columns if col not in first_cols])
final_cols = first_cols + other_cols
df_display = df_filtered[final_cols]

# =====================
# Show data
# =====================
st.subheader("📋 Displayed Clients")
st.dataframe(df_display, use_container_width=True)
st.markdown(f"🔎 Total displayed: **{len(df_display):,}** / 100,000")
