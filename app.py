import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Page Configuration
st.set_page_config(page_title="Agri-Risk Dashboard", page_icon="🌾", layout="wide")

# 2. Load data from the CSV in your data folder (Cloud-Friendly!)
@st.cache_data
def load_data():
    # Looks inside the 'data' folder we created for the dataset
    df = pd.read_csv("data/final_agri_risk_predictions.csv") 
    return df

df = load_data()

# 3. Build the User Interface (UI)
st.title("🌾 Agricultural Risk & Profitability Interactive Dashboard")
st.markdown("**DWDM Project Web Interface** - Use the sidebar to filter the data in real-time!")

# --- Sidebar Filters ---
st.sidebar.header("🔍 Filter Data")
selected_season = st.sidebar.multiselect("Select Season(s):", df['season'].unique(), default=df['season'].unique())
risk_filter = st.sidebar.radio("Select Risk Level:", ("All", "High Risk Only", "Low Risk Only"))

# --- Apply Filters ---
filtered_df = df[df['season'].isin(selected_season)]
if risk_filter == "High Risk Only":
    filtered_df = filtered_df[filtered_df['is_high_risk'] == 1]
elif risk_filter == "Low Risk Only":
    filtered_df = filtered_df[filtered_df['is_high_risk'] == 0]

# --- Top Level Metrics ---
col1, col2, col3 = st.columns(3)
col1.metric(label="Total Records Analysed", value=f"{len(filtered_df):,}")
col2.metric(label="Average Profit", value=f"₹ {filtered_df['profit'].mean():,.2f}")
col3.metric(label="Average Cost", value=f"₹ {filtered_df['estimated_cost'].mean():,.2f}")

st.divider()

# --- Interactive Charts (Now with Log Scales!) ---
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Top Crops by Average Profit (Log Scale)")
    crop_profit = filtered_df.groupby('crop')['profit'].mean().sort_values(ascending=False).head(10)
    
    fig_crop, ax_crop = plt.subplots(figsize=(6, 4))
    sns.barplot(x=crop_profit.values, y=crop_profit.index, ax=ax_crop, hue=crop_profit.index, palette='viridis', legend=False)
    ax_crop.set_xscale('symlog') # <--- LOG SCALE APPLIED
    ax_crop.set_xlabel("Average Profit (Log Scale)")
    ax_crop.set_ylabel("")
    st.pyplot(fig_crop)

with col_right:
    st.subheader("Incidents by Season (Log Scale)")
    season_counts = filtered_df['season'].value_counts()
    
    fig_season, ax_season = plt.subplots(figsize=(6, 4))
    sns.barplot(x=season_counts.index, y=season_counts.values, ax=ax_season, hue=season_counts.index, palette='magma', legend=False)
    ax_season.set_yscale('log') # <--- LOG SCALE APPLIED
    ax_season.set_ylabel("Number of Incidents (Log Scale)")
    ax_season.set_xlabel("")
    plt.xticks(rotation=45)
    st.pyplot(fig_season)

st.divider()

# --- Raw Data Explorer ---
st.subheader("🗄️ Explore the Dataset")
st.dataframe(filtered_df.head(100), width='stretch')