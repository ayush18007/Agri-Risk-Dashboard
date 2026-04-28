import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns

print("Connecting to database...")
engine = create_engine('postgresql://postgres:Ayush%40141819@localhost:5432/agri_risk_db')

print("Fetching data from Data Warehouse...")
df = pd.read_sql("SELECT * FROM fact_crop_risk", engine)

print("Building advanced dashboard...")
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Data Mining Project: Agricultural Risk & Profitability', fontsize=22, fontweight='bold')

# --- Panel 1: Top 10 Crops by AVERAGE Profit ---
# Changed to mean() instead of sum() for a fairer comparison between crops
top_crops = df.groupby('crop')['profit'].mean().sort_values(ascending=False).head(10)
sns.barplot(x=top_crops.values, y=top_crops.index, ax=axes[0, 0], hue=top_crops.index, legend=False, palette='viridis')
axes[0, 0].set_title('Top 10 Most Profitable Crops (Average)', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Average Profit (Log Scale)')
axes[0, 0].set_ylabel('')
axes[0, 0].set_xscale('symlog') # <--- THIS FIXES THE SQUASHED BARS

# --- Panel 2: Risk Distribution (Pie Chart) ---
risk_counts = df['is_high_risk'].value_counts().rename(index={0: 'Low Risk', 1: 'High Risk'})
axes[0, 1].pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], startangle=90, textprops={'fontsize': 12})
axes[0, 1].set_title('Overall Crop Risk Distribution', fontsize=14, fontweight='bold')

# --- Panel 3: Average Profit by Season ---
season_profit = df.groupby('season')['profit'].mean().sort_values(ascending=False)
sns.barplot(x=season_profit.index, y=season_profit.values, ax=axes[1, 0], hue=season_profit.index, legend=False, palette='magma')
axes[1, 0].set_title('Average Profit by Season', fontsize=14, fontweight='bold')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].set_ylabel('Average Profit (Log Scale)')
axes[1, 0].set_xlabel('')
axes[1, 0].set_yscale('symlog') # <--- THIS FIXES THE SQUASHED SEASON BARS

# --- Panel 4: Revenue vs Cost (Scatter Plot) ---
df_sample = df.sample(n=min(5000, len(df)), random_state=42)
sns.scatterplot(data=df_sample, x='estimated_cost', y='revenue', hue='is_high_risk', palette={0:'#2ecc71', 1:'#e74c3c'}, alpha=0.6, ax=axes[1, 1])
axes[1, 1].set_title('Revenue vs Estimated Cost (Sampled)', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Estimated Cost (Log Scale)')
axes[1, 1].set_ylabel('Revenue (Log Scale)')
axes[1, 1].set_xscale('symlog') # <--- Better visibility for outliers
axes[1, 1].set_yscale('symlog') # <--- Better visibility for outliers

# Fix the legend
handles, labels = axes[1, 1].get_legend_handles_labels()
axes[1, 1].legend(handles=handles, labels=['Low Risk', 'High Risk'], title='Risk Level')

plt.tight_layout()
plt.savefig('DWDM_Advanced_Dashboard.png', dpi=300)
print("SUCCESS: Check your folder for the updated 'DWDM_Advanced_Dashboard.png'!")