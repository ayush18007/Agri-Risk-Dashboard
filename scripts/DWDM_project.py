import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

print("1. Loading Datasets...")
df_complete = pd.read_csv('final_complete_crop_dataset.csv')
df_prod = pd.read_csv('crop_production_FIXED.csv').dropna()
# We drop NA here because large Excel conversions sometimes leave thousands of empty rows at the bottom
df_rain = pd.read_csv('rainfall_state_mapped_fixed.csv').dropna(subset=['state_name', 'year']) 

print("2. Cleaning Market Price Data...")
year_cols = [c for c in df_complete.columns if '-' in c]
df_price = df_complete.melt(id_vars=['commodity', 'category', 'variety'], 
                            value_vars=year_cols, 
                            var_name='year_range', 
                            value_name='price')

df_price['crop_year'] = df_price['year_range'].apply(lambda x: int(x.split('-')[0]))
df_price['commodity'] = df_price['commodity'].str.lower().str.strip()
df_prod['crop'] = df_prod['crop'].str.lower().str.strip()

# Average price per commodity per year
df_price_avg = df_price.groupby(['commodity', 'crop_year'])['price'].mean().reset_index()

print("3. Formatting Rainfall Data...")
# Standardize strings
df_rain['state_name'] = df_rain['state_name'].astype(str).str.lower().str.strip()
df_rain['year'] = df_rain['year'].astype(int)

# Select only the columns we need
df_rain = df_rain[['state_name', 'year', 'annual']].rename(columns={
    'year': 'crop_year',
    'annual': 'rainfall_mm'
})

print("4. Merging Yield, Market, and Rainfall Data...")
# First merge: Production + Price (matches on Crop and Year)
df_merged = pd.merge(df_prod, df_price_avg, left_on=['crop', 'crop_year'], right_on=['commodity', 'crop_year'], how='inner')

# Second merge: Add Rainfall (matches on State and Year)
df_merged = pd.merge(df_merged, df_rain, on=['state_name', 'crop_year'], how='left')

# Fill missing rainfall data intelligently using the median rainfall of that state
df_merged['rainfall_mm'] = df_merged.groupby('state_name')['rainfall_mm'].transform(lambda x: x.fillna(x.median()))
# Catch-all for any completely blank states
df_merged['rainfall_mm'] = df_merged['rainfall_mm'].fillna(0)

print("5. Feature Engineering: Financial Risk Target...")
# Simulate Financials
df_merged['revenue'] = df_merged['production'] * df_merged['price']
df_merged['revenue_per_area'] = df_merged['revenue'] / df_merged['area']

# Assume Cost is 85% of the historical median revenue per area for that specific crop
crop_costs = df_merged.groupby('crop')['revenue_per_area'].median() * 0.85
crop_costs.name = 'cost_per_area'

df_merged = df_merged.merge(crop_costs, on='crop', how='left')
df_merged['estimated_cost'] = df_merged['area'] * df_merged['cost_per_area']
df_merged['profit'] = df_merged['revenue'] - df_merged['estimated_cost']

# The Target Variable: 1 if Loss (High Risk), 0 if Profit (Low Risk)
df_merged['is_high_risk'] = (df_merged['profit'] < 0).astype(int)

print("6. Training Machine Learning Model...")
# Select our ML features (Notice 'rainfall_mm' is included!)
features = ['state_name', 'district_name', 'season', 'crop', 'crop_year', 'area', 'yield', 'price', 'rainfall_mm']
X = df_merged[features].copy()
y = df_merged['is_high_risk']

# Encode Categorical Variables
label_encoders = {}
cat_cols = ['state_name', 'district_name', 'season', 'crop']
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
    X, y, df_merged.index, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print("\n--- Model Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("7. Exporting Results...")
# Prepare clean DataFrame to export
results_df = df_merged.loc[indices_test].copy()
results_df['predicted_risk'] = y_pred

# Save the final file
results_df.to_csv('final_agri_risk_predictions.csv', index=False)
print(f"SUCCESS: {len(results_df)} rows saved to 'final_agri_risk_predictions.csv'!")