import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

print("Connecting to Data Warehouse...")
engine = create_engine('postgresql://postgres:Ayush%40141819@localhost:5432/agri_risk_db')

print("Fetching data for Machine Learning...")
df = pd.read_sql("SELECT crop, season, estimated_cost, is_high_risk FROM fact_crop_risk", engine)

print("Preprocessing data...")
X = pd.get_dummies(df[['crop', 'season', 'estimated_cost']], drop_first=True)
y = df['is_high_risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Random Forest AI Model...")
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
rf_model.fit(X_train, y_train)

print("Generating Predictions...")
y_pred = rf_model.predict(X_test)

print("Plotting ML Results...")
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Data Mining: Random Forest Predictive Model', fontsize=18, fontweight='bold')

# --- Panel 1: Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
axes[0].set_title('Confusion Matrix (AI Accuracy)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Predicted Risk')
axes[0].set_ylabel('Actual Risk')
axes[0].set_xticklabels(['Low Risk', 'High Risk'])
axes[0].set_yticklabels(['Low Risk', 'High Risk'])

# --- Panel 2: Feature Importance ---
importances = rf_model.feature_importances_
feature_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False).head(10)
sns.barplot(x='Importance', y='Feature', data=feature_df, ax=axes[1], palette='rocket', hue='Feature', legend=False)
axes[1].set_title('Top 10 Drivers of Crop Risk', fontsize=14, fontweight='bold')
axes[1].set_xscale('log') # <--- THE FIX: Applying Log Scale to see smaller features!
axes[1].set_xlabel('Relative Importance (Log Scale)')

plt.tight_layout()
plt.savefig('ML_Analysis_Dashboard.png', dpi=300)
print("SUCCESS: Check your folder for the updated 'ML_Analysis_Dashboard.png'!")