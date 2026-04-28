import pandas as pd

print("Cleaning data for PostgreSQL...")
df = pd.read_csv('final_agri_risk_predictions.csv')

# Drop the 17 rows with blank spaces
df = df.dropna()

# Save a squeaky-clean version just for pgAdmin
df.to_csv('final_agri_risk_predictions_CLEAN.csv', index=False)
print("SUCCESS: Saved as final_agri_risk_predictions_CLEAN.csv!")