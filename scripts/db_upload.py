import pandas as pd
from sqlalchemy import create_engine

print("Loading clean data...")
df = pd.read_csv('final_agri_risk_predictions_CLEAN.csv')

# Connect directly to your PostgreSQL server
# Format: postgresql://username:password@localhost:5432/database_name
# Note: 'postgres' is the default username. Just swap out your password!
engine = create_engine('postgresql://postgres:Ayush%40141819@localhost:5432/agri_risk_db')

print("Pushing data to the Data Warehouse (this might take 10-20 seconds)...")
# This command completely bypasses pgAdmin. It creates the table AND inserts the data perfectly.
df.to_sql('fact_crop_risk', engine, if_exists='replace', index=False)

print("SUCCESS: 100% of the data is now safely in your PostgreSQL database!")