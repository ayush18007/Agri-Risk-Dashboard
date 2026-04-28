from sqlalchemy import create_engine, text

print("Connecting to Data Warehouse...")
engine = create_engine('postgresql://postgres:Ayush%40141819@localhost:5432/agri_risk_db')

print("Generating OLAP Views...")
# We use a transaction to safely create the views
with engine.begin() as conn:
    
    # View 1: High-Risk Crop Analysis
    conn.execute(text("""
        CREATE OR REPLACE VIEW vw_high_risk_analysis AS
        SELECT crop, season, 
               ROUND(AVG(estimated_cost)::numeric, 2) as avg_cost, 
               ROUND(AVG(profit)::numeric, 2) as avg_profit,
               COUNT(*) as incident_count
        FROM fact_crop_risk
        WHERE is_high_risk = 1
        GROUP BY crop, season
        ORDER BY incident_count DESC;
    """))
    
    # View 2: Seasonal Profitability Cube
    conn.execute(text("""
        CREATE OR REPLACE VIEW vw_seasonal_summary AS
        SELECT season, 
               COUNT(*) as total_plantings, 
               ROUND(AVG(profit)::numeric, 2) as average_profit
        FROM fact_crop_risk
        GROUP BY season
        ORDER BY average_profit DESC;
    """))

print("SUCCESS: OLAP Views have been permanently added to your Data Warehouse!")