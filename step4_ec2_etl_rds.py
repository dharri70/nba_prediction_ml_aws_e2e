import pandas as pd
from sqlalchemy import create_engine
import urllib.parse


# Must merge csv files on ec2 instance before completing this step

# Database connection
DB_HOST = ""
DB_NAME = ""
DB_USER = ""
DB_PASSWORD = "" 
TABLE_NAME = ""
db_driver = ""

# conn_str = f"mssql+pyodbc://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}?driver={urllib.parse.quote_plus(db_driver)}"

conn_str = f"mssql+pymssql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
db_engine = create_engine(conn_str)

# Load CSV and insert into RDS

df = pd.read_csv("/mnt/data/merged_data.csv")
df.to_sql(TABLE_NAME, db_engine, schema="nba",  if_exists="append", index=False, chunksize=1000)
print("Data inserted successfully!")
