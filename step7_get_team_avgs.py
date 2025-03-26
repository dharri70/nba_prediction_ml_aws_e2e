import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
from sqlalchemy import create_engine


def get_team_stats(url: str, table_id: str) -> pd.DataFrame:
    """
    Fetch and parse the specified stats table from the given URL using exact column names.
    """
    response = requests.get(url, timeout=10)
    response.raise_for_status()  # Ensure we got a valid response
    
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': table_id})
    
    if not table:
        raise ValueError(f"Table with id '{table_id}' not found on the page.")

    # Extract headers: Use visible column names from <th> elements
    headers = []
    data_stat_map = {}
    
    for th in table.find('thead').find_all('th'):
        data_stat = th.get('data-stat', '').strip()
        column_name = th.text.strip()  # Use the actual column name from the webpage
        
        if data_stat:  # Ensure we only include columns with a valid data-stat
            headers.append(column_name)
            data_stat_map[data_stat] = column_name  # Map data-stat to column names
    
    # Extract table rows using 'data-stat' mapping
    rows = []
    for row in table.find('tbody').find_all('tr'):
        row_data = {column_name: None for column_name in headers}  # Initialize with None
        
        for td in row.find_all(['th', 'td']):
            data_stat = td.get('data-stat', '').strip()
            if data_stat in data_stat_map:  # Ensure only valid columns are included
                row_data[data_stat_map[data_stat]] = td.text.strip()
        
        rows.append(row_data)

    # Create DataFrame with corrected column names
    df = pd.DataFrame(rows, columns=headers)
    
    return df

def main():
    base_url = "https://www.basketball-reference.com/leagues/NBA_2025.html"
    
    # Fetch per-game stats using exact column names
    per_game_df = get_team_stats(base_url, "per_game-team")

    # Fetch advanced stats using exact column names
    advanced_df = get_team_stats(base_url, "advanced-team")
    
    # Check which column exists in both DataFrames
    merge_column = 'Tm' if 'Tm' in per_game_df.columns and 'Tm' in advanced_df.columns else 'Team'

    # Merge using the determined column
    merged_df = pd.merge(per_game_df, advanced_df, on=merge_column, how='inner', suffixes=('_PerGame', '_Advanced'))

    
    # Drop unwanted columns
    columns_to_drop = ["Rk_PerGame", "Offense Four Factors", "Defense Four Factors", "Rk_Advanced", "Arena", "Attend.", "Attend./G"]
    merged_df.drop(columns=[col for col in columns_to_drop if col in merged_df.columns], inplace=True)
    
    # Drop fully blank columns
    merged_df.dropna(axis=1, how='all', inplace=True)
    
    # Drop columns with blank column names
    merged_df = merged_df.loc[:, merged_df.columns.notnull()]
    merged_df.columns = [col if col else f"Unnamed_{i}" for i, col in enumerate(merged_df.columns)]
    merged_df = merged_df.iloc[:, :-5]

    
    # Save merged DataFrame to a single CSV
    csv_filename = "nba_team_stats.csv"
    merged_df.to_csv(csv_filename, index=False)
    print(f"NBA team stats saved to {csv_filename}")
    
    # Database connection using SQLAlchemy for SQL Server with environment variables
    db_user = ""
    db_password = ""
    db_server = ""
    db_name = ""
    
    database_url = f"mssql+pymssql://{db_user}:{db_password}@{db_server}/{db_name}"
    engine = create_engine(database_url)
    
    # Insert data into the database (schema: nba, table: team_average, replace if exists)
    merged_df.to_sql("team_average", engine, schema="nba", if_exists="replace", index=False)
    print("Data successfully inserted into the nba.team_average table.")

if __name__ == "__main__":
    main()
