import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import urllib.parse
from sklearn.model_selection import train_test_split
import pymssql
import logging
import os
import time
import numpy as np
import joblib
from typing import Dict, List, Tuple
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import requests
import urllib.parse
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText

# Database connection variables
db_user = ""
db_password = ""
db_host = ""  # Example: "nba-database.xyz.us-east-1.rds.amazonaws.com"
db_name = ""
db_schema = ""  # Change if needed

# Encode password (to handle special characters)
encoded_password = urllib.parse.quote_plus(db_password)

# Create connection string for SQLAlchemy
conn_str = f"mssql+pymssql://{db_user}:{encoded_password}@{db_host}/{db_name}"

# Create SQLAlchemy engine
engine = create_engine(conn_str)

# Load NBA data from SQL Server into a Pandas DataFrame
query = f"SELECT * FROM {db_schema}.game_data"
df = pd.read_sql(query, engine)


print("Dropped Columns:", df.columns[:6])
print("Included Columns:", df.columns[6:])

df = df.iloc[:, 6:]
df = df.drop(columns=['Total Points'])

# Close connection
engine.dispose()

# First dataframe (Regular X/Y split)
#X1 = df.drop(columns=['PTS'])
#y1 = df['PTS']

# Second dataframe (Dropping specified columns in X)
X2 = df.drop(columns=['PTS', 'ORtg', 'ORtg (FF)', 'FG', 'TS%', 'Q1 Points', 'Q2 Points', 'Q3 Points', 'Q4 Points', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'USG%'])
y2 = df['PTS']

# Train/Test split for both dataframes
#X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Show Completion
print("Variables are loaded")

# Initialize the scaler
scaler1 = StandardScaler()
scaler2 = StandardScaler()

# Fit and transform the training data, transform the test data
#X1_train_scaled = scaler1.fit_transform(X1_train)
#X1_test_scaled = scaler1.transform(X1_test)

X2_train_scaled = scaler2.fit_transform(X2_train)
X2_test_scaled = scaler2.transform(X2_test)


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_models() -> Dict[str, object]:
    """Define regression models."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.01),
        "SGDRegressor": SGDRegressor(max_iter=1000, tol=1e-3, random_state=42),
        "Support Vector Regression": SVR(kernel='linear', C=100, epsilon=0.1, max_iter=5000),
        "Multi-Layer Perceptron": MLPRegressor(hidden_layer_sizes=(50,), max_iter=200, random_state=42)
    }

def train_and_evaluate(models: Dict[str, object], X_train: np.ndarray, X_test: np.ndarray, 
                      y_train: np.ndarray, y_test: np.ndarray, dataset_name: str) -> List[Tuple]:
    """Train models, evaluate performance, and save trained models and scaler."""
    results = []
    model_dir = "saved_models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    scaler_filename = os.path.join(model_dir, "feature_scaler.joblib")
    joblib.dump(scaler, scaler_filename)
    logging.info(f"Saved feature scaler to {scaler_filename}")
    
    for name, model in models.items():
        logging.info(f"Training & Evaluating: {name} on {dataset_name}")
        start_time = time.time()
        
        try:
            model.fit(X_train_scaled, y_train)
            model_filename = os.path.join(model_dir, f"{name.replace(' ', '_').lower()}.joblib")
            joblib.dump(model, model_filename)
            logging.info(f"Saved {name} model to {model_filename}")

            y_pred = model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            results.append((name, dataset_name, mae, mse, rmse, r2))
            
            logging.info(f"{name} completed in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logging.error(f"Error training {name}: {e}")
    
    return results

def print_results(results: List[Tuple]):
    """Print evaluation results in a formatted table."""
    header = f"{'Model':<30}{'Dataset':<40}{'MAE':<10}{'MSE':<15}{'RMSE':<15}{'R² Score':<10}"
    print(f"\n{header}")
    print("=" * 120)
    for name, dataset, mae, mse, rmse, r2 in results:
        print(f"{name:<30}{dataset:<40}{mae:<10.4f}{mse:<15.4f}{rmse:<15.4f}{r2:<10.4f}")
    print("\n")


"""Main execution function using Feature-Reduced dataset."""
dataset = ("Feature-Reduced X/Y Split Dataset", (X2_train, X2_test, y2_train, y2_test))  # Use unscaled data here
models = get_models()
results = train_and_evaluate(models, *dataset[1], dataset[0])
print_results(results)

STATS = [
    'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB',
    'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'eFG%', '3PAr', 'FTr', 'ORB%',
    'TOV%', 'DRtg', 'Pace', 'eFG% (FF)', 'TOV% (FF)', 'ORB% (FF)', 'FT/FGA'
]

MODEL_DIR = "saved_models"  # Adjust as needed
# engine = create_engine(...)  # Add your database connection

# Email configuration (update these)
SMTP_LOGIN = ""  # Used for authentication
SENDER_EMAIL = ""  # Must be a verified sender in Brevo
SENDER_PASSWORD = ""  # Your SMTP key from Brevo
RECIPIENTS = ""
SMTP_SERVER = ""
SMTP_PORT = 587

def get_current_nba_schedule_url():
    today = datetime.today()
    month = today.strftime('%B').lower()
    year = today.year if today.month < 10 else today.year + 1
    return f"https://www.basketball-reference.com/leagues/NBA_{year}_games-{month}.html"

def fetch_games_today():
    url = get_current_nba_schedule_url()
    #today_str = (datetime.today() - timedelta(days=1)).strftime('%a, %b %d, %Y')
    today_str = datetime.today().strftime('%a, %b %d, %Y')
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        rows = soup.find_all('tr')
        games_today = [(row.find('td', {'data-stat': 'visitor_team_name'}).text.strip(),
                       row.find('td', {'data-stat': 'home_team_name'}).text.strip())
                      for row in rows if row.find('th', {'data-stat': 'date_game'}) and
                      row.find('th', {'data-stat': 'date_game'}).text.strip() == today_str]
        return games_today
    except requests.RequestException as e:
        print(f"Error fetching schedule: {e}")
        return []

def query_team_stats(team_name, connection):
    query = text("""
        SELECT 
            [FGA], [FG%], [3P], [3PA], [3P%], [FT], [FTA], [FT%], [ORB], [DRB], 
            [TRB], [AST], [STL], [BLK], [TOV], [PF], [eFG%], [3PAr], [FTr], 
            [ORB%], [TOV%], [DRtg], [Pace], [eFG%], [TOV%], [ORB%], [FT/FGA]
        FROM [nba_data].[nba].[team_average]
        WHERE [Team] = :team_name
    """)
    
    with connection.connect() as conn:
        result = conn.execute(query, {"team_name": team_name}).fetchone()
        if result is None:
            print(f"⚠️ No stats found for team: {team_name}")
            return None
        print(f"✅ Retrieved stats for {team_name}")
        return result

def load_models_and_scaler():
    models = {}
    scaler = None
    for file in os.listdir(MODEL_DIR):
        if file.endswith(".joblib"):
            filepath = os.path.join(MODEL_DIR, file)
            if file == "feature_scaler.joblib":
                scaler = joblib.load(filepath)
            else:
                model_name = file.replace(".joblib", "")
                models[model_name] = joblib.load(filepath)
    return models, scaler

def make_predictions(games, connection, models, scaler):
    predictions = []
    
    if not scaler:
        print("⚠️ Feature scaler not found")
        return predictions
    if not models:
        print("⚠️ No models found")
        return predictions

    for visitor, home in games:
        visitor_stats = query_team_stats(visitor, connection)
        home_stats = query_team_stats(home, connection)

        if visitor_stats is None or home_stats is None:
            print(f"⚠️ Skipping game {visitor} vs {home} due to missing stats")
            continue

        visitor_data = pd.DataFrame([visitor_stats], columns=STATS)
        home_data = pd.DataFrame([home_stats], columns=STATS)

        visitor_data_scaled = scaler.transform(visitor_data)
        home_data_scaled = scaler.transform(home_data)

        game_prediction = {"game": f"{visitor} vs {home}", "visitor": visitor, "home": home}
        
        for model_name, model in models.items():
            visitor_score = model.predict(visitor_data_scaled)[0]
            home_score = model.predict(home_data_scaled)[0]
            game_prediction[f"{visitor}_score_{model_name}"] = visitor_score
            game_prediction[f"{home}_score_{model_name}"] = home_score

        predictions.append(game_prediction)

    return predictions

def format_predictions_for_email(predictions):
    email_body = f"NBA Game Predictions for {datetime.today().strftime('%Y-%m-%d')}\n\n"
    for prediction in predictions:
        visitor = prediction["visitor"]
        home = prediction["home"]
        email_body += f"{'='*60}\n"
        email_body += f"Game: {prediction['game']}\n".center(60)
        email_body += f"{'='*60}\n"
        email_body += f"{'Team':<25} {'Model':<25} {'Predicted Score':<15}\n"
        email_body += f"{'-'*25} {'-'*25} {'-'*15}\n"
        
        visitor_scores = {k: v for k, v in prediction.items() if k.startswith(f"{visitor}_score_")}
        for key, score in visitor_scores.items():
            model_name = key.replace(f"{visitor}_score_", "").replace("_", " ").title()
            email_body += f"{visitor:<25} {model_name:<25} {score:>10.1f}\n"
        
        home_scores = {k: v for k, v in prediction.items() if k.startswith(f"{home}_score_")}
        for key, score in home_scores.items():
            model_name = key.replace(f"{home}_score_", "").replace("_", " ").title()
            email_body += f"{home:<25} {model_name:<25} {score:>10.1f}\n"
        
        email_body += f"{'='*60}\n\n"
    return email_body

def send_email(subject, body, smtp_login, sender_email, recipients, password):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender_email  # Use the verified sender
    msg['To'] = ", ".join(recipients)
    
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(smtp_login, password)  # Use SMTP login for auth
            server.sendmail(sender_email, recipients, msg.as_string())
            print("✅ Email sent successfully")
    except Exception as e:
        print(f"⚠️ Error sending email: {e}")


def main():
    games_today = fetch_games_today()
    if not games_today:
        print("No games today.")
        return
        
    try:
        models, scaler = load_models_and_scaler()
        predictions = make_predictions(games_today, engine, models, scaler)
        print_formatted_predictions(predictions)
        email_subject = f"NBA Game Predictions - {datetime.today().strftime('%Y-%m-%d')}"
        email_body = format_predictions_for_email(predictions)
        send_email(email_subject, email_body, SMTP_LOGIN, SENDER_EMAIL, RECIPIENTS, SENDER_PASSWORD)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
            engine.dispose()
        except NameError:
            pass

def print_formatted_predictions(predictions):  # Keep this for console output
    for prediction in predictions:
        visitor = prediction["visitor"]
        home = prediction["home"]
        print(f"\n{'='*60}")
        print(f"Game: {prediction['game']}".center(60))
        print(f"{'='*60}")
        print(f"{'Team':<25} {'Model':<25} {'Predicted Score':<15}")
        print(f"{'-'*25} {'-'*25} {'-'*15}")
        visitor_scores = {k: v for k, v in prediction.items() if k.startswith(f"{visitor}_score_")}
        for key, score in visitor_scores.items():
            model_name = key.replace(f"{visitor}_score_", "").replace("_", " ").title()
            print(f"{visitor:<25} {model_name:<25} {score:>10.1f}")
        home_scores = {k: v for k, v in prediction.items() if k.startswith(f"{home}_score_")}
        for key, score in home_scores.items():
            model_name = key.replace(f"{home}_score_", "").replace("_", " ").title()
            print(f"{home:<25} {model_name:<25} {score:>10.1f}")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

