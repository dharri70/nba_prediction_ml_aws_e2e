import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from sqlalchemy import text, create_engine
import urllib.parse
import joblib
import smtplib
from email.mime.text import MIMEText
import boto3
from io import BytesIO

# AWS S3 Configuration
s3_client = boto3.client('s3',
	aws_access_key_id='',
	aws_secret_access_key=''
)
S3_BUCKET = "your-s3-bucket"
S3_FOLDER = "saved_models/"

# Database connection variables
db_user = ""
db_password = ""
db_host = ""
db_name = ""
db_schema = ""

# Encode password and create engine once
encoded_password = urllib.parse.quote_plus(db_password)
conn_str = f"mssql+pymssql://{db_user}:{encoded_password}@{db_host}/{db_name}"
engine = create_engine(conn_str)

# Email configuration
SMTP_LOGIN = ""
SENDER_EMAIL = ""
SENDER_PASSWORD = ""
RECIPIENTS = [""]
SMTP_SERVER = ""
SMTP_PORT = 587

STATS = [
    'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB',
    'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'eFG%', '3PAr', 'FTr', 'ORB%',
    'TOV%', 'DRtg', 'Pace', 'eFG% (FF)', 'TOV% (FF)', 'ORB% (FF)', 'FT/FGA'
]

def get_current_nba_schedule_url():
    today = datetime.today()
    month = today.strftime('%B').lower()
    year = today.year if today.month < 10 else today.year + 1
    return f"https://www.basketball-reference.com/leagues/NBA_{year}_games-{month}.html"

def fetch_games_today():
    url = get_current_nba_schedule_url()
    today_str = datetime.today().strftime('%a, %b %d, %Y')
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        rows = soup.find_all('tr')
        return [(row.find('td', {'data-stat': 'visitor_team_name'}).text.strip(),
                 row.find('td', {'data-stat': 'home_team_name'}).text.strip())
                for row in rows if row.find('th', {'data-stat': 'date_game'}) and
                row.find('th', {'data-stat': 'date_game'}).text.strip() == today_str]
    except requests.RequestException as e:
        print(f"Error fetching schedule: {e}")
        return []

def query_team_stats(team_name, connection):

    # Check which column exists in the database
    column_check_query = text("""
    SELECT COLUMN_NAME
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME = 'team_average' AND COLUMN_NAME IN ('Tm', 'Team')
    """)

    # Execute the column check query
    with engine.connect() as conn:
        result = conn.execute(column_check_query).fetchone()

# Determine the correct column name
    team_column = result[0] if result else 'Tm'  # Default to 'Tm' if neither is found


    query = text(f"""
        SELECT 
            [FGA], [FG%], [3P], [3PA], [3P%], [FT], [FTA], [FT%], [ORB], [DRB], 
            [TRB], [AST], [STL], [BLK], [TOV], [PF], [eFG%], [3PAr], [FTr], 
            [ORB%], [TOV%], [DRtg], [Pace], [eFG%], [TOV%], [ORB%], [FT/FGA]
        FROM [nba_data].[nba].[team_average]
        WHERE [{team_column}] = :team_name
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
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_FOLDER)
    if 'Contents' not in response:
        print(f"⚠️ No files found in s3://{S3_BUCKET}/{S3_FOLDER}")
        return models, scaler
    for obj in response['Contents']:
        file_key = obj['Key']
        if file_key.endswith('.joblib'):
            file_obj = BytesIO()
            s3_client.download_fileobj(S3_BUCKET, file_key, file_obj)
            file_obj.seek(0)
            if file_key.endswith('feature_scaler.joblib'):
                scaler = joblib.load(file_obj)
            else:
                model_name = file_key.split('/')[-1].replace('.joblib', '')
                models[model_name] = joblib.load(file_obj)
    return models, scaler

def make_predictions(games, connection, models, scaler):
    predictions = []
    if not scaler or not models:
        print("⚠️ Scaler or models missing")
        return predictions
    for visitor, home in games:
        visitor_stats = query_team_stats(visitor, connection)
        home_stats = query_team_stats(home, connection)
        if visitor_stats is None or home_stats is None:
            print(f"⚠️ Skipping game {visitor} vs {home}")
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

def send_email(subject, body):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = SENDER_EMAIL
    msg['To'] = ", ".join(RECIPIENTS)
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_LOGIN, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECIPIENTS, msg.as_string())
        print("✅ Email sent successfully")

def main():
    games_today = fetch_games_today()
    if not games_today:
        print("No games today.")
        return
    models, scaler = load_models_and_scaler()
    predictions = make_predictions(games_today, engine, models, scaler)
    if predictions:
        email_subject = f"NBA Game Predictions - {datetime.today().strftime('%Y-%m-%d')}"
        email_body = format_predictions_for_email(predictions)
        send_email(email_subject, email_body)
    engine.dispose()

if __name__ == "__main__":
    main()
