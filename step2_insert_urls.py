import pyodbc

# ✅ RDS Connection Details
server = ""
database = ""
username = ""
password = ""
driver = ""  # Ensure this matches your system

# ✅ File Path for URLs
file_path = "boxscore_urls.txt"  # Ensure this file is in the correct directory

print("🚀 Starting the URL insertion process...")

# ✅ Read URLs from File
try:
    with open(file_path, "r") as file:
        urls = [line.strip() for line in file if line.strip()]
    print(f"✅ Loaded {len(urls)} URLs from the file.")
except FileNotFoundError:
    print("❌ Error: The file nba_urls.txt was not found.")
    exit()


# ✅ Initialize connection variables
conn = None
cursor = None

# ✅ Connect to RDS
try:
    print("🔗 Connecting to AWS RDS...")
    conn = pyodbc.connect(
        f"DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}"
    )
    cursor = conn.cursor()
    print("✅ Connected to RDS successfully!")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    exit()

# ✅ Insert URLs into RDS
inserted_count = 0
failed_count = 0

print("📤 Inserting URLs into the database...")

for index, url in enumerate(urls, start=1):
    try:
        cursor.execute(
            "INSERT INTO nba.game_urls (game_url) VALUES (?)",
            (url,),
        )
        inserted_count += 1
        if index % 1000 == 0:  # Print progress every 1000 rows
            print(f"✅ {index}/{len(urls)} URLs inserted...")
    except Exception as e:
        failed_count += 1
        print(f"❌ Failed to insert URL {index}: {url} → Error: {e}")

# ✅ Commit and Close Connection
conn.commit()
cursor.close()
conn.close()

print(f"\n🎯 Process Completed!")
print(f"✅ Successfully inserted: {inserted_count} URLs")
print(f"❌ Failed inserts: {failed_count}")
