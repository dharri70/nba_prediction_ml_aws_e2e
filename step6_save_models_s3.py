import boto3
import os


s3_bucket = "your-s3-bucket"
s3_folder = "saved_models/"  # Folder in S3 where files will be uploaded
local_folder = "saved_models/"  # Local folder in SageMaker instance

s3 = boto3.client("s3")

for root, dirs, files in os.walk(local_folder):
    for file in files:
        local_file_path = os.path.join(root, file)
        s3_key = os.path.join(s3_folder, file)  # Define S3 object key

        s3.upload_file(local_file_path, s3_bucket, s3_key)
        print(f"Uploaded {file} to s3://{s3_bucket}/{s3_key}")

response = s3.list_objects_v2(Bucket=s3_bucket, Prefix=s3_folder)

if "Contents" in response:
    for obj in response["Contents"]:
        print(obj["Key"])
else:
    print("No files found in S3.")