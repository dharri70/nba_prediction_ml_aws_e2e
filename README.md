# NBA Score Prediction System

## Overview
This project is an **end-to-end automated pipeline** for predicting NBA game scores using **machine learning and AWS cloud services**. It collects historical game data, processes and cleans the data, trains multiple regression models, and generates daily predictions.

## Features
- **Automated Data Collection**: Uses Scrapy to gather historical NBA game data from 1976 to the present.
- **Cloud-Based Data Storage**: Stores raw and processed data in AWS services (EFS, RDS, S3).
- **Machine Learning Models**: Implements multiple regression models to predict game scores.
- **Daily Predictions**: Fetches the latest team stats, applies models, and emails predictions every day at 10:30 AM.
- **Scalable Cloud Infrastructure**: Leverages AWS services for efficient and automated processing.

## Tech Stack
- **Programming Languages**: Python, SQL
- **Cloud Services**: AWS (EFS, EBS, S3, EC2, Lambda, EventBridge, RDS, SageMaker, IAM)
- **Machine Learning Models**:
  - Lasso Regression
  - Linear Regression
  - Multi-Layer Perceptron (MLP)
  - Ridge Regression
  - SGDRegressor
  - Support Vector Regression (SVR)
- **Email Service**: Brevo

## Data Pipeline
1. **Data Collection**:
   - Scrapy collects all relevant URLs for NBA game data.
   - Scrapy extracts game data from 1976 onward via an AWS EC2 instance.
   - Extracted URLs are stored in AWS RDS to track all scraped URLs.
   - Raw data is stored as CSV files in AWS EFS.
   
2. **Data Processing & Storage**:
   - CSV files from EFS are merged into a single dataset.
   - Data is inserted into AWS RDS after merging.
   - Cleaning process removes incomplete records.
   
3. **Model Training & Prediction**:
   - Feature engineering is performed to optimize model accuracy and avoid overfitting.
   - Machine learning models are trained on historical data via AWS SageMaker Studio.
   - Trained models are stored in AWS S3.
   - A script fetches daily team stats, applies models, and predicts scores.
   
4. **Automated Prediction & Notification**:
   - An AWS Lambda script runs daily via AWS EventBridge to:
     - Collect the day's games
     - Retrieve current team stats
     - Apply machine learning models
     - Send prediction results via email at 12:15 PM EST.
   
## Areas for Improvement
- **Expand Data Inputs**: Incorporate player-level statistics to refine predictions.
- **Reinforcement Learning**: Collect game outcomes to improve model learning over time.
- **Sports Betting Analysis**: Integrate sportsbook data for additional predictive insights.
- **Model Performance Tracking**: Implement a system to track model accuracy over time.

## Next Steps
- Develop an **automated accuracy tracking system** to measure prediction performance.
- Implement reinforcement learning techniques for continuous model improvement.
- Enhance prediction models with **real-time injury reports and betting lines**.
- Use different seasonal data to create more diverse, and potentially more accurate models.

## Getting Started
### Prerequisites
- Python 3.x
- AWS Account with configured IAM roles
- Required Python libraries 


## Contact
For questions or collaborations, feel free to reach out or connect with me on [LinkedIn] https://www.linkedin.com/in/danielcharris01/.

---

This `README.md` ensures clarity and professionalism while highlighting the complexity of your project. Let me know if you’d like any modifications!
