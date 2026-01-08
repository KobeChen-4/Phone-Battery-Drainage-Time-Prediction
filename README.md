# Battery Drain Prediction

A machine learning project that predicts **daily smartphone battery drain (mAh/day)** based on user behavior and device features.

## Features
- Linear regression model
- User Behavior Class as a major predictive factor
- Android / iOS OS encoding
- Interactive Streamlit web app

## Input Features
- App Usage Time
- Screen On Time
- Number of Apps
- Data Usage
- Age
- User Behavior Class
- Operating System (Android / iOS)

## Model
- Scikit-learn Linear Regression
- Trained with train/test split
- Evaluated using MAE and RMSE

## Run the App Locally
```bash
pip install -r requirements.txt
streamlit run app.py
