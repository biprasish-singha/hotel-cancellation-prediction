# 🏨 Hotel Cancellation Prediction

A machine learning project that predicts whether a hotel booking is likely to be cancelled based on customer and booking information.

## 🚀 Live Demo

👉 [Try the Hotel Cancellation Prediction App](https://hotel-cancellation-prediction-ctizcvsr5xgfm8kdscqra2.streamlit.app/)

## 📌 Project Overview

The objective of this project is to build a machine learning model that predicts hotel booking cancellations.

The project covers:

- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Handling categorical variables
- Label encoding for high-cardinality categorical features
- One-hot encoding for categorical features
- Feature scaling using StandardScaler
- Model training
- Model evaluation
- Streamlit deployment

## 📊 Dataset

The project uses the Hotel Booking Demand dataset.

The dataset contains information about hotel bookings, including:

- Hotel type
- Lead time
- Arrival date
- Number of adults, children and babies
- Previous cancellations
- Previous bookings
- Market segment
- Distribution channel
- Deposit type
- Customer type
- Special requests
- Room type
- Country
- Agent
- Booking changes

The target variable is:

`is_canceled`

where:

- `0` = Booking not cancelled
- `1` = Booking cancelled

## 🤖 Model

The final model used for prediction is stored as:

`model_hotel_cancellation_prediction.pkl`

The application also uses the following saved preprocessing objects:

- `scaler_hotel_cancellation_prediction.pkl`
- `label_encoders_hotel_cancellation.pkl`
- `columns_hotel_cancellation_prediction.pkl`

These files ensure that the same preprocessing applied during model training is applied to new user inputs.

## 📈 Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 84.64 |
| Precision | 78 |
| Recall | 61.47 |
| F1 Score | 68.75 |

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook
- Git & GitHub

## 📂 Project Structure

```text
hotel-cancellation-prediction/
│
├── hotel_cancellation_predictor.ipynb
├── hotel_cancellation_predictor.py
│
├── model_hotel_cancellation_prediction.pkl
├── scaler_hotel_cancellation_prediction.pkl
├── label_encoders_hotel_cancellation.pkl
├── columns_hotel_cancellation_prediction.pkl
│
├── .gitignore
├── .gitattributes
└── README.md
