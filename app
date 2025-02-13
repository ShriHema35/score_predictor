import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv("t20i_info.csv")

# Data Preprocessing
categorical_columns = ['batting_team', 'bowling_team', 'city']
numerical_columns = ['current_score', 'wickets_left', 'current_run_rate', 'balls_left']

preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy='mean'), numerical_columns),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Prepare Data
X = df[['batting_team', 'bowling_team', 'current_score', 'wickets_left', 'current_run_rate', 'city', 'balls_left']]
y = df['runs']

# Train model
model.fit(X, y)

# Save Model
with open("t20_score_predictor.pkl", "wb") as f:
    pickle.dump(model, f)

# Streamlit UI
st.title("🏏 T20 Match Score Predictor")
st.sidebar.header("Enter Match Details")

# User Input Fields
teams = df['batting_team'].unique().tolist()
cities = df['city'].dropna().unique().tolist()

batting_team = st.sidebar.selectbox("Select Batting Team", teams)
bowling_team = st.sidebar.selectbox("Select Bowling Team", teams)
city = st.sidebar.selectbox("Match Venue", cities)
current_score = st.sidebar.number_input("Current Score", min_value=0, max_value=300, value=50)
wickets_left = st.sidebar.slider("Wickets Left", 0, 10, 5)
current_run_rate = st.sidebar.number_input("Current Run Rate", min_value=0.0, max_value=15.0, value=6.0)
balls_left = st.sidebar.number_input("Balls Left", min_value=0, max_value=120, value=60)

# Load Trained Model
with open("t20_score_predictor.pkl", "rb") as f:
    loaded_model = pickle.load(f)

# Predict Score
input_data = pd.DataFrame({
    'batting_team': [batting_team],
    'bowling_team': [bowling_team],
    'current_score': [current_score],
    'wickets_left': [wickets_left],
    'current_run_rate': [current_run_rate],
    'city': [city],
    'balls_left': [balls_left]
})

predicted_score = loaded_model.predict(input_data)[0]
st.subheader(f"Predicted Final Score: {round(predicted_score)}")
