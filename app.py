
# app.py
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load the trained model
with open('model_penguin_66130701702.pkl', 'rb') as f:
    model = pickle.load(f)

# Title of the web app
st.title('Penguin Species Prediction')

# Instructions for the user
st.write("This app predicts the species of penguins based on their physical characteristics.")

# Input fields for new data (new penguin data)
culmen_length = st.number_input('Culmen Length (mm)', min_value=10.0, max_value=100.0, value=39.1)
culmen_depth = st.number_input('Culmen Depth (mm)', min_value=10.0, max_value=100.0, value=18.7)
flipper_length = st.number_input('Flipper Length (mm)', min_value=100.0, max_value=250.0, value=181.0)
body_mass = st.number_input('Body Mass (g)', min_value=2000.0, max_value=6000.0, value=3750.0)  # เปลี่ยน min_value และ max_value เป็น float

input_features = np.array([culmen_length,culmen_depth,flipper_length,body_mass])
# Create a DataFrame with the input data
#input_data = pd.DataFrame({
 #   'culmen_length_mm': [culmen_length],
  #  'culmen_depth_mm': [culmen_depth],
   # 'flipper_length_mm': [flipper_length],
    #'body_mass_g': [body_mass]
#})

# Standardize the input features using the same scaler as during training


# Make a prediction using the trained model
if st.button('Predict'):
    prediction = model.predict(input_features)  # Make the prediction
    predicted_species = prediction[0]  # Get the predicted species
    
    # Display the prediction
    st.write(f"The predicted species of the penguin is: {predicted_species}")
