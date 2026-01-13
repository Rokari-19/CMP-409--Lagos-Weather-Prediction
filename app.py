import streamlit as st
import joblib
import numpy as np

# 1. Load the trained model
model = joblib.load('weather_model.pkl')

# 2. App Title and Description
st.title("🌤️ Lagos Weather Predictor")
st.write("Enter today's weather conditions to predict **Tomorrow's Max Temperature**.")

# 3. Create Input Fields (The GUI)
st.sidebar.header("Input Today's Weather")

# We use sliders and number inputs for better user experience
# The default values are averages from your dataset
temp_max = st.sidebar.number_input("Max Temperature (°C)", min_value=20.0, max_value=45.0, value=30.0)
temp_min = st.sidebar.number_input("Min Temperature (°C)", min_value=15.0, max_value=35.0, value=25.0)
humidity = st.sidebar.slider("Humidity (%)", min_value=0, max_value=100, value=80)
windspeed = st.sidebar.slider("Wind Speed (km/h)", min_value=0.0, max_value=50.0, value=15.0)
precip = st.sidebar.number_input("Precipitation (mm)", min_value=0.0, max_value=200.0, value=0.0)

# 4. Predict Button
if st.button("Predict Tomorrow's Weather"):
    # Format the input as a numpy array (must match the order of 'features' in train_model.py)
    # ['tempmax', 'tempmin', 'precip', 'humidity', 'windspeed']
    user_input = np.array([[temp_max, temp_min, precip, humidity, windspeed]])
    
    # Make prediction
    prediction = model.predict(user_input)[0]
    
    # 5. Display Result
    st.success(f"Predicted Max Temp for Tomorrow: **{prediction:.1f} °C**")
    
    # Add some logic for "interpretation"
    if prediction > 32:
        st.warning("It's going to be a hot day! 🥵")
    elif prediction < 25:
        st.info("It might be a bit cool. 🧥")
    else:
        st.info("The weather will be pleasant. 😊")

# Optional: Show data stats
st.markdown("---")
st.caption("Model trained on historical Lagos weather data (2002-2024).")