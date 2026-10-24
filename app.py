
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/traffic_model.pkl")

st.set_page_config(
    page_title="AI Traffic Prediction",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 AI Traffic Congestion Prediction")
st.write(
    "Predict hourly traffic volume using a trained "
    "Random Forest machine learning model."
)

st.info(
    "This demonstration uses historical US highway data. "
    "It is not a live traffic monitoring system."
)

st.header("Enter Traffic Conditions")

col1, col2 = st.columns(2)

with col1:
    hour = st.slider("Hour of the day", 0, 23, 8)
    day_of_week = st.selectbox(
        "Day of the week",
        options=list(range(7)),
        format_func=lambda x: [
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"
        ][x]
    )
    month = st.selectbox(
        "Month",
        options=list(range(1, 13)),
        format_func=lambda x: [
            "January", "February", "March", "April",
            "May", "June", "July", "August",
            "September", "October", "November", "December"
        ][x - 1]
    )
    weather = st.selectbox(
        "Weather condition",
        [
            "Clear", "Clouds", "Rain", "Drizzle",
            "Mist", "Fog", "Snow", "Haze",
            "Thunderstorm", "Smoke"
        ]
    )

with col2:
    temp_c = st.slider(
        "Temperature (°C)", -20.0, 40.0, 20.0
    )
    rain = st.number_input(
        "Rainfall (mm/hour)", min_value=0.0,
        max_value=100.0, value=0.0
    )
    snow = st.number_input(
        "Snowfall (mm/hour)", min_value=0.0,
        max_value=1.0, value=0.0
    )
    clouds = st.slider(
        "Cloud coverage (%)", 0, 100, 40
    )

if st.button("Predict Traffic", type="primary"):
    # Dataset temperature is in Kelvin
    temp_kelvin = temp_c + 273.15

    input_data = pd.DataFrame([{
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        "temp": temp_kelvin,
        "rain_1h": rain,
        "snow_1h": snow,
        "clouds_all": clouds,
        "weather_main": weather
    }])

    prediction = max(0, float(model.predict(input_data)[0]))

    st.subheader("Prediction Result")
    st.metric(
        "Predicted hourly traffic volume",
        f"{prediction:,.0f} vehicles"
    )

    # Illustrative thresholds, not official congestion standards
    if prediction < 1500:
        level = "Low"
    elif prediction < 3500:
        level = "Moderate"
    elif prediction < 5000:
        level = "High"
    else:
        level = "Very High"

    st.write(f"### Estimated congestion: {level}")
    st.caption(
        "Congestion categories use illustrative traffic-volume "
        "thresholds, not measured road speed or official standards."
    )
