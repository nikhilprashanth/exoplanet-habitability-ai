import streamlit as st
import pandas as pd
import joblib

# ✅ Load the trained model
model = joblib.load('habitability_model.pkl')

st.title("🌍 Exoplanet Habitability Predictor")
st.write("This AI predicts the probability of an exoplanet being habitable based on its properties.")

# ✅ User inputs
planet_radius = st.number_input("Planet Radius (Earth radii)", min_value=0.1, max_value=20.0, value=1.0)
planet_period = st.number_input("Orbital Period (days)", min_value=0.1, max_value=10000.0, value=365.0)
star_temp = st.number_input("Star Temperature (K)", min_value=2000.0, max_value=10000.0, value=5500.0)
semi_major_axis = st.number_input("Semi-major Axis (AU)", min_value=0.01, max_value=10.0, value=1.0)

# ✅ Predict button
if st.button("Predict Habitability"):
    input_data = [[planet_radius, planet_period, star_temp, semi_major_axis]]
    prob = model.predict_proba(input_data)[0][1]
    st.write(f"**Predicted Habitability Probability:** {prob:.2f}")
    if prob > 0.5:
        st.success("✅ This planet is likely to be habitable!")
    else:
        st.warning("❌ This planet is probably not habitable.")

# ✅ Show Top 10 Habitable Planets (optional)
if st.button("Show Top 10 Habitable Planets"):
    df = pd.read_csv('phl_exoplanet_catalog_2019.csv.zip')
    df = df[['P_RADIUS','P_PERIOD','S_TEMPERATURE','P_SEMI_MAJOR_AXIS_EST','P_NAME']].dropna()
    probs = model.predict_proba(df[['P_RADIUS','P_PERIOD','S_TEMPERATURE','P_SEMI_MAJOR_AXIS_EST']])[:,1]
    df['Habitability_Prob'] = probs
    top_planets = df.sort_values(by='Habitability_Prob', ascending=False).head(10)
    st.write("### 🌟 Top 10 Most Habitable Planets (Predicted)")
    st.table(top_planets[['P_NAME','Habitability_Prob']])
