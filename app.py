import streamlit as st
import pandas as pd
import pickle

# ✅ Load the trained model
with open("habitability_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🌍 Exoplanet Habitability Predictor")
st.write("This AI predicts the probability of an exoplanet being habitable based on its properties.")

# ✅ Learn More Section
with st.expander("ℹ️ Learn More About This Project"):
    st.write("""
    ### 🔭 What does this AI do?
    It predicts the **probability of habitability** of an exoplanet based on:
    - **Planet Radius (Earth radii)** – Too large → gas giant; Too small → no atmosphere  
    - **Orbital Period (days)** – Linked to distance from its star  
    - **Star Temperature (Kelvin)** – Too hot or too cold affects habitability  
    - **Semi-major Axis (AU)** – Distance from the star, affects the habitable zone  

    ### 🧪 Why these factors matter?
    - **Habitable Zone**: The range where liquid water can exist  
    - **Planet Size**: Rocky planets (like Earth) are more likely habitable than gas giants  
    - **Star Type**: Cooler/red dwarf stars vs hotter stars affect stability of orbit  

    ### 🤖 How does the AI work?
    - It uses a **Balanced Random Forest Model** trained on ~4,000 exoplanets  
    - Handles **severely imbalanced data** (very few known habitable planets)  
    - Outputs a **habitability probability** between 0 and 1  

    ### ⚠️ Limitations
    - A high habitability score ≠ proof of life  
    - It only considers basic parameters, not atmosphere, magnetic field, etc.  

    *This tool helps prioritize interesting planets for future study!*  
    """)

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

    # ✅ Give explanation WHY
    st.write("### Why this result?")
    reasons = []

    # Planet size
    if 0.5 < planet_radius < 2.5:
        reasons.append("✅ Planet size is Earth-like → likely a rocky surface")
    else:
        reasons.append("❌ Planet size is unusual (too big/small for rocky surface)")

    # Star temperature
    if 2500 < star_temp < 7000:
        reasons.append("✅ Star temperature is within habitable range for liquid water")
    else:
        reasons.append("❌ Star is too hot or too cold for stable habitability")

    # Orbit distance
    if 0.5 < semi_major_axis < 2.0:
        reasons.append("✅ Orbit is in the star’s habitable zone (liquid water possible)")
    else:
        reasons.append("❌ Orbit is outside the habitable zone")

    # Orbital period (rough check)
    if planet_period < 1000:
        reasons.append("✅ Reasonable orbital period for a stable climate")
    else:
        reasons.append("❌ Very long orbital period → possibly too cold")

    for r in reasons:
        st.write("- " + r)

    if prob > 0.5:
        st.success("✅ This planet is **likely habitable!**")
    else:
        st.warning("❌ This planet is **probably not habitable.**")

# ✅ Show Top 10 Habitable Planets (optional)
if st.button("Show Top 10 Habitable Planets"):
    df = pd.read_csv('phl_exoplanet_catalog_2019.csv.zip')
    df = df[['P_RADIUS','P_PERIOD','S_TEMPERATURE','P_SEMI_MAJOR_AXIS_EST','P_NAME']].dropna()
    probs = model.predict_proba(df[['P_RADIUS','P_PERIOD','S_TEMPERATURE','P_SEMI_MAJOR_AXIS_EST']])[:,1]
    df['Habitability_Prob'] = probs
    top_planets = df.sort_values(by='Habitability_Prob', ascending=False).head(10)
    st.write("### 🌟 Top 10 Most Habitable Planets (Predicted)")
    st.table(top_planets[['P_NAME','Habitability_Prob']])
