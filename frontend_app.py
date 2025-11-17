import requests
import streamlit as st

API_URL = "https://mental-health-ml.onrender.com/predict"

st.title("Mental Health Prediction Demo")
st.write("Fill out the survey to predict the likelihood of seeking mental health treatment.")

# --- Input Form ---
with st.form(key="survey_form"):
    age = st.slider("Age", min_value=15, max_value=80, value=30)

    gender = st.selectbox("Gender", ["male", "female", "other"])
    country = st.text_input("Country (e.g., United States, Japan)")  # Country varies too much for a dropdown
    self_employed = st.selectbox("Are you self-employed?", ["yes", "no"])
    family_history = st.selectbox("Family history of mental illness?", ["yes", "no"])
    work_interfere = st.selectbox("Work interference with mental health", ["never", "rarely", "sometimes", "often"])
    no_employees = st.selectbox("Company size", ["1-5", "6-25", "26-100", "100-500", "500-1000", "more than 1000"])
    remote_work = st.selectbox("Remote work?", ["yes", "no"])
    tech_company = st.selectbox("Tech company?", ["yes", "no"])
    benefits = st.selectbox("Mental health benefits provided?", ["yes", "no", "don't know"])
    care_options = st.selectbox("Care options known?", ["yes", "no", "not sure"])
    wellness_program = st.selectbox("Wellness program available?", ["yes", "no", "don't know"])
    seek_help = st.selectbox("Resources to seek help available?", ["yes", "no", "don't know"])
    anonymity = st.selectbox("Is anonymity protected?", ["yes", "no", "don't know"])
    leave = st.selectbox("Ease of taking mental health leave", 
                         ["very easy", "easy", "neutral", "difficult", "very difficult"])
    mental_health_consequence = st.selectbox("Consequences of discussing mental health", 
                                             ["none", "maybe", "yes"])
    phys_health_consequence = st.selectbox("Consequences of discussing physical health", 
                                           ["none", "maybe", "yes"])
    coworkers = st.selectbox("Would you talk to coworkers?", ["yes", "no", "some of them"])
    supervisor = st.selectbox("Would you talk to your supervisor?", ["yes", "no", "some of them"])
    mental_health_interview = st.selectbox("Discuss mental health in interview?", ["yes", "no", "maybe"])
    phys_health_interview = st.selectbox("Discuss physical health in interview?", ["yes", "no", "maybe"])
    mental_vs_physical = st.selectbox("Is mental health as important as physical health?", 
                                      ["yes", "no", "don't know"])
    obs_consequence = st.selectbox("Observed negative mental health consequences at work?", ["yes", "no"])

    submit_button = st.form_submit_button("Predict")

# --- API Call & Display Results ---
if submit_button:
    payload = {
        "Age": age,
        "Gender": gender,
        "Country": country,
        "self_employed": self_employed,
        "family_history": family_history,
        "work_interfere": work_interfere,
        "no_employees": no_employees,
        "remote_work": remote_work,
        "tech_company": tech_company,
        "benefits": benefits,
        "care_options": care_options,
        "wellness_program": wellness_program,
        "seek_help": seek_help,
        "anonymity": anonymity,
        "leave": leave,
        "mental_health_consequence": mental_health_consequence,
        "phys_health_consequence": phys_health_consequence,
        "coworkers": coworkers,
        "supervisor": supervisor,
        "mental_health_interview": mental_health_interview,
        "phys_health_interview": phys_health_interview,
        "mental_vs_physical": mental_vs_physical,
        "obs_consequence": obs_consequence,
    }

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        result = response.json()

        pred = result.get("prediction")
        prob_yes = result.get("probability_yes")
        model_name = result.get("model")

        st.subheader("Prediction Result")
        if pred == 1:
            st.success("This person is **likely to seek mental health treatment**.")
        else:
            st.info("This person is **less likely to seek mental health treatment**.")

        if prob_yes is not None:
            st.write(f"Probability of seeking treatment (Yes): **{prob_yes * 100:.1f}%**")

        if model_name:
            st.caption(f"Model used: {model_name}")

        with st.expander("View raw JSON response"):
            st.json(result)

    except requests.exceptions.RequestException as e:
        st.error(f"API call error: {e}")
