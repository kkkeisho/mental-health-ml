import requests
import streamlit as st

API_URL = "https://mental-health-ml.onrender.com/predict"

st.title("Mental Health Prediction Demo")

st.write("Fill out the survey to predict the likelihood of seeking mental health treatment.")

# --- Input Form ---
with st.form(key="survey_form"):
    age = st.slider("Age", min_value=15, max_value=80, value=30, step=1)

    gender = st.text_input("Gender (e.g., male / female / other)")
    country = st.text_input("Country (e.g., United States, Japan)")
    self_employed = st.text_input("Are you self-employed? (yes / no)")
    family_history = st.text_input("Family history of mental illness? (yes / no)")
    work_interfere = st.text_input("How often does work interfere with mental health? (never / sometimes / often / always)")
    no_employees = st.text_input("Company size (e.g., 1-5, 6-25, 26-100, ...)")
    remote_work = st.text_input("Do you work remotely? (yes / no)")
    tech_company = st.text_input("Is it a tech company? (yes / no)")
    benefits = st.text_input("Are mental health benefits provided? (yes / no / don't know)")
    care_options = st.text_input("Do you know care options? (yes / no / not sure)")
    wellness_program = st.text_input("Is there a wellness program? (yes / no / don't know)")
    seek_help = st.text_input("Are resources available to seek help? (yes / no / don't know)")
    anonymity = st.text_input("Is anonymity protected? (yes / no / don't know)")
    leave = st.text_input("How easy is it to take mental health leave? (very easy / easy / neutral / difficult / very difficult)")
    mental_health_consequence = st.text_input("Consequences of discussing mental health? (none / maybe / yes)")
    phys_health_consequence = st.text_input("Consequences of discussing physical health? (none / maybe / yes)")
    coworkers = st.text_input("Would you talk to coworkers? (yes / no / some of them)")
    supervisor = st.text_input("Would you talk to your supervisor? (yes / no / some of them)")
    mental_health_interview = st.text_input("Would you discuss mental health in an interview? (yes / no / maybe)")
    phys_health_interview = st.text_input("Would you discuss physical health in an interview? (yes / no / maybe)")
    mental_vs_physical = st.text_input("Is mental health as important as physical health? (yes / no / don't know)")
    obs_consequence = st.text_input("Have you observed negative consequences regarding mental health at work? (yes / no)")

    submit_button = st.form_submit_button(label="Predict")

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

        with st.expander("View raw response (JSON)"):
            st.json(result)

    except requests.exceptions.RequestException as e:
        st.error(f"Error while calling the API: {e}")
