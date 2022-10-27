# Libaries
import streamlit as st
import pandas as pd
import pickle
import joblib


st.title("Vorhersagetool")
st.markdown("### Einleitung")
st.markdown("Vorhersagetool für die Bestimmung der Haltbarkeit von Pizza. "
            "Es kann zwischen zwei Sensoren gewählt werden. CO2 und VOC. "
            "Bitte wählen sie einen Datensatz und das Verfahren aus. "
            "Zur Auswahl steht eine simple Klassifikation oder eine Regression für die Vorhersage."
            )

st.session_state.choice_sensor = st.radio("Welchen Sensor wollen Sie verwenden?",
                      ("CO2", "VOC"), index=0)

st.session_state.choice_method = st.radio("Welches Verfahren wollen Sie verwenden?",
                      ("Klassifikation", "Regression"), index=0)

uploaded_file = st.file_uploader("Wählen Sie Ihre Daten aus!")
if uploaded_file is not None:
    # Read file
    df = pd.read_csv(uploaded_file, sep="\t", skiprows=9)

if st.session_state.choice_sensor == "CO2":
    # Name columns
    df.columns = ["Time", "Time2", "CO2", "Temp", "Humidity"]

    # Filter the relevant data for CO2
    df = df[["CO2", "Temp", "Humidity"]]

    # Load the prediction model
    if st.session_state.choice_method == "Klassifikation":
        scaler = joblib.load(open('./model/scaler_rf_co2.gz', 'rb'))
        model = joblib.load(open('./model/rf_co2.gz', 'rb'))
    elif st.session_state.choice_method == "Regression":
        scaler = joblib.load(open('./model/scaler_rf_reg_co2.gz', 'rb'))
        model = joblib.load(open('./model/rf_reg_co2.gz', 'rb'))
    else:
        st.markdown("Modeltyp was not selected")

elif st.session_state.choice_sensor == "VOC":
    # Name columns
    df.columns = ["Time", "Time2", "Humidity", "Temp", "Index_VOC", "Humidity2", "Temp2", "VOC"]

    # Filter the relevant data for CO2
    df = df[["Humidity", "Temp", "VOC"]]

    # Load the prediction model
    if st.session_state.choice_method == "Klassifikation":
        scaler = joblib.load(open('./model/scaler_rf_voc.gz', 'rb'))
        model = joblib.load(open('./model/rf_voc.gz', 'rb'))
    elif st.session_state.choice_method == "Regression":
        scaler = joblib.load(open('./model/scaler_rf_reg_voc.gz', 'rb'))
        model = joblib.load(open('./model/rf_reg_voc.gz', 'rb'))
    else:
        st.markdown("Modeltyp was not selected")


start_prognose = st.button("Starte Vorhersage")
# Prognose
if start_prognose:
    # Get last values
    current_data = df.tail(1)

    # Fit current data with scaler from the model
    try:
        scaler.fit(current_data)
    except:
        st.markdown("Scaler is missing!")

    # Use model for prediction
    prediction = model.predict(current_data)

    # Display the result
    if st.session_state.choice_method == "Klassifikation":
        st.markdown("The possible outcomes are: E for eatable, N for not eatable and U for undefined!")
        st.write(prediction)
    elif st.session_state.choice_method == "Regression":
        st.markdown("The outcome is the amount of days relative to the best-before-date. "
              "E.g. 7 means you have 7 days before expiring. "
              "-1 means you are already 1 day over the expiration. ")
        st.write(prediction)
    else:
        st.markdown("Modeltyp was not selected")



