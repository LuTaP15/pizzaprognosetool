'''
Run Befehl ist:
streamlit run "C:/Users/Paul Wunderlich/PycharmProjects/IP3/Demo/streamlit_demo_advanced.py"

'''
# Libaries
import streamlit as st
import pandas as pd
import pickle
import joblib

####################################################################################################

""":arg

        # Load the prediction model
        if choice_method == "Klassifikation":
            scaler = joblib.load(open('./models/scaler_rf_co2.gz', 'rb'))
            model = joblib.load(open('./models/rf_co2.gz', 'rb'))
        elif choice_method == "Regression":
            scaler = joblib.load(open('./models/scaler_rf_reg_co2.gz', 'rb'))
            model = joblib.load(open('./models/rf_reg_co2.gz', 'rb'))
        else:
            st.markdown("Modeltyp was not selected")

    elif choice_sensor == "VOC":
        # Name columns
        df.columns = ["Time", "Time2", "Humidity", "Temp", "Index_VOC", "Humidity2", "Temp2", "VOC"]

        # Filter the relevant data for CO2
        df = df[["Humidity", "Temp", "VOC"]]

        # Load the prediction model
        if choice_method == "Klassifikation":
            scaler = joblib.load(open('./models/scaler_rf_voc.gz', 'rb'))
            model = joblib.load(open('./models/rf_voc.gz', 'rb'))
        elif choice_method == "Regression":
            scaler = joblib.load(open('./models/scaler_rf_reg_voc.gz', 'rb'))
            model = joblib.load(open('./models/rf_reg_voc.gz', 'rb'))
        else:
            st.markdown("Modeltyp was not selected")
    else:
        st.markdown("Fehler!")

    # Get last values
    current_data = df.tail(1)

    # Fit current data with scaler from the model
    try:
        scaler.fit(current_data)
    except:
        st.markdown("Scaler is missing!")

    # Use model for prediction
    prediction = model.predict(current_data)
    return prediction
"""


####################################################################################################

st.title("Vorhersagetool")
st.markdown("### Einleitung")
st.markdown("Vorhersagetool für die Bestimmung der Haltbarkeit von Pizza. "
            "Es kann zwischen zwei Sensoren gewählt werden. CO2 und VOC. "
            "Bitte wählen sie einen Datensatz und das Verfahren aus. "
            "Zur Auswahl steht eine simple Klassifikation oder eine Regression für die Vorhersage."
            )

choice_sensor = st.radio("Welchen Sensor wollen Sie verwenden?",
                      ("CO2 Sensor", "VOC Sensor"), index=0)

choice_method = st.radio("Welches Verfahren wollen Sie verwenden?",
                      ("Klassifikation", "Regression"), index=0)

uploaded_file = st.file_uploader("Wählen Sie Ihre Daten aus!")
if uploaded_file is not None:
    # Read file
    df = pd.read_csv(uploaded_file, sep="\t", skiprows=9)


start_prognose = st.button("Starte Vorhersage")
# Prognose
if start_prognose:
    df.columns = ["Time", "Time2", "CO2", "Temp", "Humidity"]

    # Filter the relevant data for CO2
    df = df[["CO2", "Temp", "Humidity"]]

    scaler = joblib.load(open('./models/scaler_rf_voc.gz', 'rb'))
    model = joblib.load(open('./models/rf_voc.gz', 'rb'))

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
    if choice_method == "Classification":
        st.markdown("The possible outcomes are: E for eatable, N for not eatable and U for undefined!")
        st.write(prediction)
    elif choice_method == "Regression":
        st.markdown("The outcome is the amount of days relative to the best-before-date. "
              "E.g. 7 means you have 7 days before expiring. "
              "-1 means you are already 1 day over the expiration. ")
        st.write(prediction)
    else:
        st.markdown("Modeltyp was not selected")


