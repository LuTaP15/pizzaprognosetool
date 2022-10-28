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

st.markdown("Bitte Laden Sie zunächst eine Datei hoch bevor Sie die Prognose starten!")

uploaded_file = st.file_uploader("Wählen Sie Ihre Daten aus!", type=(["edf"]))
if uploaded_file is not None:
    # Read file
    df = pd.read_csv(uploaded_file, sep="\t", skiprows=9)


start_prognose = st.button("Starte Vorhersage")
# Prognose
if start_prognose and uploaded_file is not None:

    if st.session_state.choice_sensor == "CO2" and len(df.columns)==5:
        # Name columns
        df.columns = ["Time", "Time2", "CO2", "Temp", "Humidity"]

        # Filter the relevant data for CO2
        df = df[["CO2", "Temp", "Humidity"]]

        # Load the prediction model
        if st.session_state.choice_method == "Klassifikation":
            scaler = joblib.load(open('./models/scaler_rf_co2.gz', 'rb'))
            model = joblib.load(open('./models/rf_co2.gz', 'rb'))
        elif st.session_state.choice_method == "Regression":
            scaler = joblib.load(open('./models/scaler_rf_reg_co2.gz', 'rb'))
            model = joblib.load(open('./models/rf_reg_co2.gz', 'rb'))
        else:
            st.markdown("Modeltyp was not selected")

    elif st.session_state.choice_sensor == "VOC" and len(df.columns)==8:
        # Name columns
        df.columns = ["Time", "Time2", "Humidity", "Temp", "Index_VOC", "Humidity2", "Temp2", "VOC"]

        # Filter the relevant data for CO2
        df = df[["Humidity", "Temp", "VOC"]]

        # Load the prediction model
        if st.session_state.choice_method == "Klassifikation":
            scaler = joblib.load(open('./models/scaler_rf_voc.gz', 'rb'))
            model = joblib.load(open('./models/rf_voc.gz', 'rb'))
        elif st.session_state.choice_method == "Regression":
            scaler = joblib.load(open('./models/scaler_rf_reg_voc.gz', 'rb'))
            model = joblib.load(open('./models/rf_reg_voc.gz', 'rb'))
        else:
            st.markdown("Modeltyp was not selected")
    else:
        st.markdown("Falsche Sensordaten!")

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
        st.markdown("Sie haben sich für das Verfahren Klassifikation entschieden!")
        st.markdown("Das heißt das Model sagt Ihnen in welchem der drei Zustände sich die Pizza befindet.")
        st.markdown("- E für essbar,")
        st.markdown("- N für nicht essbar,")
        st.markdown("- U für undefiniert")

        if prediction.item(0) == "E":
            st.write(f"Ihre Pizza hat den Zustand {prediction.item(0)} und ist noch essbar!")
        elif prediction.item(0) == "N":
            st.write(f"Ihre Pizza hat den Zustand {prediction.item(0)} und ist leider nicht mehr essbar!")
        elif prediction.item(0) == "U":
            st.write(f"Der Zustand Ihrer Pizza ist {prediction.item(0)} und es kann somit keine Ausage getroffen werden!")

    elif st.session_state.choice_method == "Regression":
        st.markdown("Die Ausgabe gibt die Tage relativ zum Mindesthaltbarkeitsdatum an.")
        if prediction.item(0) >= 0:
            st.write(f"Sie haben noch {prediction.item(0)} Tage bis zum Mindesthaltbarkeitsdatum!")
        elif prediction.item(0) < 0:
            st.write(f"Ihre Pizza ist bereits {prediction.item(0)} Tage über dem Mindesthaltbarkeitsdatum!")
    else:
        st.markdown("Modeltyp wurde nicht ausgewählt!")




