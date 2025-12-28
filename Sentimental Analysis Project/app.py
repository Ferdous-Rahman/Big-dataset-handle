import streamlit as st
import joblib
import pandas as pd

model = joblib.load("model.pkl")

st.set_page_config(page_title="Mental Health Predictor", layout="centered")
st.title("🧠 Mental Health Prediction Web App")

text = st.text_area("Enter your text:")
language = st.selectbox("Language", ["Bangla", "English", "Banglish"])
has_emoji = st.selectbox("Has Emoji?", [True, False])

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        input_df = pd.DataFrame([{
            "Text": text,
            "Language": language,
            "has_emoji": has_emoji
        }])
        prediction = model.predict(input_df)[0]
        st.success(f"✅ Prediction: {prediction}")
