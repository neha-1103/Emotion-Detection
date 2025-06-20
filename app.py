import streamlit as st
import joblib
from utils import clean_text

model_path = "models/emotion_model.pkl"
vectorizer_path = "models/tfidf_vectorizer.pkl"

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

st.set_page_config(page_title="Emotion Detector", layout="centered")
st.title("Emotion Detection from Text")
st.write("Type any sentence to detect its emotion:")

user_input = st.text_area("Enter your text here", height=150)

if st.button("🔍 Predict Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        st.success(f"Predicted Emotion: {prediction.upper()}")

