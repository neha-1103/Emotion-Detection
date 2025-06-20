import streamlit as st
import whisper
import joblib
import tempfile
from utils import clean_text

model = joblib.load("models/emotion_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

whisper_model = whisper.load_model("base")

emojis = {
    "joy": "😊", "sadness": "😢", "anger": "😡",
    "fear": "😨", "love": "❤️", "surprise": "😲"
}

st.set_page_config(page_title="Emotion Detection with Voice", layout="centered")
st.title(" Emotion Detection from Voice & Text")

tab1, tab2 = st.tabs(["📝 Text Input", "🎙️ Voice Input"])

with tab1:
    text_input = st.text_area("Type your message here", height=150)
    if st.button(" Predict from Text"):
        if not text_input.strip():
            st.warning("Please enter some text.")
        else:
            cleaned = clean_text(text_input)
            vector = vectorizer.transform([cleaned])
            prediction = model.predict(vector)[0]
            emoji = emojis.get(prediction, "")
            st.success(f"**Predicted Emotion:** {prediction.upper()} {emoji}")

with tab2:
    audio_file = st.file_uploader("Upload an audio file (.mp3, .wav, .m4a)", type=["mp3", "wav", "m4a"])
    if audio_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name
        st.audio(tmp_path)
        st.info("Transcribing...")
        result = whisper_model.transcribe(tmp_path)
        transcribed_text = result["text"]
        st.write("📝 Transcribed Text:", transcribed_text)

        if st.button(" Predict from Audio"):
            cleaned = clean_text(transcribed_text)
            vector = vectorizer.transform([cleaned])
            prediction = model.predict(vector)[0]
            emoji = emojis.get(prediction, "")
            st.success(f"**Predicted Emotion:** {prediction.upper()} {emoji}")


