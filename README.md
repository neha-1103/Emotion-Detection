<h1>Emotion Detection from Speech using Streamlit (Offline)</h1>

A local, browser-based emotion detection app that records your voice, transcribes it using <b>Whisper</b>, and predicts the emotion using a pre-trained <b>Scikit-learn ML model</b>.The model has been trained on kaggle dataset consisting 16000 rows of sentences with their labelled emotions. 


This app runs <b>entirely offline</b> — no API calls, no internet dependency — and features a clean <b>Streamlit interface</b>.

<h1>Emotions Detected</h1>

😊 Joy  
😢 Sadness  
😡 Anger  
😨 Fear  
❤️ Love  
😲 Surprise

<h1>Dependencies</h1>
python<br>
Whisper and requirements<br>
To install the required packages,run pip install -r requirements.txt

<h1> Features </h1>

1. Upload mp3, m4a, wav file.<br>
2. Transcribe speech using OpenAI's Whisper(runs locally)<br>  
3. Detect emotions using trained `Scikit-learn` model  <br>
4. Display emoji with prediction  <br>
5. Simple and clean Streamlit UI  <br>
6. 100% Offline — no APIs required

(output_images/Demo1.png)
(output_images/Demo2.png)

<h1>How to Run </h1>
1. First,clone the repository.<br>
2. Enter the folder.<br>
3. Download requirements.<br>
4. Type "streamlit run app.py"<br>

