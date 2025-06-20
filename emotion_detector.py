import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
from utils import clean_text
import joblib, pathlib

df = pd.read_csv("data/train.csv")

df['clean_text'] = df['text'].apply(clean_text)

sns.countplot(data=df, x='label')
plt.title("Emotion Label Distribution")
plt.show()

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['clean_text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

print("[INFO] current working dir:", pathlib.Path().resolve())

joblib.dump(model, "models/emotion_model.pkl")
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
