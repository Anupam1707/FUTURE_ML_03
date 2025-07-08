import json
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

with open("Intents.json", "r") as file:
    data = json.load(file)

training_data = []
for intent in data["intents"]:
    label = intent["tag"]
    for phrase in intent["patterns"]:
        training_data.append((phrase, label))

df = pd.DataFrame(training_data, columns=["text", "intent"])

X_train, X_test, y_train, y_test = train_test_split(df["text"], df["intent"], test_size=0.2, random_state=42)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

joblib.dump(pipeline, "chatbot_intent_model.pkl")
print("Model saved")