# src/predict.py
import pickle
from src.preprocess import clean_text

# Load models and vectorizers
with open("models/url_model.pkl", "rb") as f:
    url_model = pickle.load(f)
with open("models/url_vectorizer.pkl", "rb") as f:
    url_vectorizer = pickle.load(f)
with open("models/msg_model.pkl", "rb") as f:
    msg_model = pickle.load(f)
with open("models/msg_vectorizer.pkl", "rb") as f:
    msg_vectorizer = pickle.load(f)

def predict_spam(text):
    """Detect whether input is URL or message and classify as Spam/Not Spam."""
    cleaned = clean_text(text)

    # Identify URL based on basic pattern
    if any(x in text for x in ["http", "www", ".", "/"]):
        vector = url_vectorizer.transform([cleaned])
        prediction = url_model.predict(vector)[0]
        return f"Detected: URL → {prediction.upper()}"
    else:
        vector = msg_vectorizer.transform([cleaned])
        prediction = msg_model.predict(vector)[0]
        return f"Detected: MESSAGE → {prediction.upper()}"

if __name__ == "__main__":
    text = input("Enter URL or message: ")
    print(predict_spam(text))
