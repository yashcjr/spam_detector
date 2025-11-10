# src/train_msg_model.py
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from src.preprocess import clean_text, create_vectorizer


def sms_model():
    # Load dataset
    df = pd.read_csv("data/merged_sms_dataset.csv")

    # Clean messages
    df['sms'] = df['sms'].apply(clean_text)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['sms'], df['label'], test_size=0.2, random_state=42
    )

    # Vectorize
    vectorizer, X_train_vec = create_vectorizer(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    # Evaluate
    y_pred = model.predict(X_test_vec)
    # print("Message Model Accuracy:", accuracy_score(y_test, y_pred))

    # Save model & vectorizer
    with open("models/msg_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("models/msg_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    # print("Message model and vectorizer saved successfully.")
