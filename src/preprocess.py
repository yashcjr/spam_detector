# src/preprocess.py
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text):
    """Clean text by removing punctuation, digits, and extra spaces."""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def create_vectorizer(texts):
    """Create and fit a TF-IDF vectorizer."""
    vectorizer = TfidfVectorizer(max_features=3000)
    X = vectorizer.fit_transform(texts)
    return vectorizer, X
