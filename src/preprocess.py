# src/preprocess.py
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text_sms(text):
    """Clean text by removing punctuation, digits, and extra spaces."""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def create_vectorizer_sms(texts):
    """Create and fit a TF-IDF vectorizer."""
    vectorizer = TfidfVectorizer(max_features=4000)
    X = vectorizer.fit_transform(texts)
    return vectorizer, X

def clean_text_url(url):
    """
    Simple cleaning: lowercase and remove common scheme/prefix.
    This helps the vectorizer focus on the core URL structure.
    """
    # Remove http:// or https:// and optional www.
    url = re.sub(r'https?://(www\.)?', '', str(url)).lower()
    return url

def create_vectorizer_url(X_train):
    """
    Creates and fits a TfidfVectorizer using character n-grams.
    Character n-grams from 3 to 5 are highly effective for URL structural analysis.
    """
    vectorizer = TfidfVectorizer(
        analyzer='char', # Crucial: Analyze at the character level
        ngram_range=(3, 5), # Use 3-grams up to 5-grams
        max_features=50000 # Limit features to a reasonable number
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    return vectorizer, X_train_vec

def clean_text(text):
    return clean_text_url(clean_text_sms(text))
