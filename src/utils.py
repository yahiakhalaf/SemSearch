# utils.py

import re
import spacy
from contractions import fix

# Load spaCy model once globally
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def clean_text(text: str) -> str:
    """
    Basic text normalization:
    - Lowercase
    - Remove punctuation, newlines, and extra whitespace
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'\n+', ' ', text)         # remove newlines
    text = re.sub(r'\s+', ' ', text)         # collapse multiple spaces
    text = re.sub(r'[^\w\s]', '', text)      # remove punctuation
    return text.strip()


def preprocess_text(text: str):
    """
    Full preprocessing pipeline:
    1. Clean text (normalize case, punctuation, whitespace)
    2. Expand contractions
    3. Lemmatize with spaCy
    4. Remove stopwords and short tokens

    Returns:
        list of lemmatized tokens
    """
    if not isinstance(text, str):
        return []

    text = clean_text(text)
    text = fix(text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop and len(token) > 2]

    return tokens
