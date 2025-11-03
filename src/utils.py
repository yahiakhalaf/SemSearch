# utils.py

import re
import spacy
from contractions import fix
import os
import faiss
from src.logger_config import load_logger
from src.config import load_config

config = load_config()
logger = load_logger()

# Load spaCy model once globally (disabled components for speed)
nlp = spacy.load(config['models']['spacy']['small_model'], disable=["parser", "ner"])


def clean_text(text: str) -> str:
    """
    Basic text normalization.

    Steps:
    - Convert to lowercase
    - Replace newlines with space
    - Collapse multiple spaces
    - Remove punctuation

    Args:
        text (str): Raw input text.

    Returns:
        str: Cleaned text.
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
    Full preprocessing pipeline for lexical search (BM25).

    Steps:
    1. Clean text
    2. Expand contractions
    3. Lemmatize using spaCy
    4. Remove stopwords and short tokens

    Args:
        text (str): Input text.

    Returns:
        list[str]: List of lemmatized tokens.
    """
    if not isinstance(text, str):
        return []

    text = clean_text(text)
    text = fix(text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop and len(token) > 2]

    return tokens


def build_or_load_faiss_index(index_path, doc_vectors):
    """
    Build or load a FAISS inner-product index for normalized vectors.

    Args:
        index_path (str): File path to save/load the index.
        doc_vectors (np.ndarray): Document embedding matrix (float32, L2-normalized).

    Returns:
        faiss.Index: FAISS index ready for search.
    """
    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    # Load index if exists
    if os.path.exists(index_path):
        logger.info(f"Loading existing FAISS index from {index_path}")
        index = faiss.read_index(index_path)
        return index

    # Build new FAISS index
    logger.info("Building new FAISS index...")
    doc_vectors = doc_vectors.astype("float32")
    faiss.normalize_L2(doc_vectors)
    index = faiss.IndexFlatIP(doc_vectors.shape[1])
    index.add(doc_vectors)

    # Save index
    faiss.write_index(index, index_path)
    logger.info(f"FAISS index built and saved to {index_path}")

    return index