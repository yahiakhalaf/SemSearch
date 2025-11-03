import os
import numpy as np
import faiss
import spacy
import pandas as pd
from src.utils import clean_text
from src.logger_config import load_logger
from src.config import load_config

config = load_config()
logger = load_logger()

# Load spaCy model for vector extraction
nlp = spacy.load(config['models']['spacy']['model_name'])


def compute_spacy_doc_vectors(df, text_col="text", save_path=None):
    """
    Compute and cache document vectors using spaCy's word vectors.

    Args:
        df (pd.DataFrame): Article dataset.
        text_col (str): Column with text.
        save_path (str, optional): Cache path.

    Returns:
        np.ndarray: Document vectors (N x dim).
    """
    save_path = save_path or config['models']['spacy']['embeddings_path']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        logger.info(f"Loading cached document vectors from {save_path}")
        return np.load(save_path)
    
    logger.info("Computing spaCy document vectors...")
    doc_vectors = np.vstack([nlp(text).vector for text in df[text_col]])
    np.save(save_path, doc_vectors)
    logger.info(f"Saved document vectors to {save_path}")
    return doc_vectors


def search_articles_spacy(query, df, index, top_n=5):
    """
    Perform semantic search using spaCy vectors and FAISS.

    Args:
        query (str): Search query.
        df (pd.DataFrame): Article dataset.
        index (faiss.Index): FAISS inner-product index.
        top_n (int): Number of results.

    Returns:
        list[dict]: Ranked results with similarity.
    """
    try:
        logger.info(f"Running spaCy semantic search for query: '{query}'")

        query = clean_text(query)
        query_vec = nlp(query).vector.astype("float32").reshape(1, -1)
        faiss.normalize_L2(query_vec)

        distances, indices = index.search(query_vec, top_n)
        results = df.iloc[indices[0]].copy()
        results["similarity"] = distances[0]
        results["rank"] = np.arange(1, len(results) + 1)

        results_json = results[["rank", "id", "category", "similarity", "text"]].to_dict(orient="records")
        logger.info(f"Top {top_n} results retrieved for query '{query}'")

        return results_json

    except Exception as e:
        logger.error(f"Error during spaCy semantic search for query '{query}': {str(e)}")
        raise