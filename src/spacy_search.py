import os
import numpy as np
import faiss
import spacy
import pandas as pd
from src.utils import clean_text
from src.logger_config import load_logger

logger = load_logger()

# Load spaCy model
nlp = spacy.load("en_core_web_lg")


def compute_spacy_doc_vectors(df, text_col="text", save_path="../data/embeddings/spacy_doc_vectors.npy"):
    """
    Compute and cache document embeddings using spaCy.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        logger.info(f"Loading cached document vectors from {save_path}")
        return np.load(save_path)
    
    logger.info("Computing spaCy document vectors...")
    doc_vectors = np.vstack([nlp(text).vector for text in df[text_col]])
    np.save(save_path, doc_vectors)
    logger.info(f"Saved document vectors to {save_path}")
    return doc_vectors


def build_or_load_faiss_index(df,text_col, index_path,doc_vectors_path,):
    """
    Build or load a FAISS index with spaCy embeddings.
    Returns: FAISS index and document vectors.
    """
    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    # Load or compute document vectors
    doc_vectors = compute_spacy_doc_vectors(df, text_col=text_col, save_path=doc_vectors_path)

    # Load index if exists
    if os.path.exists(index_path):
        logger.info(f"Loading existing FAISS index from {index_path}")
        index = faiss.read_index(index_path)
        return index, doc_vectors

    # Build new FAISS index
    logger.info("Building new FAISS index...")
    doc_vectors = doc_vectors.astype("float32")
    faiss.normalize_L2(doc_vectors)
    index = faiss.IndexFlatIP(doc_vectors.shape[1])
    index.add(doc_vectors)

    # Save index
    faiss.write_index(index, index_path)
    logger.info(f"FAISS index built and saved to {index_path}")

    return index, doc_vectors


def search_articles_spacy(query, df, index, top_n=5):
    """
    Perform semantic search using FAISS + spaCy.
    Returns top N most similar articles as JSON with rank.
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
