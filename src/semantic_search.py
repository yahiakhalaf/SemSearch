import os
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from src.utils import clean_text
from src.logger_config import load_logger
from src.config import load_config

config = load_config()
logger = load_logger()

# Load SentenceTransformer model (shared across calls)
embedder = SentenceTransformer(config['models']['transformer']['model_name'])


def compute_transformer_doc_vectors(df, text_col, save_path=None):
    """
    Compute and cache SentenceTransformer embeddings for all documents.

    Args:
        df (pd.DataFrame): DataFrame with article text.
        text_col (str): Column name containing article text.
        save_path (str, optional): Path to cache embeddings. Defaults to config.

    Returns:
        np.ndarray: Embedding matrix (N_docs x dim).
    """
    save_path = save_path or config['models']['transformer']['embeddings_path']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        logger.info(f"Loading cached transformer embeddings from {save_path}")
        embeddings = np.load(save_path)
    else:
        logger.info("Computing transformer embeddings for documents...")
        embeddings = embedder.encode(df[text_col].tolist(), convert_to_numpy=True, show_progress_bar=True)
        np.save(save_path, embeddings)
        logger.info(f"Saved transformer embeddings to {save_path}")

    return embeddings


def search_articles_semantic(query, df, index, top_n=5):
    """
    Perform semantic search using transformer embeddings and FAISS.

    Args:
        query (str): User search query.
        df (pd.DataFrame): Article dataset.
        index (faiss.Index): Pre-built FAISS index (inner product).
        top_n (int): Number of results to return.

    Returns:
        list[dict]: Top-N results with rank, id, category, similarity, and text.
    """
    try:
        logger.info(f"Running semantic search for query: '{query}'")

        query = clean_text(query)
        query_vec = embedder.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_vec)

        distances, indices = index.search(query_vec, top_n)
        results = df.iloc[indices[0]].copy()
        results["similarity"] = distances[0]
        results["rank"] = np.arange(1, len(results) + 1)

        results_json = results[["rank", "id", "category", "similarity", "text"]].to_dict(orient="records")
        logger.info(f"Top {top_n} results retrieved for query '{query}'")

        return results_json

    except Exception as e:
        logger.error(f"Error during semantic search for query '{query}': {str(e)}")
        raise