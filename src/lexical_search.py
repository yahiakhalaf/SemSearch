import os
import pickle
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from src.utils import preprocess_text
from src.logger_config import load_logger
from src.config import load_config

config = load_config()
logger = load_logger()


def build_bm25(df, text_col, corpus_path=None, model_path=None):
    """
    Build and cache BM25 model and tokenized corpus.
    If saved versions exist, they are loaded from disk.
    """
    corpus_path = corpus_path or config['models']['bm25']['corpus_path']
    model_path = model_path or config['models']['bm25']['model_path']
    os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Load cached corpus if available
    if os.path.exists(corpus_path):
        logger.info(f"Loading cached corpus from {corpus_path}")
        with open(corpus_path, "rb") as f:
            corpus = pickle.load(f)
    else:
        logger.info("Preprocessing text and building corpus...")
        corpus = df[text_col].apply(preprocess_text).tolist()
        with open(corpus_path, "wb") as f:
            pickle.dump(corpus, f)
        logger.info(f"Corpus saved to {corpus_path}")

    # Load cached BM25 model if available
    if os.path.exists(model_path):
        logger.info(f"Loading cached BM25 model from {model_path}")
        with open(model_path, "rb") as f:
            bm25 = pickle.load(f)
    else:
        logger.info("Training BM25 model...")
        bm25 = BM25Okapi(corpus)
        with open(model_path, "wb") as f:
            pickle.dump(bm25, f)
        logger.info(f"BM25 model saved to {model_path}")

    return bm25, corpus


def search_articles_bm25(query_sentence, df, bm25, top_n=5):
    """
    Retrieve the most relevant articles for a query using BM25.

    Args:
        query_sentence (str): The search query.
        df (pd.DataFrame): The dataset containing text.
        bm25 (BM25Okapi): The trained BM25 model.
        top_n (int): Number of results to return.

    Returns:
        list[dict]: JSON-like list of top N matching articles with rank and score.
    """
    try:
        logger.info(f"Processing search query: '{query_sentence}'")

        # Preprocess query
        query_tokens = preprocess_text(query_sentence)

        # Compute BM25 scores
        scores = bm25.get_scores(query_tokens)

        # Sort and select top-N
        top_n_idx = np.argsort(scores)[::-1][:top_n]
        results = df.iloc[top_n_idx].copy()
        results["bm25_score"] = scores[top_n_idx]
        results["rank"] = np.arange(1, len(results) + 1)

        # Reorder columns for clarity
        results = results[["rank", "id", "category", "bm25_score", "text"]]

        # Convert DataFrame to JSON-compatible list of dicts
        results_json = results.to_dict(orient="records")

        logger.info(f"Top {top_n} results retrieved for query: '{query_sentence}'")

        return results_json

    except Exception as e:
        logger.error(f"Error during BM25 search for query '{query_sentence}': {str(e)}")
        raise
