import os
import pandas as pd
import numpy as np
from src.evaluation_utils import (
    precision_at_k,
    ndcg_at_k,
    average_precision,
    load_ground_truth
)
from src.lexical_search import build_bm25, search_articles_bm25
from src.spacy_search import compute_spacy_doc_vectors, search_articles_spacy
from src.semantic_search import compute_transformer_doc_vectors, search_articles_semantic
from src.utils import build_or_load_faiss_index
from src.config import load_config
from src.logger_config import load_logger


config = load_config()
logger = load_logger()


def evaluate_model(search_function, ground_truth, df, model_name="Model", k=5, **kwargs):
    """
    Evaluate a search model using Precision@K, NDCG@K, and MAP.
    """
    precisions, ndcgs, aps = [], [], []

    logger.info(f"Evaluating model: {model_name}")

    for query, relevant_ids in ground_truth.items():
        results = search_function(query, df, top_n=k, **kwargs)
        retrieved_ids = [result["id"] for result in results]

        precisions.append(precision_at_k(retrieved_ids, relevant_ids, k))
        ndcgs.append(ndcg_at_k(retrieved_ids, relevant_ids, k))
        aps.append(average_precision(retrieved_ids, relevant_ids))

    result = {
        "Model": model_name,
        f"Precision@{k}": np.mean(precisions),
        f"NDCG@{k}": np.mean(ndcgs),
        "MAP": np.mean(aps),
    }

    logger.info(f"✅ {model_name} results: {result}")
    return result


def run_complete_evaluation(df, bm25, index_spacy, index_semantic, k=5):
    """
    Run evaluation for all three models, using paths from config.
    Saves results to config["data"]["evaluation_results"].
    """
    eval_dir = config["data"]["evaluation_dir"]
    results_path = config["data"]["evaluation_results"]

    if not os.path.exists(eval_dir):
        raise FileNotFoundError(f"Evaluation directory not found: {eval_dir}")

    eval_files = [f for f in os.listdir(eval_dir) if f.endswith(".csv")]
    if not eval_files:
        raise FileNotFoundError(f"No evaluation CSV found in {eval_dir}")

    eval_csv_path = os.path.join(eval_dir, eval_files[0])
    logger.info(f"Loading evaluation data from: {eval_csv_path}")

    ground_truth = load_ground_truth(eval_csv_path)
    results = []

    # Evaluate models - FIXED: Added bm25=bm25 parameter
    results.append(evaluate_model(search_articles_bm25, ground_truth, df, model_name="Lexical Model", k=k, bm25=bm25))
    results.append(evaluate_model(search_articles_spacy, ground_truth, df, model_name="spaCy Model", k=k, index=index_spacy))
    results.append(evaluate_model(search_articles_semantic, ground_truth, df, model_name="Semantic Model", k=k, index=index_semantic))

    comparison_df = pd.DataFrame(results)

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    comparison_df.to_csv(results_path, index=False)
    logger.info(f"📊 Evaluation results saved to: {results_path}")

    return comparison_df


def main():
    """Run full evaluation pipeline end-to-end."""
    logger.info("🚀 Starting evaluation pipeline...")

    # Load dataset
    articles_path = config["data"]["articles_csv"]
    df = pd.read_csv(articles_path)
    logger.info(f"Loaded {len(df)} articles from {articles_path}")

    # Compute BM25 model
    bm25_corpus_path = config["models"]["bm25"]["corpus_path"]
    bm25_model_path = config["models"]["bm25"]["model_path"]
    bm25_model, _ = build_bm25(df, "text", bm25_corpus_path, bm25_model_path)

    # Compute and index spaCy vectors
    spacy_embeddings_path = config["models"]["spacy"]["embeddings_path"]
    spacy_faiss_path = config["models"]["spacy"]["faiss_index_path"]
    spacy_vectors = compute_spacy_doc_vectors(df, "text", spacy_embeddings_path)
    spacy_index = build_or_load_faiss_index(spacy_faiss_path, spacy_vectors)

    # Compute and index transformer vectors
    transformer_embeddings_path = config["models"]["transformer"]["embeddings_path"]
    transformer_faiss_path = config["models"]["transformer"]["faiss_index_path"]
    transformer_vectors = compute_transformer_doc_vectors(df, "text", transformer_embeddings_path)
    transformer_index = build_or_load_faiss_index(transformer_faiss_path, transformer_vectors)

    # Run evaluation
    comparison_df = run_complete_evaluation(
        df,
        bm25_model,
        spacy_index,
        transformer_index,
        k=5,
    )

    logger.info("✅ Evaluation completed successfully.")
    print("\n=== Evaluation Results ===")
    print(comparison_df)
    print("\nSaved results to:", config["data"]["evaluation_results"])


if __name__ == "__main__":
    main()