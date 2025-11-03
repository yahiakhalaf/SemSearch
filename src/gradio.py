# gradio.py
import gradio as gr
import pandas as pd
import os
from src.logger_config import setup_logger
from src.hot_keywords import (
    extract_hot_keywords_yake,
    extract_hot_keywords_keybert
)
from src.lexical_search import build_bm25, search_articles_bm25
from src.semantic_search import compute_transformer_doc_vectors, search_articles_semantic
from src.spacy_search import compute_spacy_doc_vectors, search_articles_spacy
from src.utils import build_or_load_faiss_index
from src.config import load_config

config = load_config()
logger = setup_logger()

# === File Paths ===
ARTICLES_PATH = config['data']['articles_csv']
BM25_CORPUS_PATH = config['models']['bm25']['corpus_path']
BM25_MODEL_PATH = config['models']['bm25']['model_path']
TRANSFORMER_EMBEDDINGS_PATH = config['models']['transformer']['embeddings_path']
SPACY_EMBEDDINGS_PATH = config['models']['spacy']['embeddings_path']
FAISS_TRANSFORMER_INDEX_PATH = config['models']['transformer']['faiss_index_path']
FAISS_SPACY_INDEX_PATH = config['models']['spacy']['faiss_index_path']

HOT_KEYWORDS_DIR = config['data']['hot_keywords_dir']
YAKE_OUTPUT_PATH = os.path.join(HOT_KEYWORDS_DIR, "yake_hot_keywords.json")
KEYBERT_OUTPUT_PATH = os.path.join(HOT_KEYWORDS_DIR, "keybert_hot_keywords.json")

# === Global State ===
df = None
bm25_model = None
transformer_index = None
spacy_index = None
hot_keywords_cache = {}


def initialize_models():
    """Initialize all models and pre-compute hot keywords."""
    global df, bm25_model, transformer_index, spacy_index, hot_keywords_cache

    logger.info("Initializing models and pre-computing hot keywords...")

    # Load dataset
    df = pd.read_csv(ARTICLES_PATH)
    logger.info(f"Loaded {len(df)} articles from {ARTICLES_PATH}")

    # Load BM25
    bm25_model, _ = build_bm25(df, "text", BM25_CORPUS_PATH, BM25_MODEL_PATH)

    # Load Transformer + FAISS
    transformer_embeddings = compute_transformer_doc_vectors(df, "text", TRANSFORMER_EMBEDDINGS_PATH)
    transformer_index = build_or_load_faiss_index(FAISS_TRANSFORMER_INDEX_PATH, transformer_embeddings)

    # Load spaCy + FAISS
    spacy_embeddings = compute_spacy_doc_vectors(df, "text", SPACY_EMBEDDINGS_PATH)
    spacy_index = build_or_load_faiss_index(FAISS_SPACY_INDEX_PATH, spacy_embeddings)

    # Pre-compute hot keywords
    os.makedirs(HOT_KEYWORDS_DIR, exist_ok=True)

    if not os.path.exists(YAKE_OUTPUT_PATH):
        logger.info(f"Pre-computing YAKE hot keywords to {YAKE_OUTPUT_PATH}")
        extract_hot_keywords_yake(ARTICLES_PATH, YAKE_OUTPUT_PATH)
    else:
        logger.info(f"YAKE keywords exist: {YAKE_OUTPUT_PATH}")

    if not os.path.exists(KEYBERT_OUTPUT_PATH):
        logger.info(f"Pre-computing KeyBERT hot keywords to {KEYBERT_OUTPUT_PATH}")
        extract_hot_keywords_keybert(ARTICLES_PATH, KEYBERT_OUTPUT_PATH)
    else:
        logger.info(f"KeyBERT keywords exist: {KEYBERT_OUTPUT_PATH}")

    # Load into memory
    def load_keywords_from_file(file_path):
        df_kw = pd.read_json(file_path)
        return dict(zip(df_kw["id"], df_kw["hot_keywords"]))

    hot_keywords_cache["YAKE"] = load_keywords_from_file(YAKE_OUTPUT_PATH)
    hot_keywords_cache["KeyBERT"] = load_keywords_from_file(KEYBERT_OUTPUT_PATH)

    logger.info("Hot keywords pre-loaded into memory.")
    logger.info("All models initialized successfully!")


def search_and_display(query, hot_keyword_method, search_method):
    """
    Perform search and render results with hot keywords.

    Args:
        query (str): User query.
        hot_keyword_method (str): "YAKE" or "KeyBERT".
        search_method (str): Search backend.

    Returns:
        str: HTML-formatted results.
    """
    if not query or not query.strip():
        return "<p style='text-align: center; color: var(--color-accent);'>Warning: Please enter a search query.</p>"

    try:
        logger.info(f"Query: '{query}' | Keywords: {hot_keyword_method} | Search: {search_method}")

        if search_method == "Lexical Search":
            results = search_articles_bm25(query, df, bm25_model, top_n=5)
        elif search_method == "Spacy Search":
            results = search_articles_spacy(query, df, spacy_index, top_n=5)
        else:
            results = search_articles_semantic(query, df, transformer_index, top_n=5)

        if not results:
            return "<p style='text-align: center;'>No results found.</p>"

        kw_lookup = hot_keywords_cache.get(hot_keyword_method, {})

        html = f"""
        <div style="margin-bottom: 24px; padding-bottom: 16px; border-bottom: 2px solid var(--border-color-primary);">
            <h2 style="margin: 0 0 8px 0;">Search: {query}</h2>
            <p style="color: var(--body-text-color-subdued); margin: 0;">
                <strong>Method:</strong> {search_method} | <strong>Keywords:</strong> {hot_keyword_method}
            </p>
        </div>
        """

        for result in results:
            rank = result["rank"]
            article_id = result["id"]
            text = result["text"]
            keywords = kw_lookup.get(article_id, [])
            keywords_str = ", ".join(keywords) if keywords else "No keywords"

            html += f"""
            <div style="margin-bottom: 24px; padding: 20px; border-radius: 8px; 
                        background: var(--block-background-fill); 
                        border: 1px solid var(--block-border-color);">
                <div style="display: inline-block; padding: 4px 12px; border-radius: 4px; 
                           background: var(--color-accent); color: white; font-weight: 600; 
                           margin-bottom: 12px;">Rank {rank}</div>

                <div style="margin: 16px 0; padding: 12px; border-radius: 6px; 
                           background: var(--background-fill-secondary); 
                           border-left: 3px solid var(--color-accent);">
                    <div style="font-weight: 600; margin-bottom: 8px;">Hot Keywords</div>
                    <div style="font-style: italic; color: var(--body-text-color-subdued);">{keywords_str}</div>
                </div>

                <div style="margin-top: 16px; padding-top: 16px; border-top: 1px solid var(--border-color-primary);">
                    <div style="font-weight: 600; margin-bottom: 12px;">Article</div>
                    <div style="text-align: justify; line-height: 1.7;">{text}</div>
                </div>
            </div>
            """

        return html

    except Exception as e:
        logger.error(f"Error during search: {str(e)}")
        return f"<p style='text-align: center; color: var(--color-accent);'>Warning: Error: {str(e)}</p>"


def create_interface():
    """Build and return the Gradio interface."""
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        font=gr.themes.GoogleFont("Inter")
    )

    css = """
    .gradio-container {
        max-width: 1400px !important;
        margin: auto !important;
    }
    footer {display: none !important;}
    """

    with gr.Blocks(title="SemSearch", theme=theme, css=css) as demo:
        gr.Markdown("# SemSearch\n### Advanced Article Search Engine")

        with gr.Row():
            query_input = gr.Textbox(
                label="Search Query",
                placeholder="Enter your search query...",
                lines=2,
                scale=2
            )
            with gr.Column(scale=1):
                hot_keyword_method = gr.Dropdown(
                    choices=["YAKE", "KeyBERT"],
                    value="YAKE",
                    label="Keywords Method"
                )
                search_method = gr.Dropdown(
                    choices=["Lexical Search", "Spacy Search", "Semantic Search"],
                    value="Semantic Search",
                    label="Search Method"
                )

        search_button = gr.Button("Search", variant="primary", size="lg")
        output_display = gr.HTML(label="Results")

        search_button.click(
            fn=search_and_display,
            inputs=[query_input, hot_keyword_method, search_method],
            outputs=output_display
        )
        query_input.submit(
            fn=search_and_display,
            inputs=[query_input, hot_keyword_method, search_method],
            outputs=output_display
        )

        gr.Markdown("""
        ---
        **Lexical Search** - BM25 keyword matching | 
        **Spacy Search** - spaCy embeddings | 
        **Semantic Search** - Transformer-based
        """)

    return demo


if __name__ == "__main__":
    initialize_models()
    demo = create_interface()
    demo.launch(
        share=config['interface']['server']['share'],
        server_name=config['interface']['server']['host'],
        server_port=config['interface']['server']['port'],
        show_error=True
    )