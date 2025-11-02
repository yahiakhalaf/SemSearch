import gradio as gr
import pandas as pd
from src.logger_config import setup_logger
from src.hot_keywords import get_yake_keywords_from_text, get_keybert_keywords_from_text
from src.lexical_search import build_bm25, search_articles_bm25
from src.semantic_search import compute_transformer_doc_vectors, search_articles_semantic
from src.spacy_search import compute_spacy_doc_vectors, search_articles_spacy
from src.utils import build_or_load_faiss_index

logger = setup_logger()

# File paths
ARTICLES_PATH = "data/processed/articles.csv"
BM25_CORPUS_PATH = "data/embeddings/bm25_corpus.pkl"
BM25_MODEL_PATH = "data/embeddings/bm25_model.pkl"
TRANSFORMER_EMBEDDINGS_PATH = "data/embeddings/transformer_embeddings.npy"
SPACY_EMBEDDINGS_PATH = "data/embeddings/spacy_doc_vectors.npy"
FAISS_TRANSFORMER_INDEX_PATH = "data/embeddings/faiss_transformer_index.bin"
FAISS_SPACY_INDEX_PATH = "data/embeddings/faiss_spacy_index.bin"

# Global variables
df = None
bm25_model = None
transformer_index = None
spacy_index = None


def initialize_models():
    """Initialize all models and load dataset."""
    global df, bm25_model, transformer_index, spacy_index
    
    logger.info("Initializing models...")
    
    # Load dataset
    df = pd.read_csv(ARTICLES_PATH)
    logger.info(f"Loaded {len(df)} articles")
    
    # Load BM25
    bm25_model, _ = build_bm25(df, "text", BM25_CORPUS_PATH, BM25_MODEL_PATH)
    
    # Load Transformer embeddings and FAISS
    transformer_embeddings = compute_transformer_doc_vectors(df, "text", TRANSFORMER_EMBEDDINGS_PATH)
    transformer_index = build_or_load_faiss_index(FAISS_TRANSFORMER_INDEX_PATH, transformer_embeddings)
    
    # Load spaCy embeddings and FAISS
    spacy_embeddings = compute_spacy_doc_vectors(df, "text", SPACY_EMBEDDINGS_PATH)
    spacy_index = build_or_load_faiss_index(FAISS_SPACY_INDEX_PATH, spacy_embeddings)
    
    logger.info("Models initialized successfully!")


def search_and_display(query, hot_keyword_method, search_method):
    """Perform search and return formatted HTML results."""
    if not query or not query.strip():
        return "<p style='text-align: center; color: var(--color-accent);'>⚠️ Please enter a search query.</p>"
    
    try:
        logger.info(f"Query: '{query}' | Keywords: {hot_keyword_method} | Search: {search_method}")
        
        # Perform search
        if search_method == "Lexical Search":
            results = search_articles_bm25(query, df, bm25_model, top_n=5)
        elif search_method == "Spacy Search":
            results = search_articles_spacy(query, df, spacy_index, top_n=5)
        else:  # Semantic Search
            results = search_articles_semantic(query, df, transformer_index, top_n=5)
        
        if not results:
            return "<p style='text-align: center;'>No results found.</p>"
        
        # Build HTML output
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
            text = result["text"]
            
            # Extract keywords
            if hot_keyword_method == "YAKE":
                keywords = get_yake_keywords_from_text(text, top_n=10)
            else:
                keywords = get_keybert_keywords_from_text(text, top_n=10)
            
            keywords_str = ", ".join(keywords) if keywords else "No keywords extracted"
            
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
        return f"<p style='text-align: center; color: var(--color-accent);'>⚠️ Error: {str(e)}</p>"


def create_interface():
    """Create Gradio interface."""
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
        gr.Markdown("# 📰 SemSearch\n### Advanced Article Search Engine")
        
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
        
        search_button = gr.Button("🔍 Search", variant="primary", size="lg")
        output_display = gr.HTML(label="Results")
        
        # Connect events
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
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )