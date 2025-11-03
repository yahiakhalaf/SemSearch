# 📰 SemSearch - Advanced Article Search Engine

A sophisticated semantic search system that enables efficient retrieval of news articles using multiple search methodologies and keyword extraction techniques.

## 📖 Overview

SemSearch is a comprehensive information retrieval system designed to search through news articles using state-of-the-art NLP techniques. The system implements and compares three different search approaches:

- **Traditional lexical search** using BM25 algorithm for keyword-based matching
- **Neural semantic search** using spaCy's pre-trained word embeddings
- **Deep semantic search** using transformer-based sentence embeddings

Each search method can be combined with automatic keyword extraction (YAKE or KeyBERT) to highlight the most important terms in retrieved articles. The system provides an intuitive web interface built with Gradio, allowing users to easily compare different search strategies and explore results interactively.

The project processes 2,225 BBC news articles across 5 categories (Business, Entertainment, Politics, Sport, Tech) and uses FAISS for efficient similarity search on large embedding spaces.

## 🌟 Features

- **Three Search Methods**:
  - **Lexical Search**: BM25-based keyword matching for traditional IR
  - **spaCy Search**: Semantic search using spaCy word embeddings
  - **Semantic Search**: Transformer-based embeddings (SentenceTransformers) for deep semantic understanding

- **Keyword Extraction**:
  - **YAKE**: Unsupervised keyword extraction
  - **KeyBERT**: Transformer-based keyword extraction

- **Interactive Web Interface**: Built with Gradio for easy querying and result visualization

- **FAISS Integration**: Fast similarity search with indexed embeddings

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Results](#results)

## 🚀 Installation

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yahiakhalaf/SemSearch.git
cd SemSearch
```

2. Create a virtual environment and install uv locally:
```bash
python -m venv .venv
# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

pip install uv
```

3. Sync dependencies and create virtual environment:
```bash
uv sync
```

4. Download spaCy models:
```bash
uv run python -m spacy download en_core_web_sm
uv run python -m spacy download en_core_web_lg
```

## 📊 Dataset

The project uses the **BBC News Articles** dataset containing articles across 5 categories:
- Business
- Entertainment
- Politics
- Sport
- Tech

### Data Processing

Run the data processing script to prepare the dataset:
```bash
python -m src.process_data
```

This will:
- Load raw text files from `data/raw/NewsArticles/`
- Combine them into a structured CSV
- Save to `data/processed/articles.csv`

## 💻 Usage

### Web Interface

Launch the Gradio interface:
```bash
python -m src.gradio
```

Access the application at `http://127.0.0.1:7860`

The interface allows you to:
- Enter search queries
- Select search method (Lexical/spaCy/Semantic)
- Choose keyword extraction method (YAKE/KeyBERT)
- View ranked results with extracted keywords

## 🏗️ Architecture

### Search Pipeline

1. **Text Preprocessing**:
   - Lowercasing
   - Contraction expansion
   - Lemmatization
   - Stop word removal

2. **Indexing**:
   - BM25 corpus tokenization
   - spaCy embedding generation
   - Transformer embedding generation
   - FAISS index creation

3. **Query Processing**:
   - Query preprocessing
   - Vector encoding (for semantic methods)
   - Similarity computation
   - Result ranking

### Models Used

- **BM25**: `rank-bm25` library
- **spaCy**: `en_core_web_lg` (300-dim word vectors)
- **SentenceTransformers**: `all-MiniLM-L6-v2` (384-dim embeddings)
- **KeyBERT**: `all-MiniLM-L6-v2`
- **YAKE**: Unsupervised keyword extraction

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
# Configuration file for SemSearch Article Search System

# Data paths
data:
  raw_dir: "data/raw/NewsArticles"
  processed_dir: "data/processed"
  articles_csv: "data/processed/articles.csv"
  embeddings_dir: "data/embeddings"
  hot_keywords_dir: "data/hot_keywords"
  evaluation_dir: "data/evaluation/"
  evaluation_results: "evaluation/evaluation_results.csv"
# Model paths
models:
  bm25:
    corpus_path: "models/bm25_corpus.pkl"
    model_path: "models/bm25_model.pkl"
  
  transformer:
    embeddings_path: "data/embeddings/transformer_embeddings.npy"
    faiss_index_path: "data/embeddings/faiss_transformer_index.bin"
    model_name: "all-MiniLM-L6-v2"
  
  spacy:
    embeddings_path: "data/embeddings/spacy_doc_vectors.npy"
    faiss_index_path: "data/embeddings/faiss_spacy_index.bin"
    model_name: "en_core_web_lg"
    small_model: "en_core_web_sm"

  keybert:
    model_name: "all-MiniLM-L6-v2"

# Keyword extraction settings
keywords:
  yake:
    language: "en"
    n_gram: 3  # Extract up to 3-word phrases
    dedup_lim: 0.9
    dedup_func: "seqm"
    window_size: 1
    top_n: 10
  
  keybert:
    ngram_range: [1, 3]  # 1-3 word phrases
    stop_words: "english"
    top_n: 10


# Gradio interface settings
interface:
  server:
    host: "127.0.0.1"
    port: 7860
    share: false

# Logging settings
logging:
  dir: "logs"
  level: "INFO"


```

## 📈 Evaluation

The system is evaluated using three metrics:

- **Precision@5**: Proportion of relevant documents in top-5 results
- **NDCG@5**: Normalized Discounted Cumulative Gain at position 5
- **MAP**: Mean Average Precision

### Running Evaluation

```python
python -m src.evaluation 
```

### Sample Results

| Model          | Precision@5 | NDCG@5 | MAP    |
|----------------|-------------|--------|--------|
| Lexical (BM25) | 0.229       | 0.889  | 0.203  |
| spaCy          | 0.187       | 0.613  | 0.142  |
| Semantic       | **0.244**   | **0.932** | **0.218** |

## 📁 Project Structure

```
semsearch/
├── src/
│   ├── evaluation_utils.py
│   ├── evaluation.py         
│   ├── gradio.py         
│   ├── config.py              # Configuration loader
│   ├── logger_config.py       # Logging setup
│   ├── process_data.py        # Data processing
│   ├── utils.py               # Utility functions
│   ├── lexical_search.py      # BM25 search
│   ├── spacy_search.py        # spaCy-based search
│   ├── semantic_search.py     # Transformer-based search
│   └── hot_keywords.py        # Keyword extraction
├── evaluation/
├── notebooks/
├── data/
│   ├── raw/                   # Raw article files
│   ├── processed/             # Processed CSV
│   ├── embeddings/            # Cached embeddings & indices
│   ├── hot_keywords/
│   └── evaluation/            # Ground truth data
├── config.yaml                # Configuration file
├── pyproject.toml             # Dependencies
└── README.md
```

## 🔧 Key Dependencies

- `gradio>=5.49.1` - Web interface
- `sentence-transformers` - Transformer embeddings
- `spacy>=3.8.7` - NLP processing
- `faiss-cpu>=1.12.0` - Similarity search
- `rank-bm25>=0.2.2` - BM25 implementation
- `yake>=0.6.0` - Keyword extraction
- `keybert>=0.9.0` - Transformer-based keywords
- `pandas>=2.3.3` - Data manipulation

## 🎯 Use Cases

- News article recommendation
- Research paper discovery
- Document retrieval systems
- Content-based filtering
- Semantic search applications

## 📄 License

This project is licensed under the MIT License.

---

**Built with ❤️ using Python, spaCy, and SentenceTransformers**

