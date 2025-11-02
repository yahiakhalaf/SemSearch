import os
import pandas as pd
import yake
from keybert import KeyBERT
from src.logger_config import load_logger


# Initialize logger
logger = load_logger()


def get_yake_keywords_from_text(text, top_n=10):
    """
    Extract top keywords from a single text using YAKE.

    Args:
        text (str): The input article text.
        top_n (int): Number of keywords to extract.

    Returns:
        list[str]: List of extracted keyword strings.
    """
    kw_extractor = yake.KeywordExtractor(
        lan="en",
        n=3,                      # Extract up to 3-word phrases
        dedupLim=0.9,             # High deduplication threshold
        dedupFunc="seqm",
        windowsSize=1,
        top=top_n,
        features=None
    )
    return [kw[0] for kw in kw_extractor.extract_keywords(text)]


def extract_hot_keywords_yake(input_path, output_path, top_n=10):
    """
    Extract hot keywords for each article using YAKE and save them to JSON.

    Args:
        input_path (str): Path to the input CSV containing article texts.
        output_path (str): Path to save the output JSON file.
        top_n (int): Number of keywords to extract per article.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load data
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} articles from {input_path}")

    # Extract keywords for each article
    logger.info("Extracting hot keywords using YAKE...")
    df["hot_keywords"] = df["text"].apply(lambda text: get_yake_keywords_from_text(text, top_n=top_n))

    # Save results
    df[["id", "hot_keywords"]].to_json(
        output_path,
        orient="records",
        indent=2,
        force_ascii=False
    )

    logger.info(f"Saved YAKE hot keywords to {output_path}")



# Load KeyBERT model 
kw_model = KeyBERT(model='all-MiniLM-L6-v2')

def get_keybert_keywords_from_text(text, top_n=10):
    """
    Extract keywords for a single document using KeyBERT.

    Args:
        text (str): Input text.
        top_n (int): Number of top keywords to extract.

    Returns:
        list[str]: List of extracted keywords.
    """
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 3),  # 1–3 word phrases
        stop_words='english',
        top_n=top_n
    )
    return [kw[0] for kw in keywords]


def extract_hot_keywords_keybert(input_path, output_path, top_n=10):
    """
    Extract hot keywords for each article using KeyBERT and save to JSON.

    Args:
        input_path (str): Path to input CSV containing article texts.
        output_path (str): Path to save the output JSON file.
        top_n (int): Number of keywords per article.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load dataset
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} articles from {input_path}")

    # Extract keywords
    logger.info("Extracting hot keywords using KeyBERT...")
    df["hot_keywords"] = df["text"].apply(lambda t: get_keybert_keywords_from_text(t, top_n=top_n))

    # Save results to JSON
    df[["id", "hot_keywords"]].to_json(
        output_path,
        orient="records",
        indent=2,
        force_ascii=False
    )
    logger.info(f"Saved KeyBERT hot keywords to {output_path}")
