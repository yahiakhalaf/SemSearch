import os
import pandas as pd
import yake
from keybert import KeyBERT
from src.logger_config import load_logger
from src.config import load_config

config = load_config()
logger = load_logger()

YAKE_CFG = config['keywords']['yake']
KEYBERT_CFG = config['keywords']['keybert']
KEYBERT_MODEL = config['models']['keybert']['model_name']


def get_yake_keywords_from_text(text, top_n=None):
    """
    Extract top keywords from a single text using YAKE.

    Args:
        text (str): The input article text.
        top_n (int): Number of keywords to extract.

    Returns:
        list[str]: List of extracted keyword strings.
    """
    top_n = top_n or YAKE_CFG['top_n']
    kw_extractor = yake.KeywordExtractor(
        lan="en",
        n=3,                      # Extract up to 3-word phrases
        dedupLim=0.2,             # High deduplication threshold
        dedupFunc="seqm",
        windowsSize=3,
        use_stemmer=True,
        top=top_n,
        features=None
    )
    return [kw[0] for kw in kw_extractor.extract_keywords(text)]


def extract_hot_keywords_yake(input_path, output_path, top_n=None):
    """
    Extract hot keywords for each article using YAKE and save them to JSON.

    Args:
        input_path (str): Path to the input CSV containing article texts.
        output_path (str): Path to save the output JSON file.
        top_n (int): Number of keywords to extract per article.
    """
    top_n = top_n or YAKE_CFG['top_n']
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
kw_model = KeyBERT(model=KEYBERT_MODEL)

def get_keybert_keywords_from_text(text, top_n=None):
    """
    Extract keywords for a single document using KeyBERT.

    Args:
        text (str): Input text.
        top_n (int): Number of top keywords to extract.

    Returns:
        list[str]: List of extracted keywords.
    """
    top_n = top_n or KEYBERT_CFG['top_n']
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=tuple(KEYBERT_CFG['ngram_range']), 
        stop_words=KEYBERT_CFG['stop_words'],
        top_n=top_n
    )
    return [kw[0] for kw in keywords]


def extract_hot_keywords_keybert(input_path, output_path, top_n=None):
    """
    Extract hot keywords for each article using KeyBERT and save to JSON.

    Args:
        input_path (str): Path to input CSV containing article texts.
        output_path (str): Path to save the output JSON file.
        top_n (int): Number of keywords per article.
    """
    top_n = top_n or KEYBERT_CFG['top_n']
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
