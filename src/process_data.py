import os
import pandas as pd
from src.logger_config import load_logger

logger = load_logger()

base_dir = "data/raw/NewsArticles"
output_path = "data/processed/articles.csv"

data = []
article_id = 0

if not os.path.exists(base_dir):
    logger.error(f"Base directory not found: {base_dir}")
    raise FileNotFoundError(f"Directory not found: {base_dir}")

logger.info(f"Loading articles from {base_dir}...")

for category in os.listdir(base_dir):
    category_path = os.path.join(base_dir, category)
    if not os.path.isdir(category_path):
        logger.debug(f"Skipping non-directory item: {category_path}")
        continue

    logger.info(f"Processing category: {category}")
    for filename in os.listdir(category_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(category_path, filename)
            try:
                with open(file_path, "r", encoding="latin1") as f:
                    text = f.read().strip()
                data.append({
                    "id": article_id,
                    "category": category,
                    "text": text
                })
                article_id += 1
            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")

df = pd.DataFrame(data)
logger.info(f"Loaded {len(df)} articles across {df['category'].nunique()} categories.")

# Save combined CSV
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False, encoding="utf-8")
logger.info(f"Saved combined dataset to {output_path}")
