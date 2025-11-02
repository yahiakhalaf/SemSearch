import logging
from datetime import datetime
from pathlib import Path
from src.config import load_config

config= load_config()

LOG_DIR = Path(config['logging']['dir'])
LOG_DIR.mkdir(exist_ok=True)


def setup_logger() -> logging.Logger:
    """
    Create a new timestamped log file and return a configured logger.
    """
    logger = logging.getLogger("app_logger")
    logger.setLevel(config['logging']['level'])
    logger.handlers.clear()

    # Timestamped log file
    log_file = LOG_DIR / f"app_{datetime.now():%Y%m%d_%H%M%S}.log"

    # File handler and formatter
    fmt = '%(asctime)s - [%(module)s] - %(levelname)s - %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt, datefmt))
    logger.addHandler(fh)

    logger.propagate = False
    logger.info(f"Logger initialized → {log_file}")

    return logger


def load_logger() -> logging.Logger:
    """
    Load the latest existing log file. If none exist, create a new one.
    """
    log_files = sorted(LOG_DIR.glob("app_*.log"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not log_files:
        return setup_logger()

    latest = log_files[0]

    logger = logging.getLogger("app_logger")
    logger.setLevel(config['logging']['level'])
    logger.handlers.clear()

    fmt = '%(asctime)s - [%(module)s] - %(levelname)s - %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'

    fh = logging.FileHandler(latest, mode='a', encoding='utf-8')
    fh.setFormatter(logging.Formatter(fmt, datefmt))
    logger.addHandler(fh)

    logger.propagate = False
    logger.info(f"Resuming logging in {latest}")

    return logger
