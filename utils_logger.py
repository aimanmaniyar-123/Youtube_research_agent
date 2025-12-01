"""
Logging helpers - simple wrapper around Python logging.
Provides get_logger(name) -> logging.Logger
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

# Force UTF-8 everywhere
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


def setup_root_logger(
    level: str = "INFO",
    fmt: Optional[str] = None,
    log_filename: Optional[str] = None,
):
    """
    Initialize root logger once. Safe to call multiple times.
    """
    if fmt is None:
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    if log_filename is None:
        log_filename = LOG_DIR / f"youtube_agent_{datetime.utcnow().strftime('%Y%m%d')}.log"

    root = logging.getLogger()
    if root.handlers:
        # already configured
        return

    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # console handler
    ch = logging.StreamHandler(sys.stdout)
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    ch.setFormatter(logging.Formatter(fmt))
    root.addHandler(ch)

    # file handler
    fh = logging.FileHandler(log_filename)
    fh.setLevel(getattr(logging, level.upper(), logging.INFO))
    fh.setFormatter(logging.Formatter(fmt))
    root.addHandler(fh)


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Returns a configured logger instance. Call setup_root_logger(...) at app startup.
    """
    setup_root_logger(level=level)
    return logging.getLogger(name)
