"""
Database Service – FINAL VERSION (No ORM Models Required)
Safely handles:
✔ save_channel()
✔ save_videos()
✔ save_transcripts()
✔ never crashes even if tables/models don't exist
"""

from typing import Dict, List
from utils_logger import get_logger

logger = get_logger("database_service")


class DatabaseService:
    """
    A lightweight database service that *pretends* to save data.
    Your project does NOT actually use ORM tables,
    so this service simply logs the operations and avoids crashes.
    """

    def __init__(self):
        pass  # No DB session required

    # ------------------------------------------------------
    # CHANNEL SAVE
    # ------------------------------------------------------
    def save_channel(self, channel: Dict):
        """
        Save channel metadata (simulated).
        Prevents crash when ORM models do not exist.
        """
        try:
            logger.info(f"[DB] Channel saved (simulated): {channel.get('id')}")
        except Exception as e:
            logger.error(f"[DB] Failed to save channel: {e}")

    # ------------------------------------------------------
    # VIDEOS SAVE
    # ------------------------------------------------------
    def save_videos(self, videos: List[Dict]):
        """
        Save video objects (simulated).
        """
        try:
            logger.info(f"[DB] Saved {len(videos)} videos (simulated)")
        except Exception as e:
            logger.error(f"[DB] Failed to save videos: {e}")

    # ------------------------------------------------------
    # TRANSCRIPTS SAVE
    # ------------------------------------------------------
    def save_transcripts(self, transcripts: Dict[str, str]):
        """
        Save transcripts (simulated).
        """
        try:
            logger.info(f"[DB] Saved {len(transcripts)} transcripts (simulated)")
        except Exception as e:
            logger.error(f"[DB] Failed to save transcripts: {e}")

    # ------------------------------------------------------
    def close(self):
        """Nothing to close; included for compatibility"""
        pass
