"""
FAST MODE YouTube API Service
(Uses YouTube Data API v3 – metadata only)
"""

import requests
import re
from typing import Optional, Dict, List
from config_settings import settings
from utils_logger import get_logger

logger = get_logger("YouTubeAPIService")


class YouTubeAPIService:
    """
    FAST MODE version:
    - Metadata only (no transcripts, no captions)
    - Very low quota usage (1–2 units per request)
    - Compatible with: from services_youtube_client import YouTubeAPIService
    """

    BASE_URL = "https://www.googleapis.com/youtube/v3"

    def __init__(self):
        self.api_key = settings.youtube_api_key
        if not self.api_key:
            raise ValueError("YouTube API key missing")

    # ======================================================================
    # ID / HANDLE / NAME RESOLUTION
    # ======================================================================
    def _extract_channel_id(self, identifier: str) -> str:
        """Extract channelId, handle, or name."""

        identifier = identifier.strip()

        # CASE 1: Already a channelId (UCxxxxxx)
        if identifier.startswith("UC") and len(identifier) >= 24:
            return identifier

        # CASE 2: Handle @something
        if identifier.startswith("@"):
            return identifier  # handle

        # CASE 3: YouTube URLs
        patterns = {
            "channel": r"youtube\.com/channel/([UC][A-Za-z0-9_-]{22})",
            "handle": r"youtube\.com/@([A-Za-z0-9_-]+)",
            "custom": r"youtube\.com/c/([A-Za-z0-9_-]+)",
            "user": r"youtube\.com/user/([A-Za-z0-9_-]+)"
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, identifier)
            if match:
                extracted = match.group(1)
                if key == "handle":
                    return f"@{extracted}"
                return extracted

        # DEFAULT: treat as name (searchable)
        return identifier

    # ======================================================================
    # SEARCH CHANNEL BY HANDLE / NAME
    # ======================================================================
    def get_channel_by_name_or_handle(self, query: str) -> Optional[Dict]:

        query = self._extract_channel_id(query)
        logger.info(f"[YT] Resolving channel: {query}")

        # 1. Direct handle lookup (NEW YouTube API param)
        if query.startswith("@"):
            url = f"{self.BASE_URL}/channels"
            params = {
                "part": "snippet,statistics,contentDetails",
                "forHandle": query[1:],   # remove "@"
                "key": self.api_key,
            }
            resp = requests.get(url, params=params).json()
            
            if resp.get("items"):
                return resp["items"][0]

        # 2. Direct channelId
        if query.startswith("UC") and len(query) >= 24:
            return self.get_channel_details(query)

        # 3. Fallback search using name
        url = f"{self.BASE_URL}/search"
        params = {
            "part": "snippet",
            "type": "channel",
            "q": query,
            "maxResults": 5,
            "key": self.api_key,
        }
        resp = requests.get(url, params=params).json()
        
        for item in resp.get("items", []):
            cid = item["id"]["channelId"]
            return self.get_channel_details(cid)

        return None


    # Handle search: youtube.com/@handle
    def _search_exact_handle(self, handle: str) -> Optional[Dict]:
        search_query = handle.replace("@", "")

        url = f"{self.BASE_URL}/search"
        params = {
            "part": "snippet",
            "type": "channel",
            "q": search_query,
            "maxResults": 5,
            "key": self.api_key,
        }

        resp = requests.get(url, params=params).json()
        items = resp.get("items", [])
        if not items:
            return None

        # Find matching customUrl
        for it in items:
            cid = it["id"]["channelId"]
            full = self.get_channel_details(cid)

            custom_url = full["snippet"].get("customUrl", "")
            if custom_url.lower() == search_query.lower():
                return full

        return None

    # General channel search
    def _search_channel_generic(self, name: str) -> Optional[Dict]:
        url = f"{self.BASE_URL}/search"
        params = {
            "part": "snippet",
            "type": "channel",
            "q": name,
            "maxResults": 1,
            "key": self.api_key,
        }

        resp = requests.get(url, params=params).json()
        items = resp.get("items", [])
        if not items:
            return None

        channel_id = items[0]["id"]["channelId"]
        return self.get_channel_details(channel_id)

    # ======================================================================
    # CHANNEL DETAILS
    # ======================================================================
    def get_channel_details(self, channel_id: str) -> Optional[Dict]:
        url = f"{self.BASE_URL}/channels"
        params = {
            "part": "snippet,statistics,brandingSettings,contentDetails",
            "id": channel_id,
            "key": self.api_key,
        }

        resp = requests.get(url, params=params).json()
        items = resp.get("items", [])
        if not items:
            return None

        return items[0]

    # ======================================================================
    # VIDEOS – METADATA ONLY (FAST MODE)
    # ======================================================================
    def get_channel_videos(self, channel_id: str, max_results: int = 25) -> List[Dict]:
        playlist_id = self._get_uploads_playlist(channel_id)
        if not playlist_id:
            return []

        url = f"{self.BASE_URL}/playlistItems"
        params = {
            "part": "snippet,contentDetails",
            "playlistId": playlist_id,
            "maxResults": max_results,
            "key": self.api_key,
        }

        resp = requests.get(url, params=params).json()
        items = resp.get("items", [])

        videos = []
        for item in items:
            snip = item.get("snippet", {})
            videos.append({
                "id": item.get("contentDetails", {}).get("videoId"),
                "title": snip.get("title"),
                "description": snip.get("description"),
                "publishedAt": snip.get("publishedAt"),
            })

        return videos


    def _get_uploads_playlist(self, channel_id: str) -> Optional[str]:
        url = f"{self.BASE_URL}/channels"
        params = {
            "part": "contentDetails",
            "id": channel_id,
            "key": self.api_key,
        }

        resp = requests.get(url, params=params).json()
        items = resp.get("items", [])

        if not items:
            return None

        return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
