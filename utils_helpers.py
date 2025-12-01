import re
import aiohttp
import asyncio
from utils_logger import get_logger

logger = get_logger("utils")


# =====================================================================================
# 1. EXTRACT CHANNEL FROM USER TEXT
# =====================================================================================
def extract_channel_from_text(text: str) -> str:
    """
    Extracts YouTube channel handle / name / URL from natural text.

    Supports:
    - @handles → "@MrBeast"
    - https://youtube.com/@handle
    - https://youtube.com/channel/UC12345
    - Raw text containing channel name: "Analyze MrBeast channel"
    """

    text = text.strip()

    # ----------------------------------------------------------
    # CASE 1: @handle
    # ----------------------------------------------------------
    handle_match = re.search(r"@([A-Za-z0-9_]+)", text)
    if handle_match:
        handle = f"@{handle_match.group(1)}"
        logger.info(f"extract_channel_from_text → handle={handle}")
        return handle

    # ----------------------------------------------------------
    # CASE 2: Full URL
    # ----------------------------------------------------------
    url_match = re.search(r"(https?://[^\s]+)", text)
    if url_match:
        url = url_match.group(1)
        logger.info(f"extract_channel_from_text → URL found: {url}")

        # youtube.com/@handle
        handle = extract_handle_from_url(url)
        if handle:
            return handle

        # youtube.com/channel/UCxxxx
        channel_id = extract_channel_id_from_url(url)
        if channel_id:
            return channel_id

    # ----------------------------------------------------------
    # CASE 3: Look for probable channel name – simple heuristic
    # ----------------------------------------------------------
    words = text.split()
    if len(words) <= 5:
        guess = " ".join(words)
        logger.info(f"extract_channel_from_text → guess channel name={guess}")
        return guess

    # fallback
    logger.info("extract_channel_from_text → fallback to original text")
    return text


# =====================================================================================
# 2. Extract handle from URL
# =====================================================================================
def extract_handle_from_url(url: str) -> str:
    """
    Example:
      https://youtube.com/@MrBeast → @MrBeast
    """
    m = re.search(r"/@([A-Za-z0-9_\-.]+)", url)
    if m:
        return f"@{m.group(1)}"
    return None


# =====================================================================================
# 3. Extract channelId from URL
# =====================================================================================
def extract_channel_id_from_url(url: str) -> str:
    """
    Example:
      https://youtube.com/channel/UC1234567 → UC1234567
    """
    m = re.search(r"/channel/([A-Za-z0-9_-]+)", url)
    if m:
        return m.group(1)
    return None


# =====================================================================================
# 4. GET TRANSCRIPT (real API optional)
# =====================================================================================
async def get_transcript_or_empty(video_id: str) -> str:
    """
    In FAST MODE → no real transcripts needed.
    Always returns empty string (placeholder).
    """

    return ""


# =====================================================================================
# 5. OPTIONAL: async request helper (for future use)
# =====================================================================================
async def fetch_json(url: str, params=None, headers=None, timeout=10):
    """
    Safe async HTTP GET wrapper.
    """

    if params is None:
        params = {}
    if headers is None:
        headers = {}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers, timeout=timeout) as resp:
                if resp.status != 200:
                    logger.warning(f"fetch_json failed {url} → status {resp.status}")
                    return None
                return await resp.json()

    except Exception as e:
        logger.error(f"fetch_json error for {url}: {e}")
        return None
