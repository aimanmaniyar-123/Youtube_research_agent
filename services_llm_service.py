"""
LLM Service for YouTube Research Agent
FAST MODE (Groq + single prompt)
"""

'''import json
import logging
from typing import Dict, Any
from groq import Groq
from utils_logger import get_logger

logger = get_logger("llm_service")

# ---------------------------------------------------------
# SINGLETON INSTANCE
# ---------------------------------------------------------
_llm_service_instance = None


def get_llm_service(config=None):
    """Return global singleton."""
    global _llm_service_instance
    if _llm_service_instance is None:
        _llm_service_instance = LLMService(config)
    return _llm_service_instance


# ---------------------------------------------------------
# LLM SERVICE
# ---------------------------------------------------------
class LLMService:
    def __init__(self, config: Dict[str, Any] = None):

        if config:
            self.api_key = config.get("groq_api_key")
            self.model = config.get("groq_model", "llama-3.1-8b-instant")
        else:
            raise ValueError("LLMService requires configuration on first load.")

        if not self.api_key:
            raise ValueError("❌ Groq API key missing")

        self.client = Groq(api_key=self.api_key)

    # -----------------------------------------------------
    async def test_connection(self):
        """Quick connectivity test."""
        try:
            res = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Say OK"}],
                max_tokens=3,
            )
            return "OK" in res.choices[0].message.content

        except Exception as e:
            logger.error(f"Groq connection failed: {e}")
            return False

    # -----------------------------------------------------
    #  FAST MODE — ONLY 1 ARGUMENT: prompt
    # -----------------------------------------------------
    async def generate_analysis(self, prompt: str) -> Dict[str, Any]:
        """
        FAST MODE LLM wrapper.
        Accepts ONLY a single prompt string.
        Output MUST be structured JSON.
        """

        system_prompt = """
You are an expert YouTube analytics engine.
Respond ONLY with valid JSON. 
Do NOT add extra text or explanations.
If data is missing, return "Not Available".
"""

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                max_tokens=1500,
                messages=[
                    {"role": "system", "content": system_prompt.strip()},
                    {"role": "user", "content": prompt.strip()},
                ],
            )

            raw_text = completion.choices[0].message.content.strip()

            # Try direct JSON parsing
            try:
                return json.loads(raw_text)

            except Exception:
                logger.error("LLM returned non-JSON. Attempting cleanup...")

                cleaned = self._extract_json(raw_text)
                return json.loads(cleaned)

        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            raise

    # -----------------------------------------------------
    # JSON extraction fallback
    # -----------------------------------------------------
    def _extract_json(self, text: str) -> str:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON object found in LLM output.")
        return text[start:end + 1]'''


"""
services_llm_service.py

LLM Service for YouTube Research Agent — FAST MODE (Hybrid Analytics)
- Accepts: channel_info, videos, semantic_texts, hybrid_analytics
- Builds a compact, safe prompt (trims long fields)
- Calls Groq chat completions and returns strict JSON
"""

import json
import logging
from typing import Dict, List, Any
from groq import Groq
from utils_logger import get_logger

logger = get_logger("llm_service")

# ---------------------------------------------------------
# SINGLETON INSTANCE
# ---------------------------------------------------------
_llm_service_instance = None


def get_llm_service(config: Dict[str, Any] = None):
    """Return global singleton. First call must provide config."""
    global _llm_service_instance
    if _llm_service_instance is None:
        _llm_service_instance = LLMService(config)
    return _llm_service_instance


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def _shorten_str(s: str, limit: int = 400) -> str:
    if s is None:
        return ""
    s = str(s)
    if len(s) <= limit:
        return s
    return s[: limit - 3] + "..."


def _safe_json_dumps(obj: Any, indent: int = 2, limit: int = 1200) -> str:
    try:
        j = json.dumps(obj, ensure_ascii=False, indent=indent)
    except Exception:
        j = str(obj)
    if len(j) <= limit:
        return j
    return j[: limit - 3] + "..."


# ---------------------------------------------------------
# LLM Service
# ---------------------------------------------------------
class LLMService:
    def __init__(self, config: Dict[str, Any] = None):
        if config:
            self.api_key = config.get("groq_api_key")
            self.model = config.get("groq_model", "llama-3.1-8b-instant")
        else:
            raise ValueError("LLMService requires configuration on first load.")

        if not self.api_key:
            raise ValueError("❌ Groq API key missing")

        self.client = Groq(api_key=self.api_key)

    # -----------------------------------------------------
    async def test_connection(self) -> bool:
        """Quick connectivity test."""
        try:
            res = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Say OK"}],
                max_tokens=3,
            )
            out = res.choices[0].message.content
            return "OK" in (out or "")
        except Exception as e:
            logger.error(f"Groq test failed: {e}")
            return False

    # -----------------------------------------------------
    async def generate_analysis(
        self,
        channel_info: Dict[str, Any],
        videos: List[Dict[str, Any]],
        semantic_texts: List[str],
        hybrid_analytics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate structured analysis JSON using:
        - channel_info (dict)
        - a sample list of videos (list of dicts)
        - semantic_texts (list of short strings)
        - hybrid_analytics (precomputed numeric analytics)
        """

        # Safety limits
        MAX_VIDEOS = 8
        MAX_SEMANTICS = 4
        MAX_PROMPT_CHARS = 5200  # rough safe limit

        try:
            # Prepare compact channel block
            channel_block = {
                "title": _shorten_str(channel_info.get("snippet", {}).get("title", "")),
                "description": _shorten_str(channel_info.get("snippet", {}).get("description", ""), 300),
                "subscriberCount": channel_info.get("statistics", {}).get("subscriberCount", "Not Available"),
                "viewCount": channel_info.get("statistics", {}).get("viewCount", "Not Available"),
                "videoCount": channel_info.get("statistics", {}).get("videoCount", "Not Available"),
                "country": channel_info.get("snippet", {}).get("country", "Not Available"),
            }
            channel_json = _safe_json_dumps(channel_block, limit=800)

            # Prepare videos list (trim fields)
            short_videos = []
            for v in videos[:MAX_VIDEOS]:
                short_videos.append(
                    {
                        "id": v.get("id"),
                        "title": _shorten_str(v.get("title", ""), 200),
                        "description": _shorten_str(v.get("description", ""), 300),
                        "publishedAt": v.get("publishedAt"),
                        "viewCount": v.get("viewCount", "0"),
                    }
                )
            videos_json = _safe_json_dumps(short_videos, limit=1600)

            # Prepare semantic texts
            sems = [ _shorten_str(s, 300) for s in (semantic_texts or []) ][:MAX_SEMANTICS]
            semantic_block = "\n".join(sems) if sems else "None"

            # Hybrid analytics compacted
            hybrid_json = _safe_json_dumps(hybrid_analytics, limit=1600)

            # Build system + user prompts (compact)
            system_prompt = """
You are an expert YouTube channel analyst. You will produce STRICT JSON only.
Do NOT include any explanatory text outside the JSON object.
If a value is missing, use "Not Available".
"""

            user_prompt = (
                "ANALYSIS TASK: produce a JSON object with these keys:\n"
                "executive_summary, metrics, themes, insights, recommendations,\n"
                "engagement_summary, engagement_insights, trends\n\n"
                "INPUT DATA (trimmed):\n\n"
                f"CHANNEL: {channel_json}\n\n"
                f"VIDEOS (sample): {videos_json}\n\n"
                f"SEMANTIC_EXTRACTS (sample): {semantic_block}\n\n"
                f"HYBRID_ANALYTICS: {hybrid_json}\n\n"
                "OUTPUT SCHEMA (required):\n"
                "{\n"
                '  "executive_summary": "...",\n'
                '  "metrics": {"subscriber_count": <number|Not Available>, "total_views": <number|Not Available>, "total_videos": <number|Not Available>, "average_views": <number|Not Available>, "engagement_rate": <number|Not Available>},\n'
                '  "themes": [{"name":"...","frequency":<number>,"engagement":"High|Medium|Low"}],\n'
                '  "insights": [{"text":"...","confidence":"Low|Medium|High","category":"Performance|Content|Engagement|Audience"}],\n'
                '  "recommendations": [{"title":"...","description":"...","priority":"Low|Medium|High","impact":"Low|Medium|High"}],\n'
                '  "engagement_summary": {...},\n'
                '  "engagement_insights": [{"topic":"...","relative_engagement":"High|Medium|Low","reason":"..."}],\n'
                '  "trends": {...}\n'
                "}\n\n"
                "IMPORTANT: Keep responses concise and ensure valid JSON only."
            )

            # Safety: final prompt length check
            final_text = system_prompt + "\n" + user_prompt
            if len(final_text) > MAX_PROMPT_CHARS:
                # aggressive trimming - reduce videos/semantics sizes
                logger.warning("LLM prompt too large; applying aggressive trimming.")
                # reduce videos to 4 and semantics to 2
                short_videos = short_videos[:4]
                videos_json = _safe_json_dumps(short_videos, limit=800)
                sems = sems[:2]
                semantic_block = "\n".join(sems) if sems else "None"
                user_prompt = user_prompt.replace(
                    f"VIDEOS (sample): {videos_json}", f"VIDEOS (sample): {videos_json}"
                )

            # Call Groq
            completion = self.client.chat.completions.create(
                model=self.model,
                temperature=0.15,
                max_tokens=1500,
                messages=[
                    {"role": "system", "content": system_prompt.strip()},
                    {"role": "user", "content": user_prompt.strip()},
                ],
            )

            raw_text = completion.choices[0].message.content.strip()

            # Try JSON parse
            try:
                parsed = json.loads(raw_text)
                return parsed
            except Exception:
                logger.error("LLM returned non-JSON, attempting to extract JSON body.")
                cleaned = self._extract_json(raw_text)
                parsed = json.loads(cleaned)
                return parsed

        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            # re-raise so orchestrator can handle/log and fallback if needed
            raise

    # -----------------------------------------------------
    # JSON extraction fallback
    # -----------------------------------------------------
    def _extract_json(self, text: str) -> str:
        """
        Extract the first JSON object from a potentially noisy LLM output.
        """
        if not text or "{" not in text:
            raise ValueError("No JSON object found in LLM output.")

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Could not locate JSON boundaries in LLM output.")

        candidate = text[start:end + 1]

        # Basic bracket matching to ensure balanced braces
        stack = []
        for i, ch in enumerate(candidate):
            if ch == "{":
                stack.append(i)
            elif ch == "}":
                if stack:
                    stack.pop()
        if stack:
            # fallback: attempt to find the matching closing brace further in original text
            idx = start
            count = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    count += 1
                elif text[i] == "}":
                    count -= 1
                    if count == 0:
                        end = i
                        break
            if end <= start:
                raise ValueError("Unbalanced JSON braces in LLM output.")
            candidate = text[start:end+1]

        return candidate
