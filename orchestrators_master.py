"""
Master Orchestrator – FAST MODE (Optimized)
Handles:
✔ Chunking & semantic indexing
✔ Vector search (FAISS)
✔ Reduced semantic context (TPM safe)
✔ LLM summarization with small prompts
"""

'''import asyncio
from typing import Dict, List
from utils_logger import get_logger

from services_llm_service import get_llm_service
from services_vector_database_service import get_vector_db_service

logger = get_logger("orchestrator")


class MasterOrchestrator:
    def __init__(self):
        self.llm = get_llm_service()
        self.vector_db = get_vector_db_service()

    # -------------------------------------------------------------
    # Chunk Text (FAST MODE - metadata only)
    # -------------------------------------------------------------
    def chunk_text(self, text: str, max_size: int = 500):
        """
        Lightweight chunker:
        breaks metadata into blocks for semantic search.
        """
        chunks = []
        words = text.split()
        buf = []

        for w in words:
            buf.append(w)
            if len(" ".join(buf)) > max_size:
                chunks.append(" ".join(buf))
                buf = []

        if buf:
            chunks.append(" ".join(buf))

        return chunks

    # -------------------------------------------------------------
    # Create vector DB documents
    # -------------------------------------------------------------
    def prepare_documents(self, transcripts: Dict[str, str]):
        """
        transcripts = { video_id: synthetic text }
        Returns list of {"text": "...", "metadata": {...}}
        """

        docs = []

        for vid, text in transcripts.items():
            chunks = self.chunk_text(text)

            for i, chunk in enumerate(chunks):
                docs.append({
                    "text": chunk,
                    "metadata": {
                        "video_id": vid,
                        "chunk_id": i
                    }
                })

        return docs

    # -------------------------------------------------------------
    # Build a Groq-safe small LLM prompt
    # -------------------------------------------------------------
    def build_safe_prompt(self, channel, semantic_chunks: List[str]):
        snippet = channel.get("snippet", {})
        stats = channel.get("statistics", {})

        channel_title = snippet.get("title", "Unknown Channel")
        subscribers = stats.get("subscriberCount", "N/A")
        total_views = stats.get("viewCount", "N/A")
        total_videos = stats.get("videoCount", "N/A")

        semantic_text = "\n".join(semantic_chunks)

        prompt = f"""
You are an expert YouTube channel analyst.

CHANNEL OVERVIEW:
Title: {channel_title}
Subscribers: {subscribers}
Total Views: {total_views}
Total Videos: {total_videos}

SEMANTIC CONTENT SUMMARY (TOP THEMES):
{semantic_text}

TASK:
Using only the above information, generate a structured JSON analysis with:

1. executive_summary
2. metrics (subscriber_count, total_views, total_videos, average_views, engagement_rate)
3. themes (list of objects with name, frequency, engagement)
4. insights (observations with confidence + category)
5. recommendations (title + description + priority + impact)

Ensure the output is strict JSON.
"""

        # Hard safety limit (Groq TPM = 6000 tokens)
        if len(prompt) > 5000:
            prompt = prompt[:5000] + "\n...(trimmed for size)"

        return prompt

    # -------------------------------------------------------------
    # MAIN PROCESSOR
    # -------------------------------------------------------------
    async def process(self, channel, videos, transcripts):
        try:
            logger.info("Preparing documents for FAISS...")
            docs = self.prepare_documents(transcripts)

            if not docs:
                raise ValueError("No documents to embed.")

            # 1. Add to FAISS
            logger.info(f"Adding {len(docs)} text chunks to FAISS...")
            await self.vector_db.add_documents(docs)

            # 2. Retrieve semantic chunks (TPM-safe)
            logger.info("Performing semantic search (k=3)...")
            results = await self.vector_db.search(
                "key themes in this content",
                k=3  # reduced for safety
            )

            # 3. Trim chunks for LLM prompt
            def trim(t, limit=400):
                return t[:limit] + ("..." if len(t) > limit else "")

            semantic_chunks = [trim(r["text"]) for r in results]

            # 4. Build reduced-size prompt
            prompt = self.build_safe_prompt(channel, semantic_chunks)

            # 5. Generate final analysis
            logger.info("Calling LLM for analysis...")
            analysis = await self.llm.generate_analysis(prompt)

            # 6. Build response
            response = {
                "executive_summary": analysis.get("executive_summary", ""),
                "metrics": analysis.get("metrics", {}),
                "themes": analysis.get("themes", []),
                "insights": analysis.get("insights", []),
                "recommendations": analysis.get("recommendations", []),
            }

            logger.info("Analysis complete.")
            return response

        except Exception as e:
            logger.error(f"LLM generate_analysis failed: {e}")
            raise'''


import numpy as np
from datetime import datetime
from utils_logger import get_logger
from services_embedding_service import get_embedding_service
from services_vector_database_service import get_vector_db_service
from services_llm_service import get_llm_service

logger = get_logger("orchestrator")

class MasterOrchestrator:

    def __init__(self):
        self.embedder = get_embedding_service()
        self.vector_db = get_vector_db_service()
        self.llm = get_llm_service()

    # --------------------------------------------------------------
    # Utility helpers
    # --------------------------------------------------------------
    def _days_since(self, date_str):
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
            return max((datetime.utcnow() - dt).days, 1)
        except:
            return 1

    def _compute_velocities(self, videos):
        velocities = {}
        for v in videos:
            vid = v.get("id")
            views = int(v.get("viewCount", 0))
            days = self._days_since(v.get("publishedAt"))
            velocities[vid] = round(views / days, 2)
        return velocities

    def _group_by_theme(self, videos, themes, velocities):
        theme_scores = {}

        for t in themes:
            tname = t.get("name")
            if not tname:
                continue

            theme_scores[tname] = []

            for v in videos:
                title = v.get("title", "").lower()
                desc = v.get("description", "").lower()

                if tname.lower() in title or tname.lower() in desc:
                    vid = v.get("id")
                    theme_scores[tname].append(velocities.get(vid, 0))

        # Average score
        for k in theme_scores:
            if theme_scores[k]:
                theme_scores[k] = round(sum(theme_scores[k])/len(theme_scores[k]), 2)
            else:
                theme_scores[k] = 0.0

        return theme_scores

    def _detect_formats(self, videos):
        formats = {
            "shorts": [],
            "explainer": [],
            "vlog": [],
            "reaction": [],
            "debate": [],
            "documentary": []
        }

        for v in videos:
            title = v.get("title", "").lower()
            vid = v.get("id")

            if "short" in title or "shorts" in title:
                formats["shorts"].append(vid)
            if "explained" in title or "explain" in title:
                formats["explainer"].append(vid)
            if "vlog" in title:
                formats["vlog"].append(vid)
            if "react" in title:
                formats["reaction"].append(vid)
            if "debate" in title:
                formats["debate"].append(vid)
            if "documentary" in title:
                formats["documentary"].append(vid)

        return formats

    def _compute_format_scores(self, formats, velocities):
        scores = {}
        for fmt, vids in formats.items():
            vals = [velocities.get(v, 0) for v in vids]
            if vals:
                scores[fmt] = round(sum(vals)/len(vals), 2)
            else:
                scores[fmt] = 0.0
        return scores

    def _compute_upload_frequency(self, videos):
        if not videos:
            return "Not enough data"

        dates = [
            datetime.strptime(v.get("publishedAt"), "%Y-%m-%dT%H:%M:%SZ")
            for v in videos
            if v.get("publishedAt")
        ]

        dates.sort(reverse=True)

        diffs = []
        for i in range(len(dates)-1):
            gap = (dates[i] - dates[i+1]).days
            diffs.append(gap)

        if not diffs:
            return "Not enough data"

        avg_gap = sum(diffs) / len(diffs)
        per_week = round(7 / avg_gap, 2) if avg_gap > 0 else "N/A"

        return f"{per_week} videos/week"

    # --------------------------------------------------------------
    # MAIN PROCESSING PIPELINE
    # --------------------------------------------------------------
    async def process(self, channel, videos, transcripts):

        logger.info("Starting Orchestrator pipeline (FAST MODE + Hybrid Analytics)")

        # ----------------------------------------------------------
        # 1. Chunk + Store Transcripts → FAISS
        # ----------------------------------------------------------
        text_chunks = []
        for vid, text in transcripts.items():
            text_chunks.append({
                "text": text,
                "metadata": {"video_id": vid}
            })

        if text_chunks:
            logger.info("Embedding transcripts and storing in FAISS...")
            await self.vector_db.add_documents(text_chunks)
            logger.info("Vector embeddings stored successfully.")

        # ----------------------------------------------------------
        # 2. Semantic retrieval (FAST MODE baseline)
        # ----------------------------------------------------------
        semantic_results = await self.vector_db.search("key topics themes", k=8)
        semantic_texts = [r["text"] for r in semantic_results]

        # ----------------------------------------------------------
        # 3. Compute Hybrid Engagement Analytics
        # ----------------------------------------------------------
        velocities = self._compute_velocities(videos)
        formats = self._detect_formats(videos)
        format_scores = self._compute_format_scores(formats, velocities)

        trends = {
            "upload_frequency": self._compute_upload_frequency(videos),
            "top_recent_videos": sorted(
                [
                    {
                        "title": v.get("title"),
                        "views": v.get("viewCount"),
                        "velocity": velocities.get(v.get("id"), 0)
                    }
                    for v in videos
                ],
                key=lambda x: x["velocity"],
                reverse=True
            )[:5]
        }

        # premature theme extraction (LLM later refines)
        preliminary_themes = [{"name": t} for t in set(
            word.capitalize()
            for v in videos
            for word in v.get("title", "").split()
            if len(word) > 4
        )]

        theme_scores = self._group_by_theme(videos, preliminary_themes, velocities)

        engagement_summary = {
            "overall_engagement": "High" if np.mean(list(velocities.values())) > 50000 else
                                  "Medium" if np.mean(list(velocities.values())) > 10000 else "Low",
            "strong_topics": [k for k,v in theme_scores.items() if v > 50000],
            "weak_topics": [k for k,v in theme_scores.items() if v < 10000],
            "top_formats": [k for k,v in format_scores.items() if v > 50000],
            "low_performing_formats": [k for k,v in format_scores.items() if v < 10000],
        }

        hybrid_analytics = {
            "velocities": velocities,
            "theme_scores": theme_scores,
            "format_scores": format_scores,
            "engagement_summary": engagement_summary,
            "trends": trends
        }

        # ----------------------------------------------------------
        # 4. Call LLM to produce final analysis
        # ----------------------------------------------------------
        result = await self.llm.generate_analysis(
            channel_info=channel,
            videos=videos,
            semantic_texts=semantic_texts,
            hybrid_analytics=hybrid_analytics
        )

        return result
