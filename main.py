"""
YouTube Research Agent API - FAST MODE (Complete + Fixed)
"""

import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from utils_helpers import extract_channel_from_text
from services_youtube_client import YouTubeAPIService
from orchestrators_master import MasterOrchestrator
from services_llm_service import get_llm_service
from services_embedding_service import get_embedding_service
from services_vector_database_service import create_vector_db_service, get_vector_db_service
from utils_logger import get_logger
from config_settings import settings

logger = get_logger("main")

app = FastAPI(title="YouTube Research Agent API (FAST MODE)")

# ------------------------------------------------------
# CORS
# ------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------
# Startup
# ------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting YouTube Research Agent API (FAST MODE)...")

    # Initialize LLM
    from config_settings import settings

    llm = get_llm_service({
        "groq_api_key": settings.groq_api_key,
        "groq_model": settings.groq_model
    })


    ok = await llm.test_connection()
    logger.info(f"LLM Test: {ok}")

    # Initialize embedding service
    

    get_embedding_service({
        "model": settings.sentence_transformers_model
    })

    # Initialize FAISS DB
    await create_vector_db_service({
        "dimension": 384,
        "faiss_index_path": "./data/faiss_index"
    })

    logger.info("✔ All services initialized successfully")


# ------------------------------------------------------
# INPUT MODEL
# ------------------------------------------------------
class QueryModel(BaseModel):
    query: str


# ------------------------------------------------------
# MAIN ENDPOINT
# ------------------------------------------------------
@app.post("/query")
async def process_query(body: QueryModel):
    user_query = body.query.strip()
    logger.info(f"📝 User Query Received: {user_query}")

    # Extract channel handle/name
    channel_lookup = extract_channel_from_text(user_query)
    logger.info(f"🔍 Extracted Channel: {channel_lookup}")

    try:
        yt = YouTubeAPIService()
        orchestrator = MasterOrchestrator()

        # 1. Fetch channel details
        channel = yt.get_channel_by_name_or_handle(channel_lookup)
        if not channel:
            raise HTTPException(404, "Channel not found")

        channel_id = channel.get("id")

        logger.info(f"✔ Channel Resolved → ID: {channel_id}")

        # 2. Fetch metadata-only videos
        videos = yt.get_channel_videos(channel_id, max_results=25)

        # 3. Generate FAST MODE synthetic transcripts
        synthetic_transcripts = {}
        for v in videos:
            vid = v.get("id")
            if not vid:
                continue

            synthetic_transcripts[vid] = (
                f"Title: {v.get('title','')}\n"
                f"Description: {v.get('description','')}\n"
                f"(Synthetic transcript from metadata only)"
            )

        # 4. Run Orchestrator
        result = await orchestrator.process(
            channel=channel,
            videos=videos,
            transcripts=synthetic_transcripts
        )

        # 5. Build API response for Streamlit
        vsvc = get_vector_db_service()
        semantic_count = vsvc.index.ntotal if vsvc and vsvc.index else 0

        response_payload = {
            "channel": channel,
            "videos": videos,
            "analysis": result,
            "semantic_used": semantic_count
        }

        return response_payload

    except Exception as e:
        logger.error(f"❌ Query Processing Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------
# Health Check
# ------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "mode": "FAST_METADATA_ONLY"}
