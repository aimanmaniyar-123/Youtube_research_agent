"""
API Endpoints
"""
'''from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Dict, Any
import uuid

from config_database import get_db
from models_task_models import ResearchRequest, TaskResponse, TaskStatus
from orchestrators_master import coordinate_research
from celery_app import celery_app
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/research/start", response_model=TaskResponse)
async def start_channel_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Start YouTube channel research task
    """
    try:
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Start the research task asynchronously
        task = celery_app.send_task(
            'orchestrators.master.coordinate_research',
            args=[task_id, request.channel_identifier, request.dict()],
            task_id=task_id
        )
        
        logger.info(f"Started research task {task_id} for channel {request.channel_identifier}")
        
        return TaskResponse(
            task_id=task_id,
            status="initiated",
            message="Research task started successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to start research task: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/research/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str, db: Session = Depends(get_db)):
    """
    Get research task status and progress
    """
    try:
        # Check Celery task status
        task_result = celery_app.AsyncResult(task_id)
        
        # Get additional info from database
        from config_database import ResearchTask
        db_task = db.query(ResearchTask).filter(ResearchTask.task_id == task_id).first()
        
        if not db_task and task_result.state == 'PENDING':
            raise HTTPException(status_code=404, detail="Task not found")
        
        status_mapping = {
            'PENDING': 'pending',
            'STARTED': 'running', 
            'SUCCESS': 'completed',
            'FAILURE': 'failed',
            'RETRY': 'running',
            'REVOKED': 'cancelled'
        }
        
        return TaskStatus(
            task_id=task_id,
            status=status_mapping.get(task_result.state, 'unknown'),
            progress=db_task.progress if db_task else 0.0,
            result=task_result.result if task_result.successful() else None,
            error_message=str(task_result.info) if task_result.failed() else None,
            created_at=db_task.created_at if db_task else None,
            completed_at=db_task.completed_at if db_task else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/research/{task_id}/result")
async def get_research_result(task_id: str, db: Session = Depends(get_db)):
    """
    Get complete research results for a task
    """
    try:
        task_result = celery_app.AsyncResult(task_id)
        
        if not task_result.successful():
            raise HTTPException(
                status_code=400, 
                detail="Task not completed successfully"
            )
        
        return {
            "task_id": task_id,
            "status": "completed",
            "result": task_result.result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get research result: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/research/{task_id}")
async def cancel_task(task_id: str):
    """
    Cancel a running research task
    """
    try:
        celery_app.control.revoke(task_id, terminate=True)
        logger.info(f"Cancelled task {task_id}")
        
        return {"message": f"Task {task_id} cancelled successfully"}
        
    except Exception as e:
        logger.error(f"Failed to cancel task: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/research/")
async def list_research_tasks(
    skip: int = 0, 
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    List all research tasks with pagination
    """
    try:
        from config_database import ResearchTask
        tasks = db.query(ResearchTask).offset(skip).limit(limit).all()
        
        return {
            "tasks": [
                {
                    "task_id": task.task_id,
                    "channel_id": task.channel_id,
                    "status": task.status,
                    "progress": task.progress,
                    "created_at": task.created_at,
                    "completed_at": task.completed_at
                }
                for task in tasks
            ],
            "total": db.query(ResearchTask).count()
        }
        
    except Exception as e:
        logger.error(f"Failed to list tasks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/channels/{channel_id}/analysis")
async def get_channel_analysis(channel_id: str, db: Session = Depends(get_db)):
    """
    Get stored analysis results for a specific channel
    """
    try:
        from config_database import Analysis
        
        analyses = db.query(Analysis).filter(
            Analysis.channel_id == channel_id
        ).order_by(Analysis.created_at.desc()).all()
        
        if not analyses:
            raise HTTPException(status_code=404, detail="No analysis found for this channel")
        
        return {
            "channel_id": channel_id,
            "analyses": [
                {
                    "analysis_type": analysis.analysis_type,
                    "result": analysis.result,
                    "confidence_score": analysis.confidence_score,
                    "created_at": analysis.created_at
                }
                for analysis in analyses
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get channel analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system/stats")
async def get_system_stats():
    """
    Get system statistics and health metrics
    """
    try:
        # Get Celery stats
        inspect = celery_app.control.inspect()
        active_tasks = inspect.active()
        scheduled_tasks = inspect.scheduled()
        
        stats = {
            "celery": {
                "active_tasks": sum(len(tasks) for tasks in (active_tasks or {}).values()),
                "scheduled_tasks": sum(len(tasks) for tasks in (scheduled_tasks or {}).values()),
                "workers": list((active_tasks or {}).keys())
            },
            "queues": {
                "data_collection": 0,  # Would need Redis inspection
                "rag_processing": 0,
                "analysis": 0,
                "orchestration": 0
            }
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get system stats: {str(e)}")
        return {"error": "Unable to retrieve system statistics"}'''

"""
Main API - Updated for Gemini + FAISS Pipeline
"""
"""
Natural Language Query API
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from orchestrators_master import process_user_query

app = FastAPI(title="YouTube Research Agent - Natural Language Interface")

class NaturalQueryRequest(BaseModel):
    query: str  # "I want video summary of this YouTube channel of 5 videos"
    conversation_context: Optional[List[str]] = None
    user_id: Optional[str] = None

class QueryResponse(BaseModel):
    status: str  # "completed", "clarification_needed", "error"
    message: Optional[str] = None
    response: Optional[str] = None
    project_id: Optional[str] = None
    suggestions: Optional[List[Dict[str, str]]] = None

@app.post("/query", response_model=QueryResponse)
async def process_natural_language_query(
    request: NaturalQueryRequest,
    background_tasks: BackgroundTasks
):
    """
    Process natural language queries like:
    - "I want video summary of this YouTube channel of 5 videos"
    - "Analyze content themes from recent 10 videos of TechChannel"  
    - "Give me insights about @username recent content"
    """
    
    try:
        result = await process_user_query(
            user_query=request.query,
            conversation_context=request.conversation_context,
            user_id=request.user_id
        )
        
        if result["status"] == "clarification_needed":
            return QueryResponse(
                status="clarification_needed",
                message=result["message"],
                suggestions=result.get("suggestions", [])
            )
        elif result["status"] == "completed":
            return QueryResponse(
                status="completed",
                response=result["response"],
                project_id=result["project_id"]
            )
        else:
            return QueryResponse(
                status="error",
                message=result.get("message", "Unknown error")
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/specify-channel")
async def specify_channel_for_query(
    original_query: str,
    channel_url: str,
    user_id: Optional[str] = None
):
    """Handle channel specification after clarification"""
    
    # Process the query with the specified channel
    # Implementation similar to above but with explicit channel
    pass

@app.get("/")
async def root():
    return {
        "message": "YouTube Research Agent - Natural Language Interface",
        "example_queries": [
            "I want video summary of this YouTube channel of 5 videos",
            "Analyze content themes from recent videos",
            "Give me insights about TechChannel",
            "Compare the performance of last 10 videos"
        ]
    }
