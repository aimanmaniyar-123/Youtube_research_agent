"""
Task-related Pydantic Models
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class ResearchScope(str, Enum):
    BASIC = "basic"          # Channel info + recent videos
    STANDARD = "standard"    # + transcripts + basic analysis  
    COMPREHENSIVE = "comprehensive"  # + deep analysis + trends


class ResearchRequest(BaseModel):
    """Request model for starting channel research"""
    channel_identifier: str = Field(
        ..., 
        description="YouTube channel ID, handle, or URL",
        examples=["@channelhandle", "UCxxx", "https://youtube.com/@channel"]
    )
    scope: ResearchScope = ResearchScope.STANDARD
    max_videos: Optional[int] = Field(50, ge=1, le=200)
    include_transcripts: bool = True
    include_comments: bool = False
    analysis_depth: str = Field("standard", pattern="^(basic|standard|deep)$")
    custom_queries: Optional[List[str]] = Field(None, max_items=10)
    
    @validator('channel_identifier')
    def validate_channel_identifier(cls, v):
        if not v or len(v.strip()) < 3:
            raise ValueError('Channel identifier must be at least 3 characters')
        return v.strip()


class TaskResponse(BaseModel):
    """Response model for task creation"""
    task_id: str
    status: str
    message: str
    estimated_completion: Optional[datetime] = None


class TaskStatus(BaseModel):
    """Task status response model"""
    task_id: str
    status: str  # pending, running, completed, failed, cancelled
    progress: float = Field(0.0, ge=0.0, le=100.0)
    current_step: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_remaining: Optional[int] = None  # seconds


class AgentTask(BaseModel):
    """Internal agent task model"""
    task_id: str
    agent_type: str
    input_data: Dict[str, Any]
    config: Dict[str, Any] = {}
    parent_task_id: Optional[str] = None
    priority: int = Field(1, ge=1, le=10)
    retry_count: int = 0
    max_retries: int = 3


class AgentResult(BaseModel):
    """Agent execution result"""
    task_id: str
    agent_id: str
    status: str  # completed, failed, partial
    result: Dict[str, Any]
    execution_time: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = {}


class ProgressUpdate(BaseModel):
    """Progress update model"""
    task_id: str
    progress: float
    current_step: str
    details: Optional[str] = None
    agent_results: Optional[List[AgentResult]] = None