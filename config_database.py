"""
Database configuration and models for YouTube Research Agent
"""
'''import logging
from typing import Optional, Any, Dict,List
from datetime import datetime
from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, String, 
    DateTime, Text, JSON, Float, Boolean, ForeignKey, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship,Session
from sqlalchemy.dialects.postgresql import UUID
from databases import Database
import uuid

from config_settings import settings

logger = logging.getLogger(__name__)

# Database setup
DATABASE_URL = settings.database_url
#ASYNC_DATABASE_URL = settings.async_database_url

# SQLAlchemy setup
engine = create_engine(DATABASE_URL, echo=settings.debug)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Async database setup
#database = Database(ASYNC_DATABASE_URL)

# Base class for models
Base = declarative_base()


class ResearchProject(Base):
    """Research project model"""
    __tablename__ = "research_projects"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False)
    description = Column(Text)
    channel_url = Column(String(255), nullable=False)
    channel_id = Column(String(100))
    status = Column(String(50), default="pending")  # pending, running, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime)
    
    # Configuration
    config = Column(JSON)
    
    # Results
    results = Column(JSON)
    report_path = Column(String(500))
    
    # Relationships
    tasks = relationship("AgentTask", back_populates="project", cascade="all, delete-orphan")
    videos = relationship("VideoData", back_populates="project", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_research_projects_status', 'status'),
        Index('idx_research_projects_channel_id', 'channel_id'),
        Index('idx_research_projects_created_at', 'created_at'),
    )


class AgentTask(Base):
    """Agent task tracking model"""
    __tablename__ = "agent_tasks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey('research_projects.id'), nullable=False)
    
    # Task identification
    task_type = Column(String(100), nullable=False)  # orchestrator, agent type
    agent_id = Column(String(100), nullable=False)
    task_name = Column(String(255))
    
    # Task status
    status = Column(String(50), default="pending")  # pending, running, completed, failed
    priority = Column(Integer, default=5)
    
    # Timing
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Task data
    input_data = Column(JSON)
    output_data = Column(JSON)
    config = Column(JSON)
    
    # Error handling
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    
    # Relationships
    project = relationship("ResearchProject", back_populates="tasks")
    
    # Indexes
    __table_args__ = (
        Index('idx_agent_tasks_project_id', 'project_id'),
        Index('idx_agent_tasks_status', 'status'),
        Index('idx_agent_tasks_task_type', 'task_type'),
        Index('idx_agent_tasks_agent_id', 'agent_id'),
        Index('idx_agent_tasks_created_at', 'created_at'),
    )


class VideoData(Base):
    """Video data model"""
    __tablename__ = "video_data"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey('research_projects.id'), nullable=False)
    
    # Video identification
    video_id = Column(String(100), unique=True, nullable=False)
    channel_id = Column(String(100), nullable=False)
    
    # Basic metadata
    title = Column(String(500))
    description = Column(Text)
    published_at = Column(DateTime)
    duration = Column(String(50))
    
    # Statistics
    view_count = Column(Integer, default=0)
    like_count = Column(Integer, default=0)
    comment_count = Column(Integer, default=0)
    
    # Content
    transcript = Column(Text)
    captions = Column(JSON)
    
    # Analysis results
    themes = Column(JSON)
    sentiment = Column(Float)
    relevance_score = Column(Float)
    
    # Metadata
    raw_data = Column(JSON)  # Store complete API response
    processed_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    project = relationship("ResearchProject", back_populates="videos")
    content_vectors = relationship("ContentVector", back_populates="video", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_video_data_project_id', 'project_id'),
        Index('idx_video_data_video_id', 'video_id'),
        Index('idx_video_data_channel_id', 'channel_id'),
        Index('idx_video_data_published_at', 'published_at'),
        Index('idx_video_data_relevance_score', 'relevance_score'),
    )


class ContentVector(Base):
    """Content vector embeddings model"""
    __tablename__ = "content_vectors"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id = Column(UUID(as_uuid=True), ForeignKey('video_data.id'), nullable=False)
    
    # Vector information
    content_type = Column(String(50))  # title, description, transcript, summary
    text_content = Column(Text)
    vector_provider = Column(String(50))  # openai, sentence_transformers, etc.
    vector_model = Column(String(100))
    
    # Vector storage reference (actual vectors stored in vector DB)
    vector_id = Column(String(255))  # ID in vector database
    vector_dimension = Column(Integer)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    extra_metadata = Column("metadata", JSON)

    
    # Relationships
    video = relationship("VideoData", back_populates="content_vectors")
    
    # Indexes
    __table_args__ = (
        Index('idx_content_vectors_video_id', 'video_id'),
        Index('idx_content_vectors_content_type', 'content_type'),
        Index('idx_content_vectors_vector_provider', 'vector_provider'),
    )


class ChannelData(Base):
    """Channel metadata model"""
    __tablename__ = "channel_data"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Channel identification
    channel_id = Column(String(100), unique=True, nullable=False)
    channel_url = Column(String(255))
    channel_handle = Column(String(100))
    
    # Basic information
    title = Column(String(255))
    description = Column(Text)
    custom_url = Column(String(255))
    
    # Statistics
    subscriber_count = Column(Integer, default=0)
    video_count = Column(Integer, default=0)
    view_count = Column(Integer, default=0)
    
    # Metadata
    country = Column(String(10))
    language = Column(String(10))
    created_at_youtube = Column(DateTime)
    
    # Branding
    thumbnail_url = Column(String(500))
    banner_url = Column(String(500))
    
    # Analysis results
    content_categories = Column(JSON)
    upload_schedule = Column(JSON)
    audience_demographics = Column(JSON)
    
    # System metadata
    raw_data = Column(JSON)
    last_updated = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_channel_data_channel_id', 'channel_id'),
        Index('idx_channel_data_subscriber_count', 'subscriber_count'),
        Index('idx_channel_data_last_updated', 'last_updated'),
    )


class ResearchReport(Base):
    """Generated research reports model"""
    __tablename__ = "research_reports"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey('research_projects.id'), nullable=False)
    
    # Report information
    report_type = Column(String(100))  # executive_summary, full_report, custom
    title = Column(String(255))
    format = Column(String(50))  # pdf, html, json, markdown
    
    # Content
    content = Column(Text)
    summary = Column(Text)
    
    # File information
    file_path = Column(String(500))
    file_size = Column(Integer)
    
    # Generation metadata
    generated_by = Column(String(100))  # agent that generated this
    generation_config = Column(JSON)
    
    # System metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_research_reports_project_id', 'project_id'),
        Index('idx_research_reports_report_type', 'report_type'),
        Index('idx_research_reports_created_at', 'created_at'),
    )


class SystemMetrics(Base):
    """System performance and metrics model"""
    __tablename__ = "system_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Metric information
    metric_type = Column(String(100))  # api_usage, task_performance, error_rate
    metric_name = Column(String(255))
    
    # Values
    value_int = Column(Integer)
    value_float = Column(Float)
    value_string = Column(String(255))
    value_json = Column(JSON)
    
    # Context
    source = Column(String(100))  # agent_id, service_name
    category = Column(String(100))
    tags = Column(JSON)
    
    # Timing
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_system_metrics_metric_type', 'metric_type'),
        Index('idx_system_metrics_source', 'source'),
        Index('idx_system_metrics_timestamp', 'timestamp'),
    )


# Database connection management
async def connect_database():
    """Connect to the database"""
    try:
        await database.connect()
        logger.info("Database connected successfully")
    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}")
        raise


async def disconnect_database():
    """Disconnect from the database"""
    try:
        await database.disconnect()
        logger.info("Database disconnected successfully")
    except Exception as e:
        logger.error(f"Error disconnecting from database: {str(e)}")


def create_tables():
    """Create all tables"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {str(e)}")
        raise


def get_db() -> Session:
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Database utility functions
async def create_research_project(
    title: str,
    channel_url: str,
    description: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> str:
    """Create a new research project"""
    
    query = """
    INSERT INTO research_projects (id, title, description, channel_url, config, created_at)
    VALUES (:id, :title, :description, :channel_url, :config, :created_at)
    RETURNING id
    """
    
    project_id = str(uuid.uuid4())
    values = {
        "id": project_id,
        "title": title,
        "description": description,
        "channel_url": channel_url,
        "config": config or {},
        "created_at": datetime.utcnow()
    }
    
    try:
        await database.execute(query, values)
        logger.info(f"Created research project: {project_id}")
        return project_id
    except Exception as e:
        logger.error(f"Failed to create research project: {str(e)}")
        raise


async def update_research_project_status(project_id: str, status: str, results: Optional[Dict] = None):
    """Update research project status"""
    
    query = """
    UPDATE research_projects 
    SET status = :status, updated_at = :updated_at, results = :results
    WHERE id = :project_id
    """
    
    values = {
        "project_id": project_id,
        "status": status,
        "updated_at": datetime.utcnow(),
        "results": results
    }
    
    if status == "completed":
        query = """
        UPDATE research_projects 
        SET status = :status, updated_at = :updated_at, completed_at = :completed_at, results = :results
        WHERE id = :project_id
        """
        values["completed_at"] = datetime.utcnow()
    
    try:
        await database.execute(query, values)
        logger.info(f"Updated project {project_id} status to {status}")
    except Exception as e:
        logger.error(f"Failed to update project status: {str(e)}")
        raise


async def create_agent_task(
    project_id: str,
    task_type: str,
    agent_id: str,
    input_data: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
    priority: int = 5
) -> str:
    """Create a new agent task"""
    
    query = """
    INSERT INTO agent_tasks (id, project_id, task_type, agent_id, input_data, config, priority, created_at)
    VALUES (:id, :project_id, :task_type, :agent_id, :input_data, :config, :priority, :created_at)
    RETURNING id
    """
    
    task_id = str(uuid.uuid4())
    values = {
        "id": task_id,
        "project_id": project_id,
        "task_type": task_type,
        "agent_id": agent_id,
        "input_data": input_data,
        "config": config or {},
        "priority": priority,
        "created_at": datetime.utcnow()
    }
    
    try:
        await database.execute(query, values)
        logger.info(f"Created agent task: {task_id}")
        return task_id
    except Exception as e:
        logger.error(f"Failed to create agent task: {str(e)}")
        raise


async def update_agent_task_status(
    task_id: str, 
    status: str, 
    output_data: Optional[Dict] = None,
    error_message: Optional[str] = None
):
    """Update agent task status"""
    
    query = """
    UPDATE agent_tasks 
    SET status = :status, output_data = :output_data, error_message = :error_message
    """
    
    values = {
        "task_id": task_id,
        "status": status,
        "output_data": output_data,
        "error_message": error_message
    }
    
    if status == "running":
        query += ", started_at = :started_at"
        values["started_at"] = datetime.utcnow()
    elif status in ["completed", "failed"]:
        query += ", completed_at = :completed_at"
        values["completed_at"] = datetime.utcnow()
    
    query += " WHERE id = :task_id"
    
    try:
        await database.execute(query, values)
        logger.info(f"Updated task {task_id} status to {status}")
    except Exception as e:
        logger.error(f"Failed to update task status: {str(e)}")
        raise


async def store_video_data(project_id: str, video_data: Dict[str, Any]) -> str:
    """Store video data"""
    
    query = """
    INSERT INTO video_data (
        id, project_id, video_id, channel_id, title, description, 
        published_at, duration, view_count, like_count, comment_count,
        raw_data, processed_at
    ) VALUES (
        :id, :project_id, :video_id, :channel_id, :title, :description,
        :published_at, :duration, :view_count, :like_count, :comment_count,
        :raw_data, :processed_at
    ) ON CONFLICT (video_id) DO UPDATE SET
        title = EXCLUDED.title,
        view_count = EXCLUDED.view_count,
        like_count = EXCLUDED.like_count,
        comment_count = EXCLUDED.comment_count,
        raw_data = EXCLUDED.raw_data,
        processed_at = EXCLUDED.processed_at
    RETURNING id
    """
    
    record_id = str(uuid.uuid4())
    
    # Parse video data
    snippet = video_data.get('snippet', {})
    statistics = video_data.get('statistics', {})
    
    values = {
        "id": record_id,
        "project_id": project_id,
        "video_id": video_data.get('id'),
        "channel_id": snippet.get('channelId'),
        "title": snippet.get('title'),
        "description": snippet.get('description'),
        "published_at": datetime.fromisoformat(snippet.get('publishedAt', '').replace('Z', '+00:00')) if snippet.get('publishedAt') else None,
        "duration": video_data.get('contentDetails', {}).get('duration'),
        "view_count": int(statistics.get('viewCount', 0)),
        "like_count": int(statistics.get('likeCount', 0)),
        "comment_count": int(statistics.get('commentCount', 0)),
        "raw_data": video_data,
        "processed_at": datetime.utcnow()
    }
    
    try:
        result = await database.fetch_one(query, values)
        video_record_id = result['id'] if result else record_id
        logger.info(f"Stored video data: {video_record_id}")
        return video_record_id
    except Exception as e:
        logger.error(f"Failed to store video data: {str(e)}")
        raise


async def get_project_videos(project_id: str) -> List[Dict[str, Any]]:
    """Get all videos for a project"""
    
    query = """
    SELECT * FROM video_data 
    WHERE project_id = :project_id 
    ORDER BY published_at DESC
    """
    
    try:
        rows = await database.fetch_all(query, {"project_id": project_id})
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Failed to get project videos: {str(e)}")
        raise
class Transcript(Base):
    """Video Transcript Model"""
    __tablename__ = "transcripts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id = Column(String(50), index=True)
    language = Column(String(10))
    text = Column(Text)
    is_auto_generated = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Analysis(Base):
    """Analysis Results Model"""
    __tablename__ = "analysis_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_id = Column(String(100), index=True)
    channel_id = Column(String(50), index=True)
    analysis_type = Column(String(50))  # summary, themes, insights, trends
    result = Column(JSON)
    confidence_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


async def log_system_metric(
    metric_type: str,
    metric_name: str,
    value: Any,
    source: str,
    category: Optional[str] = None,
    tags: Optional[Dict] = None
):
    """Log system metric"""
    
    query = """
    INSERT INTO system_metrics (
        id, metric_type, metric_name, source, category, tags, timestamp,
        value_int, value_float, value_string, value_json
    ) VALUES (
        :id, :metric_type, :metric_name, :source, :category, :tags, :timestamp,
        :value_int, :value_float, :value_string, :value_json
    )
    """
    
    # Determine value type
    value_int = value if isinstance(value, int) else None
    value_float = value if isinstance(value, float) else None
    value_string = str(value) if isinstance(value, str) else None
    value_json = value if isinstance(value, (dict, list)) else None
    
    values = {
        "id": str(uuid.uuid4()),
        "metric_type": metric_type,
        "metric_name": metric_name,
        "source": source,
        "category": category,
        "tags": tags,
        "timestamp": datetime.utcnow(),
        "value_int": value_int,
        "value_float": value_float,
        "value_string": value_string,
        "value_json": value_json
    }
    
    try:
        await database.execute(query, values)
    except Exception as e:
        logger.error(f"Failed to log system metric: {str(e)}")


# Database health check
async def database_health_check() -> Dict[str, Any]:
    """Check database health"""
    try:
        query = "SELECT 1 as health_check"
        result = await database.fetch_one(query)
        
        if result and result['health_check'] == 1:
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "database_url": settings.postgres_host
            }
        else:
            return {
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": "Health check query failed"
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }'''

"""
Database configuration and models for YouTube Research Agent
"""
import logging
from typing import Optional, Any, Dict, List
from datetime import datetime
import uuid

from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, String,
    DateTime, Text, JSON, Float, Boolean, ForeignKey, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.dialects.postgresql import UUID

from config_settings import settings

logger = logging.getLogger(__name__)

# Database setup using synchronous SQLAlchemy engine
DATABASE_URL = settings.database_url
engine = create_engine(DATABASE_URL, echo=getattr(settings, "debug", False))
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# Models
class ResearchProject(Base):
    __tablename__ = "research_projects"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False)
    description = Column(Text)
    channel_url = Column(String(255), nullable=False)
    channel_id = Column(String(100))
    status = Column(String(50), default="pending")  # pending, running, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime)

    config = Column(JSON)
    results = Column(JSON)
    report_path = Column(String(500))

    tasks = relationship("AgentTask", back_populates="project", cascade="all, delete-orphan")
    videos = relationship("VideoData", back_populates="project", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_research_projects_status', 'status'),
        Index('idx_research_projects_channel_id', 'channel_id'),
        Index('idx_research_projects_created_at', 'created_at'),
    )


class AgentTask(Base):
    __tablename__ = "agent_tasks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey('research_projects.id'), nullable=False)
    task_type = Column(String(100), nullable=False)
    agent_id = Column(String(100), nullable=False)
    task_name = Column(String(255))
    status = Column(String(50), default="pending")  # pending, running, completed, failed
    priority = Column(Integer, default=5)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    input_data = Column(JSON)
    output_data = Column(JSON)
    config = Column(JSON)
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)

    project = relationship("ResearchProject", back_populates="tasks")

    __table_args__ = (
        Index('idx_agent_tasks_project_id', 'project_id'),
        Index('idx_agent_tasks_status', 'status'),
        Index('idx_agent_tasks_task_type', 'task_type'),
        Index('idx_agent_tasks_agent_id', 'agent_id'),
        Index('idx_agent_tasks_created_at', 'created_at'),
    )


class VideoData(Base):
    __tablename__ = "video_data"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey('research_projects.id'), nullable=False)
    video_id = Column(String(100), unique=True, nullable=False)
    channel_id = Column(String(100), nullable=False)
    title = Column(String(500))
    description = Column(Text)
    published_at = Column(DateTime)
    duration = Column(String(50))
    view_count = Column(Integer, default=0)
    like_count = Column(Integer, default=0)
    comment_count = Column(Integer, default=0)
    transcript = Column(Text)
    captions = Column(JSON)
    themes = Column(JSON)
    sentiment = Column(Float)
    relevance_score = Column(Float)
    raw_data = Column(JSON)
    processed_at = Column(DateTime, default=datetime.utcnow)

    project = relationship("ResearchProject", back_populates="videos")
    content_vectors = relationship("ContentVector", back_populates="video", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_video_data_project_id', 'project_id'),
        Index('idx_video_data_video_id', 'video_id'),
        Index('idx_video_data_channel_id', 'channel_id'),
        Index('idx_video_data_published_at', 'published_at'),
        Index('idx_video_data_relevance_score', 'relevance_score'),
    )


class ContentVector(Base):
    __tablename__ = "content_vectors"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id = Column(UUID(as_uuid=True), ForeignKey('video_data.id'), nullable=False)
    content_type = Column(String(50))  # title, description, transcript, summary
    text_content = Column(Text)
    vector_provider = Column(String(50))  # openai, sentence_transformers, etc.
    vector_model = Column(String(100))
    vector_id = Column(String(255))  # ID in vector database
    vector_dimension = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    extra_metadata = Column("metadata", JSON)

    video = relationship("VideoData", back_populates="content_vectors")

    __table_args__ = (
        Index('idx_content_vectors_video_id', 'video_id'),
        Index('idx_content_vectors_content_type', 'content_type'),
        Index('idx_content_vectors_vector_provider', 'vector_provider'),
    )


class ChannelData(Base):
    __tablename__ = "channel_data"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    channel_id = Column(String(100), unique=True, nullable=False)
    channel_url = Column(String(255))
    channel_handle = Column(String(100))
    title = Column(String(255))
    description = Column(Text)
    custom_url = Column(String(255))
    subscriber_count = Column(Integer, default=0)
    video_count = Column(Integer, default=0)
    view_count = Column(Integer, default=0)
    country = Column(String(10))
    language = Column(String(10))
    created_at_youtube = Column(DateTime)
    thumbnail_url = Column(String(500))
    banner_url = Column(String(500))
    content_categories = Column(JSON)
    upload_schedule = Column(JSON)
    audience_demographics = Column(JSON)
    raw_data = Column(JSON)
    last_updated = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_channel_data_channel_id', 'channel_id'),
        Index('idx_channel_data_subscriber_count', 'subscriber_count'),
        Index('idx_channel_data_last_updated', 'last_updated'),
    )


class ResearchReport(Base):
    __tablename__ = "research_reports"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey('research_projects.id'), nullable=False)
    report_type = Column(String(100))  # executive_summary, full_report, custom
    title = Column(String(255))
    format = Column(String(50))  # pdf, html, json, markdown
    content = Column(Text)
    summary = Column(Text)
    file_path = Column(String(500))
    file_size = Column(Integer)
    generated_by = Column(String(100))  # agent that generated this
    generation_config = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_research_reports_project_id', 'project_id'),
        Index('idx_research_reports_report_type', 'report_type'),
        Index('idx_research_reports_created_at', 'created_at'),
    )


class SystemMetrics(Base):
    __tablename__ = "system_metrics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    metric_type = Column(String(100))  # api_usage, task_performance, error_rate
    metric_name = Column(String(255))
    value_int = Column(Integer)
    value_float = Column(Float)
    value_string = Column(String(255))
    value_json = Column(JSON)
    source = Column(String(100))  # agent_id, service_name
    category = Column(String(100))
    tags = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_system_metrics_metric_type', 'metric_type'),
        Index('idx_system_metrics_source', 'source'),
        Index('idx_system_metrics_timestamp', 'timestamp'),
    )

class Transcript(Base):
    """Video Transcript Model"""
    __tablename__ = "transcripts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id = Column(String(50), index=True)
    language = Column(String(10))
    text = Column(Text)
    is_auto_generated = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Analysis(Base):
    """Analysis Results Model"""
    __tablename__ = "analysis_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_id = Column(String(100), index=True)
    channel_id = Column(String(50), index=True)
    analysis_type = Column(String(50))  # summary, themes, insights, trends
    result = Column(JSON)
    confidence_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


# Database session management
def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Create tables synchronously
def create_tables():
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {str(e)}")
        raise


# Synchronous database utility functions for your application
def create_research_project(title: str, channel_url: str, description: Optional[str] = None,
                            config: Optional[Dict[str, Any]] = None) -> str:
    db = SessionLocal()
    try:
        project_id = str(uuid.uuid4())
        project = ResearchProject(
            id=project_id,
            title=title,
            description=description,
            channel_url=channel_url,
            config=config or {},
            status="pending",
            created_at=datetime.utcnow()
        )
        db.add(project)
        db.commit()
        return project_id
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def update_research_project_status(project_id: str, status: str, results: Optional[Dict] = None):
    db = SessionLocal()
    try:
        project = db.query(ResearchProject).filter(ResearchProject.id == project_id).first()
        if not project:
            raise Exception("Project Not Found")
        project.status = status
        project.updated_at = datetime.utcnow()
        if results:
            project.results = results
        if status == "completed":
            project.completed_at = datetime.utcnow()
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def create_agent_task(project_id: str, task_type: str, agent_id: str, input_data: Dict[str, Any],
                      config: Optional[Dict[str, Any]] = None, priority: int = 5) -> str:
    db = SessionLocal()
    try:
        task_id = str(uuid.uuid4())
        task = AgentTask(
            id=task_id,
            project_id=project_id,
            task_type=task_type,
            agent_id=agent_id,
            input_data=input_data,
            config=config or {},
            priority=priority,
            status="pending",
            created_at=datetime.utcnow()
        )
        db.add(task)
        db.commit()
        return task_id
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def update_agent_task_status(task_id: str, status: str, output_data: Optional[Dict] = None,
                             error_message: Optional[str] = None):
    db = SessionLocal()
    try:
        task = db.query(AgentTask).filter(AgentTask.id == task_id).first()
        if not task:
            raise Exception("Task Not Found")
        task.status = status
        if output_data:
            task.output_data = output_data
        if error_message:
            task.error_message = error_message
        if status == "running":
            task.started_at = datetime.utcnow()
        elif status in ["completed", "failed"]:
            task.completed_at = datetime.utcnow()
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def store_video_data(project_id: str, video_data: Dict[str, Any]) -> str:
    db = SessionLocal()
    try:
        record_id = str(uuid.uuid4())
        snippet = video_data.get('snippet', {})
        statistics = video_data.get('statistics', {})
        video = db.query(VideoData).filter(VideoData.video_id == video_data.get('id')).first()
        if video:
            video.title = snippet.get('title')
            video.view_count = int(statistics.get('viewCount', 0))
            video.like_count = int(statistics.get('likeCount', 0))
            video.comment_count = int(statistics.get('commentCount', 0))
            video.raw_data = video_data
            video.processed_at = datetime.utcnow()
            db.commit()
            return video.id
        else:
            video = VideoData(
                id=record_id,
                project_id=project_id,
                video_id=video_data.get('id'),
                channel_id=snippet.get('channelId'),
                title=snippet.get('title'),
                description=snippet.get('description'),
                published_at=datetime.fromisoformat(snippet.get('publishedAt').replace('Z', '+00:00')) if snippet.get('publishedAt') else None,
                duration=video_data.get('contentDetails', {}).get('duration'),
                view_count=int(statistics.get('viewCount', 0)),
                like_count=int(statistics.get('likeCount', 0)),
                comment_count=int(statistics.get('commentCount', 0)),
                raw_data=video_data,
                processed_at=datetime.utcnow()
            )
            db.add(video)
            db.commit()
            return record_id
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_project_videos(project_id: str) -> List[Dict[str, Any]]:
    db = SessionLocal()
    try:
        videos = db.query(VideoData).filter(VideoData.project_id == project_id).order_by(VideoData.published_at.desc()).all()
        return [video.__dict__ for video in videos]
    finally:
        db.close()


def database_health_check() -> Dict[str, Any]:
    db = SessionLocal()
    try:
        result = db.execute("SELECT 1 as health_check").scalar()
        if result == 1:
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "database_url": settings.database_url
            }
        else:
            return {
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": "Health check query failed"
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }
    finally:
        db.close()
