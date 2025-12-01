# YouTube Channel Research Agent

A scalable multi-agent system for comprehensive YouTube channel research and analysis.

## 🎯 Overview

This system automatically collects, analyzes, and generates insights about YouTube channels using a sophisticated multi-agent architecture with RAG (Retrieval-Augmented Generation) processing.

### Key Features

- **🤖 Multi-Agent Architecture**: Hierarchical system with specialized agents for different tasks
- **📊 Comprehensive Data Collection**: Channel metadata, videos, transcripts, engagement metrics
- **🧠 RAG Processing**: Semantic analysis using vector embeddings and similarity search
- **📈 Advanced Analytics**: Content themes, trends, audience insights, performance metrics
- **📋 Automated Reports**: Structured insights with actionable recommendations
- **⚡ Scalable Infrastructure**: Event-driven workflow with horizontal scaling capabilities

## 🏗️ Architecture

```
Master Orchestrator
├── Data Collection Orchestrator
│   ├── Channel Validator
│   ├── Metadata Extractor  
│   ├── Video Discovery
│   ├── Transcript Harvester
│   └── Content Enricher
├── RAG Processing Orchestrator
│   ├── Vectorization Agent
│   ├── Index Builder
│   ├── Query Processor
│   ├── Retrieval Agent
│   └── Relevance Scorer
└── Analysis Orchestrator
    ├── Content Summarizer
    ├── Theme Identifier
    ├── Insight Synthesizer
    ├── Trend Analyzer
    └── Report Generator
```

## 🛠️ Technology Stack

- **Backend**: FastAPI (async Python web framework)
- **Task Queue**: Celery + Redis/RabbitMQ
- **Database**: PostgreSQL + pgvector for embeddings
- **Vector DB**: Qdrant (open-source vector database)
- **APIs**: YouTube Data API v3, OpenAI API
- **ML/AI**: HuggingFace Transformers, sentence-transformers
- **Containerization**: Docker + Docker Compose

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- YouTube Data API v3 key
- OpenAI API key (optional, for enhanced analysis)

### 1. Clone and Setup

```bash
git clone <repository-url>
cd youtube-research-agent
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env file with your API keys
```

Required environment variables:
```env
YOUTUBE_API_KEY=your_youtube_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Start the System

```bash
# Make start script executable
chmod +x start.sh

# Start all services
./start.sh
```

Or manually with Docker Compose:
```bash
docker-compose up --build -d
```

### 4. Verify Installation

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Celery Monitoring**: http://localhost:5555
- **Vector Database**: http://localhost:6333

## 📖 API Usage

### Start Channel Research

```bash
curl -X POST "http://localhost:8000/api/v1/research/start" \
  -H "Content-Type: application/json" \
  -d '{
    "channel_identifier": "@channelhandle",
    "scope": "standard",
    "max_videos": 50,
    "include_transcripts": true,
    "analysis_depth": "standard"
  }'
```

### Check Task Status

```bash
curl "http://localhost:8000/api/v1/research/{task_id}"
```

### Get Results

```bash
curl "http://localhost:8000/api/v1/research/{task_id}/result"
```

## 🔧 Development Setup

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up database
docker-compose up -d db redis qdrant

# Run migrations (if using Alembic)
alembic upgrade head

# Start FastAPI development server
uvicorn main:app --reload

# Start Celery worker (in separate terminal)
celery -A celery_app.celery_app worker --loglevel=info
```

### Running Tests

```bash
pytest tests/ -v
```

## 📊 System Components

### Data Collection Agents

1. **Channel Validator**: Validates channel existence and accessibility
2. **Metadata Extractor**: Collects channel and video metadata
3. **Video Discovery**: Discovers and catalogs channel videos
4. **Transcript Harvester**: Extracts video transcripts using multiple methods
5. **Content Enricher**: Enhances data with additional context

### RAG Processing Agents

1. **Vectorization Agent**: Converts text to embeddings
2. **Index Builder**: Builds and maintains vector indices
3. **Query Processor**: Processes semantic queries
4. **Retrieval Agent**: Retrieves relevant content
5. **Relevance Scorer**: Scores content relevance

### Analysis Agents

1. **Content Summarizer**: Generates content summaries
2. **Theme Identifier**: Identifies dominant themes
3. **Insight Synthesizer**: Synthesizes actionable insights
4. **Trend Analyzer**: Analyzes temporal trends
5. **Report Generator**: Generates comprehensive reports

## 💰 Cost Analysis

| Component | Monthly Cost Range | Notes |
|-----------|-------------------|-------|
| YouTube API | $0-50 | Free tier: 10K units/day |
| OpenAI API | $20-100 | Based on analysis volume |
| Vector DB (Qdrant Cloud) | $25-100 | Managed service option |
| Infrastructure | $20-100 | Cloud hosting costs |
| **Total** | **$65-250** | Scales with usage |

## ⚡ Performance & Scalability

- **Throughput**: Process 100+ channels/hour
- **Concurrent Tasks**: 50+ simultaneous research tasks
- **Storage**: Handles millions of video transcripts
- **Scaling**: Horizontal scaling via additional workers

## 🔒 Rate Limiting & Quotas

### YouTube API Quotas
- Default: 10,000 units/day (free tier)
- Channel info: 1 unit per request
- Video details: 1 unit per request
- Captions: 50 units per request
- Search: 100 units per request

### OpenAI API
- Rate limits apply based on your plan
- Automatic retry with exponential backoff
- Cost optimization through prompt engineering

## 🐛 Troubleshooting

### Common Issues

1. **YouTube API Quota Exceeded**
   - Wait for quota reset (daily at midnight PT)
   - Upgrade to paid quota if needed
   - Use transcript fallback methods

2. **Vector Database Connection Issues**
   - Ensure Qdrant service is running
   - Check firewall settings
   - Verify connection string in .env

3. **Celery Workers Not Processing Tasks**
   - Check Redis connection
   - Restart worker processes
   - Monitor with Flower dashboard

### Logs and Monitoring

```bash
# View all service logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f app
docker-compose logs -f celery-worker

# Check system stats
curl http://localhost:8000/api/v1/system/stats
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write comprehensive tests
- Update documentation for new features

## 📄 License

[License information here]

## 🙏 Acknowledgments

- YouTube Data API for content access
- OpenAI for language model capabilities
- Qdrant for vector search functionality
- The open-source community for foundational tools

## 📞 Support

- **Documentation**: [Link to detailed docs]
- **Issues**: [GitHub Issues]
- **Discussions**: [GitHub Discussions]
- **Email**: [Support email]

---

**Note**: This system is designed for research and analysis purposes. Please ensure compliance with YouTube's Terms of Service and API usage policies.