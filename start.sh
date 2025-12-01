#!/bin/bash
# Start FastAPI backend in background
uvicorn main:app --host 0.0.0.0 --port 8000 &

# Wait for backend to start
sleep 10

# Start streamlit on Render's assigned port
streamlit run streamlit_dashboard.py --host 0.0.0.0 --port $PORT
