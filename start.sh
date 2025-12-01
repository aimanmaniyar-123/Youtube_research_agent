#!/bin/bash
# Start FastAPI backend in background
uvicorn main:app --host 0.0.0.0 --port 8000 &


