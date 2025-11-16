@echo off
title Real-Time Object Detection Launcher

echo ==============================================
echo       Starting Backend (FastAPI YOLO)...
echo ==============================================
start "" cmd /k "uvicorn backend:app --reload --port 8000"

echo Waiting for backend to start...
timeout /t 3 >nul

echo Opening Backend Docs in Browser...
start http://127.0.0.1:8000/docs

echo ==============================================
echo       Starting Streamlit Frontend...
echo ==============================================
start "" cmd /k "streamlit run streamlit_app.py"

echo Waiting for Streamlit to start...
timeout /t 3 >nul

echo Opening Streamlit App in Browser...
start http://localhost:8501

echo ==============================================
echo        ALL SYSTEMS STARTED SUCCESSFULLY!
echo ==============================================
pause
