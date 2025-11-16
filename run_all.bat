@echo off
cd /d "%~dp0"

echo Activating virtual environment...
call .venv\Scripts\activate

echo Starting Backend...
start "" cmd /k "uvicorn backend:app --reload --port 8000"
timeout /t 3 >nul
start http://127.0.0.1:8000/docs

echo Starting Streamlit frontend...
start "" cmd /k "streamlit run streamlit_app.py --server.headless true"
timeout /t 3 >nul
start http://localhost:8501

echo All systems started!
pause
