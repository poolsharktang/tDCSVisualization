@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo Virtual environment not found: .venv
    echo Create it with:
    echo     python -m venv .venv
    echo     .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"
python tdcs_app.py
if errorlevel 1 pause
