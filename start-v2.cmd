@echo off
cd /d "%~dp0"
set "FGIR_PYTHON=%~dp0.venv-d-v2\Scripts\python.exe"
if not exist "%FGIR_PYTHON%" set "FGIR_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%FGIR_PYTHON%" (
    echo Run install.cmd first to install the application.
    pause
    exit /b 1
)
"%FGIR_PYTHON%" -m streamlit run app.py -- --config configs/demo-v2.json --device auto
if errorlevel 1 pause
