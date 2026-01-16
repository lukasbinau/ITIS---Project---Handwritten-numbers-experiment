@echo off
setlocal

echo ==========================================
echo   MNIST Model Tester - Streamlit Launcher
echo ==========================================
echo.

REM Go to the folder where THIS .bat file is located
cd /d "%~dp0"
echo Running from: %cd%
echo.

REM Check venv activation script exists
if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: Could not find .venv\Scripts\activate.bat
    echo.
    echo Make sure this run_app.bat is in the SAME folder as:
    echo   - main.py
    echo   - app.py
    echo   - .venv\
    echo.
    echo Contents of current folder:
    dir
    echo.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"

echo.
echo Launching Streamlit...
python -m streamlit run app.py

pause

