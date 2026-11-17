@echo off
REM ============================================================
REM  STEP 1: Environment Setup
REM  Run this ONCE before anything else.
REM  Double-click this file OR run it in your terminal.
REM ============================================================

echo [1/3] Creating virtual environment...
python -m venv env
if errorlevel 1 goto :fail

echo [2/3] Activating environment...
call env\Scripts\activate

echo [3/3] Installing Python dependencies (uses a prebuilt llama-cpp-python wheel, no compiler needed)...
python -m pip install -r requirements-dev.txt
if errorlevel 1 goto :fail

echo.
echo Done! Now activate the environment in your terminal with:
echo     .\env\Scripts\activate
echo and then run:  python step2_download_model.py
pause
exit /b 0

:fail
echo.
echo SETUP FAILED - see the error above. Do not continue to the next step.
pause
exit /b 1
