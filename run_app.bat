@echo off
REM Launches the Peatland Trend Analysis Streamlit app.
REM First run: create the environment once with:
REM   cd /d "%~dp0"
REM   python -m venv .venv
REM   .venv\Scripts\activate
REM   pip install -r requirements.txt
REM Then this script just activates the env and starts the app.

cd /d "%~dp0"
call .venv\Scripts\activate.bat
streamlit run app.py
pause
