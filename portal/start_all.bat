@echo off
setlocal

REM ============================================================
REM   AI Market Risk Management Platform - One-click Launcher
REM
REM   Uses each project's own conda environment:
REM     CSI 300 Prediction  -> index_predictor   (port 8000)
REM     HK Connect Monitor  -> ht_proj           (port 8001)
REM     Podcast Analysis    -> podcast_agent     (port 8501)
REM
REM   All project paths are resolved relative to this script,
REM   so the repo works from any location.
REM   If your conda path differs, edit the PY_* variables below.
REM ============================================================

REM This script lives in <repo>\portal\ , so the repo root is its parent.
for %%I in ("%~dp0..") do set "ROOT=%%~fI"

set PY_CSI=C:\software\miniforge\envs\index_predictor\python.exe
set PY_HK=C:\software\miniforge\envs\ht_proj\python.exe
set PY_POD=C:\software\miniforge\envs\podcast_agent\python.exe

echo ============================================================
echo   AI Market Risk Management Platform
echo   Starting three systems...
echo ============================================================
echo.

echo [1/3] CSI 300 Prediction System  (port 8000)
start "CSI300-Predict" /D "%ROOT%\index_adjustment_predictor" cmd /k "%PY_CSI% main.py --api"

echo [2/3] HK Connect Monitor  (port 8001)
start "HK-Connect-Monitor" /D "%ROOT%\hk_connect_monitor" cmd /k "%PY_HK% main_chat.py --api --port 8001"

echo [3/3] Podcast Analysis System  (port 8501)
start "Podcast-Analysis" /D "%ROOT%\podcast_public_opinion_analysis\podcast_agent" cmd /k "%PY_POD% -m streamlit run src/podcast_agent/web/app.py --server.port 8501 --server.headless true"

echo.
echo Opening the portal page in your browser...
start "" "%~dp0index.html"

echo.
echo Done. Keep the three service windows open while using the portal.
echo Close each service window to stop that system.
echo.
pause
endlocal
