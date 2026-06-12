@echo off
cd /d "%~dp0"
echo ==========================================================
echo   Yifan the Starbound Nightfarer
echo   Quant Console v2 (bridge-aware)
echo ==========================================================
echo Launching...
python launcher.py
if errorlevel 1 (
  echo.
  echo Launch failed. Make sure Python is installed and you have run:
  echo   pip install yfinance pandas numpy scipy statsmodels
  echo.
  echo If the Portfolio button is greyed out: the bridge/ folder is
  echo missing next to this file. Export it from Claude, or pull the
  echo latest from your portfolio-ops repo.
  echo.
  pause
)
