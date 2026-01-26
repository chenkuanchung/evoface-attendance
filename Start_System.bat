@echo off
title EvoFace Launcher
color 0A

:: === Configuration: Check your venv folder name ===
:: If your venv folder is not .venv (e.g., venv), change the path below
set VENV_ACTIVATE=.venv\Scripts\activate.bat

:: Check if venv exists
if not exist "%VENV_ACTIVATE%" (
    echo [ERROR] Virtual environment activation script not found!
    echo Please check if the .venv folder exists.
    echo Expected path: %CD%\%VENV_ACTIVATE%
    pause
    exit
)

echo =================================================
echo       EvoFace AI Attendance System V2.3
echo           (Running in Virtual Environment)
echo =================================================
echo.

echo [1/3] Starting Web Portal...
start "EvoFace Web" /min cmd /k "call %VENV_ACTIVATE% && streamlit run webapp.py"

echo [2/3] Waiting for Web Server...
timeout /t 3 /nobreak >nul

echo [3/3] Starting Main Application...
echo.
echo System is running! Please do not close this window.
echo.

:: Start Main App
call %VENV_ACTIVATE%
python main.py

:: Kill python processes on exit
taskkill /F /IM python.exe /T
echo System Shutdown.
pause