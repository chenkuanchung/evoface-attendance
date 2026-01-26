@echo off
title EvoFace Admin Launcher
color 0B

set VENV_ACTIVATE=.venv\Scripts\activate.bat

if not exist "%VENV_ACTIVATE%" (
    echo [ERROR] Virtual environment not found!
    pause
    exit
)

echo Starting Admin Interface...
call %VENV_ACTIVATE%
python admin.py

pause