@echo off
cd /d "%~dp0"
title Ahlul Skull Bot

echo Stopping any existing instances...
taskkill /F /IM python.exe >nul 2>&1
timeout /t 2 >nul

echo Starting bot (3-layer auto-recovery: watchdog > autofix > bot)...
echo Close this window to stop everything.
echo.
py watchdog.py
