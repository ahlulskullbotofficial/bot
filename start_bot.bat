@echo off
cd /d "%~dp0"
title Ahlul Skull Bot - AutoFix Watchdog

echo Stopping any existing bot instances...
taskkill /F /IM python.exe >nul 2>&1
timeout /t 2 >nul

echo Starting bot with AutoFix watchdog...
echo The bot will auto-fix and restart itself if it crashes.
echo Close this window to stop the bot.
echo.
py autofix.py
