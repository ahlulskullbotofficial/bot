@echo off
:: Creates a Windows Task Scheduler entry that starts the bot automatically
:: on login, with a 30-second delay to let the network connect first.

echo Installing Ahlul Skull Bot auto-start...

schtasks /create /tn "AhlulSkullBot" /tr "cmd /c cd /d \"%~dp0\" && py autofix.py >> \"%~dp0autofix_log.txt\" 2>&1" /sc onlogon /delay 0000:30 /ru "%USERNAME%" /rl HIGHEST /f

if %errorlevel% == 0 (
    echo.
    echo SUCCESS! The bot will now start automatically every time you log into Windows.
    echo You can also start it manually by running RUN_BOT.cmd
    echo.
    echo To remove auto-start, run: schtasks /delete /tn "AhlulSkullBot" /f
) else (
    echo.
    echo Something went wrong. Try running this file as Administrator.
    echo Right-click install_autostart.bat and choose "Run as administrator"
)

pause
