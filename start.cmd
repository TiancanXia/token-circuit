@echo off
setlocal
cd /d "%~dp0"
python -c "import sys, tkinter; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if not errorlevel 1 (
    python "%~dp0game.py" %*
) else (
    py -3 -c "import sys, tkinter; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
    if errorlevel 1 (
        echo Python 3.10+ with Tcl/Tk was not found.
        echo Download: https://www.python.org/downloads/
        pause
        exit /b 1
    )
    py -3 "%~dp0game.py" %*
)
if errorlevel 1 pause
