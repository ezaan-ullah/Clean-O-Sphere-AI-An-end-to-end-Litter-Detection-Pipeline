@echo off
REM Batch script to run the Flask app with correct Python environment
set PYENV_ROOT=%USERPROFILE%\.pyenv\pyenv-win
set PYENV_HOME=%PYENV_ROOT%
set Path=%PYENV_ROOT%\bin;%PYENV_ROOT%\shims;%Path%

echo Using Python:
python --version
echo.
echo Starting Flask server...
python app.py
pause

