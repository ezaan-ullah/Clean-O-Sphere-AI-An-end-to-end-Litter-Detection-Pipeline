# PowerShell script to run the Flask app with correct Python environment
$env:PYENV_ROOT = "$env:USERPROFILE\.pyenv\pyenv-win"
$env:PYENV_HOME = "$env:PYENV_ROOT"
$env:Path = "$env:PYENV_ROOT\bin;$env:PYENV_ROOT\shims;$env:Path"

Write-Host "Using Python: $(python --version)" -ForegroundColor Green
Write-Host "Starting Flask server..." -ForegroundColor Green

python app.py

