# Restart Flask server with correct Python environment
$env:PYENV_ROOT = "$env:USERPROFILE\.pyenv\pyenv-win"
$env:PYENV_HOME = "$env:PYENV_ROOT"
$env:Path = "$env:PYENV_ROOT\bin;$env:PYENV_ROOT\shims;$env:Path"

Write-Host "Stopping any existing Flask servers..." -ForegroundColor Yellow
Get-Process | Where-Object {$_.ProcessName -like "*python*"} | Stop-Process -Force -ErrorAction SilentlyContinue

Start-Sleep -Seconds 2

Write-Host "Starting Flask server..." -ForegroundColor Green
python app.py

