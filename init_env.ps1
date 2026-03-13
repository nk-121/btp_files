# PowerShell script to create and activate a Python virtual environment
# Usage: .\init_env.ps1 [env_name]

param(
    [string]$envName = "venv"
)

if (Test-Path $envName) {
    Write-Host "Virtual environment '$envName' already exists." -ForegroundColor Yellow
} else {
    Write-Host "Creating virtual environment '$envName'..." -ForegroundColor Green
    python -m venv $envName
}

Write-Host "To activate the environment run:`n`n    .\$envName\Scripts\Activate.ps1`n" -ForegroundColor Cyan
Write-Host "Then install dependencies with:`n    pip install -r requirements.txt`n" -ForegroundColor Cyan
