$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendPython = Join-Path $root '.backend-venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $backendPython)) {
    $backendPython = Join-Path $root '.venv\Scripts\python.exe'
}

$backendCommand = "Set-Location '$root'; & '$backendPython' -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"
Start-Process powershell -ArgumentList "-NoExit","-Command",$backendCommand | Out-Null
Start-Process powershell -ArgumentList "-NoExit","-Command","Set-Location '$root'; npm run dev" | Out-Null

Write-Output "Backend and frontend are starting. Open http://localhost:5173 to view the dashboard."
