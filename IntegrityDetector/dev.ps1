# dev.ps1 - start the API and the web UI together.
#
#   Run from the project root:    .\dev.ps1
#   If PowerShell blocks scripts:  powershell -ExecutionPolicy Bypass -File .\dev.ps1
#
# The API opens in its own window (so you can see its logs). The Vite UI runs
# here in the foreground. Press Ctrl+C to stop the UI; the API is stopped on
# exit. This guarantees the backend is up before the UI calls it - no more
# "ECONNREFUSED / proxy error" from running them separately.
#
# Ports: if 5000 is busy the API picks the next free port and the UI proxy is
# pointed at it automatically. Vite picks the next free UI port on its own.

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

# Returns $true if something is already listening on the port.
function Test-PortInUse([int]$Port) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $client.Connect("127.0.0.1", $Port)
        $client.Close()
        return $true
    } catch {
        return $false
    }
}

# First free port at or after $Start.
function Get-FreePort([int]$Start) {
    for ($p = $Start; $p -lt ($Start + 50); $p++) {
        if (-not (Test-PortInUse $p)) { return $p }
    }
    throw "No free port found starting at $Start"
}

$apiHost = if ($env:FLASK_HOST) { $env:FLASK_HOST } else { "127.0.0.1" }
$apiPort = Get-FreePort 5000
$healthUrl = "http://${apiHost}:${apiPort}/api/health"

Write-Host ""
Write-Host "  Integrity Detector - starting API + UI" -ForegroundColor Magenta
Write-Host "  ----------------------------------------" -ForegroundColor DarkGray
Write-Host "  API will listen on port $apiPort" -ForegroundColor DarkGray

# 1) Start the Flask API in its own window on the chosen port.
$env:FLASK_HOST = $apiHost
$env:FLASK_PORT = "$apiPort"
$api = Start-Process -FilePath "python" -ArgumentList "scripts/run_api.py" `
    -WorkingDirectory $root -PassThru

# 2) Wait until the API answers its health check.
Write-Host "  Waiting for the API at $healthUrl ..." -ForegroundColor DarkGray
$healthy = $false
for ($i = 0; $i -lt 60; $i++) {
    if ($api.HasExited) {
        Write-Host "  API process exited early (code $($api.ExitCode)). Check its window for the error." -ForegroundColor Red
        exit 1
    }
    try {
        $r = Invoke-WebRequest -UseBasicParsing -Uri $healthUrl -TimeoutSec 2
        if ($r.StatusCode -eq 200) { $healthy = $true; break }
    } catch {
        Start-Sleep -Milliseconds 500
    }
}
if ($healthy) {
    Write-Host "  API is up." -ForegroundColor Green
} else {
    Write-Host "  API not responding yet - starting the UI anyway." -ForegroundColor Yellow
}

# 3) Run the Vite dev server in the foreground, with its proxy pointed at the
#    API port we actually chose.
try {
    Push-Location (Join-Path $root "frontend")
    if (-not (Test-Path "node_modules")) {
        Write-Host "  Installing frontend dependencies (first run)..." -ForegroundColor DarkGray
        npm install
    }
    $env:VITE_API_TARGET = "http://${apiHost}:${apiPort}"
    Write-Host "  Starting the UI - open the Local URL that Vite prints below." -ForegroundColor Cyan
    Write-Host "  (usually http://localhost:5173, or the next free port)." -ForegroundColor DarkGray
    Write-Host ""
    npm run dev
} finally {
    Pop-Location
    Write-Host ""
    Write-Host "  Stopping API..." -ForegroundColor Magenta
    if ($api -and -not $api.HasExited) {
        Stop-Process -Id $api.Id -Force -ErrorAction SilentlyContinue
    }
}
