# Restart the existing local demo. Does not train, transfer data or approve models.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
$dockerExe = Join-Path $env:LOCALAPPDATA 'Programs\DockerDesktop\resources\bin\docker.exe'
if (Test-Path -LiteralPath 'artifacts/serving/deployment.lock') {
    throw 'A deployment lock exists. Finish the active demo or inspect the interrupted deployment before restarting.'
}
$engineReady = $false
try { & $dockerExe info --format '{{.ServerVersion}}' 2>$null; $engineReady = $LASTEXITCODE -eq 0 } catch {}
if (-not $engineReady) {
    Start-Process -FilePath (Join-Path $env:LOCALAPPDATA 'Programs\DockerDesktop\Docker Desktop.exe') -WindowStyle Hidden
    $ready = $false
    for ($attempt=0; $attempt -lt 24; $attempt++) {
        Start-Sleep -Seconds 5
        try { & $dockerExe info --format '{{.ServerVersion}}' 2>$null; $ready = $LASTEXITCODE -eq 0 } catch { $ready=$false }
        if ($ready) { break }
    }
    if (-not $ready) { throw 'Docker did not become ready. Open Docker Desktop and inspect its status.' }
}
& $dockerExe compose -f compose.monitoring.yaml up -d
if ($LASTEXITCODE -ne 0) { throw 'Monitoring startup failed.' }
& $dockerExe start visual-analytics-api
if ($LASTEXITCODE -ne 0) { throw 'The saved API container is missing. Restore its verified deployment before continuing.' }
$expected = Get-Content -LiteralPath 'artifacts/serving/deployment.json' -Raw | ConvertFrom-Json
$healthy = $false
for ($attempt=0; $attempt -lt 30; $attempt++) {
    try {
        $health = Invoke-RestMethod 'http://127.0.0.1:8000/health' -TimeoutSec 3
        if ($health.status -eq 'ready' -and $health.checkpoint_sha256 -eq $expected.model.checkpoint_sha256 -and $health.model_version -eq $expected.model.model_version) { $healthy=$true; break }
    } catch {}
    Start-Sleep -Seconds 2
}
if (-not $healthy) { throw 'The API did not load the expected model. Inspect Docker logs before presenting.' }
$logDirectory = Join-Path $projectRoot 'reports/startup'
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
foreach ($service in @(
    @{Port=5000; Path='/health'; Arguments=@('-m','src.tracking.mlflow_utils','serve'); Name='mlflow'},
    @{Port=8767; Path='/project/status'; Arguments=@('-m','src.serving.console'); Name='console'}
)) {
    $running=$false
    try { $response=Invoke-WebRequest "http://127.0.0.1:$($service.Port)$($service.Path)" -UseBasicParsing -TimeoutSec 3; $running=$response.StatusCode -eq 200 } catch {}
    if (-not $running) {
        $stamp=Get-Date -Format 'yyyyMMdd-HHmmss'
        Start-Process -FilePath $pythonExe -ArgumentList $service.Arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDirectory "$($service.Name)-$stamp.log") -RedirectStandardError (Join-Path $logDirectory "$($service.Name)-$stamp.err")
        for ($attempt=0; $attempt -lt 20; $attempt++) {
            Start-Sleep -Seconds 2
            try { $response=Invoke-WebRequest "http://127.0.0.1:$($service.Port)$($service.Path)" -UseBasicParsing -TimeoutSec 3; $running=$response.StatusCode -eq 200; if ($running) { break } } catch {}
        }
    }
    if (-not $running) { throw "$($service.Name) did not become ready. Inspect reports/startup logs." }
}
Write-Host "Ready: Model $($health.model_version). Main page: http://127.0.0.1:8767/"
Write-Host 'Grafana: http://127.0.0.1:3000/   MLflow: http://127.0.0.1:5000/'
