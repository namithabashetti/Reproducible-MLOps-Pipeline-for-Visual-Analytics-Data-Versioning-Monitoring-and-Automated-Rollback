# Stop serving safely; training and MLflow remain available.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$jobFile = Join-Path $projectRoot 'reports/live-console-job.json'
if (Test-Path -LiteralPath $jobFile) {
    $jobState = Get-Content -LiteralPath $jobFile -Raw | ConvertFrom-Json
    if ($jobState.directory -notmatch '^live-demo-[0-9a-f]{12}$') { throw 'Invalid recovery session path.' }
    $reportDirectory = Join-Path $projectRoot ('reports/' + $jobState.directory)
    $resultFile = Join-Path $reportDirectory 'result.json'
    if (-not (Test-Path -LiteralPath $resultFile)) {
        $state = Invoke-RestMethod 'http://127.0.0.1:8767/demo/status' -TimeoutSec 10
        if (-not $state.running -or -not $state.automatic) { throw 'Inspect the unfinished recovery session before stopping.' }
        New-Item -Path (Join-Path $reportDirectory 'stop') -ItemType File -Force | Out-Null
        for ($attempt=0; $attempt -lt 90 -and -not (Test-Path -LiteralPath $resultFile); $attempt++) { Start-Sleep -Seconds 2 }
        if (-not (Test-Path -LiteralPath $resultFile)) { throw 'Recovery is still stopping. Do not kill it during restoration.' }
    }
    $result = Get-Content -LiteralPath $resultFile -Raw | ConvertFrom-Json
    if ($result.restoration_error) { throw 'Original deployment restoration failed; inspect the session result.' }
}
$listener = Get-NetTCPConnection -LocalPort 8767 -State Listen -ErrorAction SilentlyContinue
if ($listener) {
    $server = Get-CimInstance Win32_Process -Filter "ProcessId = $($listener[0].OwningProcess)"
    $parent = Get-CimInstance Win32_Process -Filter "ProcessId = $($server.ParentProcessId)"
    $expectedPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
    if ($server.CommandLine -notmatch '\-m src.serving.console$' -or $parent.ExecutablePath -ne $expectedPython) {
        throw 'Port 8767 is not owned by the expected project console.'
    }
    Stop-Process -Id $server.ProcessId
    Stop-Process -Id $parent.ProcessId -ErrorAction SilentlyContinue
}
$dockerExe = Join-Path $env:LOCALAPPDATA 'Programs\DockerDesktop\resources\bin\docker.exe'
& $dockerExe stop --time 10 visual-analytics-api
if ($LASTEXITCODE -ne 0) { throw 'API stop failed.' }
& $dockerExe compose -f compose.monitoring.yaml stop
if ($LASTEXITCODE -ne 0) { throw 'Monitoring stop failed.' }
Write-Host 'Detection, recovery and monitoring stopped. Training and MLflow were left running.'
