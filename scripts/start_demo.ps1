param(
    [switch]$Reload,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
Set-Location $repoRoot

$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$pidFile = Join-Path $PSScriptRoot ".demo_pids.json"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    Write-Error "Virtualenv Python was not found at $python. Create .venv and install the project dependencies first."
    exit 1
}

function Get-PortOwner($port) {
    $connection = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -eq $connection) {
        return $null
    }
    $process = Get-Process -Id $connection.OwningProcess -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        return [PSCustomObject]@{ Port = $port; Pid = $connection.OwningProcess; Name = "unknown" }
    }
    return [PSCustomObject]@{ Port = $port; Pid = $process.Id; Name = $process.ProcessName }
}

foreach ($port in @(8002, 8010)) {
    $owner = Get-PortOwner $port
    if ($null -ne $owner) {
        Write-Host ("Port {0} is already in use by {1} (PID {2}). Nothing was started." -f $owner.Port, $owner.Name, $owner.Pid) -ForegroundColor Red
        exit 2
    }
}

function Start-VisibleService($title, $command) {
    return Start-Process -FilePath "powershell.exe" -ArgumentList @(
        "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $command
    ) -PassThru
}

function Stop-StartedProcess($processId) {
    if ($null -ne $processId) {
        & taskkill.exe /PID $processId /T /F 2>$null | Out-Null
    }
}

function Wait-Http200($url, $label) {
    $deadline = (Get-Date).AddSeconds(30)
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 3
            if ($response.StatusCode -eq 200) {
                Write-Host ("PASS: {0} is ready ({1})" -f $label, $url) -ForegroundColor Green
                return $true
            }
        } catch {
        }
        Start-Sleep -Milliseconds 500
    }
    Write-Host ("FAIL: {0} did not respond with HTTP 200 within 30 seconds." -f $label) -ForegroundColor Red
    return $false
}

$policyCommand = "`$Host.UI.RawUI.WindowTitle = 'LogiShield policy service :8002'; Set-Location '$repoRoot'; & '$python' -m uvicorn api.routes:app --host 127.0.0.1 --port 8002"
$policyProcess = Start-VisibleService "LogiShield policy service :8002" $policyCommand

if (-not (Wait-Http200 "http://127.0.0.1:8002/" "Policy service")) {
    Stop-StartedProcess $policyProcess.Id
    exit 3
}

$env:POLICY_URL = "http://127.0.0.1:8002"
$reloadArgument = if ($Reload) { " --reload" } else { "" }
$dashboardCommand = "`$Host.UI.RawUI.WindowTitle = 'LogiShield dashboard :8010'; Set-Location '$repoRoot'; `$env:POLICY_URL = 'http://127.0.0.1:8002'; & '$python' -m uvicorn tokenization_ai.app:app --host 127.0.0.1 --port 8010$reloadArgument"
$dashboardProcess = Start-VisibleService "LogiShield dashboard :8010" $dashboardCommand

if (-not (Wait-Http200 "http://127.0.0.1:8010/health" "Dashboard")) {
    Stop-StartedProcess $dashboardProcess.Id
    Stop-StartedProcess $policyProcess.Id
    exit 4
}

$smokeBody = @{
    policy_text = "Administrators can delete user accounts."
    policy_id = "demo-policy"
} | ConvertTo-Json -Compress
try {
    $smokeResponse = Invoke-WebRequest -Uri "http://127.0.0.1:8010/api/v1/verify-text" `
        -Method Post -ContentType "application/json" -Body $smokeBody -UseBasicParsing -TimeoutSec 10
    if ($smokeResponse.StatusCode -eq 200) {
        Write-Host "PASS: dashboard policy proxy smoke check returned HTTP 200." -ForegroundColor Green
    } else {
        Write-Host ("WARNING: dashboard policy proxy smoke check returned HTTP {0}." -f $smokeResponse.StatusCode) -ForegroundColor Yellow
    }
} catch {
    $status = if ($_.Exception.Response) { [int]$_.Exception.Response.StatusCode } else { "no response" }
    Write-Host ("WARNING: dashboard policy proxy smoke check failed with HTTP {0}. Servers remain running." -f $status) -ForegroundColor Yellow
}

$pidData = @{
    policy_pid = $policyProcess.Id
    dashboard_pid = $dashboardProcess.Id
    repo_root = $repoRoot
    created_at = (Get-Date).ToString("o")
}
$pidData | ConvertTo-Json | Set-Content -LiteralPath $pidFile -Encoding UTF8

if (-not $NoBrowser) {
    Start-Process "http://127.0.0.1:8010/dashboard#/policy-verification"
}

Write-Host ""
Write-Host "LogiShield demo is running." -ForegroundColor Cyan
Write-Host "Policy service: http://127.0.0.1:8002/"
Write-Host "Dashboard:      http://127.0.0.1:8010/dashboard#/policy-verification"
Write-Host ("Policy PID:     {0}" -f $policyProcess.Id)
Write-Host ("Dashboard PID:  {0}" -f $dashboardProcess.Id)
Write-Host "Run scripts\stop_demo.ps1 to stop."
