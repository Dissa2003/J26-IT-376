$ErrorActionPreference = "SilentlyContinue"
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$pidFile = Join-Path $PSScriptRoot ".demo_pids.json"
$stopped = @()

function Stop-Tree($processId, $label) {
    if ($null -eq $processId) {
        return
    }
    $process = Get-Process -Id ([int]$processId) -ErrorAction SilentlyContinue
    if ($null -ne $process) {
        & taskkill.exe /PID $process.Id /T /F 2>$null | Out-Null
        $script:stopped += ("{0} (PID {1})" -f $label, $process.Id)
    }
}

if (Test-Path -LiteralPath $pidFile -PathType Leaf) {
    try {
        $saved = Get-Content -LiteralPath $pidFile -Raw | ConvertFrom-Json
        Stop-Tree $saved.dashboard_pid "Dashboard"
        Stop-Tree $saved.policy_pid "Policy service"
    } catch {
        Write-Host "The PID file could not be read; using the safe port fallback." -ForegroundColor Yellow
    }
    Remove-Item -LiteralPath $pidFile -Force
} else {
    foreach ($port in @(8002, 8010)) {
        $connections = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
        foreach ($connection in $connections) {
            $process = Get-CimInstance Win32_Process -Filter ("ProcessId = {0}" -f $connection.OwningProcess) -ErrorAction SilentlyContinue
            $expectedPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
            $commandLineUsesRepoVenv = $null -ne $process -and $process.CommandLine -like ("*" + $expectedPython + "*")
            $executableIsRepoVenv = $null -ne $process -and $process.ExecutablePath -ieq $expectedPython
            if ($null -ne $process -and $process.Name -ieq "python.exe" -and ($commandLineUsesRepoVenv -or $executableIsRepoVenv)) {
                Stop-Tree $connection.OwningProcess ("Python service on port {0}" -f $port)
            }
        }
    }
}

if ($stopped.Count -eq 0) {
    Write-Host "No demo processes were running."
} else {
    foreach ($item in $stopped) {
        Write-Host ("Stopped {0}." -f $item)
    }
}
exit 0
