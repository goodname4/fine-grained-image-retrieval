$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
python -m venv .venv
if ($LASTEXITCODE -ne 0) { throw "Python 3.11+ is required to create .venv." }
$demoPython = Join-Path $root ".venv\Scripts\python.exe"
& $demoPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Failed to install pip." }
$bundledWheel = Get-ChildItem -Path (Join-Path $root "wheels\*.whl") -ErrorAction SilentlyContinue | Select-Object -First 1
if ($bundledWheel) {
    & $demoPython -m pip install ($bundledWheel.FullName + "[demo]")
} else {
    & $demoPython -m pip install -e ".[demo]"
}
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed; check network and retry." }
Write-Host "Demo installed. Start it with .\scripts\start-demo.ps1"
