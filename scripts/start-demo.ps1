$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$demoPython = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $demoPython)) {
    throw "Demo is not installed. Run .\scripts\install-demo.ps1 first."
}
& $demoPython -m streamlit run app.py -- --config configs/demo.json --device cpu
if ($LASTEXITCODE -ne 0) { throw "Demo exited with an error." }
