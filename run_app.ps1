# Always launch Streamlit with this repo's venv, not system Python.
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Python = Join-Path $Root "venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Error "venv not found at $Python. Create it in the repo root first."
    exit 1
}
& $Python -m streamlit run (Join-Path $Root "app.py")
