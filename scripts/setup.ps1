$ErrorActionPreference = "Stop"
py -3 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }
& .\.venv\Scripts\python.exe scripts/check_environment.py
Write-Host "CPU environment ready. For NVIDIA training install a compatible CUDA PyTorch wheel using the official PyTorch selector, then rerun the hardware report and probe."
