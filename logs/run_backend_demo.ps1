$ErrorActionPreference = "Stop"
$Host.UI.RawUI.WindowTitle = "Lens backend 8000"
Set-Location "C:\Users\user\lens"
$env:PYTHONPATH="C:\Users\user\lens\backend"
$env:BACKEND_CORS_ORIGINS="http://localhost:3000,http://127.0.0.1:3000"
& "C:\Users\user\lens\.venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
