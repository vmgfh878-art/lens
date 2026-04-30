$ErrorActionPreference = "Stop"
$Host.UI.RawUI.WindowTitle = "Lens frontend 3000"
Set-Location "C:\Users\user\lens\frontend"
$env:NEXT_PUBLIC_BACKEND_URL="http://127.0.0.1:8000"
& "C:\Program Files\nodejs\npm.cmd" run dev -- --hostname 127.0.0.1 --port 3000
