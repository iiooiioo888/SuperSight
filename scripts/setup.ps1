# SuperSight V2.1 - Windows 快速安裝腳本 (PowerShell)
# 使用方法: 以管理員身份運行 PowerShell，然後:
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
#   .\scripts\setup.ps1

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  🔍 SuperSight V2.1 - 安裝腳本 (Windows)" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# 檢查管理員權限
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "⚠️  建議以管理員身份運行此腳本" -ForegroundColor Yellow
    Write-Host "   右鍵 PowerShell -> 以管理員身份執行" -ForegroundColor Yellow
    Write-Host ""
}

# 檢查 Python 版本
Write-Host "📋 檢查 Python 版本..." -ForegroundColor Green
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ $pythonVersion"
} catch {
    Write-Host "❌ 未檢測到 Python，請先安裝 Python 3.10+" -ForegroundColor Red
    Write-Host "   下載地址: https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host "   安裝時請勾選 'Add Python to PATH'"
    exit 1
}

# 檢查 Visual Studio Build Tools
Write-Host "📋 檢查 C++ 編譯環境..." -ForegroundColor Green
$vcInstalled = $false
$vsPaths = @(
    "${env:ProgramFiles}\Microsoft Visual Studio\2022\Community",
    "${env:ProgramFiles}\Microsoft Visual Studio\2022\Professional",
    "${env:ProgramFiles}\Microsoft Visual Studio\2022\Enterprise",
    "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2019\Community",
    "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2019\Professional"
)
foreach ($path in $vsPaths) {
    if (Test-Path $path) {
        $vcInstalled = $true
        break
    }
}
if ($vcInstalled) {
    Write-Host "✅ Visual Studio 已安裝" -ForegroundColor Green
} else {
    Write-Host "⚠️  未檢測到 Visual Studio Build Tools" -ForegroundColor Yellow
    Write-Host "   InsightFace 編譯需要 C++ 環境" -ForegroundColor Yellow
    Write-Host "   下載地址: https://visualstudio.microsoft.com/zh-hans/downloads/" -ForegroundColor Yellow
    Write-Host "   安裝時請勾選 'Desktop development with C++'" -ForegroundColor Yellow
    Write-Host ""
    $continue = Read-Host "是否繼續安裝？(y/n, 默認 y)"
    if ($continue -eq "n") {
        exit 1
    }
}

# 檢查 Ollama
Write-Host "📋 檢查 Ollama..." -ForegroundColor Green
try {
    $ollamaVersion = ollama --version 2>&1
    Write-Host "✅ Ollama 已安裝: $ollamaVersion"
} catch {
    Write-Host "⚠️  Ollama 未安裝" -ForegroundColor Yellow
    Write-Host "   下載地址: https://ollama.com/download/windows" -ForegroundColor Yellow
    Write-Host "   請手動下載並安裝，然後重新運行此腳本" -ForegroundColor Yellow
}

# 創建虛擬環境
Write-Host ""
Write-Host "📦 創建虛擬環境..." -ForegroundColor Green
if (-not (Test-Path "venv")) {
    python -m venv venv
    Write-Host "✅ 虛擬環境已創建" -ForegroundColor Green
} else {
    Write-Host "✅ 虛擬環境已存在" -ForegroundColor Green
}

# 激活虛擬環境
& .\venv\Scripts\Activate.ps1

# 升級 pip
Write-Host "📦 升級 pip..." -ForegroundColor Green
python -m pip install --upgrade pip

# 安裝依賴
Write-Host "📦 安裝 Python 依賴..." -ForegroundColor Green
pip install -r requirements.txt

# 安裝 onnxruntime（GPU 版本）
Write-Host "📦 安裝 ONNX Runtime GPU 版本..." -ForegroundColor Green
pip install onnxruntime-gpu --no-build-isolation

# 確保 insightface 安裝成功
Write-Host "📦 確保 InsightFace 安裝..." -ForegroundColor Green
pip install insightface --no-build-isolation

# 拉取 VLM 模型
Write-Host ""
Write-Host "🤖 拉取 Qwen2.5-VL 模型..." -ForegroundColor Green
ollama pull qwen2.5-vl:7b-instruct-q4_k_m

# 設置環境變量
Write-Host ""
Write-Host "🔐 設置安全配置..." -ForegroundColor Green
$userPwd = $env:SUPERSIGHT_PASSWORD
if (-not $userPwd) {
    $userPwd = Read-Host "請設置管理員密碼（留空將生成隨機密碼）"
    if ($userPwd) {
        [Environment]::SetEnvironmentVariable("SUPERSIGHT_PASSWORD", $userPwd, "User")
        Write-Host "✅ 密碼已設置為用戶環境變量" -ForegroundColor Green
    }
}

# 創建必要目錄
Write-Host ""
Write-Host "📁 創建數據目錄..." -ForegroundColor Green
@("memories/episodes", "memories/profiles", "memories/vector_store", "logs", "uploads") | ForEach-Object {
    New-Item -ItemType Directory -Force -Path $_ | Out-Null
}
Write-Host "✅ 數據目錄已創建" -ForegroundColor Green

# 完成
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  ✅ SuperSight V2.1 安裝完成！" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "🚀 啟動服務: python app.py" -ForegroundColor Green
Write-Host "🌐 訪問地址: http://127.0.0.1:7860" -ForegroundColor Green
Write-Host ""
Write-Host "📌 注意事項:" -ForegroundColor Yellow
Write-Host "  - 確保 Ollama 服務正在運行" -ForegroundColor Yellow
Write-Host "  - 使用前設置 SUPERSIGHT_PASSWORD 環境變量" -ForegroundColor Yellow
Write-Host "  - 首次啟動可能需要下載 bge-m3 嵌入模型" -ForegroundColor Yellow
Write-Host ""