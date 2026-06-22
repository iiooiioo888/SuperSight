#!/bin/bash
# SuperSight V2.1 - Linux/macOS 快速安裝腳本
# 使用方法: chmod +x scripts/setup.sh && ./scripts/setup.sh

set -e

echo "============================================"
echo "  🔍 SuperSight V2.1 - 安裝腳本 (Linux/macOS)"
echo "============================================"
echo ""

# 檢查 Python 版本
echo "📋 檢查 Python 版本..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo "❌ 需要 Python 3.10+，當前版本: $PYTHON_VERSION"
    exit 1
fi
echo "✅ Python $PYTHON_VERSION"

# 檢查 CUDA（可選）
if command -v nvidia-smi &> /dev/null; then
    echo "✅ 檢測到 NVIDIA GPU"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
else
    echo "⚠️  未檢測到 NVIDIA GPU（將使用 CPU 模式）"
fi

# 檢查 Ollama
if command -v ollama &> /dev/null; then
    echo "✅ Ollama 已安裝"
else
    echo "📥 正在安裝 Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "✅ Ollama 安裝完成"
fi

# 創建虛擬環境
echo ""
echo "📦 創建虛擬環境..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# 升級 pip
echo "📦 升級 pip..."
pip install --upgrade pip

# 安裝依賴
echo "📦 安裝 Python 依賴..."
pip install -r requirements.txt

# 安裝 PyTorch GPU 版本（若檢測到 CUDA）
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    echo "📦 檢測到 CUDA driver: $CUDA_VERSION"
    # 安裝 CUDA 12.1 版本 (兼容 11.8+)
    pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
fi

# 拉取 VLM 模型
echo ""
echo "🤖 拉取 Qwen2.5-VL 模型..."
ollama pull qwen2.5-vl:7b-instruct-q4_k_m

# 設置環境變量
echo ""
echo "🔐 設置安全配置..."
if [ -z "$SUPERSIGHT_PASSWORD" ]; then
    echo ""
    echo "請設置管理員密碼（留空將生成隨機密碼）:"
    read -s -p "密碼: " USER_PWD
    echo ""
    if [ -n "$USER_PWD" ]; then
        export SUPERSIGHT_PASSWORD="$USER_PWD"
    fi
fi

# 創建必要目錄
echo ""
echo "📁 創建數據目錄..."
mkdir -p memories/episodes memories/profiles memories/vector_store logs uploads

# 完成
echo ""
echo "============================================"
echo "  ✅ SuperSight V2.1 安裝完成！"
echo "============================================"
echo ""
echo "啟動服務: python app.py"
echo "訪問地址: http://127.0.0.1:7860"
echo ""
echo "注意: 請確保安裝以下依賴（若使用 GPU）:"
echo "  - NVIDIA Driver >= 525.60.13"
echo "  - CUDA 11.8+"
echo "  - cuDNN 8.6+"
echo ""