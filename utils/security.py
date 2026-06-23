"""
SuperSight V3.0 - 安全工具模組
提供文件校驗、身份鑑權、審計日誌等安全功能。
"""
import os
import logging
import secrets
from pathlib import Path
from typing import Tuple, Optional
from datetime import datetime

from config.settings import settings


# ─── 文件校驗 ────────────────────────────────────

def validate_file(file_path: str) -> Tuple[bool, str]:
    """
    校驗上傳文件的安全性。
    
    規則：
    1. 擴展名必須在白名單內
    2. 文件大小不得超過限制
    3. 必須是真實的圖片文件（通過魔術字節檢查）
    
    Args:
        file_path: 文件路徑字串
    
    Returns:
        (is_valid: bool, error_message: str)
    """
    file_path = Path(file_path)
    
    # 檢查擴展名
    ext = file_path.suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        return False, f"不支持的文件格式 '{ext}'，允許格式：{', '.join(settings.ALLOWED_EXTENSIONS)}"
    
    # 檢查文件大小
    try:
        file_size = file_path.stat().st_size
        max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
        if file_size > max_bytes:
            return False, f"文件大小超過限制 ({settings.MAX_FILE_SIZE_MB}MB)"
        if file_size == 0:
            return False, "文件為空"
    except OSError as e:
        return False, f"無法讀取文件: {e}"
    
    # 魔術字節檢查（防止偽裝成圖片的惡意文件）
    try:
        with open(file_path, "rb") as f:
            header = f.read(12)
    
        magic_bytes = {
            b"\xff\xd8": ("JPEG", [".jpg", ".jpeg"]),
            b"\x89PNG": ("PNG", [".png"]),
            b"RIFF": ("WEBP", [".webp"]),  # WEBP 以 RIFF 開頭，需進一步驗證
        }
        
        is_valid_magic = False
        for magic, (fmt, valid_exts) in magic_bytes.items():
            if header.startswith(magic):
                # WEBP 需要進一步驗證（RIFF....WEBP）
                if fmt == "WEBP":
                    if len(header) < 12 or header[8:12] != b"WEBP":
                        continue  # 不是真正的 WEBP 文件
                
                is_valid_magic = True
                # 簡單驗證擴展名與魔術字節一致
                if ext not in valid_exts:
                    return False, f"文件實際為 {fmt} 格式，但擴展名為 '{ext}'"
                break
        
        if not is_valid_magic:
            return False, "文件內容不是有效的圖片格式"
    
    except Exception as e:
        return False, f"文件校驗失敗: {e}"
    
    return True, ""


# ─── 身份鑑權 ────────────────────────────────────

def get_auth_credentials() -> Tuple[str, str]:
    """
    獲取登錄憑證。
    若未設置環境變量或使用了默認弱密碼，生成隨機臨時密碼。
    """
    return settings.AUTH_CREDENTIALS


def generate_api_token() -> str:
    """生成 API 訪問令牌"""
    return secrets.token_hex(32)


# ─── 審計日誌 ────────────────────────────────────

def setup_logging() -> logging.Logger:
    """
    配置審計日誌系統。
    日誌文件位於 logs/access.log，同時輸出到控制台。
    
    Returns:
        配置好的 root logger
    """
    log_dir = settings.LOGS_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / "access.log"
    
    logger = logging.getLogger("SuperSight")
    logger.setLevel(logging.INFO)
    
    # 避免重複添加 handler
    if logger.handlers:
        return logger
    
    # 文件 Handler（保留最近 30 天，每日輪替）
    from logging.handlers import TimedRotatingFileHandler
    file_handler = TimedRotatingFileHandler(
        filename=str(log_file),
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)
    
    # 控制台 Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)
    
    return logger


def log_access(logger: logging.Logger, action: str, detail: str = "", 
               user: str = "anonymous", status: str = "OK"):
    """記錄訪問日誌"""
    logger.info(f"[{user}] {action} | {detail} | Status: {status}")