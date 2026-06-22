"""
SuperSight V2.1 - 配置模組
集中管理所有系統配置、模型路徑、安全參數與性能閾值。
"""
import os
import secrets
from pathlib import Path
from typing import Optional, List


class Settings:
    """應用全局配置，支援環境變量覆蓋默認值。"""

    # ─── 專案基礎 ────────────────────────────────────
    PROJECT_NAME: str = "SuperSight"
    VERSION: str = "V2.1"
    BASE_DIR: Path = Path(__file__).resolve().parent.parent

    # ─── 安全配置 ────────────────────────────────────
    # 強密碼策略：從環境變量讀取，若未設置則生成臨時密碼
    SUPERSIGHT_USERNAME: str = os.getenv("SUPERSIGHT_USERNAME", "admin")
    _raw_password: Optional[str] = os.getenv("SUPERSIGHT_PASSWORD")
    
    @property
    def SUPERSIGHT_PASSWORD(self) -> str:
        if not self._raw_password or self._raw_password == "change_me_first":
            return secrets.token_urlsafe(16)  # 生成隨機強密碼
        return self._raw_password
    
    @property
    def AUTH_CREDENTIALS(self) -> tuple:
        return (self.SUPERSIGHT_USERNAME, self.SUPERSIGHT_PASSWORD)
    
    # ─── 服務器配置 ──────────────────────────────────
    SERVER_HOST: str = "127.0.0.1"  # 強制本地綁定
    SERVER_PORT: int = int(os.getenv("SUPERSIGHT_PORT", "7860"))
    SHARE_MODE: bool = False  # 禁止公網分享
    
    # ─── 文件校驗 ────────────────────────────────────
    ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".webp"]
    MAX_FILE_SIZE_MB: int = 10  # 單文件上限 10MB
    MAX_BATCH_SIZE: int = 50    # 批量上傳上限
    
    # ─── 模型配置 ────────────────────────────────────
    # Qwen-VL 模型
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    VLM_MODEL_NAME: str = os.getenv("VLM_MODEL_NAME", "qwen2.5-vl:7b-instruct-q4_k_m")
    VLM_MAX_TOKENS: int = 512
    VLM_TEMPERATURE: float = 0.1
    
    # InsightFace 配置
    INSIGHTFACE_MODEL: str = "buffalo_l"
    INSIGHTFACE_DET_SIZE: tuple = (640, 640)
    
    # ChromaDB 配置
    CHROMA_DB_PATH: str = str(BASE_DIR / "memories" / "vector_store")
    COLLECTION_NAME: str = "user_memories"
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_DEVICE: str = "cuda"  # 或 "cpu"
    TOP_K_RETRIEVAL: int = 5
    
    # ─── 資源管理 ────────────────────────────────────
    GPU_DEVICE_ID: int = int(os.getenv("CUDA_VISIBLE_DEVICES", "0"))
    # 顯存閾值 (GB)，用於 OOM 防護
    VRAM_WARNING_THRESHOLD_GB: float = 2.0   # 剩餘少於此值時警告
    VRAM_CRITICAL_THRESHOLD_GB: float = 1.0  # 剩餘少於此值時降級
    # 預估模型顯存佔用 (GB)
    VLM_VRAM_ESTIMATE_GB: float = 6.0
    FACE_VRAM_ESTIMATE_GB: float = 1.5
    
    # ─── 存儲路徑 ────────────────────────────────────
    EPISODES_DIR: Path = BASE_DIR / "memories" / "episodes"
    PROFILE_DIR: Path = BASE_DIR / "memories" / "profiles"
    LOGS_DIR: Path = BASE_DIR / "logs"
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    
    def __post_init__(self):
        """確保所有依賴目錄存在"""
        for d in [self.EPISODES_DIR, self.PROFILE_DIR, self.LOGS_DIR, self.UPLOAD_DIR]:
            d.mkdir(parents=True, exist_ok=True)
    
    @property
    def is_password_default(self) -> bool:
        """檢查是否使用了默認或生成的臨時密碼"""
        return not self._raw_password or self._raw_password == "change_me_first"


# 全局單例
settings = Settings()
settings.__post_init__()