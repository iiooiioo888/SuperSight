"""
SuperSight V3.0 - 配置模組
2026 世代適配：NVIDIA Blackwell (RTX 50 系列) FP4 加速 + Apple M4 架構
集中管理所有系統配置、模型路徑、安全參數與性能閾值。
"""
import os
import secrets
from pathlib import Path
from typing import Optional, List


class Settings:
    """應用全局配置，支援環境變量覆蓋默認值。"""

    def __init__(self):
        # ─── 專案基礎 ────────────────────────────────────
        self.PROJECT_NAME: str = "SuperSight"
        self.VERSION: str = "V3.0"
        self.BASE_DIR: Path = Path(__file__).resolve().parent.parent

        # ─── 安全配置 ────────────────────────────────────
        self.SUPERSIGHT_USERNAME: str = os.getenv("SUPERSIGHT_USERNAME", "admin")
        self._raw_password: Optional[str] = os.getenv("SUPERSIGHT_PASSWORD")
        self._generated_password: Optional[str] = None

        # ─── 服務器配置 ──────────────────────────────────
        self.SERVER_HOST: str = "127.0.0.1"
        self.SERVER_PORT: int = int(os.getenv("SUPERSIGHT_PORT", "7860"))
        self.SHARE_MODE: bool = False

        # ─── 文件校驗 ────────────────────────────────────
        self.ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
        self.MAX_FILE_SIZE_MB: int = 20  # 4K 圖片體積更大，放寬至 20MB
        self.MAX_BATCH_SIZE: int = 50

        # ─── 模型配置 (2026 世代) ──────────────────────────
        self.OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.VLM_MODEL_NAME: str = os.getenv("VLM_MODEL_NAME", "qwen3-vl:8b-fp4")
        self.VLM_CONTEXT_WINDOW: int = 1048576  # 1M tokens 超長上下文
        self.VLM_MAX_TOKENS: int = 1024        # V3.0 可生成更長回覆
        self.VLM_TEMPERATURE: float = 0.1

        # 圖像解析度 (V3.0 原生支援 4K 不再強制縮放)
        self.MAX_IMAGE_RESOLUTION: int = 3840  # 4K 原生輸入

        # InsightFace 2026 更新版
        self.INSIGHTFACE_MODEL: str = "buffalo_m"  # V3.0 使用更新版模型包
        self.INSIGHTFACE_DET_SIZE: tuple = (640, 640)

        # ChromaDB 1.2.x 配置
        self.CHROMA_DB_PATH: str = str(self.BASE_DIR / "memories" / "vector_store")
        self.COLLECTION_NAME: str = "user_memories"
        self.EMBEDDING_MODEL: str = "BAAI/bge-m4"       # bge-m4 1024維
        self.EMBEDDING_DIMENSIONS: int = 1024            # 動態維度
        self.EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cuda")
        self.TOP_K_RETRIEVAL: int = 5

        # ─── 硬體加速配置 (2026 世代) ──────────────────────
        self.GPU_DEVICE_ID: int = int(os.getenv("CUDA_VISIBLE_DEVICES", "0"))
        self.FP4_QUANTIZATION: bool = True               # Blackwell FP4 硬體加速
        self.GPU_ARCHITECTURE: str = "blackwell"         # blackwell | ada | apple_m4 | cpu

        # 顯存閾值 (GB) — V3.0 因 FP4 量化大幅降低需求
        self.VRAM_WARNING_THRESHOLD_GB: float = 1.0
        self.VRAM_CRITICAL_THRESHOLD_GB: float = 0.5

        # 預估模型顯存佔用 (GB) — FP4 量化後顯著下降
        self.VLM_VRAM_ESTIMATE_GB: float = 4.5   # Qwen3-VL-8B FP4 約 4.5GB
        self.FACE_VRAM_ESTIMATE_GB: float = 1.2  # buffalo_m 略小於 buffalo_l

        # ─── 存儲路徑 ────────────────────────────────────
        self.EPISODES_DIR: Path = self.BASE_DIR / "memories" / "episodes"
        self.PROFILE_DIR: Path = self.BASE_DIR / "memories" / "profiles"
        self.LOGS_DIR: Path = self.BASE_DIR / "logs"
        self.UPLOAD_DIR: Path = self.BASE_DIR / "uploads"

        # 確保所有依賴目錄存在
        for d in [self.EPISODES_DIR, self.PROFILE_DIR, self.LOGS_DIR, self.UPLOAD_DIR]:
            d.mkdir(parents=True, exist_ok=True)

    @property
    def SUPERSIGHT_PASSWORD(self) -> str:
        _raw = os.getenv("SUPERSIGHT_PASSWORD")
        if not _raw or _raw in ("change_me_first", "your_strong_password_here"):
            if self._generated_password is None:
                self._generated_password = secrets.token_urlsafe(16)
            return self._generated_password
        return _raw

    @property
    def AUTH_CREDENTIALS(self) -> tuple:
        return (os.getenv("SUPERSIGHT_USERNAME", "admin"), self.SUPERSIGHT_PASSWORD)

    @property
    def is_password_default(self) -> bool:
        _raw = os.getenv("SUPERSIGHT_PASSWORD")
        return not _raw or _raw in ("change_me_first", "your_strong_password_here")


# 全局單例
settings = Settings()
