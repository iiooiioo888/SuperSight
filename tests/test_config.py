"""
SuperSight V2.1 - 配置模組測試
驗證 Settings 類的正確性：環境變量讀取、強密碼策略、默認值。
"""
import os
import secrets
from unittest.mock import patch

import pytest

from config.settings import Settings


class TestSettings:
    """配置類核心功能測試"""

    def test_default_values(self):
        """測試默認配置值"""
        settings = Settings()
        assert settings.PROJECT_NAME == "SuperSight"
        assert settings.VERSION == "V2.1"
        assert settings.SERVER_HOST == "127.0.0.1"
        assert settings.SERVER_PORT == 7860
        assert settings.SHARE_MODE is False
        assert settings.MAX_BATCH_SIZE == 50
        assert settings.TOP_K_RETRIEVAL == 5

    def test_password_from_env(self):
        """測試從環境變量讀取密碼"""
        with patch.dict(os.environ, {"SUPERSIGHT_PASSWORD": "MySecureP@ss123"}):
            settings = Settings()
            assert settings.SUPERSIGHT_PASSWORD == "MySecureP@ss123"
            assert settings.is_password_default is False

    def test_password_generation_when_empty(self):
        """測試當密碼為空時生成隨機密碼"""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            pwd = settings.SUPERSIGHT_PASSWORD
            # 應該生成 16 字節的 token_urlsafe = 22 字符
            assert len(pwd) >= 20
            assert settings.is_password_default is True

    def test_password_generation_when_default(self):
        """測試當密碼為 'change_me_first' 時生成隨機密碼"""
        with patch.dict(os.environ, {"SUPERSIGHT_PASSWORD": "change_me_first"}):
            settings = Settings()
            pwd = settings.SUPERSIGHT_PASSWORD
            assert len(pwd) >= 20
            assert settings.is_password_default is True

    def test_auth_credentials_tuple(self):
        """測試 AUTH_CREDENTIALS 返回 (user, pass) 元組"""
        with patch.dict(os.environ, {
            "SUPERSIGHT_USERNAME": "jerry",
            "SUPERSIGHT_PASSWORD": "Test@123"
        }):
            settings = Settings()
            username, password = settings.AUTH_CREDENTIALS
            assert username == "jerry"
            assert password == "Test@123"

    def test_port_from_env(self):
        """測試端口從環境變量讀取"""
        with patch.dict(os.environ, {"SUPERSIGHT_PORT": "8080"}):
            settings = Settings()
            assert settings.SERVER_PORT == 8080

    def test_allowed_extensions(self):
        """測試文件白名單"""
        settings = Settings()
        assert ".jpg" in settings.ALLOWED_EXTENSIONS
        assert ".jpeg" in settings.ALLOWED_EXTENSIONS
        assert ".png" in settings.ALLOWED_EXTENSIONS
        assert ".webp" in settings.ALLOWED_EXTENSIONS
        assert ".exe" not in settings.ALLOWED_EXTENSIONS
        assert ".pdf" not in settings.ALLOWED_EXTENSIONS

    def test_max_file_size(self):
        """測試文件大小限制"""
        settings = Settings()
        assert settings.MAX_FILE_SIZE_MB == 10

    def test_vram_thresholds(self):
        """測試顯存閾值配置"""
        settings = Settings()
        assert settings.VRAM_WARNING_THRESHOLD_GB == 2.0
        assert settings.VRAM_CRITICAL_THRESHOLD_GB == 1.0
        assert settings.VLM_VRAM_ESTIMATE_GB == 6.0
        assert settings.FACE_VRAM_ESTIMATE_GB == 1.5

    def test_model_config(self):
        """測試模型配置默認值"""
        settings = Settings()
        assert settings.VLM_MODEL_NAME == "qwen2.5-vl:7b-instruct-q4_k_m"
        assert settings.VLM_MAX_TOKENS == 512
        assert settings.VLM_TEMPERATURE == 0.1
        assert settings.INSIGHTFACE_MODEL == "buffalo_l"
        assert settings.EMBEDDING_MODEL == "BAAI/bge-m3"

    def test_directory_paths(self):
        """測試目錄路徑配置"""
        settings = Settings()
        # 這些目錄應該在 BASE_DIR 之下
        base = settings.BASE_DIR
        assert settings.EPISODES_DIR == base / "memories" / "episodes"
        assert settings.PROFILE_DIR == base / "memories" / "profiles"
        assert settings.LOGS_DIR == base / "logs"
        assert settings.UPLOAD_DIR == base / "uploads"

    def test_gpu_device_id_from_env(self):
        """測試 GPU 設備 ID 從環境變量讀取"""
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2"}):
            settings = Settings()
            assert settings.GPU_DEVICE_ID == 2


class TestSettingsEdgeCases:
    """邊界情況測試"""

    def test_ollama_url_from_env(self):
        """測試 Ollama URL 可配置"""
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://192.168.1.100:11434"}):
            settings = Settings()
            assert settings.OLLAMA_BASE_URL == "http://192.168.1.100:11434"

    def test_embedding_device_from_env(self):
        """測試 Embedding 設備可配置"""
        with patch.dict(os.environ, {"EMBEDDING_DEVICE": "cpu"}):
            settings = Settings()
            assert settings.EMBEDDING_DEVICE == "cpu"

    def test_vlm_model_from_env(self):
        """測試 VLM 模型可配置"""
        with patch.dict(os.environ, {"VLM_MODEL_NAME": "llava:7b"}):
            settings = Settings()
            assert settings.VLM_MODEL_NAME == "llava:7b"