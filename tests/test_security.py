"""
SuperSight V2.1 - 安全工具模組測試
驗證文件校驗（白名單、魔術字節）、強密碼、審計日誌。
"""
import os
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from utils.security import validate_file, get_auth_credentials, setup_logging, log_access
from config.settings import Settings


class TestValidateFile:
    """文件校驗功能測試"""

    def test_valid_jpg(self, test_image_jpg: Path):
        """有效的 JPEG 文件應通過校驗"""
        is_valid, msg = validate_file(str(test_image_jpg))
        assert is_valid is True
        assert msg == ""

    def test_valid_png(self, test_image_png: Path):
        """有效的 PNG 文件應通過校驗"""
        is_valid, msg = validate_file(str(test_image_png))
        assert is_valid is True
        assert msg == ""

    def test_valid_webp(self, test_image_webp: Path):
        """有效的 WebP 文件應通過校驗"""
        is_valid, msg = validate_file(str(test_image_webp))
        assert is_valid is True
        assert msg == ""

    def test_invalid_extension(self, temp_dir: str):
        """不支援的擴展名應被拒絕"""
        file_path = Path(temp_dir) / "test.pdf"
        file_path.write_text("fake content")
        is_valid, msg = validate_file(str(file_path))
        assert is_valid is False
        assert "不支援" in msg or "不支持" in msg

    def test_empty_file(self, empty_file: Path):
        """空文件應被拒絕"""
        is_valid, msg = validate_file(str(empty_file))
        assert is_valid is False
        assert "空" in msg or "empty" in msg.lower()

    def test_fake_exe_as_image(self, fake_exe_file: Path):
        """偽裝成圖片的 exe 應被魔術字節檢測攔截"""
        is_valid, msg = validate_file(str(fake_exe_file))
        assert is_valid is False
        assert "不是有效的圖片" in msg

    def test_non_existent_file(self):
        """不存在的文件應返回錯誤"""
        is_valid, msg = validate_file(r"Z:\nonexistent\file.jpg")
        assert is_valid is False
        assert "無法讀取" in msg or "No such file" in msg or "系統找不到" in msg


class TestValidateFileEdgeCases:
    """文件校驗邊界情況測試"""

    def test_jpeg_magic_but_png_extension(self, temp_dir: str):
        """JPEG 魔術字節但 .png 擴展名應被拒絕"""
        from tests.conftest import create_test_image_file
        file_path = Path(temp_dir) / "fake.png"
        # 寫入 JPEG 內容但用 .png 擴展名
        img_data = create_test_image_file(temp_dir, "temp.jpg")
        file_path.write_bytes(img_data.read_bytes())
        is_valid, msg = validate_file(str(file_path))
        assert is_valid is False
        assert "JPEG" in msg and "png" in msg.lower()

    def test_oversized_file(self, temp_dir: str):
        """超過大小限制的文件應被拒絕"""
        from tests.conftest import create_test_image_file
        file_path = create_test_image_file(temp_dir, "big.jpg", width=5000, height=5000)
        # 修改 size 回傳值來模擬超大文件（MAX_FILE_SIZE_MB = 20）
        with patch.object(Path, "stat") as mock_stat:
            mock_stat.return_value.st_size = 30 * 1024 * 1024  # 30MB > 20MB
            is_valid, msg = validate_file(str(file_path))
            assert is_valid is False
            assert "超過" in msg or "exceeds" in msg.lower()


class TestAuthCredentials:
    """身份鑑權測試"""

    def test_credentials_from_settings(self):
        """get_auth_credentials 應返回 settings 中的憑證"""
        with patch.dict(os.environ, {
            "SUPERSIGHT_USERNAME": "test_user",
            "SUPERSIGHT_PASSWORD": "TestP@ss123"
        }):
            username, password = get_auth_credentials()
            assert username == "test_user"
            assert password == "TestP@ss123"


class TestLogging:
    """審計日誌測試"""

    def test_setup_logging(self, temp_dir: str):
        """setup_logging 應返回 logger 並創建日誌文件"""
        with patch("utils.security.settings") as mock_settings:
            mock_settings.LOGS_DIR = Path(temp_dir)
            logger = setup_logging()
            assert logger is not None
            assert logger.name == "SuperSight"
            assert logger.level == logging.INFO

    def test_log_access_creates_log(self, temp_dir: str):
        """log_access 應寫入日誌"""
        with patch("utils.security.settings") as mock_settings:
            mock_settings.LOGS_DIR = Path(temp_dir)
            # 重置 logger handlers 以便為測試創建新日誌文件
            test_logger = logging.getLogger("SuperSight.Test")
            test_logger.handlers.clear()
            test_logger.setLevel(logging.INFO)
            
            from logging.handlers import TimedRotatingFileHandler
            file_handler = TimedRotatingFileHandler(
                filename=str(Path(temp_dir) / "access.log"),
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
            test_logger.addHandler(file_handler)
            
            log_access(test_logger, "test_action", "test detail", user="tester")
            
            # 檢查日誌文件已創建
            log_file = Path(temp_dir) / "access.log"
            assert log_file.exists()
            content = log_file.read_text(encoding="utf-8")
            assert "test_action" in content
            assert "test detail" in content
            assert "tester" in content