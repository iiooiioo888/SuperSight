"""
SuperSight V2.1 - 測試共用固件 (Fixtures)
提供 mock 物件、測試圖片生成、臨時目錄管理等。
"""
import os
import io
import json
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any
from unittest.mock import patch, MagicMock

import pytest
from PIL import Image

from config.settings import Settings


# ─── 輔助函數 ────────────────────────────────────

def create_test_image(width: int = 640, height: int = 480, 
                      format: str = "JPEG") -> bytes:
    """
    生成測試用圖片（純色）。
    
    Args:
        width: 圖片寬度
        height: 圖片高度
        format: 圖片格式 (JPEG/PNG/WEBP)
    
    Returns:
        圖片的 bytes 數據
    """
    img = Image.new("RGB", (width, height), color=(73, 109, 137))
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


def create_test_image_file(temp_dir: str, filename: str = "test.jpg",
                           width: int = 640, height: int = 480,
                           format: str = "JPEG") -> Path:
    """
    生成測試圖片文件。
    
    Returns:
        文件路徑
    """
    file_path = Path(temp_dir) / filename
    img_data = create_test_image(width, height, format)
    file_path.write_bytes(img_data)
    return file_path


# ─── pytest Fixtures ─────────────────────────────

@pytest.fixture(scope="function")
def temp_dir() -> Generator[str, None, None]:
    """臨時目錄，測試結束後自動清理"""
    tmp = tempfile.mkdtemp()
    yield tmp
    shutil.rmtree(tmp)


@pytest.fixture(scope="function")
def test_image_jpg(temp_dir: str) -> Path:
    """標準 JPEG 測試圖片"""
    return create_test_image_file(temp_dir, "test.jpg")


@pytest.fixture(scope="function")
def test_image_png(temp_dir: str) -> Path:
    """PNG 測試圖片"""
    return create_test_image_file(temp_dir, "test.png", format="PNG")


@pytest.fixture(scope="function")
def test_image_webp(temp_dir: str) -> Path:
    """WebP 測試圖片"""
    return create_test_image_file(temp_dir, "test.webp", format="WEBP")


@pytest.fixture(scope="function")
def fake_exif_image(temp_dir: str) -> Path:
    """
    包含 EXIF 信息的測試圖片（使用 Pillow 寫入 EXIF）。
    """
    from PIL.ExifTags import Base
    
    img = Image.new("RGB", (800, 600), color=(255, 0, 0))
    exif_data = img.getexif()
    
    # 寫入基本 EXIF 字段
    exif_data[Base.DateTimeOriginal] = "2024:06:15 14:30:00"
    exif_data[Base.Make] = "TestCamera"
    exif_data[Base.Model] = "ModelX"
    exif_data[Base.Software] = "SuperSightTest"
    exif_data[Base.Orientation] = 1
    exif_data[Base.ISOSpeedRatings] = 200
    exif_data[Base.FNumber] = (28, 10)  # F2.8
    
    file_path = Path(temp_dir) / "exif_test.jpg"
    img.save(file_path, exif=exif_data)
    
    return file_path


@pytest.fixture(scope="function")
def empty_file(temp_dir: str) -> Path:
    """空文件（用於校驗測試）"""
    file_path = Path(temp_dir) / "empty.jpg"
    file_path.write_text("")
    return file_path


@pytest.fixture(scope="function")
def fake_exe_file(temp_dir: str) -> Path:
    """偽裝成圖片的可執行文件（魔術字節測試）"""
    file_path = Path(temp_dir) / "fake_virus.jpg"
    # Windows PE 文件頭 MZ
    file_path.write_bytes(b"MZ" + b"\x00" * 100)
    return file_path


@pytest.fixture(scope="function")
def mock_settings() -> Settings:
    """返回默認配置實例（不修改全局 settings）"""
    return Settings()


@pytest.fixture(scope="function")
def mock_ollama_response() -> Dict[str, Any]:
    """模擬 Ollama API 的 JSON 回應"""
    return {
        "response": """```json
{
    "scene_description": "一個陽光明媚的戶外公園場景",
    "main_content": "草地上有幾個人在野餐",
    "ocr_text": "NO PARKING",
    "activity_inference": "家庭週末聚會",
    "tags": ["戶外", "公園", "野餐", "陽光"]
}
```""",
        "done": True
    }


@pytest.fixture(scope="function")
def mock_face_result() -> Dict[str, Any]:
    """模擬人臉分析結果"""
    return {
        "has_face": True,
        "face_count": 2,
        "faces": [
            {
                "age": 30,
                "gender": "Male",
                "emotion": "happy",
                "emotion_scores": {"happy": 0.95, "neutral": 0.03, "sad": 0.02},
                "bbox": [100, 100, 200, 200],
                "confidence": 0.98,
            },
            {
                "age": 28,
                "gender": "Female",
                "emotion": "happy",
                "emotion_scores": {"happy": 0.88, "neutral": 0.10, "sad": 0.02},
                "bbox": [300, 150, 400, 250],
                "confidence": 0.97,
            }
        ]
    }


@pytest.fixture(scope="function")
def mock_scene_result() -> Dict[str, Any]:
    """模擬場景分析結果"""
    return {
        "success": True,
        "scene_description": "一個陽光明媚的戶外公園",
        "main_content": "人們在草地上野餐",
        "ocr_text": "",
        "activity_inference": "休閒聚會",
        "tags": ["戶外", "公園", "野餐"]
    }