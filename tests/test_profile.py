"""
SuperSight V2.1 - 用戶畫像管理器測試
驗證統計更新、Counter 序列化、查詢接口。
"""
import json
from pathlib import Path
from collections import Counter
from unittest.mock import patch, MagicMock

import pytest

from memory.profile_manager import ProfileManager


class TestProfileInitialization:
    """畫像管理器初始化測試"""

    def test_default_profile_creation(self, temp_dir: str):
        """初始化應創建默認畫像"""
        with patch("memory.profile_manager.settings") as mock_settings:
            mock_settings.PROFILE_DIR = Path(temp_dir)
            profile = ProfileManager(user_id="test_user")
            
            assert profile._profile["stats"]["total_analyses"] == 0
            assert profile._profile["stats"]["total_faces_detected"] == 0

    def test_load_existing_profile(self, temp_dir: str):
        """應從文件加載已存在的畫像"""
        with patch("memory.profile_manager.settings") as mock_settings:
            mock_settings.PROFILE_DIR = Path(temp_dir)
            
            # 先創建一個畫像
            profile = ProfileManager(user_id="test_user")
            profile._profile["stats"]["total_analyses"] = 10
            profile._save_profile()
            
            # 重新加載
            profile2 = ProfileManager(user_id="test_user")
            assert profile2._profile["stats"]["total_analyses"] == 10

    def test_profile_dir_creation(self, temp_dir: str):
        """初始化應創建畫像目錄"""
        with patch("memory.profile_manager.settings") as mock_settings:
            profile_dir = Path(temp_dir) / "test_user"  # settings.PROFILE_DIR / user_id
            mock_settings.PROFILE_DIR = Path(temp_dir)
            
            ProfileManager(user_id="test_user")
            assert profile_dir.exists()


class TestProfileUpdate:
    """畫像更新測試"""

    @pytest.fixture(autouse=True)
    def setup(self, temp_dir: str):
        with patch("memory.profile_manager.settings") as mock_settings:
            mock_settings.PROFILE_DIR = Path(temp_dir)
            yield

    def test_update_basic_stats(self, temp_dir: str):
        """更新應增加總分析次數"""
        profile = ProfileManager(user_id="test_user")
        
        episode_data = {
            "timestamp": "2024-06-15T14:30:00",
            "face_analysis": {"has_face": False},
            "scene_analysis": {"success": True, "tags": []},
            "metadata": {}
        }
        
        profile.update(episode_data)
        assert profile._profile["stats"]["total_analyses"] == 1
        assert profile._profile["stats"]["total_images_without_faces"] == 1

    def test_update_with_face_data(self, temp_dir: str):
        """含人臉數據的更新應累加統計"""
        profile = ProfileManager(user_id="test_user")
        
        episode_data = {
            "timestamp": "2024-06-15T14:30:00",
            "face_analysis": {
                "has_face": True,
                "faces": [
                    {"age": 30, "gender": "Male", "emotion": "happy"},
                    {"age": 28, "gender": "Female", "emotion": "happy"},
                ]
            },
            "scene_analysis": {"success": True, "tags": ["聚會"]},
            "metadata": {}
        }
        
        profile.update(episode_data)
        stats = profile._profile["stats"]
        assert stats["total_analyses"] == 1
        assert stats["total_faces_detected"] == 2
        assert stats["total_images_with_faces"] == 1
        
        # 性別分布
        gender = profile._profile["face_stats"]["gender_distribution"]
        assert gender["Male"] == 1
        assert gender["Female"] == 1
        
        # 情緒分布
        emotion = profile._profile["face_stats"]["emotion_distribution"]
        assert emotion["happy"] == 2

    def test_update_with_tags(self, temp_dir: str):
        """場景標籤應被累積"""
        profile = ProfileManager(user_id="test_user")
        
        episode_data = {
            "timestamp": "2024-06-15T14:30:00",
            "face_analysis": {"has_face": False},
            "scene_analysis": {
                "success": True,
                "tags": ["戶外", "公園", "野餐", "陽光"]
            },
            "metadata": {}
        }
        
        profile.update(episode_data)
        tags = profile._profile["scene_stats"]["top_tags"]
        assert tags["戶外"] == 1
        assert tags["公園"] == 1
        assert tags["野餐"] == 1

    def test_multiple_updates(self, temp_dir: str):
        """多次更新應累加所有統計"""
        profile = ProfileManager(user_id="test_user")
        
        for i in range(5):
            episode_data = {
                "timestamp": f"2024-06-{15+i}T14:30:00",
                "face_analysis": {"has_face": False},
                "scene_analysis": {"success": True, "tags": ["標籤"]},
                "metadata": {}
            }
            profile.update(episode_data)
        
        assert profile._profile["stats"]["total_analyses"] == 5
        assert profile._profile["stats"]["total_images_without_faces"] == 5


class TestProfileQueries:
    """畫像查詢接口測試"""

    @pytest.fixture(autouse=True)
    def setup(self, temp_dir: str):
        with patch("memory.profile_manager.settings") as mock_settings:
            mock_settings.PROFILE_DIR = Path(temp_dir)
            
            # 建立帶數據的畫像
            self.profile = ProfileManager(user_id="test_user")
            
            # 添加一些測試數據
            for i in range(3):
                episode_data = {
                    "timestamp": f"2024-06-{15+i}T14:30:00",
                    "face_analysis": {
                        "has_face": True,
                        "faces": [{"age": 30, "gender": "Male", "emotion": "happy"}]
                    },
                    "scene_analysis": {
                        "success": True,
                        "tags": ["測試", f"標籤{i}"]
                    },
                    "metadata": {}
                }
                self.profile.update(episode_data)
            
            yield

    def test_get_summary(self, temp_dir: str):
        """get_summary 應返回格式化摘要"""
        summary = self.profile.get_summary()
        assert summary["user_id"] == "test_user"
        assert summary["stats"]["total_analyses"] == 3
        assert summary["stats"]["total_faces_detected"] == 3
        assert "top_tags" in summary
        assert "gender_ratio" in summary

    def test_get_top_tags(self):
        """get_top_tags 應返回最常出現的標籤"""
        tags = self.profile.get_top_tags(n=5)
        assert "測試" in tags
        assert len(tags) >= 3

    def test_get_emotion_trend(self):
        """get_emotion_trend 應返回情緒分布"""
        trend = self.profile.get_emotion_trend()
        assert trend["happy"] == 3

    def test_get_active_hours(self):
        """get_active_hours 應返回時段統計"""
        hours = self.profile.get_active_hours()
        assert len(hours) > 0


class TestProfileAgeCategorization:
    """年齡分組測試"""

    @pytest.fixture(autouse=True)
    def setup(self, temp_dir: str):
        with patch("memory.profile_manager.settings") as mock_settings:
            mock_settings.PROFILE_DIR = Path(temp_dir)
            self.profile = ProfileManager(user_id="test_user")

    def test_child_age(self):
        assert self.profile._categorize_age(5) == "child"
        
    def test_teen_age(self):
        assert self.profile._categorize_age(15) == "teen"
        
    def test_young_adult(self):
        assert self.profile._categorize_age(25) == "young_adult"
        
    def test_middle_aged(self):
        assert self.profile._categorize_age(45) == "middle_aged"
        
    def test_senior(self):
        assert self.profile._categorize_age(70) == "senior"


class TestProfileSerialization:
    """畫像序列化測試"""

    @pytest.fixture(autouse=True)
    def setup(self, temp_dir: str):
        with patch("memory.profile_manager.settings") as mock_settings:
            mock_settings.PROFILE_DIR = Path(temp_dir)
            self.profile = ProfileManager(user_id="test_user")

    def test_counter_to_dict(self):
        """包含 Counter 的畫像應被正確序列化為 JSON"""
        self.profile._profile["face_stats"]["emotion_distribution"]["happy"] += 5
        self.profile._profile["scene_stats"]["top_tags"]["戶外"] += 3
        
        serializable = self.profile._make_serializable(self.profile._profile)
        
        # 應能成功轉為 JSON
        json_str = json.dumps(serializable, ensure_ascii=False)
        assert len(json_str) > 0
        
        # 重新解析應得到相同數據
        parsed = json.loads(json_str)
        assert parsed["face_stats"]["emotion_distribution"]["happy"] == 5
        assert parsed["scene_stats"]["top_tags"]["戶外"] == 3

    def test_atomic_write(self, temp_dir: str):
        """保存應使用原子寫入（先寫 tmp 再 replace）"""
        self.profile._profile["stats"]["total_analyses"] = 99
        self.profile._save_profile()
        
        # 檢查最終文件
        with open(self.profile.profile_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["stats"]["total_analyses"] == 99
        
        # tmp 文件應被清理
        tmp_file = self.profile.profile_path.with_suffix(".tmp")
        assert not tmp_file.exists()


class TestProfileReset:
    """畫像重置測試"""

    def test_reset_clears_data(self, temp_dir: str):
        with patch("memory.profile_manager.settings") as mock_settings:
            mock_settings.PROFILE_DIR = Path(temp_dir)
            profile = ProfileManager(user_id="test_user")
            
            profile._profile["stats"]["total_analyses"] = 100
            profile.reset()
            
            assert profile._profile["stats"]["total_analyses"] == 0