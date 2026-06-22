"""
SuperSight V2.1 - 應用整合測試
驗證 app.py 的核心功能函數：analyze_images、search_memories、view_profile。
"""
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from app import analyze_images, search_memories, view_profile


class TestAnalyzeImages:
    """圖片分析功能整合測試"""

    def test_no_files(self):
        """未上傳文件應返回提示"""
        result = analyze_images([], query="")
        assert "上傳" in result or "請先" in result

    @pytest.mark.asyncio
    async def test_single_file_success(self, test_image_jpg: Path):
        """單張圖片分析應返回報告"""
        # 使用 mock 避免實際初始化 agent
        mock_file = MagicMock()
        mock_file.name = str(test_image_jpg)
        
        with patch('app.get_agent') as mock_get_agent:
            mock_agent = MagicMock()
            mock_agent.process.return_value = "📸 場景分析\n👤 人臉分析"
            mock_agent.last_aggregated_data = {"test": "data"}
            mock_get_agent.return_value = mock_agent
            
            with patch('app.get_profile') as mock_get_profile:
                mock_profile = MagicMock()
                mock_get_profile.return_value = mock_profile
                
                result = analyze_images([mock_file], query="測試查詢")
                assert len(result) > 0
        
    def test_invalid_file(self, temp_dir: str):
        """無效文件應返回錯誤"""
        mock_file = MagicMock()
        mock_file.name = str(Path(temp_dir) / "bad.exe")
        
        # 創建一個無效擴展名的文件
        bad_file = Path(temp_dir) / "bad.exe"
        bad_file.write_text("not an image")
        
        with patch('app.validate_file', return_value=(False, "不支持的文件格式")):
            result = analyze_images([mock_file], query="")
            assert "不支持" in result or "❌" in result

    def test_batch_limit_exceeded(self):
        """超過批量上限應返回限制提示"""
        files = [MagicMock() for _ in range(100)]  # MAX_BATCH_SIZE = 50
        result = analyze_images(files, query="")
        assert "超過" in result or "限制" in result or "50" in result


class TestSearchMemories:
    """記憶檢索功能整合測試"""

    def test_empty_query(self):
        """空查詢應返回提示"""
        result = search_memories(query="", user_id="test")
        assert "請輸入" in result

    def test_no_results(self):
        """無結果應返回提示"""
        with patch('app.get_memory') as mock_get_memory:
            mock_memory = MagicMock()
            mock_memory.search.return_value = []
            mock_get_memory.return_value = mock_memory
            
            result = search_memories(query="不存在", user_id="test")
            assert "未找到" in result

    def test_with_results(self):
        """有結果應返回格式化列表"""
        with patch('app.get_memory') as mock_get_memory:
            mock_memory = MagicMock()
            mock_memory.search.return_value = [
                {"id": "1", "content": "海邊度假", "metadata": {"timestamp": "2024-06-15"}, "distance": 0.15},
                {"id": "2", "content": "公園野餐", "metadata": {"timestamp": "2024-07-20"}, "distance": 0.25}
            ]
            mock_get_memory.return_value = mock_memory
            
            result = search_memories(query="海邊", top_k=5, user_id="test")
            assert "海邊度假" in result
            assert "公園野餐" in result
            assert "相關度" in result


class TestViewProfile:
    """用戶畫像功能整合測試"""

    def test_returns_valid_output(self):
        """view_profile 應返回格式化的畫像信息"""
        with patch('app.get_profile') as mock_get_profile:
            mock_profile = MagicMock()
            mock_profile.get_summary.return_value = {
                "user_id": "test_user",
                "stats": {
                    "total_analyses": 10,
                    "total_faces_detected": 15,
                    "total_images_with_faces": 8,
                    "total_images_without_faces": 2,
                },
                "top_tags": {"戶外": 5, "公園": 3},
                "mood_distribution": {"happy": 10},
                "gender_ratio": {"Male": 8, "Female": 7},
                "age_distribution": {"young_adult": 10, "senior": 5},
                "total_episodes": 10,
            }
            mock_get_profile.return_value = mock_profile
            
            result = view_profile(user_id="test_user")
            assert "test_user" in result
            assert "10" in result  # total_analyses
            assert "戶外" in result
            assert "Male" in result
            assert "happy" in result


class TestGlobalInstances:
    """全局實例延遲加載測試"""

    def test_get_agent_lazy_load(self):
        """get_agent 應在首次調用時創建實例"""
        from app import get_agent, _agent
        original = _agent
        # 重置
        import app
        app._agent = None
        
        agent = get_agent("test_user")
        assert agent is not None
        assert agent.user_id == "test_user"

    def test_get_memory_lazy_load(self):
        """get_memory 應在首次調用時創建實例"""
        from app import get_memory, _memory
        import app
        app._memory = None
        
        memory = get_memory("test_user")
        assert memory is not None

    def test_get_profile_lazy_load(self):
        """get_profile 應在首次調用時創建實例"""
        from app import get_profile, _profile
        import app
        app._profile = None
        
        profile = get_profile("test_user")
        assert profile is not None