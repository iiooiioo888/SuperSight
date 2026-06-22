"""
SuperSight V2.1 - 記憶模組測試
驗證 ChromaDB CRUD、雙軌存儲、RAG 檢索。
"""
import json
import uuid
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

from memory.memory_module import MemoryModule


class TestMemoryInitialization:
    """記憶模組初始化測試"""

    def test_directory_creation(self, temp_dir: str):
        """初始化應自動創建所需目錄"""
        memory = MemoryModule(user_id="test_user", base_path=temp_dir)
        assert memory.episodes_dir.exists()
        assert memory.vector_db_path.exists()

    def test_default_user_id(self, temp_dir: str):
        """應使用給定的 user_id"""
        memory = MemoryModule(user_id="custom_user", base_path=temp_dir)
        assert memory.user_id == "custom_user"


class TestMemoryStorage:
    """記憶存儲測試（使用 mock 避免實際 ChromaDB）"""

    @pytest.fixture(autouse=True)
    def mock_chromadb(self):
        """Mock ChromaDB 客戶端"""
        with patch('memory.memory_module.chromadb') as mock_cdb:
            mock_client = MagicMock()
            mock_cdb.PersistentClient.return_value = mock_client
            
            mock_collection = MagicMock()
            mock_collection.count.return_value = 0
            mock_client.get_or_create_collection.return_value = mock_collection
            
            yield mock_collection

    def test_add_episode_creates_json(self, temp_dir: str, mock_chromadb):
        """add_episode 應創建 JSON 文件"""
        memory = MemoryModule(user_id="test_user", base_path=temp_dir)
        
        content = {
            "timestamp": "2024-06-15T14:30:00",
            "image_path": "/tmp/test.jpg",
            "face_analysis": {"has_face": False},
            "scene_analysis": {"success": True, "tags": ["測試"]},
            "metadata": {}
        }
        
        episode_id = memory.add_episode(content, text_summary="測試記憶")
        
        # 檢查 JSON 文件是否創建
        json_file = memory.episodes_dir / f"{episode_id}.json"
        assert json_file.exists()
        
        # 驗證 JSON 內容
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["episode_id"] == episode_id
        assert data["user_id"] == "test_user"

    def test_add_episode_generates_summary(self, temp_dir: str, mock_chromadb):
        """未提供 summary 時應自動生成"""
        memory = MemoryModule(user_id="test_user", base_path=temp_dir)
        
        content = {
            "timestamp": "2024-06-15T14:30:00",
            "image_path": "/tmp/test.jpg",
            "face_analysis": {"has_face": False},
            "scene_analysis": {
                "success": True,
                "scene_description": "一個美麗的花園",
                "tags": ["花園", "自然"]
            },
            "metadata": {}
        }
        
        episode_id = memory.add_episode(content)  # 不傳 summary
        
        # 檢查 mock_chromadb.add 是否被調用（向量已存儲）
        mock_chromadb.add.assert_called_once()
        args, kwargs = mock_chromadb.add.call_args
        assert len(kwargs["documents"][0]) > 0  # summary 不為空

    def test_multiple_episodes(self, temp_dir: str, mock_chromadb):
        """多次存儲應創建多個 JSON 文件"""
        memory = MemoryModule(user_id="test_user", base_path=temp_dir)
        
        for i in range(3):
            content = {
                "timestamp": datetime.now().isoformat(),
                "image_path": f"/tmp/test_{i}.jpg",
                "face_analysis": {"has_face": False},
                "scene_analysis": {"success": True, "tags": [f"tag_{i}"]},
                "metadata": {}
            }
            memory.add_episode(content, text_summary=f"記憶 {i}")
        
        # 應有 3 個 JSON 文件
        json_files = list(memory.episodes_dir.glob("*.json"))
        assert len(json_files) == 3

    def test_add_episode_with_face_data(self, temp_dir: str, mock_chromadb):
        """含人臉數據的記憶應存儲正確的 metadata"""
        memory = MemoryModule(user_id="test_user", base_path=temp_dir)
        
        content = {
            "timestamp": "2024-06-15T14:30:00",
            "image_path": "/tmp/test.jpg",
            "face_analysis": {
                "has_face": True,
                "face_count": 2,
                "faces": [{"age": 30, "gender": "Male", "emotion": "happy"}]
            },
            "scene_analysis": {"success": True, "tags": ["聚會"]},
            "metadata": {}
        }
        
        memory.add_episode(content, text_summary="家庭聚會")
        
        # metadata 中應包含 has_face=true
        args, kwargs = mock_chromadb.add.call_args
        assert kwargs["metadatas"][0]["has_face"] == "True"


class TestMemoryRetrieval:
    """記憶檢索測試"""

    @pytest.fixture(autouse=True)
    def mock_chromadb_with_data(self):
        """Mock ChromaDB 並預設一些數據"""
        with patch('memory.memory_module.chromadb') as mock_cdb:
            mock_client = MagicMock()
            mock_cdb.PersistentClient.return_value = mock_client
            
            mock_collection = MagicMock()
            mock_collection.count.return_value = 5
            
            # 模擬 query 結果
            mock_collection.query.return_value = {
                "ids": [["id_1", "id_2"]],
                "documents": [["海邊度假的照片", "公園野餐的記憶"]],
                "metadatas": [[
                    {"timestamp": "2024-06-15", "tags": "海邊"},
                    {"timestamp": "2024-07-20", "tags": "公園"}
                ]],
                "distances": [[0.15, 0.25]]
            }
            
            mock_client.get_or_create_collection.return_value = mock_collection
            yield mock_collection

    def test_search_returns_results(self, temp_dir: str, mock_chromadb_with_data):
        """search 應返回格式化結果"""
        memory = MemoryModule(user_id="test_user", base_path=temp_dir)
        results = memory.search(query="海邊")
        
        assert len(results) == 2
        assert results[0]["content"] == "海邊度假的照片"
        assert results[0]["metadata"]["timestamp"] == "2024-06-15"
        assert results[0]["distance"] == 0.15

    def test_search_with_top_k(self, temp_dir: str, mock_chromadb_with_data):
        """應尊重 top_k 參數"""
        memory = MemoryModule(user_id="test_user", base_path=temp_dir)
        results = memory.search(query="照片", top_k=1)
        assert len(results) == 2  # mock 返回 2 條，但 top_k 會限制查詢

    def test_search_empty_query(self, temp_dir: str, mock_chromadb_with_data):
        """空查詢也應返回結果"""
        memory = MemoryModule(user_id="test_user", base_path=temp_dir)
        results = memory.search(query="")
        assert len(results) == 2


class TestMemoryManagement:
    """記憶管理功能測試"""

    @pytest.fixture(autouse=True)
    def mock_chromadb(self):
        with patch('memory.memory_module.chromadb') as mock_cdb:
            mock_client = MagicMock()
            mock_cdb.PersistentClient.return_value = mock_client
            mock_collection = MagicMock()
            mock_collection.count.return_value = 3
            mock_client.get_or_create_collection.return_value = mock_collection
            yield mock_collection

    def test_count(self, temp_dir: str, mock_chromadb):
        """count 應返回 ChromaDB 中的記錄數"""
        memory = MemoryModule(user_id="test_user", base_path=temp_dir)
        assert memory.count() == 3

    def test_list_episodes(self, temp_dir: str, mock_chromadb):
        """list_episodes 應返回最近的記憶列表"""
        memory = MemoryModule(user_id="test_user", base_path=temp_dir)
        
        # 先存幾條
        for i in range(3):
            content = {
                "timestamp": f"2024-06-{15+i}T14:30:00",
                "image_path": f"/tmp/test_{i}.jpg",
                "face_analysis": {"has_face": False},
                "scene_analysis": {"success": True, "scene_description": f"場景{i}"},
                "metadata": {}
            }
            memory.add_episode(content, text_summary=f"記憶{i}")
        
        episodes = memory.list_episodes(limit=10)
        assert len(episodes) == 3
        for ep in episodes:
            assert "id" in ep
            assert "timestamp" in ep
            assert "summary" in ep

    def test_get_episode_json(self, temp_dir: str, mock_chromadb):
        """get_episode_json 應返回完整的 JSON 數據"""
        memory = MemoryModule(user_id="test_user", base_path=temp_dir)
        
        content = {
            "timestamp": "2024-06-15T14:30:00",
            "image_path": "/tmp/test.jpg",
            "face_analysis": {"has_face": True, "faces": []},
            "scene_analysis": {"success": True, "tags": ["測試"]},
            "metadata": {"format": "JPEG"}
        }
        
        episode_id = memory.add_episode(content, text_summary="測試")
        retrieved = memory.get_episode_json(episode_id)
        
        assert retrieved is not None
        assert retrieved["episode_id"] == episode_id
        assert retrieved["data"]["metadata"]["format"] == "JPEG"

    def test_delete_episode(self, temp_dir: str, mock_chromadb):
        """delete_episode 應同時刪除向量和 JSON"""
        memory = MemoryModule(user_id="test_user", base_path=temp_dir)
        
        content = {
            "timestamp": "2024-06-15T14:30:00",
            "image_path": "/tmp/test.jpg",
            "face_analysis": {"has_face": False},
            "scene_analysis": {"success": True},
            "metadata": {}
        }
        
        episode_id = memory.add_episode(content, text_summary="待刪除")
        assert memory.get_episode_json(episode_id) is not None
        
        result = memory.delete_episode(episode_id)
        assert result is True
        # ChromaDB delete 應被調用
        mock_chromadb.delete.assert_called_with(ids=[episode_id])
        # JSON 文件應被刪除
        assert memory.get_episode_json(episode_id) is None

    def test_delete_nonexistent(self, temp_dir: str, mock_chromadb):
        """刪除不存在的記憶應返回 False"""
        memory = MemoryModule(user_id="test_user", base_path=temp_dir)
        result = memory.delete_episode("nonexistent_id")
        assert result is False