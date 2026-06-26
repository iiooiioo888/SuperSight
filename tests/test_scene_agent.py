"""
SuperSight V2.1 - 場景理解模組測試
驗證 Ollama API 連接、Base64 編碼、JSON 回應解析。
"""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import requests

from agents.scene_agent import SceneUnderstandingAgent


class TestSceneAgentAvailability:
    """場景智能體可用性測試"""

    @pytest.fixture(autouse=True)
    def mock_requests(self):
        """Mock requests.get/post 避免實際 HTTP 調用"""
        with patch('agents.scene_agent.requests.get') as mock_get:
            with patch('agents.scene_agent.requests.post') as mock_post:
                self._mock_get = mock_get
                self._mock_post = mock_post
                yield mock_get

    def test_check_available(self, mock_requests):
        """Ollama 服務可用時應返回 True"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [{"name": "qwen3-vl:8b-fp4"}]
        }
        self._mock_get.return_value = mock_resp

        agent = SceneUnderstandingAgent()
        result = agent.check_availability()
        assert result is True
        assert agent._available is True

    def test_check_unavailable(self, mock_requests):
        """Ollama ���務不可用時應返回 False"""
        self._mock_get.side_effect = requests.ConnectionError("Connection refused")

        agent = SceneUnderstandingAgent()
        result = agent.check_availability()
        assert result is False
        assert agent._available is False

    def test_model_not_found(self, mock_requests):
        """指定模型未找到時應返回 False"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [{"name": "llama3:8b"}]
        }
        self._mock_get.return_value = mock_resp

        agent = SceneUnderstandingAgent()
        result = agent.check_availability()
        assert result is False


class TestSceneAnalysis:
    """場景分析功能測試"""

    @pytest.fixture(autouse=True)
    def setup(self):
        with patch('agents.scene_agent.requests.get') as mock_get:
            with patch('agents.scene_agent.requests.post') as mock_post:
                # 模擬 check_availability 成功
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.json.return_value = {
                    "models": [{"name": "qwen3-vl:8b-fp4"}]
                }
                mock_get.return_value = mock_resp
                
                # 模擬 analyze 回應
                mock_gen_resp = MagicMock()
                mock_gen_resp.status_code = 200
                mock_gen_resp.json.return_value = {
                    "response": """```json
{
    "scene_description": "一個陽光明媚的戶外公園",
    "main_content": "人們在草地上野餐",
    "ocr_text": "NO PARKING",
    "activity_inference": "休閒聚會",
    "tags": ["戶外", "公園", "野餐"]
}
```""",
                    "done": True
                }
                mock_post.return_value = mock_gen_resp
                
                yield mock_post

    def test_successful_analysis(self, test_image_jpg: Path, setup):
        """成功的分析應返回結構化 JSON"""
        agent = SceneUnderstandingAgent()
        agent._available = True  # 跳過可用性檢查
        
        result = agent.analyze(str(test_image_jpg))
        assert result["success"] is True
        assert "公園" in result.get("scene_description", "")
        assert result.get("tags") is not None
        assert len(result["tags"]) > 0

    def test_analysis_with_query(self, test_image_jpg: Path, setup):
        """自定義查詢應被傳遞"""
        agent = SceneUnderstandingAgent()
        agent._available = True
        
        result = agent.analyze(str(test_image_jpg), query="這些人開心嗎？")
        assert result["success"] is True

    def test_service_unavailable(self, test_image_jpg: Path):
        """服務不可用時應返回錯誤"""
        agent = SceneUnderstandingAgent()
        agent._available = False
        # mock check_availability 返回 False（不依賴 mock_requests）
        with patch.object(agent, 'check_availability', return_value=False):
            result = agent.analyze(str(test_image_jpg))
            assert result["success"] is False
            assert "不可用" in result.get("error", "")

    def test_api_error(self, test_image_jpg: Path, setup):
        """API 返回錯誤時應處理異常"""
        agent = SceneUnderstandingAgent()
        agent._available = True
        
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        setup.return_value = mock_resp
        
        result = agent.analyze(str(test_image_jpg))
        assert result["success"] is False


class TestJsonParsing:
    """JSON 回應解析測試"""

    def test_parse_valid_json(self):
        """標準 JSON 應被正確解析"""
        agent = SceneUnderstandingAgent()
        text = '{"scene_description": "公園", "tags": ["戶外"]}'
        result = agent._parse_json_response(text)
        assert result is not None
        assert result["scene_description"] == "公園"

    def test_parse_code_block_json(self):
        """```json 代碼塊包裹的 JSON 應被正確提取"""
        agent = SceneUnderstandingAgent()
        text = """```json
{
    "scene_description": "海邊"
}
```"""
        result = agent._parse_json_response(text)
        assert result is not None
        assert result["scene_description"] == "海邊"

    def test_parse_invalid_json(self):
        """無效 JSON 應返回 None"""
        agent = SceneUnderstandingAgent()
        result = agent._parse_json_response("這不是 JSON")
        assert result is None

    def test_parse_partial_json(self):
        """帶有前綴文本的 JSON 應被提取"""
        agent = SceneUnderstandingAgent()
        text = '根據分析結果：{"scene_description": "室內", "tags": []}'
        result = agent._parse_json_response(text)
        assert result is not None
        assert result["scene_description"] == "室內"


class TestBase64Encoding:
    """Base64 圖片編碼測試"""

    def test_encode_image(self, test_image_jpg: Path):
        """圖片應被正確編碼為 Base64"""
        agent = SceneUnderstandingAgent()
        encoded = agent._encode_image(str(test_image_jpg))
        assert len(encoded) > 0
        # Base64 應只包含 ASCII 字符
        assert all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=" for c in encoded)

    def test_encode_nonexistent_image(self, temp_dir: str):
        """不存在的圖片應拋出異常"""
        agent = SceneUnderstandingAgent()
        with pytest.raises(FileNotFoundError):
            agent._encode_image("nonexistent.jpg")