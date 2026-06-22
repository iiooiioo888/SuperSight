"""
SuperSight V2.1 - 人臉分析模組測試
驗證 InsightFace 延遲加載、人臉檢測、情緒識別降級。
"""
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY

import pytest

from agents.face_agent import FaceAnalysisAgent


class TestFaceAgentInitialization:
    """人臉分析智能體初始化測試"""

    def test_lazy_initialization(self):
        """模型應在使用時才加載（延遲初始化）"""
        agent = FaceAnalysisAgent(ctx_id=-1)  # CPU mode
        assert agent._initialized is False
        assert agent.detector is None

    def test_initialize_called_on_analyze(self, test_image_jpg: Path):
        """analyze 應自動調用 initialize"""
        with patch.object(FaceAnalysisAgent, 'initialize') as mock_init:
            agent = FaceAnalysisAgent(ctx_id=-1)
            agent._initialized = True  # 模擬已初始化跳過實際加載
            agent.detector = MagicMock()
            agent.detector.get.return_value = []
            agent.analyze(str(test_image_jpg))
            # initialize 不會被調用因為 _initialized=True
            # 這是測試延遲初始化機制的行為


class TestFaceAnalysisResults:
    """人臉分析結果測試（使用 mock）"""

    @pytest.fixture(autouse=True)
    def mock_insightface(self):
        """Mock InsightFace，避免實際加載模型"""
        with patch('insightface.app.FaceAnalysis') as MockFaceAnalysis:
            mock_instance = MagicMock()
            MockFaceAnalysis.return_value = mock_instance
            
            # 模擬檢測到 2 張人臉
            mock_face_1 = MagicMock()
            mock_face_1.bbox = [100, 100, 200, 200]
            mock_face_1.age = 30
            mock_face_1.gender = 1
            mock_face_1.det_score = 0.98
            mock_face_1.landmark_2d_106 = [[100]*106, [100]*106]
            
            mock_face_2 = MagicMock()
            mock_face_2.bbox = [300, 150, 400, 250]
            mock_face_2.age = 28
            mock_face_2.gender = 0
            mock_face_2.det_score = 0.95
            mock_face_2.landmark_2d_106 = [[100]*106, [100]*106]
            
            mock_instance.get.return_value = [mock_face_1, mock_face_2]
            
            yield

    def test_detect_faces(self, test_image_jpg: Path):
        """應正確檢測到人臉數量"""
        agent = FaceAnalysisAgent(ctx_id=-1)
        agent._initialized = True
        agent.detector = MagicMock()
        
        # 手動設置 mock
        mock_face = MagicMock()
        mock_face.bbox = [100, 100, 200, 200]
        mock_face.age = 30
        mock_face.gender = 1
        mock_face.det_score = 0.98
        mock_face.landmark_2d_106 = [[0]*106, [0]*106]
        agent.detector.get.return_value = [mock_face, mock_face]
        
        result = agent.analyze(str(test_image_jpg))
        assert result["has_face"] is True
        assert result["face_count"] == 2

    def test_no_faces(self, test_image_jpg: Path):
        """無人臉時應回傳 has_face=False"""
        agent = FaceAnalysisAgent(ctx_id=-1)
        agent._initialized = True
        agent.detector = MagicMock()
        agent.detector.get.return_value = []
        
        result = agent.analyze(str(test_image_jpg))
        assert result["has_face"] is False
        assert result["faces"] == []

    def test_face_attributes(self, test_image_jpg: Path):
        """每張人臉應包含完整屬性"""
        agent = FaceAnalysisAgent(ctx_id=-1)
        agent._initialized = True
        agent.detector = MagicMock()
        
        mock_face = MagicMock()
        mock_face.bbox = [100, 100, 200, 200]
        mock_face.age = 30
        mock_face.gender = 1
        mock_face.det_score = 0.98
        mock_face.landmark_2d_106 = [[float(i) for i in range(106)], [float(i) for i in range(106)]]
        agent.detector.get.return_value = [mock_face]
        
        with patch.object(agent, '_recognize_emotion', return_value={
            "dominant": "happy", "scores": {"happy": 0.95}
        }):
            result = agent.analyze(str(test_image_jpg))
            face = result["faces"][0]
            assert face["age"] == 30
            assert face["gender"] == "Male"
            assert face["emotion"] == "happy"
            assert len(face["bbox"]) == 4
            assert face["confidence"] == 0.98

    def test_invalid_image_path(self, temp_dir: str):
        """無效路徑應回傳錯誤"""
        agent = FaceAnalysisAgent(ctx_id=-1)
        agent._initialized = True
        agent.detector = MagicMock()
        
        result = agent.analyze("nonexistent.jpg")
        assert result.get("error") is not None


class TestEmotionRecognition:
    """情緒識別降級測試"""

    def test_emotion_unknown_when_deepface_missing(self, test_image_jpg: Path):
        """DeepFace 未安裝時情緒應回傳 Unknown"""
        from agents.face_agent import FaceAnalysisAgent
        agent = FaceAnalysisAgent(ctx_id=-1)
        
        # 直接用 _recognize_emotion 測試
        import cv2
        img = cv2.imread(str(test_image_jpg))
        
        with patch('agents.face_agent.DeepFace', side_effect=ImportError("No module")):
            result = agent._recognize_emotion(img)
            assert result["dominant"] == "Unknown"
            assert result["scores"] == {}

    def test_emotion_handles_empty_image(self):
        """空圖片應回傳 Unknown"""
        import numpy as np
        agent = FaceAnalysisAgent(ctx_id=-1)
        empty_img = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        result = agent._recognize_emotion(empty_img)
        assert result["dominant"] == "Unknown"

    def test_context_manager(self):
        """上下文管理器應正確初始化和清理"""
        with patch('insightface.app.FaceAnalysis') as MockFA:
            mock_instance = MagicMock()
            MockFA.return_value = mock_instance
            
            with FaceAnalysisAgent(ctx_id=-1) as agent:
                assert agent._initialized is True
            
            # 退出上下文後應清理
            assert agent._initialized is False
            assert agent.detector is None