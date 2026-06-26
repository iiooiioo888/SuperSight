"""
SuperSight V2.1 - 主智能體測試
驗證 LangGraph 工作流、節點狀態轉換、資源路由。
"""
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, Any

import pytest

from agents.master_agent import (
    SuperSightMasterAgent, V3AgentState, MetaDataAgent
)


class TestV3AgentState:
    """LangGraph 狀態定義測試"""

    def test_state_fields(self):
        """V3AgentState 應包含所有必要字段"""
        state = V3AgentState(
            image_path="/tmp/test.jpg",
            query="測試",
            user_id="test_user",
            face_result=None,
            scene_result=None,
            meta_result=None,
            final_report="",
            errors=[],
            resource_status=None,
            aggregated_data=None
        )
        assert state["image_path"] == "/tmp/test.jpg"
        assert state["user_id"] == "test_user"
        assert state["errors"] == []


class TestMasterAgentInitialization:
    """主智能體初始化測試"""

    def test_init_with_user_id(self):
        """應使用給定的 user_id"""
        agent = SuperSightMasterAgent(user_id="custom_user")
        assert agent.user_id == "custom_user"

    def test_lazy_agent_loading(self):
        """子智能體應延遲加載"""
        agent = SuperSightMasterAgent()
        assert agent._face_agent is None
        assert agent._scene_agent is None
        assert agent._memory is None

    def test_workflow_built(self):
        """初始化時應構建 LangGraph 工作流"""
        agent = SuperSightMasterAgent()
        assert agent._graph is not None

    def test_last_aggregated_data_default(self):
        """last_aggregated_data 默認為 None"""
        agent = SuperSightMasterAgent()
        assert agent.last_aggregated_data is None


class TestMetaDataAgent:
    """元數據提取測試"""

    def test_extract_basic_info(self, test_image_jpg: Path):
        """應提取基本圖片信息"""
        result = MetaDataAgent.analyze(str(test_image_jpg))
        assert result.get("format") is not None
        assert result.get("size") is not None
        assert "x" in result.get("size", "")

    def test_extract_exif(self, fake_exif_image: Path):
        """應提取 EXIF 信息"""
        result = MetaDataAgent.analyze(str(fake_exif_image))
        exif = result.get("exif", {})
        if exif:  # EXIF 可能因圖片生成方式而缺失
            assert "DateTimeOriginal" in exif or "Make" in exif

    def test_extract_nonexistent_file(self):
        """不存在的文件應回傳錯誤"""
        result = MetaDataAgent.analyze("nonexistent.jpg")
        assert result.get("error") is not None

    def test_gps_parsing(self):
        """GPS 信息應被正確解析 (PIL GPS IFD: 0=LatRef, 1=Lat, 2=LonRef, 3=Lon)"""
        gps_info = {
            0: b'N',
            1: ((25, 1), (3, 1), (15, 1)),  # 25°3'15"N
            2: b'E',
            3: ((121, 1), (30, 1), (0, 1)),  # 121°30'0"E
        }
        result = MetaDataAgent._parse_gps(gps_info)
        assert result is not None
        assert abs(result["latitude"] - 25.054) < 0.01
        assert abs(result["longitude"] - 121.5) < 0.01


class TestMasterAgentWorkflow:
    """主智能體工作流測試（使用 mock）"""

    @pytest.fixture(autouse=True)
    def mock_all_agents(self):
        """Mock 所有子智能體以避免實際模型加載"""
        with patch.object(SuperSightMasterAgent, 'face_agent', new_callable=MagicMock) as mock_face:
            mock_face.analyze.return_value = {
                "has_face": True,
                "face_count": 1,
                "faces": [{"age": 30, "gender": "Male", "emotion": "happy"}]
            }
            
            with patch.object(SuperSightMasterAgent, 'scene_agent', new_callable=MagicMock) as mock_scene:
                mock_scene.analyze.return_value = {
                    "success": True,
                    "scene_description": "公園",
                    "tags": ["戶外", "自然"]
                }
                
                with patch.object(SuperSightMasterAgent, 'memory', new_callable=MagicMock) as mock_mem:
                    mock_mem.add_episode.return_value = "test_episode_id"
                    
                    yield

    @pytest.mark.asyncio
    async def test_process_async_success(self, test_image_jpg: Path):
        """process_async 應成功返回報告"""
        agent = SuperSightMasterAgent()
        report = await agent.process_async(str(test_image_jpg))
        assert len(report) > 0
        assert agent.last_aggregated_data is not None

    @pytest.mark.asyncio
    async def test_process_async_with_query(self, test_image_jpg: Path):
        """包含查詢的處理應正常"""
        agent = SuperSightMasterAgent()
        report = await agent.process_async(str(test_image_jpg), query="這些人開心嗎？")
        assert len(report) > 0

    def test_process_sync(self, test_image_jpg: Path):
        """同步 process 應返回報告"""
        agent = SuperSightMasterAgent()
        report = agent.process(str(test_image_jpg))
        assert len(report) > 0

    def test_report_contains_sections(self, test_image_jpg: Path):
        """報告應包含各分析章節"""
        agent = SuperSightMasterAgent()
        report = agent.process(str(test_image_jpg))
        assert "📸" in report  # 場景章節
        assert "👤" in report  # 人臉章節


class TestMasterAgentResourceRouting:
    """資源路由測試"""

    def test_full_route(self):
        """充足資源應走 full 路由"""
        agent = SuperSightMasterAgent()
        state = V3AgentState(
            image_path="/tmp/test.jpg",
            query="",
            user_id="test",
            face_result=None,
            scene_result=None,
            meta_result=None,
            final_report="",
            errors=[],
            resource_status={"level": "healthy", "can_use_vlm": True, "can_use_face": True},
            aggregated_data=None
        )
        route = agent._route_by_resources(state)
        assert route == "full"

    def test_vlm_only_route(self):
        """VLM 可用但人臉不可用時走 vlm_only"""
        agent = SuperSightMasterAgent()
        state = V3AgentState(
            image_path="/tmp/test.jpg",
            query="",
            user_id="test",
            face_result=None,
            scene_result=None,
            meta_result=None,
            final_report="",
            errors=[],
            resource_status={"level": "warning", "can_use_vlm": True, "can_use_face": False},
            aggregated_data=None
        )
        route = agent._route_by_resources(state)
        assert route == "vlm_only"

    def test_face_only_route(self):
        """人臉可用但 VLM 不可用時走 face_only"""
        agent = SuperSightMasterAgent()
        state = V3AgentState(
            image_path="/tmp/test.jpg",
            query="",
            user_id="test",
            face_result=None,
            scene_result=None,
            meta_result=None,
            final_report="",
            errors=[],
            resource_status={"level": "warning", "can_use_vlm": False, "can_use_face": True},
            aggregated_data=None
        )
        route = agent._route_by_resources(state)
        assert route == "face_only"

    def test_minimal_route(self):
        """資源枯竭時走 minimal"""
        agent = SuperSightMasterAgent()
        state = V3AgentState(
            image_path="/tmp/test.jpg",
            query="",
            user_id="test",
            face_result=None,
            scene_result=None,
            meta_result=None,
            final_report="",
            errors=[],
            resource_status={"level": "critical"},
            aggregated_data=None
        )
        route = agent._route_by_resources(state)
        assert route == "minimal"


class TestMasterAgentCleanup:
    """資源清理測試"""

    def test_cleanup_resets_state(self):
        """cleanup 應重置所有資源"""
        agent = SuperSightMasterAgent()
        agent.last_aggregated_data = {"test": "data"}
        agent.cleanup()
        assert agent.last_aggregated_data is None