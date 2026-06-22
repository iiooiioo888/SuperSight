"""
SuperSight V2.1 - 主智能體 (Master Agent)
基於 LangGraph 的狀態機，負責任務分解、並行調度、結果聚合與記憶更新。
"""
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List, TypedDict, Annotated, Sequence

from langgraph.graph import StateGraph, END, add_messages
from langgraph.checkpoint import MemorySaver

from config.settings import settings
from agents.face_agent import FaceAnalysisAgent
from agents.scene_agent import SceneUnderstandingAgent
from memory.memory_module import MemoryModule
from utils.resource_monitor import ResourceMonitor, ResourceLevel


# ─── LangGraph 狀態定義 ────────────────────────────

class AgentState(TypedDict):
    """LangGraph 工作流狀態"""
    image_path: str
    query: str
    user_id: str
    
    # 各分支結果
    face_result: Optional[Dict[str, Any]]
    scene_result: Optional[Dict[str, Any]]
    meta_result: Optional[Dict[str, Any]]
    
    # 聚合結果
    final_report: str
    errors: List[str]
    
    # 資源狀態
    resource_status: Optional[Dict[str, Any]]
    
    # 聚合數據（用於存儲和中繼）
    aggregated_data: Optional[Dict[str, Any]]


# ─── 元數據提取子智能體 ──────────────────────────

class MetaDataAgent:
    """提取 EXIF/GPS/設備信息"""
    
    @staticmethod
    def analyze(image_path: str) -> Dict[str, Any]:
        """
        提取圖片元數據。
        
        Args:
            image_path: 圖片檔案路徑
        
        Returns:
            dict: 元數據信息
        """
        from PIL import Image
        from PIL.ExifTags import TAGS
        
        try:
            img = Image.open(image_path)
            result = {
                "format": img.format,
                "size": f"{img.width}x{img.height}",
                "mode": img.mode,
            }
            
            # 提取 EXIF
            exif_data = img._getexif()
            if exif_data:
                exif = {}
                for tag_id, value in exif_data.items():
                    tag_name = TAGS.get(tag_id, tag_id)
                    
                    # 只提取有用的字段
                    if tag_name in [
                        "DateTime", "DateTimeOriginal", "DateTimeDigitized",
                        "Make", "Model", "Software",
                        "GPSInfo", "Orientation",
                        "FocalLength", "FNumber", "ISOSpeedRatings",
                        "ExposureTime", "Flash"
                    ]:
                        # GPS 信息需要特殊處理
                        if tag_name == "GPSInfo" and isinstance(value, dict):
                            exif["GPS"] = MetaDataAgent._parse_gps(value)
                        else:
                            exif[tag_name] = str(value)
                
                result["exif"] = exif
            
            # 文件修改時間
            from pathlib import Path
            stat = Path(image_path).stat()
            result["file_mtime"] = datetime.fromtimestamp(stat.st_mtime).isoformat()
            
            return result
            
        except Exception as e:
            return {
                "format": None,
                "error": f"元數據提取失敗: {str(e)}"
            }
    
    @staticmethod
    def _parse_gps(gps_info: dict) -> Optional[Dict[str, Any]]:
        """解析 GPS 信息"""
        try:
            def _to_decimal(values, ref):
                d, m, s = values
                decimal = d + m / 60.0 + s / 3600.0
                if ref in ('S', 'W'):
                    decimal = -decimal
                return round(decimal, 6)
            
            gps = {}
            
            if 2 in gps_info and 1 in gps_info:
                lat = _to_decimal(gps_info[2], gps_info.get(1, 'N'))
                gps["latitude"] = lat
            
            if 4 in gps_info and 3 in gps_info:
                lon = _to_decimal(gps_info[4], gps_info.get(3, 'E'))
                gps["longitude"] = lon
            
            if 5 in gps_info:
                gps["altitude"] = float(gps_info[5])
            
            return gps if gps else None
            
        except Exception:
            return None


# ─── LangGraph 節點函數 ──────────────────────────

class SuperSightMasterAgent:
    """
    SuperSight 主智能體。
    
    基於 LangGraph 的狀態機工作流：
    1. 人臉分析分支 (Face Branch)
    2. 場景理解分支 (Scene Branch) 
    3. 元數據提取分支 (Meta Branch)
    4. 結果聚合 (Aggregation)
    5. 記憶存儲 (Memory Update)
    """
    
    def __init__(self, user_id: str = "default_user"):
        self.logger = logging.getLogger("SuperSight.MasterAgent")
        self.user_id = user_id
        
        # 子智能體（延遲加載）
        self._face_agent = None
        self._scene_agent = None
        self._memory = None
        self._resource_monitor = ResourceMonitor()
        
        # 最近一次分析的聚合數據（用於 app.py 中更新畫像）
        self.last_aggregated_data: Optional[Dict[str, Any]] = None
        
        # LangGraph 圖
        self._graph = None
        self._build_workflow()
    
    @property
    def face_agent(self) -> FaceAnalysisAgent:
        if self._face_agent is None:
            self._face_agent = FaceAnalysisAgent(ctx_id=settings.GPU_DEVICE_ID)
        return self._face_agent
    
    @property
    def scene_agent(self) -> SceneUnderstandingAgent:
        if self._scene_agent is None:
            self._scene_agent = SceneUnderstandingAgent(
                base_url=settings.OLLAMA_BASE_URL,
                model_name=settings.VLM_MODEL_NAME,
                max_tokens=settings.VLM_MAX_TOKENS,
                temperature=settings.VLM_TEMPERATURE
            )
        return self._scene_agent
    
    @property
    def memory(self) -> MemoryModule:
        if self._memory is None:
            self._memory = MemoryModule(user_id=self.user_id)
        return self._memory
    
    def _build_workflow(self):
        """構建 LangGraph 工作流"""
        workflow = StateGraph(AgentState)
        
        # 註冊節點
        workflow.add_node("check_resources", self._check_resources_node)
        workflow.add_node("analyze_face", self._face_analysis_node)
        workflow.add_node("analyze_scene", self._scene_analysis_node)
        workflow.add_node("extract_meta", self._meta_extraction_node)
        workflow.add_node("aggregate_results", self._aggregation_node)
        workflow.add_node("update_memory", self._memory_node)
        workflow.add_node("generate_report", self._report_node)
        
        # 設置入口
        workflow.set_entry_point("check_resources")
        
        # 根據資源狀態決定分支
        workflow.add_conditional_edges(
            "check_resources",
            self._route_by_resources,
            {
                "full": "analyze_face",    # 全功能模式
                "vlm_only": "analyze_scene",  # 僅 VLM
                "face_only": "analyze_face",  # 僅人臉
                "minimal": "extract_meta"     # 僅元數據
            }
        )
        
        # 串行鏈（生產環境可用 asyncio.gather 實現真並行）
        workflow.add_edge("analyze_face", "analyze_scene")
        workflow.add_edge("analyze_scene", "extract_meta")
        
        # 聚合與存儲
        workflow.add_edge("extract_meta", "aggregate_results")
        workflow.add_edge("aggregate_results", "update_memory")
        workflow.add_edge("update_memory", "generate_report")
        workflow.add_edge("generate_report", END)
        
        # 編譯圖
        self._graph = workflow.compile()
        self.logger.info("LangGraph 工作流構建完成")
    
    def _route_by_resources(self, state: AgentState) -> str:
        """根據資源狀態路由到不同分支"""
        resource_status = state.get("resource_status", {})
        level = resource_status.get("level", "healthy")
        
        if level == "critical":
            return "minimal"
        elif level == "warning":
            can_vlm = resource_status.get("can_use_vlm", True)
            can_face = resource_status.get("can_use_face", True)
            
            if can_vlm and not can_face:
                return "vlm_only"
            elif can_face and not can_vlm:
                return "face_only"
            else:
                return "full"
        else:
            return "full"
    
    def _check_resources_node(self, state: AgentState) -> dict:
        """檢查資源狀態節點"""
        self.logger.info("檢查系統資源狀態...")
        
        status = self._resource_monitor.check_resource_status(
            need_vlm=True,
            need_face=True
        )
        
        if status.level == ResourceLevel.CRITICAL:
            self.logger.warning(f"資源不足，降級運行: {status.message}")
            for suggestion in status.suggestions:
                self.logger.warning(f"  建議: {suggestion}")
        
        state["resource_status"] = {
            "level": status.level.value,
            "message": status.message,
            "can_use_vlm": status.can_use_vlm,
            "can_use_face": status.can_use_face,
            "vram_free_gb": status.vram_free_gb
        }
        
        return state
    
    def _face_analysis_node(self, state: AgentState) -> dict:
        """人臉分析節點"""
        resource = state.get("resource_status", {})
        if not resource.get("can_use_face", True):
            self.logger.info("人臉分析已跳過（資源限制）")
            state["face_result"] = {"has_face": False, "skipped": True}
            return state
        
        self.logger.info(f"開始人臉分析: {state['image_path']}")
        
        try:
            result = self.face_agent.analyze(state["image_path"])
            state["face_result"] = result
            
            if result.get("has_face"):
                count = result.get("face_count", 0)
                self.logger.info(f"人臉分析完成: {count} 張人臉")
            else:
                self.logger.info("人臉分析完成: 未檢測到人臉")
                
        except Exception as e:
            self.logger.error(f"人臉分析失敗: {e}")
            state["face_result"] = {"has_face": False, "error": str(e)}
            state.setdefault("errors", []).append(f"人臉分析: {str(e)}")
        
        return state
    
    def _scene_analysis_node(self, state: AgentState) -> dict:
        """場景理解節點"""
        resource = state.get("resource_status", {})
        if not resource.get("can_use_vlm", True):
            self.logger.info("場景分析已跳過（資源限制）")
            state["scene_result"] = {"success": False, "skipped": True}
            return state
        
        self.logger.info(f"開始場景分析: {state['image_path']}")
        
        try:
            result = self.scene_agent.analyze(
                image_path=state["image_path"],
                query=state.get("query", "")
            )
            state["scene_result"] = result
            
            if result.get("success"):
                self.logger.info("場景分析完成")
            else:
                self.logger.warning(f"場景分析未完成: {result.get('error', '')}")
                
        except Exception as e:
            self.logger.error(f"場景分析失敗: {e}")
            state["scene_result"] = {"success": False, "error": str(e)}
            state.setdefault("errors", []).append(f"場景分析: {str(e)}")
        
        return state
    
    def _meta_extraction_node(self, state: AgentState) -> dict:
        """元數據提取節點"""
        self.logger.info("開始提取元數據...")
        
        try:
            result = MetaDataAgent.analyze(state["image_path"])
            state["meta_result"] = result
            self.logger.info("元數據提取完成")
        except Exception as e:
            self.logger.error(f"元數據提取失敗: {e}")
            state["meta_result"] = {"error": str(e)}
        
        return state
    
    def _aggregation_node(self, state: AgentState) -> dict:
        """結果聚合節點"""
        self.logger.info("聚合分析結果...")
        
        face = state.get("face_result", {})
        scene = state.get("scene_result", {})
        meta = state.get("meta_result", {})
        
        aggregated = {
            "timestamp": datetime.now().isoformat(),
            "image_path": state["image_path"],
            "face_analysis": face,
            "scene_analysis": scene,
            "metadata": meta,
        }
        
        state["aggregated_data"] = aggregated
        self.logger.info("結果聚合完成")
        
        return state
    
    def _memory_node(self, state: AgentState) -> dict:
        """記憶存儲節點"""
        aggregated = state.get("aggregated_data", {})
        
        try:
            # 生成自然語言摘要
            summary = self._generate_summary(aggregated)
            
            # 存儲到記憶模組
            self.memory.add_episode(
                content=aggregated,
                text_summary=summary
            )
            
            self.logger.info("記憶存儲完成")
            
        except Exception as e:
            self.logger.error(f"記憶存儲失敗: {e}")
            state.setdefault("errors", []).append(f"記憶存儲: {str(e)}")
        
        return state
    
    def _report_node(self, state: AgentState) -> dict:
        """生成最終報告節點"""
        face = state.get("face_result", {})
        scene = state.get("scene_result", {})
        meta = state.get("meta_result", {})
        errors = state.get("errors", [])
        
        report_parts = []
        
        # 場景描述
        if scene.get("success"):
            report_parts.append(f"📸 **場景分析**")
            if scene.get("scene_description"):
                report_parts.append(f"  場景: {scene['scene_description']}")
            if scene.get("main_content"):
                report_parts.append(f"  內容: {scene['main_content']}")
            if scene.get("ocr_text"):
                report_parts.append(f"  文字: {scene['ocr_text']}")
            if scene.get("activity_inference"):
                report_parts.append(f"  活動: {scene['activity_inference']}")
            if scene.get("tags"):
                report_parts.append(f"  標籤: {', '.join(scene['tags'])}")
        
        # 人臉分析
        if face.get("has_face"):
            report_parts.append(f"\n👤 **人臉分析**")
            for i, f in enumerate(face.get("faces", [])):
                report_parts.append(
                    f"  人物 {i+1}: {f['gender']}, 約 {f['age']}歲, "
                    f"情緒: {f['emotion']}"
                )
        elif face.get("skipped"):
            report_parts.append(f"\n👤 人臉分析已跳過（資源限制）")
        elif face.get("has_face") is False:
            report_parts.append(f"\n👤 未檢測到人臉")
        
        # 元數據
        if meta:
            report_parts.append(f"\n📋 **元數據**")
            if meta.get("format"):
                report_parts.append(f"  格式: {meta['format']}, 尺寸: {meta['size']}")
            if meta.get("exif"):
                exif = meta["exif"]
                if "DateTimeOriginal" in exif:
                    report_parts.append(f"  拍攝時間: {exif['DateTimeOriginal']}")
                if "GPS" in exif:
                    gps = exif["GPS"]
                    if "latitude" in gps and "longitude" in gps:
                        report_parts.append(
                            f"  GPS: {gps['latitude']}, {gps['longitude']}"
                        )
                if "Make" in exif and "Model" in exif:
                    report_parts.append(f"  設備: {exif['Make']} {exif['Model']}")
        
        # 錯誤信息
        if errors:
            report_parts.append(f"\n⚠️ **警告**")
            for err in errors:
                report_parts.append(f"  - {err}")
        
        # 組合報告
        report = "\n".join(report_parts) if report_parts else "分析完成，但未能提取到有效信息。"
        
        state["final_report"] = report
        self.logger.info("最終報告已生成")
        
        return state
    
    def _generate_summary(self, aggregated: Dict[str, Any]) -> str:
        """從聚合數據生成自然語言摘要（用於向量檢索）"""
        parts = []
        
        scene = aggregated.get("scene_analysis", {})
        face = aggregated.get("face_analysis", {})
        meta = aggregated.get("metadata", {})
        
        if scene.get("scene_description"):
            parts.append(f"場景: {scene['scene_description']}")
        
        if scene.get("main_content"):
            parts.append(f"內容: {scene['main_content']}")
        
        if scene.get("activity_inference"):
            parts.append(f"活動: {scene['activity_inference']}")
        
        if face.get("has_face"):
            for f in face.get("faces", []):
                parts.append(f"人物: {f['gender']}, {f['age']}歲, 情緒{f['emotion']}")
        
        if meta.get("exif"):
            exif = meta["exif"]
            if "DateTimeOriginal" in exif:
                parts.append(f"時間: {exif['DateTimeOriginal']}")
        
        if scene.get("tags"):
            parts.append(f"標籤: {', '.join(scene['tags'])}")
        
        return " | ".join(parts) if parts else "圖片分析記錄"
    
    # ─── 公開接口 ──────────────────────────────────
    
    async def process_async(self, image_path: str, query: str = "") -> str:
        """
        異步處理單張圖片（主入口）。
        
        Args:
            image_path: 圖片檔案路徑
            query: 用戶查詢（可選）
        
        Returns:
            str: 分析報告
        """
        initial_state = AgentState(
            image_path=image_path,
            query=query,
            user_id=self.user_id,
            face_result=None,
            scene_result=None,
            meta_result=None,
            final_report="",
            errors=[],
            resource_status=None,
            aggregated_data=None
        )
        
        try:
            result = await self._graph.ainvoke(initial_state)
            # 保存聚合數據，供 app.py 更新畫像使用
            self.last_aggregated_data = result.get("aggregated_data")
            return result.get("final_report", "分析失敗")
        except Exception as e:
            self.logger.error(f"處理失敗: {e}")
            self.last_aggregated_data = None
            return f"❌ 分析過程中發生錯誤: {str(e)}"
    
    def process(self, image_path: str, query: str = "") -> str:
        """
        同步處理單張圖片。
        
        Args:
            image_path: 圖片檔案路徑
            query: 用戶查詢（可選）
        
        Returns:
            str: 分析報告
        """
        try:
            # 使用 asyncio 事件循環
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.process_async(image_path, query))
            loop.close()
            return result
        except Exception as e:
            self.logger.error(f"同步處理失敗: {e}")
            self.last_aggregated_data = None
            return f"❌ 分析過程中發生錯誤: {str(e)}"
    
    def cleanup(self):
        """清理資源"""
        self._resource_monitor.clear_gpu_memory()
        self._face_agent = None
        self._scene_agent = None
        self._memory = None
        self.last_aggregated_data = None