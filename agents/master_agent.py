"""
SuperSight V3.0 - 主智能體 (Master Agent)
基於 LangGraph 0.5.x 的異步並行狀態機，負責任務分解、並行調度、結果聚合與記憶更新。

V3.0 升級：
- LangGraph 0.5.x 原生異步並行分支 (Async Parallel Branches)
- 支援 bge-m4 1024維向量嵌入
- 4K 原生解析度處理
- FP4 量化硬體加速感知
"""
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List, TypedDict

from langgraph.graph import StateGraph, END

from config.settings import settings
from agents.face_agent import FaceAnalysisAgent
from agents.scene_agent import SceneUnderstandingAgent
from memory.memory_module import MemoryModule
from utils.resource_monitor import ResourceMonitor, ResourceLevel


# ─── V3.0 LangGraph 狀態定義 ──────────────────────

class V3AgentState(TypedDict):
    """V3.0 LangGraph 工作流狀態（支援更豐富的狀態類型）"""
    image_path: str
    query: str
    user_id: str
    
    face_result: Optional[Dict[str, Any]]
    scene_result: Optional[Dict[str, Any]]
    meta_result: Optional[Dict[str, Any]]
    
    final_report: str
    errors: List[str]
    resource_status: Optional[Dict[str, Any]]
    aggregated_data: Optional[Dict[str, Any]]
    memory_saved: bool


# ─── 元數據提取子智能體 ──────────────────────────

class MetaDataAgent:
    """提取 EXIF/GPS/設備信息（V3.0 支援更多 EXIF 標籤）"""
    
    @staticmethod
    def analyze(image_path: str) -> Dict[str, Any]:
        """
        提取圖片元數據。
        V3.0: 支援更多 EXIF 標籤 + 4K 圖片尺寸識別。
        """
        from PIL import Image
        from PIL.ExifTags import TAGS
        
        try:
            img = Image.open(image_path)
            result = {
                "format": img.format,
                "size": f"{img.width}x{img.height}",
                "mode": img.mode,
                "resolution": "4K" if max(img.width, img.height) >= 3840 else 
                             "2K" if max(img.width, img.height) >= 2048 else "HD",
            }
            
            # 提取 EXIF
            exif_data = img._getexif()
            if exif_data:
                exif = {}
                for tag_id, value in exif_data.items():
                    tag_name = TAGS.get(tag_id, tag_id)
                    
                    if tag_name in [
                        "DateTime", "DateTimeOriginal", "DateTimeDigitized",
                        "Make", "Model", "Software",
                        "GPSInfo", "Orientation",
                        "FocalLength", "FNumber", "ISOSpeedRatings",
                        "ExposureTime", "Flash",
                        "LensModel", "FocalLengthIn35mmFilm", # V3.0 新增
                    ]:
                        if tag_name == "GPSInfo" and isinstance(value, dict):
                            exif["GPS"] = MetaDataAgent._parse_gps(value)
                        else:
                            exif[tag_name] = str(value)
                
                result["exif"] = exif
            
            from pathlib import Path
            stat = Path(image_path).stat()
            result["file_mtime"] = datetime.fromtimestamp(stat.st_mtime).isoformat()
            result["file_size_mb"] = round(stat.st_size / (1024 * 1024), 2)
            
            return result
            
        except Exception as e:
            return {
                "format": None,
                "error": f"元數據提取失敗: {str(e)}"
            }
    
    @staticmethod
    def _parse_gps(gps_info: dict) -> Optional[Dict[str, Any]]:
        """解析 GPS 信息（PIL EXIF GPS IFD 格式）"""
        try:
            def _rational_to_float(r):
                if isinstance(r, tuple) and len(r) == 2:
                    return float(r[0]) / float(r[1]) if r[1] != 0 else 0.0
                return float(r)
            
            def _to_decimal(values, ref):
                if isinstance(values, (list, tuple)) and len(values) >= 3:
                    d = _rational_to_float(values[0])
                    m = _rational_to_float(values[1])
                    s = _rational_to_float(values[2])
                    decimal = d + m / 60.0 + s / 3600.0
                else:
                    decimal = float(values)
                
                if isinstance(ref, bytes):
                    ref = ref.decode()
                if ref in ('S', 'W'):
                    decimal = -decimal
                return round(decimal, 6)
            
            gps = {}
            # PIL GPS IFD: 0=LatRef, 1=Lat, 2=LonRef, 3=Lon, 5=Alt
            if 0 in gps_info and 1 in gps_info:
                gps["latitude"] = _to_decimal(gps_info[1], gps_info[0])
            if 2 in gps_info and 3 in gps_info:
                gps["longitude"] = _to_decimal(gps_info[3], gps_info[2])
            if 5 in gps_info:
                alt = gps_info[5]
                if isinstance(alt, tuple) and len(alt) == 2:
                    gps["altitude"] = float(alt[0]) / float(alt[1]) if alt[1] != 0 else 0.0
                else:
                    gps["altitude"] = float(alt)
            return gps if gps else None
        except Exception:
            return None


# ─── V3.0 主智能體 ──────────────────────────────

class SuperSightMasterAgent:
    """
    SuperSight V3.0 主智能體。
    
    V3.0 工作流：
    1. 資源檢查 → 條件路由
    2. 異步並行分支：
       - Branch A: 人臉分析 (InsightFace buffalo_m + DeepFace FER+ V2)
       - Branch B: 場景理解 (Qwen3-VL-8B FP4, 1M context, 4K native)
       - Branch C: 元數據提取 (EXIF/GPS/設備)
    3. 結果聚合 + bge-m4 嵌入生成
    4. 記憶存儲 (ChromaDB 1.2.x + JSON)
    5. 報告生成
    """
    
    def __init__(self, user_id: str = "default_user"):
        self.logger = logging.getLogger("SuperSight.MasterAgent")
        self.user_id = user_id
        
        self._face_agent = None
        self._scene_agent = None
        self._memory = None
        self._resource_monitor = ResourceMonitor()
        
        self.last_aggregated_data: Optional[Dict[str, Any]] = None
        
        self._graph = None
        self._build_workflow()
    
    @property
    def face_agent(self) -> FaceAnalysisAgent:
        if self._face_agent is None:
            self._face_agent = FaceAnalysisAgent(
                ctx_id=settings.GPU_DEVICE_ID,
                # V3.0: 使用 buffalo_m 模型（2026 更新版）
            )
        return self._face_agent
    
    @property
    def scene_agent(self) -> SceneUnderstandingAgent:
        if self._scene_agent is None:
            self._scene_agent = SceneUnderstandingAgent(
                base_url=settings.OLLAMA_BASE_URL,
                model_name=settings.VLM_MODEL_NAME,
                # V3.0: Qwen3-VL-8B FP4
            )
        return self._scene_agent
    
    @property
    def memory(self) -> MemoryModule:
        if self._memory is None:
            self._memory = MemoryModule(user_id=self.user_id)
        return self._memory
    
    def _build_workflow(self):
        """構建 V3.0 LangGraph 工作流（支援異步並行）"""
        workflow = StateGraph(V3AgentState)
        
        # 節點定義
        workflow.add_node("check_resources", self._check_resources_node)
        workflow.add_node("analyze_face", self._face_analysis_node)
        workflow.add_node("analyze_scene", self._scene_analysis_node)
        workflow.add_node("extract_meta", self._meta_extraction_node)
        workflow.add_node("aggregate_results", self._aggregation_node)
        workflow.add_node("update_memory", self._memory_node)
        workflow.add_node("generate_report", self._report_node)
        
        # V3.0 入口
        workflow.set_entry_point("check_resources")
        
        # V3.0 條件路由（4 路）
        # "full" 返回列表觸發 fan-out 並行執行 face + scene 分析
        workflow.add_conditional_edges(
            "check_resources",
            self._route_by_resources,
            {
                "full": ["analyze_face", "analyze_scene"],
                "vlm_only": "analyze_scene",
                "face_only": "analyze_face",
                "minimal": "extract_meta"
            }
        )
        
        # fan-in：face 和 scene 分析完畢後都進入 extract_meta
        workflow.add_edge("analyze_face", "extract_meta")
        workflow.add_edge("analyze_scene", "extract_meta")
        
        # 聚合與存儲
        workflow.add_edge("extract_meta", "aggregate_results")
        workflow.add_edge("aggregate_results", "update_memory")
        workflow.add_edge("update_memory", "generate_report")
        workflow.add_edge("generate_report", END)
        
        self._graph = workflow.compile()
        self.logger.info(f"V3.0 LangGraph 工作流構建完成 (模型: {settings.VLM_MODEL_NAME})")
    
    def _route_by_resources(self, state: V3AgentState) -> str:
        """根據資源狀態路由"""
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
    
    def _check_resources_node(self, state: V3AgentState) -> dict:
        """檢查資源狀態"""
        self.logger.info("檢查系統資源狀態...")
        self.logger.info(f"FP4 硬體加速: {'啟用' if settings.FP4_QUANTIZATION else '關閉'}")
        self.logger.info(f"GPU 架構: {settings.GPU_ARCHITECTURE}")
        
        status = self._resource_monitor.check_resource_status(
            need_vlm=True,
            need_face=True
        )
        
        if status.level == ResourceLevel.CRITICAL:
            self.logger.warning(f"資源不足，降級運行: {status.message}")
        
        # 根據實際路由方向精確設置可用標誌
        route = self._route_by_resources({"resource_status": {
            "level": status.level.value,
            "can_use_vlm": status.can_use_vlm,
            "can_use_face": status.can_use_face,
        }})
        
        state["resource_status"] = {
            "level": status.level.value,
            "message": status.message,
            "route": route,
            "can_use_vlm": route in ("full", "vlm_only"),
            "can_use_face": route in ("full", "face_only"),
            "vram_free_gb": status.vram_free_gb,
            "fp4_quantization": settings.FP4_QUANTIZATION,
        }
        
        return state
    
    def _face_analysis_node(self, state: V3AgentState) -> dict:
        """人臉分析節點"""
        resource = state.get("resource_status", {})
        if not resource.get("can_use_face", True):
            state["face_result"] = {"has_face": False, "skipped": True}
            return state
        
        try:
            result = self.face_agent.analyze(state["image_path"])
            state["face_result"] = result
            if result.get("has_face"):
                self.logger.info(f"人臉分析完成: {result.get('face_count', 0)} 張人臉")
            else:
                self.logger.info("未檢測到人臉")
        except Exception as e:
            self.logger.error(f"人臉分析失敗: {e}")
            state["face_result"] = {"has_face": False, "error": str(e)}
            state.setdefault("errors", []).append(f"人臉分析: {str(e)}")
        
        return state
    
    def _scene_analysis_node(self, state: V3AgentState) -> dict:
        """場景理解節點（V3.0: Qwen3-VL FP4）"""
        resource = state.get("resource_status", {})
        if not resource.get("can_use_vlm", True):
            state["scene_result"] = {"success": False, "skipped": True}
            return state
        
        try:
            result = self.scene_agent.analyze(
                image_path=state["image_path"],
                query=state.get("query", "")
            )
            state["scene_result"] = result
            if result.get("success"):
                self.logger.info("V3.0 場景分析完成 (Qwen3-VL + 4K 原生)")
            else:
                self.logger.warning(f"場景分析未完成: {result.get('error', '')}")
        except Exception as e:
            self.logger.error(f"場景分析失敗: {e}")
            state["scene_result"] = {"success": False, "error": str(e)}
            state.setdefault("errors", []).append(f"場景分析: {str(e)}")
        
        return state
    
    def _meta_extraction_node(self, state: V3AgentState) -> dict:
        """元數據提取節點"""
        try:
            result = MetaDataAgent.analyze(state["image_path"])
            state["meta_result"] = result
            resolution = result.get("resolution", "unknown")
            self.logger.info(f"元數據提取完成 (解析度: {resolution})")
        except Exception as e:
            self.logger.error(f"元數據提取失敗: {e}")
            state["meta_result"] = {"error": str(e)}
        
        return state
    
    def _aggregation_node(self, state: V3AgentState) -> dict:
        """結果聚合節點"""
        face = state.get("face_result", {})
        scene = state.get("scene_result", {})
        meta = state.get("meta_result", {})
        
        aggregated = {
            "timestamp": datetime.now().isoformat(),
            "image_path": state["image_path"],
            "schema_version": "3.0",
            "face_analysis": face,
            "scene_analysis": scene,
            "metadata": meta,
        }
        
        state["aggregated_data"] = aggregated
        self.logger.info("V3.0 結果聚合完成")
        
        return state
    
    def _memory_node(self, state: V3AgentState) -> dict:
        """記憶存儲節點（V3.0: bge-m4 嵌入）"""
        aggregated = state.get("aggregated_data", {})
        
        try:
            summary = self._generate_summary(aggregated)
            self.memory.add_episode(content=aggregated, text_summary=summary)
            self.logger.info("V3.0 記憶存儲完成 (bge-m4 嵌入)")
            state["memory_saved"] = True
        except Exception as e:
            self.logger.error(f"記憶存儲失敗: {e}")
            state.setdefault("errors", []).append(
                f"⚠️ 記憶存儲失敗（分析結果未持久化）: {str(e)}"
            )
            state["memory_saved"] = False
        
        return state
    
    def _report_node(self, state: V3AgentState) -> dict:
        """生成最終報告節點"""
        face = state.get("face_result", {})
        scene = state.get("scene_result", {})
        meta = state.get("meta_result", {})
        errors = state.get("errors", [])
        
        report_parts = []
        
        if scene.get("success"):
            report_parts.append(f"📸 **場景分析 (Qwen3-VL)**")
            if scene.get("scene_description"):
                report_parts.append(f"  場景: {scene['scene_description']}")
            if scene.get("main_content"):
                report_parts.append(f"  內容: {scene['main_content']}")
            if scene.get("ocr_text"):
                report_parts.append(f"  文字: {scene['ocr_text']}")
            if scene.get("activity_inference"):
                report_parts.append(f"  活動: {scene['activity_inference']}")
            if scene.get("details"):
                report_parts.append(f"  細節: {scene['details']}")
            if scene.get("tags"):
                report_parts.append(f"  標籤: {', '.join(scene['tags'])}")
        
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
        
        if meta:
            report_parts.append(f"\n📋 **元數據**")
            if meta.get("resolution"):
                report_parts.append(f"  解析度: {meta['resolution']} ({meta.get('size', '')})")
            if meta.get("file_size_mb"):
                report_parts.append(f"  文件大小: {meta['file_size_mb']}MB")
            if meta.get("exif"):
                exif = meta["exif"]
                if "DateTimeOriginal" in exif:
                    report_parts.append(f"  拍攝時間: {exif['DateTimeOriginal']}")
                if "GPS" in exif:
                    gps = exif["GPS"]
                    if "latitude" in gps and "longitude" in gps:
                        report_parts.append(f"  GPS: {gps['latitude']}, {gps['longitude']}")
                if "Make" in exif and "Model" in exif:
                    report_parts.append(f"  設備: {exif['Make']} {exif['Model']}")
        
        if errors:
            report_parts.append(f"\n⚠️ **警告**")
            for err in errors:
                report_parts.append(f"  - {err}")
        
        if state.get("memory_saved"):
            report_parts.append(f"\n💾 記憶已保存（可用自然語言檢索）")
        
        report = "\n".join(report_parts) if report_parts else "分析完成，但未能提取到有效信息。"
        state["final_report"] = report
        self.logger.info("V3.0 最終報告已生成")
        
        return state
    
    def _generate_summary(self, aggregated: Dict[str, Any]) -> str:
        """生成自然語言摘要（用於 bge-m4 1024維向量檢索）"""
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
        if scene.get("details"):
            parts.append(f"細節: {scene['details']}")
        
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
        """異步處理單張圖片"""
        initial_state = V3AgentState(
            image_path=image_path,
            query=query,
            user_id=self.user_id,
            face_result=None,
            scene_result=None,
            meta_result=None,
            final_report="",
            errors=[],
            resource_status=None,
            aggregated_data=None,
            memory_saved=False,
        )
        
        try:
            result = await self._graph.ainvoke(initial_state)
            self.last_aggregated_data = result.get("aggregated_data")
            return result.get("final_report", "分析失敗")
        except Exception as e:
            self.logger.error(f"V3.0 處理失敗: {e}")
            self.last_aggregated_data = None
            return f"❌ 分析過程中發生錯誤: {str(e)}"
    
    def process(self, image_path: str, query: str = "") -> str:
        """同步處理單張圖片"""
        import concurrent.futures
        try:
            # 檢查是否已有運行的事件循環（例如在 Gradio 中）
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            
            if loop is not None and loop.is_running():
                # 在已有事件循環的線程中（如 Gradio），在獨立線程中運行
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.process_async(image_path, query)
                    )
                    return future.result(timeout=300)  # 5 分鐘超時防止死鎖
            else:
                # 沒有運行中的循環，直接使用 asyncio.run
                return asyncio.run(self.process_async(image_path, query))
        except concurrent.futures.TimeoutError:
            self.logger.error("V3.0 處理超時 (300s)")
            self.last_aggregated_data = None
            return "❌ 分析超時，請稍後重試"
        except Exception as e:
            self.logger.error(f"V3.0 同步處理失敗: {e}")
            self.last_aggregated_data = None
            return f"❌ 分析過程中發生錯誤: {str(e)}"
    
    def cleanup(self):
        """清理資源"""
        self._resource_monitor.clear_gpu_memory()
        self._face_agent = None
        self._scene_agent = None
        self._memory = None
        self.last_aggregated_data = None