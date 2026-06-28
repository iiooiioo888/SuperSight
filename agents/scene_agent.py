"""
SuperSight V3.0 - 場景理解子智能體
基於 Qwen3-VL (via Ollama API) 進行場景描述、活動推斷、OCR 文本提取。
2026 世代：支援 1M 超長上下文 + 原生 4K 解析度 + FP4 加速。
"""
import base64
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import requests

from config.settings import settings


class SceneUnderstandingAgent:
    """
    場景理解智能體。
    
    V3.0 升級：
    1. Qwen3-VL-8B FP4 量化 (Blackwell 原生加速)
    2. 1M tokens 超長上下文窗口
    3. 原生 4K 解析度支援（不再強制縮放）
    4. 動態分辨率處理
    
    功能：
    1. 場景描述（整體環境、時間、氛圍）
    2. 活動推斷（人物在做什麼）
    3. OCR 文本提取（4K 高解析度支援）
    4. 物體檢測描述
    """
    
    def __init__(self, 
                 base_url: str = None,
                 model_name: str = None,
                 max_tokens: int = None,
                 temperature: float = 0.1):
        
        self.logger = logging.getLogger("SuperSight.SceneAgent")
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model_name = model_name or settings.VLM_MODEL_NAME
        self.max_tokens = max_tokens or settings.VLM_MAX_TOKENS
        self.temperature = temperature
        self._available = False
        
        # V3.0 系統提示詞 - 利用超長上下文進行深度分析
        self.system_prompt = """你是一個專業的高解析度圖片分析助手。請仔細觀察圖片並提供以下信息：

1. **場景描述**：詳細描述圖片中的環境（室內/室外、天氣、時間、氛圍、顏色調性）
2. **主要內容**：圖片中有什麼物體/人物？他們在做什麼？（利用高解析度優勢捕捉細節）
3. **文本內容**：圖片中出現的所有文字（如有），4K 解析度下應能清晰辨識小字
4. **活動推斷**：推測圖片的拍攝場景或正在發生的事件
5. **細節觀察**：注意到哪些有趣的細節或亮點

請用中文回答，保持客觀描述，不要猜測超出圖片信息的內容。
輸出格式為 JSON：
{
    "scene_description": "詳細場景描述",
    "main_content": "主要內容描述",
    "ocr_text": "提取的文字內容",
    "activity_inference": "活動推斷",
    "details": "其他細節觀察",
    "tags": ["標籤1", "標籤2", "標籤3"]
}"""
    
    def check_availability(self) -> bool:
        """
        檢查 Ollama 服務是否可用。
        
        Returns:
            bool: 服務是否正常
        """
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                model_names = [m["name"] for m in models]
                
                if self.model_name in model_names:
                    self._available = True
                    self.logger.info(f"Ollama 服務可用，模型 {self.model_name} 已就緒")
                    
                    # V3.0: 檢查 FP4 加速是否啟用
                    try:
                        ps_resp = requests.post(f"{self.base_url}/api/ps", timeout=3)
                        if ps_resp.status_code == 200:
                            running_models = ps_resp.json().get("models", [])
                            for m in running_models:
                                if self.model_name in m.get("name", ""):
                                    engine = m.get("engine", "unknown")
                                    self.logger.info(f"模型引擎: {engine}")
                    except Exception:
                        pass
                else:
                    self.logger.warning(
                        f"模型 {self.model_name} 未找到。"
                        f"可用模型: {model_names}\n"
                        f"請執行: ollama pull {self.model_name}"
                    )
                    self._available = False
            else:
                self.logger.warning(f"Ollama 服務返回異常狀態碼: {resp.status_code}")
                self._available = False
        except requests.ConnectionError:
            self.logger.warning(
                f"無法連接到 Ollama 服務 ({self.base_url})。\n"
                "請確保 Ollama 正在運行: ollama serve"
            )
            self._available = False
        except Exception as e:
            self.logger.error(f"檢查 Ollama 服務時出錯: {e}")
            self._available = False
        
        return self._available
    
    def _encode_image(self, image_path: str) -> str:
        """
        將圖片編碼為 Base64 字符串。
        V3.0: 支援 4K 圖片編碼，不再強制縮小。

        Args:
            image_path: 圖片檔案路徑
        
        Returns:
            Base64 編碼的圖片數據
        """
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def analyze(self, image_path: str, query: str = "") -> Dict[str, Any]:
        """
        分析圖片場景。
        V3.0: 支援 1M context + 4K 原生解析度。
        
        Args:
            image_path: 圖片檔案路徑
            query: 用戶附加查詢（可選）
        
        Returns:
            dict: 場景分析結果
        """
        # 每次都重新驗證可用性（服務可能中途停止或恢復）
        self.check_availability()
        
        if not self._available:
            return {
                "success": False,
                "error": "Ollama 服務不可用，請先啟動: ollama serve",
                "scene_description": "",
                "main_content": "",
                "ocr_text": "",
                "activity_inference": "",
                "details": "",
                "tags": []
            }
        
        try:
            # 編碼圖片 (V3.0: 保留原生解析度)
            image_base64 = self._encode_image(image_path)
            
            # 構建用戶提示
            user_prompt = "請詳細分析這張圖片。"
            if query:
                user_prompt += f"\n用戶特別關注：{query}"
            
            # V3.0: 調用 Ollama API 並利用 1M context
            payload = {
                "model": self.model_name,
                "prompt": user_prompt,
                "system": self.system_prompt,
                "stream": False,
                "options": {
                    "num_predict": self.max_tokens,
                    "temperature": self.temperature,
                    "num_ctx": settings.VLM_CONTEXT_WINDOW,  # 1M tokens
                },
                "images": [image_base64]
            }
            
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=180  # V3.0: 4K 圖片處理需要更長時間
            )
            
            if resp.status_code != 200:
                return {
                    "success": False,
                    "error": f"Ollama API 返回錯誤: {resp.status_code} {resp.text}",
                    "scene_description": "",
                    "main_content": "",
                    "ocr_text": "",
                    "activity_inference": "",
                    "details": "",
                    "tags": []
                }
            
            result = resp.json()
            response_text = result.get("response", "")
            
            # 嘗試解析 JSON 輸出
            parsed = self._parse_json_response(response_text)
            
            if parsed:
                parsed["success"] = True
                parsed["error"] = ""
                return parsed
            else:
                return {
                    "success": True,
                    "error": "",
                    "scene_description": response_text,
                    "main_content": "",
                    "ocr_text": "",
                    "activity_inference": "",
                    "details": "",
                    "tags": []
                }
                
        except requests.Timeout:
            self.logger.error("VLM 推理超時（4K 圖片可能需要更長時間）")
            return {
                "success": False,
                "error": "VLM 推理超時，圖片解析度過高或模型回應過慢",
                "scene_description": "",
                "main_content": "",
                "ocr_text": "",
                "activity_inference": "",
                "details": "",
                "tags": []
            }
        except Exception as e:
            self.logger.error(f"場景分析失敗: {e}")
            return {
                "success": False,
                "error": f"場景分析失敗: {str(e)}",
                "scene_description": "",
                "main_content": "",
                "ocr_text": "",
                "activity_inference": "",
                "details": "",
                "tags": []
            }
    
    def _parse_json_response(self, text: str) -> Optional[Dict[str, Any]]:
        """
        嘗試從模型回應中解析 JSON。
        """
        text = text.strip()
        
        # 提取 ```json ... ``` 代碼塊
        if "```json" in text:
            json_start = text.find("```json") + 7
            json_end = text.find("```", json_start)
            if json_end != -1:
                text = text[json_start:json_end].strip()
        elif "```" in text:
            json_start = text.find("```") + 3
            json_end = text.find("```", json_start)
            if json_end != -1:
                text = text[json_start:json_end].strip()
        
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end != -1:
            text = text[brace_start:brace_end + 1]
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
    
    def __enter__(self):
        self.check_availability()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass