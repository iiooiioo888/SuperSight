"""
SuperSight V2.1 - 場景理解子智能體
基於 Qwen2.5-VL (via Ollama API) 進行場景描述、活動推斷、OCR 文本提取。
"""
import base64
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import requests


class SceneUnderstandingAgent:
    """
    場景理解智能體。
    
    功能：
    1. 場景描述（整體環境、時間、氛圍）
    2. 活動推斷（人物在做什麼）
    3. OCR 文本提取
    4. 物體檢測描述
    
    通過 Ollama API 調用 Qwen2.5-VL 模型。
    """
    
    def __init__(self, 
                 base_url: str = "http://localhost:11434",
                 model_name: str = "qwen2.5-vl:7b-instruct-q4_k_m",
                 max_tokens: int = 512,
                 temperature: float = 0.1):
        
        self.logger = logging.getLogger("SuperSight.SceneAgent")
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._available = False
        
        # 構建系統提示詞（定義輸出格式）
        self.system_prompt = """你是一個專業的圖片分析助手。請仔細觀察圖片並提供以下信息：

1. **場景描述**：簡潔描述圖片中的環境（室內/室外、天氣、時間、氛圍）
2. **主要內容**：圖片中有什麼物體/人物？他們在做什麼？
3. **文本內容**：圖片中出現的所有文字（如有）
4. **活動推斷**：推測圖片的拍攝場景或正在發生的事件

請用中文回答，保持客觀描述，不要猜測超出圖片信息的內容。
輸出格式為 JSON：
{
    "scene_description": "場景描述",
    "main_content": "主要內容描述",
    "ocr_text": "提取的文字",
    "activity_inference": "活動推斷",
    "tags": ["標籤1", "標籤2"]
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
        
        Args:
            image_path: 圖片檔案路徑
            query: 用戶附加查詢（可選）
        
        Returns:
            dict: 場景分析結果
        """
        if not self._available:
            self.check_availability()
        
        if not self._available:
            return {
                "success": False,
                "error": "Ollama 服務不可用，請先啟動: ollama serve",
                "scene_description": "",
                "main_content": "",
                "ocr_text": "",
                "activity_inference": "",
                "tags": []
            }
        
        try:
            # 編碼圖片
            image_base64 = self._encode_image(image_path)
            image_url = f"data:image/jpeg;base64,{image_base64}"
            
            # 構建用戶提示
            user_prompt = "請分析這張圖片。"
            if query:
                user_prompt += f"\n用戶特別關注：{query}"
            
            # 調用 Ollama API
            payload = {
                "model": self.model_name,
                "prompt": user_prompt,
                "system": self.system_prompt,
                "stream": False,
                "options": {
                    "num_predict": self.max_tokens,
                    "temperature": self.temperature,
                },
                "images": [image_base64]
            }
            
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120  # VLM 推理可能需要較長時間
            )
            
            if resp.status_code != 200:
                return {
                    "success": False,
                    "error": f"Ollama API 返回錯誤: {resp.status_code} {resp.text}",
                    "scene_description": "",
                    "main_content": "",
                    "ocr_text": "",
                    "activity_inference": "",
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
                # JSON 解析失敗，返回原始文本
                return {
                    "success": True,
                    "error": "",
                    "scene_description": response_text,
                    "main_content": "",
                    "ocr_text": "",
                    "activity_inference": "",
                    "tags": []
                }
                
        except requests.Timeout:
            self.logger.error("VLM 推理超時")
            return {
                "success": False,
                "error": "VLM 推理超時，圖片可能過大或模型回應過慢",
                "scene_description": "",
                "main_content": "",
                "ocr_text": "",
                "activity_inference": "",
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
                "tags": []
            }
    
    def _parse_json_response(self, text: str) -> Optional[Dict[str, Any]]:
        """
        嘗試從模型回應中解析 JSON。
        
        Args:
            text: 模型原始回應文本
        
        Returns:
            解析後的 JSON 字典，或 None 若解析失敗
        """
        # 嘗試直接解析
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
        
        # 嘗試找到 { } 包裹的 JSON
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
        """清理資源"""
        pass