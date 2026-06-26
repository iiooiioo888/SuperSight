"""
SuperSight V3.0 - 人臉分析子智能體
基於 InsightFace (buffalo_m 2026 更新版) 進行人臉檢測與屬性分析，
並集成 DeepFace FER+ V2 進行情緒識別。

V3.0 升級：
- buffalo_m 模型 (2026 更新版，體積更小、速度更快)
- FER+ V2 情緒識別引擎
"""
import logging
import traceback
from typing import List, Dict, Optional, Any

import cv2
import numpy as np

# DeepFace 嘗試導入（可能不可用）
try:
    from deepface import DeepFace
    _DEEPFACE_AVAILABLE = True
except ImportError:
    DeepFace = None  # type: ignore
    _DEEPFACE_AVAILABLE = False


class FaceAnalysisAgent:
    """
    人臉分析智能體。
    
    功能：
    1. 人臉檢測與對齊 (InsightFace buffalo_m)
    2. 年齡/性別預測
    3. 情緒識別 (DeepFace FER+ V2)
    4. 面部關鍵點定位
    """
    
    def __init__(self, ctx_id: int = 0, det_size: tuple = (640, 640)):
        self.logger = logging.getLogger("SuperSight.FaceAgent")
        self._initialized = False
        self.ctx_id = ctx_id
        self.det_size = det_size
        self.detector = None
        self._deepface_available = True
        
    def initialize(self):
        """延遲初始化，僅在使用時加載模型以節省資源"""
        if self._initialized:
            return
            
        try:
            import insightface
            from insightface.app import FaceAnalysis
            
            # V3.0: 使用 buffalo_m 模型（2026 更新版，體積更小）
            self.detector = FaceAnalysis(name='buffalo_m')
            self.detector.prepare(ctx_id=self.ctx_id, det_size=self.det_size)
            self._initialized = True
            self.logger.info("InsightFace 模型加載成功 (buffalo_m, V3.0)")
            
        except ImportError as e:
            self.logger.error(
                f"InsightFace 導入失敗: {e}\n"
                "請確保安裝: pip install insightface onnxruntime"
            )
            raise
        except Exception as e:
            # V3.0: fallback 到 buffalo_l 若 buffalo_m 不可用
            try:
                import insightface
                from insightface.app import FaceAnalysis
                self.logger.warning(f"buffalo_m 加載失敗 ({e})，回退到 buffalo_l")
                self.detector = FaceAnalysis(name='buffalo_l')
                self.detector.prepare(ctx_id=self.ctx_id, det_size=self.det_size)
                self._initialized = True
            except Exception as e2:
                self.logger.error(f"InsightFace 初始化失敗: {e2}")
                raise
    
    def analyze(self, image_path: str) -> Dict[str, Any]:
        """
        分析圖片中所有人臉。
        
        Args:
            image_path: 圖片檔案路徑
        
        Returns:
            dict: 人臉分析結果
        """
        self.initialize()
        
        img = cv2.imread(image_path)
        if img is None:
            return {"has_face": False, "faces": [], "error": "無法讀取圖片，路徑可能無效"}
        
        try:
            faces = self.detector.get(img)
        except Exception as e:
            self.logger.error(f"人臉檢測失敗: {e}")
            return {"has_face": False, "faces": [], "error": f"人臉檢測失敗: {str(e)}"}
        
        if not faces:
            return {"has_face": False, "faces": []}
        
        results = []
        for face in faces:
            face_data = self._extract_face_data(img, face)
            results.append(face_data)
        
        self.logger.info(f"檢測到 {len(results)} 張人臉")
        
        return {
            "has_face": True,
            "face_count": len(results),
            "faces": results
        }
    
    def _extract_face_data(self, img: np.ndarray, face) -> Dict[str, Any]:
        """提取單張人臉的完整數據"""
        x1, y1, x2, y2 = map(int, face.bbox)
        face_img = img[max(0, y1):min(img.shape[0], y2), 
                       max(0, x1):min(img.shape[1], x2)]
        
        emotion = self._recognize_emotion(face_img)
        
        landmarks = None
        if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
            lm = face.landmark_2d_106
            landmarks = lm.tolist() if hasattr(lm, 'tolist') else lm
        elif hasattr(face, 'landmark_3d_68') and face.landmark_3d_68 is not None:
            lm = face.landmark_3d_68
            landmarks = lm.tolist() if hasattr(lm, 'tolist') else lm
        
        result = {
            "age": int(face.age) if hasattr(face, 'age') else 0,
            "gender": "Male" if (hasattr(face, 'gender') and face.gender == 1) else "Female",
            "emotion": emotion["dominant"],
            "emotion_scores": emotion["scores"],
            "bbox": [x1, y1, x2, y2],
            "confidence": float(face.det_score) if hasattr(face, 'det_score') else 0.0,
        }
        
        if landmarks:
            result["landmarks"] = landmarks
        
        return result
    
    def _recognize_emotion(self, face_img: np.ndarray) -> Dict[str, Any]:
        """使用 DeepFace FER+ 識別情緒"""
        default_result = {"dominant": "Unknown", "scores": {}}
        
        if face_img.size == 0:
            return default_result
        
        if not _DEEPFACE_AVAILABLE:
            if self._deepface_available:
                self.logger.warning("DeepFace 未安裝，情緒識別不可用。")
                self._deepface_available = False
            return default_result
        
        if DeepFace is None:
            return default_result
        
        try:
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            emotion_obj = DeepFace.analyze(
                img_path=face_rgb,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            
            if isinstance(emotion_obj, list) and len(emotion_obj) > 0:
                emotion_data = emotion_obj[0]
                dominant = emotion_data.get('dominant_emotion', 'Unknown')
                scores = emotion_data.get('emotion', {})
                
                normalized_scores = {
                    k.lower(): round(float(v), 2) 
                    for k, v in scores.items()
                }
                
                return {"dominant": dominant, "scores": normalized_scores}
                
        except Exception as e:
            self.logger.debug(f"情緒識別失敗（非致命）: {e}")
        
        return default_result
    
    def __enter__(self):
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.detector = None
        self._initialized = False