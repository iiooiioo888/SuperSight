"""
SuperSight V2.1 - 用戶畫像管理器 (Profile Manager)
管理用戶偏好統計、人物關係追蹤、長期趨勢分析。
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import Counter

from config.settings import settings


class ProfileManager:
    """
    用戶畫像管理器。
    
    功能：
    1. 統計分析（總分析次數、人臉數、常見場景等）
    2. 人物關係（出現頻率、情緒傾向）
    3. 時間維度分析（活躍時間段等）
    """
    
    def __init__(self, user_id: str):
        self.logger = logging.getLogger("SuperSight.ProfileManager")
        self.user_id = user_id
        
        self.profile_dir = settings.PROFILE_DIR / user_id
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        
        self.profile_path = self.profile_dir / "profile.json"
        self._profile = self._load_profile()
        
        self.logger.info(f"畫像管理器初始化完成 (用戶: {user_id})")
    
    def _load_profile(self) -> Dict[str, Any]:
        """從文件加載或創建默認畫像"""
        if self.profile_path.exists():
            try:
                with open(self.profile_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"加載畫像文件失敗，創建新畫像: {e}")
        
        return self._default_profile()
    
    def _default_profile(self) -> Dict[str, Any]:
        """創建默認畫像結構（全部使用普通 dict，避免 Counter 序列化不一致）"""
        return {
            "user_id": self.user_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "stats": {
                "total_analyses": 0,
                "total_faces_detected": 0,
                "total_images_with_faces": 0,
                "total_images_without_faces": 0,
            },
            "scene_stats": {
                "top_tags": {},
                "mood_distribution": {},
            },
            "face_stats": {
                "gender_distribution": {},
                "age_groups": {},
                "emotion_distribution": {},
            },
            "time_activity": {
                "by_hour": {},
                "by_month": {},
            },
            "recent_episodes": [],
        }
    
    def update(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用新的分析結果更新用戶畫像。
        
        Args:
            episode_data: 聚合後的完整分析數據
        
        Returns:
            dict: 更新後的畫像
        """
        profile = self._profile
        
        # 基本統計
        profile["stats"]["total_analyses"] += 1
        
        # 人臉統計
        face_analysis = episode_data.get("face_analysis", {})
        if face_analysis.get("has_face"):
            profile["stats"]["total_images_with_faces"] += 1
            faces = face_analysis.get("faces", [])
            profile["stats"]["total_faces_detected"] += len(faces)
            
            for face in faces:
                # 性別
                gender = face.get("gender", "Unknown")
                profile["face_stats"]["gender_distribution"][gender] = \
                    profile["face_stats"]["gender_distribution"].get(gender, 0) + 1
                
                # 年齡分組
                age = face.get("age", 0)
                age_group = self._categorize_age(age)
                profile["face_stats"]["age_groups"][age_group] = \
                    profile["face_stats"]["age_groups"].get(age_group, 0) + 1
                
                # 情緒
                emotion = face.get("emotion", "Unknown")
                profile["face_stats"]["emotion_distribution"][emotion] = \
                    profile["face_stats"]["emotion_distribution"].get(emotion, 0) + 1
        else:
            profile["stats"]["total_images_without_faces"] += 1
        
        # 場景統計
        scene_analysis = episode_data.get("scene_analysis", {})
        if scene_analysis.get("tags"):
            for tag in scene_analysis["tags"]:
                profile["scene_stats"]["top_tags"][tag] = \
                    profile["scene_stats"]["top_tags"].get(tag, 0) + 1
        
        # 時間活動
        timestamp = episode_data.get("timestamp", datetime.now().isoformat())
        try:
            dt = datetime.fromisoformat(timestamp)
            hour_key = str(dt.hour)
            month_key = f"{dt.year}-{dt.month:02d}"
            profile["time_activity"]["by_hour"][hour_key] = \
                profile["time_activity"]["by_hour"].get(hour_key, 0) + 1
            profile["time_activity"]["by_month"][month_key] = \
                profile["time_activity"]["by_month"].get(month_key, 0) + 1
        except (ValueError, TypeError):
            pass
        
        # 更新時間
        profile["updated_at"] = datetime.now().isoformat()
        
        # 保存
        self._save_profile()
        
        return profile
    
    def _categorize_age(self, age: int) -> str:
        """將年齡分組"""
        if age < 12:
            return "child"
        elif age < 18:
            return "teen"
        elif age < 35:
            return "young_adult"
        elif age < 60:
            return "middle_aged"
        else:
            return "senior"
    
    def _save_profile(self):
        """保存畫像到文件"""
        # 將 Counter 轉換為普通 dict 以便 JSON 序列化
        profile_serializable = self._make_serializable(self._profile)
        
        temp_path = self.profile_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(profile_serializable, f, ensure_ascii=False, indent=2)
        
        # 原子寫入
        temp_path.replace(self.profile_path)
    
    def _make_serializable(self, obj: Any) -> Any:
        """確保結構可 JSON 序列化（兼容舊版 Counter 數據）"""
        if isinstance(obj, Counter):
            return dict(obj.most_common())
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        return obj
    
    # ─── 查詢接口 ──────────────────────────────────
    
    def get_summary(self) -> Dict[str, Any]:
        """
        獲取畫像摘要（用於顯示）。
        
        Returns:
            dict: 清理後的畫像統計信息
        """
        profile = self._make_serializable(self._profile)
        
        # 按計數降序排列
        top_tags = dict(sorted(
            profile["scene_stats"]["top_tags"].items(),
            key=lambda x: x[1], reverse=True
        )[:10])
        mood_distribution = dict(sorted(
            profile["face_stats"]["emotion_distribution"].items(),
            key=lambda x: x[1], reverse=True
        )[:5])
        
        summary = {
            "user_id": profile["user_id"],
            "stats": profile["stats"],
            "top_tags": top_tags,
            "mood_distribution": mood_distribution,
            "gender_ratio": profile["face_stats"]["gender_distribution"],
            "age_distribution": profile["face_stats"]["age_groups"],
            "total_episodes": profile["stats"]["total_analyses"],
        }
        
        return summary
    
    def get_top_tags(self, n: int = 10) -> List[str]:
        """獲取最常見的場景標籤"""
        tags = self._profile["scene_stats"]["top_tags"]
        sorted_tags = sorted(tags.items(), key=lambda x: x[1], reverse=True)
        return [tag for tag, _ in sorted_tags[:n]]
    
    def get_emotion_trend(self) -> Dict[str, int]:
        """獲取情緒分布趨勢"""
        emotions = self._profile["face_stats"]["emotion_distribution"]
        return dict(sorted(emotions.items(), key=lambda x: x[1], reverse=True))
    
    def get_active_hours(self) -> List[tuple]:
        """獲取活躍時段排名"""
        hours = self._profile["time_activity"]["by_hour"]
        return sorted(hours.items(), key=lambda x: x[1], reverse=True)
    
    def reset(self):
        """重置畫像（危險操作）"""
        self._profile = self._default_profile()
        self._save_profile()
        self.logger.warning(f"用戶 {self.user_id} 的畫像已重置")