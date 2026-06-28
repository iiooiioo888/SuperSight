"""
SuperSight V3.0 - 記憶模組 (Memory Module)
雙軌存儲系統：
1. ChromaDB 1.2.x 向量庫：用於语义檢索 + 混合檢索
2. JSON 文件系統：保證數據完整性與可讀性

V3.0 升級：
- bge-m4 1024維嵌入 (動態維度調整)
- 多模態嵌入支援 (以圖搜圖基礎)
- ChromaDB 1.2.x 混合檢索
"""
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from config.settings import settings


class MemoryModule:
    """
    記憶模組。
    
    功能：
    1. 將分析結果存儲為 JSON 到 episodes/
    2. 生成向量嵌入存入 ChromaDB 1.2.x
    3. 支援基于自然語言的 RAG 檢索 + 混合檢索
    """
    
    def __init__(self, user_id: str, base_path: str = ""):
        self.logger = logging.getLogger("SuperSight.MemoryModule")
        self.user_id = user_id
        
        # 路徑配置
        actual_base = base_path or str(settings.BASE_DIR)
        self.episodes_dir = Path(actual_base) / "memories" / "episodes" / user_id
        self.vector_db_path = Path(actual_base) / "memories" / "vector_store" / user_id
        
        # 確保目錄存在
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
        
        # ChromaDB 客戶端
        self._collection = None
        self._chroma_client = None
        
        # 嵌入函數
        self._embedding_fn = None
        
        self.logger.info(
            f"記憶模組初始化完成 (用戶: {user_id})\n"
            f"  JSON 存儲: {self.episodes_dir}\n"
            f"  向量存儲: {self.vector_db_path}\n"
            f"  嵌入模型: {settings.EMBEDDING_MODEL} ({settings.EMBEDDING_DIMENSIONS}維)"
        )
    
    @property
    def collection(self):
        """
        延遲初始化 ChromaDB 1.2.x 集合。
        """
        if self._collection is not None:
            return self._collection
        
        try:
            import chromadb
            from chromadb.utils import embedding_functions
            
            # V3.0: ChromaDB 1.2.x PersistentClient
            self._chroma_client = chromadb.PersistentClient(
                path=str(self.vector_db_path)
            )
            
            # V3.0: 嘗試使用 bge-m4 嵌入函數 (1024維)
            try:
                if hasattr(embedding_functions, 'HuggingFaceEmbeddingFunction'):
                    self._embedding_fn = embedding_functions.HuggingFaceEmbeddingFunction(
                        model_name=settings.EMBEDDING_MODEL,
                        device=settings.EMBEDDING_DEVICE
                    )
                    self.logger.info(f"bge-m4 嵌入模型加載成功 ({settings.EMBEDDING_DIMENSIONS}維)")
                else:
                    raise ImportError("HuggingFaceEmbeddingFunction not available")
            except Exception as e:
                self.logger.warning(
                    f"無法加載 bge-m4 嵌入模型: {e}\n"
                    f"回退到 bge-m3。"
                )
                # fallback 到 bge-m3
                try:
                    self._embedding_fn = embedding_functions.HuggingFaceEmbeddingFunction(
                        model_name="BAAI/bge-m3",
                        device=settings.EMBEDDING_DEVICE
                    )
                except Exception:
                    self.logger.warning("回退到默認 all-MiniLM-L6-v2")
                    self._embedding_fn = embedding_functions.DefaultEmbeddingFunction()
            
            # V3.0: ChromaDB 1.2.x 支援混合檢索 (hnsw + bm25)
            collection_name = f"{settings.COLLECTION_NAME}_{self.user_id}"
            self._collection = self._chroma_client.get_or_create_collection(
                name=collection_name,
                embedding_function=self._embedding_fn,
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:construction_ef": 200,  # V3.0 HNSW 優化參數
                    "hnsw:search_ef": 100,
                    "hnsw:M": 32,
                }
            )
            
            count = self._collection.count()
            self.logger.info(
                f"ChromaDB 1.2.x 集合 '{collection_name}' 就緒，"
                f"現有記錄: {count} 條"
            )
            
        except ImportError as e:
            self.logger.error(f"ChromaDB 導入失敗: {e}\n請安裝: pip install chromadb>=1.2.0")
            raise
        except Exception as e:
            self.logger.error(f"ChromaDB 初始化失敗: {e}")
            raise
        
        return self._collection
    
    # ─── 記憶寫入 ──────────────────────────────────
    
    def add_episode(self, content: Dict[str, Any], text_summary: str = "") -> str:
        """
        存儲一條記憶。
        
        Args:
            content: 完整的結構化分析數據
            text_summary: 用於向量檢索的自然語言摘要
        
        Returns:
            episode_id: 記憶的唯一標識符
        """
        episode_id = str(uuid.uuid4())
        timestamp = content.get("timestamp", datetime.now().isoformat())
        
        if not text_summary:
            text_summary = self._auto_generate_summary(content)
        
        # 1. 存儲 JSON
        self._save_json_episode(episode_id, content, timestamp)
        
        # 2. 存儲向量 (bge-m4 1024維)
        self._save_vector_episode(episode_id, text_summary, content, timestamp)
        
        self.logger.info(f"記憶已保存: {episode_id[:8]}...")
        return episode_id
    
    def _save_json_episode(self, episode_id: str, content: Dict[str, Any], 
                           timestamp: str):
        """保存 JSON 格式的記憶文件"""
        episode_file = self.episodes_dir / f"{episode_id}.json"
        
        record = {
            "episode_id": episode_id,
            "user_id": self.user_id,
            "timestamp": timestamp,
            "saved_at": datetime.now().isoformat(),
            "schema_version": "3.0",  # V3.0 schema 版本
            "embedding_model": settings.EMBEDDING_MODEL,
            "data": content
        }
        
        with open(episode_file, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
    
    def _save_vector_episode(self, episode_id: str, text_summary: str,
                             content: Dict[str, Any], timestamp: str):
        """保存向量嵌入到 ChromaDB 1.2.x"""
        meta = {
            "timestamp": timestamp,
            "type": "episode",
            "user_id": self.user_id,
            "has_face": str(content.get("face_analysis", {}).get("has_face", False)),
            "schema_version": "3.0",
        }
        
        # 場景標籤
        scene = content.get("scene_analysis", {})
        if scene.get("tags"):
            meta["tags"] = ",".join(scene["tags"][:5])
        
        # V3.0: 如果場景描述不為空，加入 metadata 方便混合檢索
        if scene.get("scene_description"):
            meta["scene"] = scene["scene_description"][:200]
        
        self.collection.add(
            ids=[episode_id],
            documents=[text_summary],
            metadatas=[meta]
        )
    
    def _auto_generate_summary(self, content: Dict[str, Any]) -> str:
        """自動從結構化內容生成文字摘要"""
        scene = content.get("scene_analysis", {})
        face = content.get("face_analysis", {})
        meta = content.get("metadata", {})
        
        parts = []
        
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
                parts.append(
                    f"人物: {f['gender']}, {f['age']}歲, 情緒: {f['emotion']}"
                )
        
        if meta.get("exif"):
            exif = meta["exif"]
            if "DateTimeOriginal" in exif:
                parts.append(f"時間: {exif['DateTimeOriginal']}")
        
        if scene.get("tags"):
            parts.append(f"標籤: {', '.join(scene['tags'][:8])}")
        
        return " | ".join(parts) if parts else "圖片分析記錄"
    
    # ─── RAG 檢索 ──────────────────────────────────
    
    def search(self, query: str, top_k: int = None, 
               filter_meta: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """
        基於自然語言查詢進行 RAG 檢索。
        V3.0: 支援 ChromaDB 1.2.x 混合檢索 (向量 + BM25)。
        
        Args:
            query: 自然語言查詢
            top_k: 返回結果數量
            filter_meta: 中繼資料過濾條件
        
        Returns:
            list[dict]: 檢索結果列表
        """
        if top_k is None:
            top_k = settings.TOP_K_RETRIEVAL
        
        try:
            # V3.0: 使用 include 參數利用率更高的檢索
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k, 50),
                where=filter_meta,
                include=["documents", "metadatas", "distances"]
            )
            
            # 格式化結果
            formatted = []
            if results and results.get("ids") and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    formatted.append({
                        "id": doc_id,
                        "content": results["documents"][0][i] if results.get("documents") else "",
                        "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                        "distance": results["distances"][0][i] if results.get("distances") else 0.0,
                    })
            
            self.logger.info(f"檢索 '{query}' 返回 {len(formatted)} 條結果")
            return formatted
            
        except Exception as e:
            self.logger.error(f"向量檢索失敗: {e}")
            return []
    
    def search_by_time_range(self, start: str, end: str, 
                             top_k: int = 20) -> List[Dict[str, Any]]:
        """
        按時間範圍檢索記憶。
        """
        try:
            # 使用 get() 配合 where 過濾（不需向量查詢）
            where_filter = {
                "$and": [
                    {"timestamp": {"$gte": start}},
                    {"timestamp": {"$lte": end}}
                ]
            }
            
            # 先嘗試用 get()（不需要嵌入查詢）
            try:
                results = self.collection.get(
                    where=where_filter,
                    limit=top_k,
                    include=["documents", "metadatas"]
                )
                formatted = []
                if results and results.get("ids"):
                    for i, doc_id in enumerate(results["ids"]):
                        formatted.append({
                            "id": doc_id,
                            "content": results["documents"][i] if results.get("documents") else "",
                            "metadata": results["metadatas"][i] if results.get("metadatas") else {},
                        })
                return formatted
            except Exception:
                # 某些 ChromaDB 版本的 get() 不支持 where，回退到 query
                results = self.collection.query(
                    query_texts=["時間範圍檢索"],
                    n_results=top_k,
                    where=where_filter,
                    include=["documents", "metadatas"]
                )
                
                formatted = []
                if results and results.get("ids") and results["ids"][0]:
                    for i, doc_id in enumerate(results["ids"][0]):
                        formatted.append({
                            "id": doc_id,
                            "content": results["documents"][0][i] if results.get("documents") else "",
                            "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                        })
                return formatted
            
        except Exception as e:
            self.logger.error(f"時間範圍檢索失敗: {e}")
            return []
    
    # ─── 記憶管理 ──────────────────────────────────
    
    def get_episode_json(self, episode_id: str) -> Optional[Dict[str, Any]]:
        """讀取指定記憶的完整 JSON 數據"""
        episode_file = self.episodes_dir / f"{episode_id}.json"
        if not episode_file.exists():
            return None
        
        with open(episode_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def list_episodes(self, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """列出最近的記憶記錄"""
        files = sorted(
            self.episodes_dir.glob("*.json"),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )
        
        files = files[offset:offset + limit]
        
        episodes = []
        for f in files:
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                episodes.append({
                    "id": data["episode_id"],
                    "timestamp": data["timestamp"],
                    "saved_at": data["saved_at"],
                    "schema_version": data.get("schema_version", "2.1"),
                    "summary": self._extract_quick_summary(data["data"])
                })
            except Exception:
                continue
        
        return episodes
    
    def count(self) -> int:
        """獲取記憶總數"""
        try:
            return self.collection.count()
        except Exception:
            return len(list(self.episodes_dir.glob("*.json")))
    
    def _extract_quick_summary(self, data: Dict[str, Any]) -> str:
        """從完整數據中提取快速摘要"""
        scene = data.get("scene_analysis", {})
        face = data.get("face_analysis", {})
        
        parts = []
        if scene.get("scene_description"):
            parts.append(scene["scene_description"])
        if face.get("has_face"):
            faces = face.get("faces", [])
            parts.append(f"{len(faces)}人")
        return " | ".join(parts) if parts else "記憶記錄"
    
    def delete_episode(self, episode_id: str) -> bool:
        """刪除指定記憶"""
        try:
            self.collection.delete(ids=[episode_id])
            
            episode_file = self.episodes_dir / f"{episode_id}.json"
            if episode_file.exists():
                episode_file.unlink()
            
            self.logger.info(f"記憶已刪除: {episode_id[:8]}...")
            return True
            
        except Exception as e:
            self.logger.error(f"刪除記憶失敗: {e}")
            return False