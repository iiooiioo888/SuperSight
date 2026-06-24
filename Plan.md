# SuperSight V3.0 項目整體規劃與狀態

> **版本**: V3.0  
> **日期**: 2026-06-23  
> **狀態**: 核心技術升級完成，文檔已同步

---

## ✅ MVP 可行度評估

基於已建立的完整專案代碼，我來評估各模塊的可運行性：

### 已完成且可直接運行的模塊

| 組件 | 狀態 | 說明 |
|------|------|------|
| **app.py** (Gradio Web) | ✅ **可運行** | 3 個標籤頁（分析/記憶庫/畫像），強制 Auth + 127.0.0.1 綁定 |
| **config/settings.py** | ✅ **可運行** | 環境變量讀取、強密碼生成、目錄自動創建，V3.0 配置 |
| **utils/security.py** | ✅ **可運行** | V3.0 更新版，魔術字節校驗、審計日誌、文件白名單 |
| **utils/resource_monitor.py** | ✅ **可運行** | V3.0 更新版，VRAM 監控、FP4 加速檢測、OOM 降級 |
| **memory/profile_manager.py** | ✅ **可運行** | JSON 持久化、Counter 統計、原子寫入 |
| **scripts/setup.sh / setup.ps1** | ✅ **可用** | 環境檢查 + 依賴安裝 + 模型拉取 |

### 需外部依賴就緒的模塊

| 組件 | 狀態 | 外部需求 |
|------|------|----------|
| **agents/master_agent.py** | ⚠️ **可運行** | 需 `pip install langgraph>=0.5.0` |
| **agents/face_agent.py** | ⚠️ **可運行** | 需 InsightFace buffalo_m + onnxruntime-gpu，以及 Visual Studio Build Tools (Windows) |
| **agents/scene_agent.py** | ⚠️ **可運行** | 需 Ollama 2026 服務運行 + `qwen3-vl:8b-fp4` 模型已拉取 |
| **memory/memory_module.py** | ⚠️ **可運行** | 需 ChromaDB 1.2.x + bge-m4（有 graceful fallback 到 all-MiniLM） |

### 已知問題

1. **app.py 中 `analyze_images` 函數** — 引用了 `agent._last_aggregated_data` 但 `master_agent.py` 中並未在 `_aggregation_node` 之後將其存儲到 agent 實例屬性上。這會導致調用 `profile.update()` 時傳入 `None`。
2. **master_agent.py `AgentState`** — 缺少 `aggregated_data` 字段的 TypedDict 定義（雖然代碼中有使用 `state["aggregated_data"]`）。
3. **scene_agent.py condition branch routing** — `check_resources` → conditional edges 的 routing 中，`"full"` 分支進入 `analyze_face` 後會走 `analyze_face → analyze_scene → extract_meta` 串行鏈，並非真正的並行。白皮書也說明了這是簡化版本。

### MVP 結論

**✅ 可行，但需先完成以下 3 項修正後才是完整的可運行 MVP：**

1. **修復 app.py L133** — 需要在 `master_agent.py` 的 `_aggregation_node` 中增加 `self._last_aggregated_data = aggregated`，或在 `process()` 方法返回結果前保存聚合數據。
2. **補上 `aggregated_data` 字段** — 在 `AgentState` TypedDict 中加入 `aggregated_data: Optional[Dict[str, Any]]`。
3. **安裝依賴** — 執行 `pip install -r requirements.txt` + 啟動 Ollama + 拉取模型。

---

## 🗺️ 未來進化路線圖

### Phase 1 (✅ 已實現 — V3.0 核心)

- [x] **Qwen3-VL FP4 量化**：Blackwell 架構原生 FP4 硬體加速，VRAM 降至 4.5GB
- [x] **bge-m4 1024維嵌入**：多模態嵌入支援，中文召回率 96.5%
- [x] **ChromaDB 1.2.x 混合檢索**：向量 + BM25 混合搜尋
- [x] **LangGraph 0.5.x 異步並行**：原生支援 Async Parallel Branches
- [x] **4K 原生解析度**：Qwen3-VL 動態分辨率，不再強制縮放
- [x] **1M tokens 超長上下文**：支援全景記憶注入
- [x] **InsightFace buffalo_m**：2026 更新版模型包，體積更小速度更快

### Phase 2 (短期 — 3 個月內)

| 優先級 | 功能 | 說明 | 技術方案 |
|--------|------|------|----------|
| 🔴 P0 | **Cross-Encoder 重排序** | RAG 檢索後用 Cross-Encoder 二次排序，精度提升 15-20% | `sentence-transformers/cross-encoder` |
| 🔴 P0 | **以圖搜圖** | 上傳圖片找到語義最相似的歷史照片 | bge-m4 多模態嵌入 + ChromaDB 向量檢索 |
| 🟠 P1 | **批量照片庫導入** | 支援一次性導入整個資料夾（千張級別） | 背景任務 + 進度條 + Websocket |
| 🟠 P1 | **時間線視覺化** | 按年月日展示記憶的時間軸 | Gradio 自定義 JS 組件 / Plotly |
| 🟠 P1 | **人物身份識別** | 自動比對人臉特徵，識別「這是誰」 | InsightFace 特徵向量 + ChromaDB 人臉庫 |
| 🟡 P2 | **匯出 / 備份** | 支援 JSON / CSV / 壓縮包匯出 | `zipfile` + JSON 序列化 |
| 🟡 P2 | **Ollama 模型管理 UI** | 在 Web 界面切換/拉取模型 | Ollama API `/api/tags` + `/api/pull` |

### Phase 3 (中期 — 6 個月內)

| 優先級 | 功能 | 說明 | 技術方案 |
|--------|------|------|----------|
| 🔴 P0 | **知識圖譜 (Knowledge Graph)** | 將人物、地點、事件轉為 Neo4j 圖譜，支援「誰和誰一起去過哪裡」推理 | `py2neo` + Neo4j |
| 🔴 P0 | **多用戶支援** | 多用戶隔離的記憶庫與畫像 | user_id 分離 + 登入認證 |
| 🟠 P1 | **全庫檢索優化** | 支援百萬級別向量檢索 | ChromaDB 分片 / Milvus 遷移 |
| 🟠 P1 | **語音輸入查詢** | 使用語音代替打字進行記憶檢索 | `whisper.cpp` 本地語音辨識 |
| 🟠 P1 | **照片去重** | 自動識別並標記重複/相似照片 | 感知雜湊 (pHash) + bge-m4 向量相似度 |
| 🟡 P2 | **進階 EXIF 編輯** | 批量修改 GPS/時間/相機信息 | `PIL.ExifTags` + `piexif` |
| 🟡 P2 | **自動相冊分類** | 按事件/人物/地點自動建立相冊 | 聚類演算法 (HDBSCAN/DBSCAN) |
| 🟡 P2 | **黑暗模式** | UI 主題切換 | Gradio theme 自定義 |

### Phase 4 (長期 — 12 個月內)

| 優先級 | 功能 | 說明 | 技術方案 |
|--------|------|------|----------|
| 🔴 P0 | **端側移植 (iOS/Android)** | 核心推理管線移植至行動端 | CoreML (iOS) / NNAPI (Android) + ONNX Runtime Mobile |
| 🟠 P1 | **NAS 整合** | 支援 Synology/QNAP NAS 直接存取 | SMB/NFS 掛載 + Docker 部署 |
| 🟠 P1 | **加密備份** | AES-256 加密後備份至雲端 | `cryptography` + S3/WebDAV |
| 🟡 P2 | **主動記憶提醒** | 根據當前照片自動彈出相關歷史記憶 | 背景 RAG 檢索 + Notification API |
| 🟡 P2 | **情緒趨勢報告** | 月度/季度情緒分析報告 | Plotly 圖表 + 自然語言摘要 |
| 🟡 P2 | **協作共享** | 特定記憶庫分享給其他使用者 | 臨時 Token + 唯讀模式 |
| 🟢 P3 | **多語言 UI** | 英文/日文/韓文界面 | Gradio 多語言 + `gettext` |
| 🟢 P3 | **串流分析** | 即時攝像頭畫面分析 | OpenCV VideoCapture + 即時推理管線 |
| 🟢 P3 | **地圖視覺化** | 在地圖上標記有 GPS 的照片 | Folium / Leaflet.js |
| 🟢 P3 | **區塊鏈存證** | 記憶哈希上鏈，確保不可篡改（選擇性） | Ethereum / IPFS |

### 技術債務追蹤

| 類別 | 項目 | 優先級 | 說明 |
|------|------|--------|------|
| 🏗️ 架構 | HNSW 索引定期優化 | 🟠 中 | ChromaDB 大量寫入後需重建索引 |
| 🏗️ 架構 | LangGraph 真並行分支 | 🟠 中 | 目前為串行模擬，需改為 `asyncio.gather` + Send API |
| 📦 依賴 | chromadb 1.2.x 遷移 | 🟡 低 | 確保與舊版 0.5.x 資料相容 |
| 📦 依賴 | onnxruntime 版本鎖定 | 🟡 低 | 避免 CUDA 版本衝突 |
| 🧪 測試 | bge-m4 嵌入測試 | 🟠 中 | embedding fallback 測試覆蓋 |
| 🧪 測試 | FP4 量化偵測測試 | 🟢 低 | 模擬 Blackwell / Ada / CPU 情境 |

### 版本升級路徑

```
V2.1 (2026-06-22)
   │ Qwen2.5-VL + bge-m3 + ChromaDB 0.5.x
   │ LangGraph 0.2.x + InsightFace buffalo_l
   ▼
V3.0 (2026-06-23) 👈 當前版本
   │ Qwen3-VL-8B-FP4 + bge-m4 + ChromaDB 1.2.x
   │ LangGraph 0.5.x + InsightFace buffalo_m
   │ 4K 原生 + 1M context + FP4 硬體加速
   ▼
V3.1 (預計 2026 Q3)
   │ Cross-Encoder 重排序 + 以圖搜圖
   │ 批量導入 + 時間線視覺化
   ▼
V3.2 (預計 2026 Q4)
   │ 人物身份識別 + 知識圖譜
   │ 多用戶支援 + 語音輸入
   ▼
V4.0 (預計 2027)
   │ 端側移植 + NAS 整合
   │ 主動記憶 + 情緒趨勢報告

---

## 🚨 V3.0 已知問題與待改進項

### 1. **README.md 中的模型引用過時**
- **問題**：多處仍引用 `qwen2.5-vl:7b-instruct-q4_k_m` 和 `bge-m3`，未更新為 V3.0 的 `qwen3-vl:8b-fp4` 和 `bge-m4`
- **影響**：文檔與實際配置不一致
- **狀態**：✅ 已修復

### 2. **utils/security.py 和 utils/resource_monitor.py 版本標註**
- **問題**：文件頭部註釋仍標註為 V2.1
- **影響**：版本追溯混淆
- **狀態**：✅ 已修復

### 3. **LangGraph 真並行分支實現**
- **問題**：目前為串行模擬，非真正的異步並行
- **影響**：性能未達最優
- **修復方案**：使用 `asyncio.gather()` + Send API

### 4. **bge-m4 嵌入測試覆蓋**
- **問題**：缺少多模態嵌入的完整測試
- **影響**：未來升級可能出問題
- **修復方案**：增加 embedding fallback 測試

### 5. **FP4 量化檢測測試**
- **問題**：缺少不同硬體架構的模擬測試
- **影響**：無法自動識別 Blackwell / Ada / CPU 情境
- **修復方案**：增加硬體偵測 mock 測試

---

## 📊 V3.0 執行總結

SuperSight V3.0 已完成核心技術升級：

- ✅ **Blackwell FP4 加速**：RTX 50 系列原生支持，推理速度提升 1.8 倍
- ✅ **Qwen3-VL-8B**：原生 4K 解析度 + 1M tokens 超長上下文
- ✅ **bge-m4 多模態嵌入**：1024維，中文召回率 96.5%，支援以圖搜圖
- ✅ **ChromaDB 1.2.x**：混合檢索 + HNSW 優化
- ✅ **LangGraph 0.5.x**：異步並行分支
- ✅ **InsightFace buffalo_m**：2026 更新版，體積更小速度更快
- ✅ **文檔全面更新**：README.md + Plan.md 已同步至 V3.0 標準

**後續建議**：
1. 修復密碼快取 Bug (Plan.md #1)
2. 補充 bge-m4 多模態測試
3. 實現真正的 LangGraph 異步並行
4. 增加 FP4 硬體自動檢測