"""
SuperSight V3.0 - 主入口
基於 Gradio 的 Web 界面，提供圖片分析、記憶檢索、畫像查看功能。
2026 世代適配：Blackwell FP4 + Qwen3-VL + bge-m4
"""
import os
import sys
import logging
from pathlib import Path
from typing import List, Optional

# 確保專案根目錄在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr

from config.settings import settings
from agents.master_agent import SuperSightMasterAgent
from memory.memory_module import MemoryModule
from memory.profile_manager import ProfileManager
from utils.security import validate_file, setup_logging, log_access


# ─── 日誌初始化 ──────────────────────────────────

logger = setup_logging()


# ─── 全局實例 ────────────────────────────────────

_agent: Optional[SuperSightMasterAgent] = None
_memory: Optional[MemoryModule] = None
_profile: Optional[ProfileManager] = None


def get_agent(user_id: str = "default_user") -> SuperSightMasterAgent:
    """獲取或創建主智能體實例（延遲加載）"""
    global _agent
    if _agent is None:
        _agent = SuperSightMasterAgent(user_id=user_id)
        logger.info("主智能體初始化完成")
    return _agent


def get_memory(user_id: str = "default_user") -> MemoryModule:
    """獲取或創建記憶模組實例"""
    global _memory
    if _memory is None:
        _memory = MemoryModule(user_id=user_id)
    return _memory


def get_profile(user_id: str = "default_user") -> ProfileManager:
    """獲取或創建畫像管理器實例"""
    global _profile
    if _profile is None:
        _profile = ProfileManager(user_id=user_id)
    return _profile


# ─── 核心功能函數 ──────────────────────────────────

def analyze_images(file_objects: List, query: str = "", 
                   user_id: str = "default_user") -> str:
    """
    分析上傳的圖片。
    
    Args:
        file_objects: Gradio 文件對象列表
        query: 用戶查詢
        user_id: 用戶 ID
    
    Returns:
        str: 分析報告
    """
    if not file_objects:
        log_access(logger, "analyze", "未上傳文件", status="FAIL")
        return "⚠️ 請先上傳圖片"
    
    # 檢查批量大小
    if len(file_objects) > settings.MAX_BATCH_SIZE:
        return f"⚠️ 批量上傳數量超過限制（最多 {settings.MAX_BATCH_SIZE} 張）"
    
    agent = get_agent(user_id)
    results = []
    
    for i, file_obj in enumerate(file_objects):
        try:
            # 文件校驗
            file_path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)
            
            is_valid, error_msg = validate_file(file_path)
            if not is_valid:
                results.append(f"❌ **{file_obj.name}**: {error_msg}")
                continue
            
            # 執行分析
            logger.info(f"開始分析 [{i+1}/{len(file_objects)}]: {file_obj.name}")
            report = agent.process(image_path=file_path, query=query)
            
            results.append(f"## 📷 {file_obj.name}\n\n{report}\n\n---\n")
            
            # 更新畫像
            profile = get_profile(user_id)
            if agent.last_aggregated_data:
                profile.update(agent.last_aggregated_data)
            
            log_access(logger, "analyze", f"{file_obj.name} 分析完成", user=user_id)
            
        except Exception as e:
            error_msg = f"❌ **{file_obj.name}**: 分析失敗 - {str(e)}"
            results.append(error_msg)
            logger.error(f"分析失敗: {file_obj.name} - {e}")
            log_access(logger, "analyze", f"{file_obj.name} 失敗: {e}", 
                      user=user_id, status="FAIL")
    
    final_report = "\n".join(results)
    
    if not final_report:
        final_report = "未產生分析結果，請檢查圖片是否有效。"
    
    return final_report


def search_memories(query: str, top_k: int = 5, 
                    user_id: str = "default_user") -> str:
    """
    檢索歷史記憶。
    
    Args:
        query: 檢索查詢
        top_k: 返回結果數量
    
    Returns:
        str: 格式化後的檢索結果
    """
    if not query.strip():
        return "請輸入檢索關鍵詞"
    
    memory = get_memory(user_id)
    results = memory.search(query=query, top_k=top_k)
    
    if not results:
        return f"未找到與「{query}」相關的記憶"
    
    output = [f"🔍 **檢索結果**（共 {len(results)} 條）\n"]
    
    for i, r in enumerate(results):
        content = r.get("content", "")
        meta = r.get("metadata", {})
        distance = r.get("distance", 0)
        score = round((1 - distance) * 100, 1)  # 轉換為相似度分數
        
        output.append(f"### {i+1}. [相關度: {score}%]")
        output.append(f"   _{content}_")
        
        # 顯示中繼資料
        ts = meta.get("timestamp", "")
        tags = meta.get("tags", "")
        if ts or tags:
            meta_str = []
            if ts:
                meta_str.append(f"時間: {ts}")
            if tags:
                meta_str.append(f"標籤: {tags}")
            output.append(f"   📋 {' | '.join(meta_str)}")
        
        output.append("")  # 空行分隔
    
    log_access(logger, "search", f"查詢: '{query}', 返回 {len(results)} 條", 
              user=user_id)
    
    return "\n".join(output)


def view_profile(user_id: str = "default_user") -> str:
    """
    查看用戶畫像統計。
    
    Returns:
        str: 格式化後的畫像信息
    """
    profile = get_profile(user_id)
    summary = profile.get_summary()
    
    output = [
        f"## 👤 用戶畫像: {user_id}\n",
        f"**統計總覽**",
        f"- 總分析次數: {summary['stats']['total_analyses']}",
        f"- 含人臉圖片: {summary['stats']['total_images_with_faces']}",
        f"- 無人臉圖片: {summary['stats']['total_images_without_faces']}",
        f"- 總人臉數: {summary['stats']['total_faces_detected']}\n",
    ]
    
    # 性別比例
    if summary.get("gender_ratio"):
        output.append("**性別比例**")
        for gender, count in summary["gender_ratio"].items():
            output.append(f"- {gender}: {count}")
        output.append("")
    
    # 年齡分布
    if summary.get("age_distribution"):
        output.append("**年齡分布**")
        age_labels = {
            "child": "兒童", "teen": "青少年", 
            "young_adult": "青年", "middle_aged": "中年", "senior": "老年"
        }
        for age_group, count in summary["age_distribution"].items():
            label = age_labels.get(age_group, age_group)
            output.append(f"- {label}: {count}")
        output.append("")
    
    # 情緒分布
    if summary.get("mood_distribution"):
        output.append("**情緒分布**")
        emotion_icons = {
            "happy": "😄", "sad": "😢", "angry": "😠",
            "surprise": "😮", "fear": "😨", "disgust": "🤢",
            "neutral": "😐"
        }
        for emotion, count in summary["mood_distribution"].items():
            icon = emotion_icons.get(emotion.lower(), "")
            output.append(f"- {icon} {emotion}: {count}")
        output.append("")
    
    # 熱門標籤
    if summary.get("top_tags"):
        output.append("**熱門標籤**")
        for tag, count in summary["top_tags"].items():
            output.append(f"- #{tag}: {count}次")
    
    log_access(logger, "view_profile", "查看畫像", user=user_id)
    
    return "\n".join(output)


# ─── Web 界面構建 ──────────────────────────────────

def create_web_interface():
    """創建 Gradio Web 界面"""
    
    with gr.Blocks(
        title=f"🔍 SuperSight {settings.VERSION} - Local AI Memory",
        theme=gr.themes.Soft()
    ) as demo:
        
        gr.Markdown(
            f"# 🔍 SuperSight {settings.VERSION} - 本地 AI 記憶體代理\n"
            f"*100% 本地推理，保護您的隱私*"
        )
        
        if settings.is_password_default:
            gr.Markdown(
                f"⚠️ **檢測到未設置強密碼。**\n"
                f"本次會話臨時密碼: **`{settings.SUPERSIGHT_PASSWORD}`**\n"
                "請在環境變量中設置 `SUPERSIGHT_PASSWORD` 以固化密碼。"
            )
        
        with gr.Tabs():
            # ─── 分析標籤頁 ────────────────────────
            with gr.TabItem("📸 圖片分析"):
                with gr.Row():
                    with gr.Column(scale=1):
                        images_input = gr.File(
                            label="上傳圖片",
                            file_count="multiple",
                            file_types=["image"]
                        )
                        query_input = gr.Textbox(
                            label="指令（可選）",
                            placeholder="例如：這張照片裡的人心情如何？",
                            lines=2
                        )
                        analyze_btn = gr.Button(
                            "🚀 開始分析", 
                            variant="primary",
                            size="lg"
                        )
                        
                        with gr.Accordion("檢索相關記憶", open=False):
                            memory_query = gr.Textbox(
                                label="記憶檢索",
                                placeholder="輸入關鍵詞查找相關記憶...",
                                lines=1
                            )
                            search_btn = gr.Button("🔍 檢索", variant="secondary")
                    
                    with gr.Column(scale=2):
                        output = gr.Textbox(
                            label="分析報告",
                            lines=25,
                            show_copy_button=True
                        )
                
                analyze_btn.click(
                    fn=analyze_images,
                    inputs=[images_input, query_input],
                    outputs=output
                )
                
                search_btn.click(
                    fn=lambda q: search_memories(q),
                    inputs=[memory_query],
                    outputs=output
                )
            
            # ─── 記憶檢索標籤頁 ────────────────────
            with gr.TabItem("📚 記憶庫"):
                with gr.Row():
                    search_input = gr.Textbox(
                        label="自然語言檢索",
                        placeholder="例如：上個月在海邊的照片",
                        lines=2,
                        scale=3
                    )
                    top_k_slider = gr.Slider(
                        minimum=1, maximum=20, value=5, step=1,
                        label="返回結果數",
                        scale=1
                    )
                
                search_memory_btn = gr.Button(
                    "🔍 檢索記憶", variant="primary", size="lg"
                )
                
                memory_output = gr.Textbox(
                    label="檢索結果",
                    lines=20,
                    show_copy_button=True
                )
                
                search_memory_btn.click(
                    fn=lambda q, k: search_memories(query=q, top_k=k),
                    inputs=[search_input, top_k_slider],
                    outputs=memory_output
                )
            
            # ─── 用戶畫像標籤頁 ────────────────────
            with gr.TabItem("👤 用戶畫像"):
                refresh_btn = gr.Button(
                    "🔄 刷新畫像", variant="primary", size="lg"
                )
                profile_output = gr.Textbox(
                    label="畫像統計",
                    lines=25,
                    show_copy_button=True
                )
                
                refresh_btn.click(
                    fn=view_profile,
                    inputs=[],
                    outputs=profile_output
                )
        
        # 頁腳
        gr.Markdown(
            f"---\n"
            f"**SuperSight {settings.VERSION}** | "
            f"Powered by LangGraph + {settings.VLM_MODEL_NAME} + ChromaDB 1.2.x | "
            f"FP4: {'✅' if settings.FP4_QUANTIZATION else '❌'} | "
            f"綁定地址: `{settings.SERVER_HOST}:{settings.SERVER_PORT}`"
        )
    
    return demo


# ─── 啟動入口 ────────────────────────────────────

def main():
    """啟動 SuperSight 服務"""
    
    # 檢查依賴
    logger.info(f"SuperSight {settings.VERSION} 啟動中...")
    logger.info(f"服務地址: http://{settings.SERVER_HOST}:{settings.SERVER_PORT}")
    
    if settings.is_password_default:
        logger.warning(
            f"⚠️ 未設置 SUPERSIGHT_PASSWORD 環境變量！\n"
            f"本次會話密碼: {settings.SUPERSIGHT_PASSWORD}"
        )
    
    # 創建界面
    demo = create_web_interface()
    
    # 啟動服務（強制本地綁定 + 身份驗證）
    demo.launch(
        server_name=settings.SERVER_HOST,
        server_port=settings.SERVER_PORT,
        auth=settings.AUTH_CREDENTIALS,
        share=settings.SHARE_MODE,
        show_error=True,
        quiet=False,
    )


if __name__ == "__main__":
    main()