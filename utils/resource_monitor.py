"""
SuperSight V2.1 - 資源監控模組
提供顯存監控與 OOM 降級策略，防止顯存溢出崩潰。
"""
import logging
import gc
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

import torch

from config.settings import settings


class ResourceLevel(Enum):
    """資源健康狀態等級"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class ResourceStatus:
    """當前資源狀態快照"""
    vram_total_gb: float = 0.0
    vram_used_gb: float = 0.0
    vram_free_gb: float = 0.0
    level: ResourceLevel = ResourceLevel.HEALTHY
    message: str = ""
    can_use_vlm: bool = True
    can_use_face: bool = True
    suggestions: list = field(default_factory=list)


class ResourceMonitor:
    """
    資源監控器，負責：
    1. 查詢 GPU 顯存狀態
    2. 判斷是否接近 OOM 閾值
    3. 提供降級建議
    """
    
    def __init__(self):
        self.logger = logging.getLogger("SuperSight.ResourceMonitor")
        self._has_gpu = torch.cuda.is_available()
        
        if self._has_gpu:
            self.device_count = torch.cuda.device_count()
            self.device_id = settings.GPU_DEVICE_ID
            if self.device_id >= self.device_count:
                self.device_id = 0
                self.logger.warning(f"GPU 設備索引 {settings.GPU_DEVICE_ID} 無效，回退到 0")
            
            device_name = torch.cuda.get_device_name(self.device_id)
            self.logger.info(f"檢測到 GPU: {device_name} (ID: {self.device_id})")
        else:
            self.logger.warning("未檢測到 CUDA GPU，將使用 CPU 模式（速度會顯著降低）")
    
    def get_vram_info(self) -> Dict[str, float]:
        """
        獲取當前顯存使用情況。
        
        Returns:
            dict: {total_gb, used_gb, free_gb} 若無 GPU 則返回 0
        """
        if not self._has_gpu:
            return {"total_gb": 0, "used_gb": 0, "free_gb": 0}
        
        try:
            # torch.cuda.memory_stats 提供更詳細的信息
            total = torch.cuda.get_device_properties(self.device_id).total_memory
            reserved = torch.cuda.memory_reserved(self.device_id)
            allocated = torch.cuda.memory_allocated(self.device_id)
            
            # 實際使用量 = allocated (已分配) + 部分 reserved (預留但未使用)
            # free = total - allocated 更接近真實可用量
            total_gb = total / (1024 ** 3)
            used_gb = allocated / (1024 ** 3)
            free_gb = total_gb - used_gb
            
            return {
                "total_gb": round(total_gb, 2),
                "used_gb": round(used_gb, 2),
                "free_gb": round(free_gb, 2),
            }
        except Exception as e:
            self.logger.error(f"獲取顯存信息失敗: {e}")
            return {"total_gb": 0, "used_gb": 0, "free_gb": 0}
    
    def check_resource_status(self, 
                              need_vlm: bool = False,
                              need_face: bool = False) -> ResourceStatus:
        """
        檢查資源狀態並提供降級建議。
        
        Args:
            need_vlm: 是否將加載 VLM 模型
            need_face: 是否將加載人臉分析模型
        
        Returns:
            ResourceStatus: 包含健康狀態、是否可加載各模型等
        """
        status = ResourceStatus()
        
        if not self._has_gpu:
            status.level = ResourceLevel.WARNING
            status.message = "未檢測到 GPU，使用 CPU 模式"
            status.can_use_vlm = True  # CPU 也可運行，只是慢
            status.can_use_face = True
            status.suggestions.append("使用 CPU 模式，處理速度會顯著降低")
            return status
        
        vram = self.get_vram_info()
        status.vram_total_gb = vram["total_gb"]
        status.vram_used_gb = vram["used_gb"]
        status.vram_free_gb = vram["free_gb"]
        
        # 預估即將佔用的顯存
        estimated_need = 0.0
        if need_vlm:
            estimated_need += settings.VLM_VRAM_ESTIMATE_GB
        if need_face:
            estimated_need += settings.FACE_VRAM_ESTIMATE_GB
        
        remaining_after_load = vram["free_gb"] - estimated_need
        
        # 判斷健康狀態
        if remaining_after_load < settings.VRAM_CRITICAL_THRESHOLD_GB:
            status.level = ResourceLevel.CRITICAL
            status.message = (
                f"顯存不足！剩餘 {vram['free_gb']:.1f}GB，"
                f"加載模型需約 {estimated_need:.1f}GB"
            )
            
            # 降級策略
            if need_vlm and need_face:
                # 優先保留 VLM（核心功能），關閉人臉分析
                status.can_use_vlm = True
                status.can_use_face = False
                status.suggestions.append("關閉人臉分析以節省顯存")
                status.suggestions.append("嘗試清空 CUDA 緩存: torch.cuda.empty_cache()")
            
            # 檢查是否需要強制 GC
            _force_garbage_collect()
            
        elif remaining_after_load < settings.VRAM_WARNING_THRESHOLD_GB:
            status.level = ResourceLevel.WARNING
            status.message = (
                f"顯存緊張：剩餘 {vram['free_gb']:.1f}GB，"
                f"加載後預計剩餘 {remaining_after_load:.1f}GB"
            )
            status.can_use_vlm = need_vlm
            status.can_use_face = need_face
            status.suggestions.append("考慮降低批量大小")
            status.suggestions.append("使用 Q4_K_M 或更小的量化模型")
        else:
            status.level = ResourceLevel.HEALTHY
            status.message = f"資源充足：剩餘 {vram['free_gb']:.1f}GB"
            status.can_use_vlm = need_vlm
            status.can_use_face = need_face
        
        return status
    
    def clear_gpu_memory(self):
        """清理 GPU 緩存，在模型卸載後調用"""
        if self._has_gpu:
            torch.cuda.empty_cache()
            gc.collect()
            self.logger.info("GPU 緩存已清理")


def _force_garbage_collect():
    """強制垃圾回收，嘗試釋放記憶體"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def optimize_batch_size(requested_size: int, status: ResourceStatus) -> int:
    """
    根據資源狀態優化批量大小。
    
    Args:
        requested_size: 請求的批量大小
        status: 當前資源狀態
    
    Returns:
        優化後的批量大小
    """
    if status.level == ResourceLevel.CRITICAL:
        return 1  # 串行處理
    elif status.level == ResourceLevel.WARNING:
        return max(1, requested_size // 2)
    return min(requested_size, settings.MAX_BATCH_SIZE)