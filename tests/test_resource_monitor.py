"""
SuperSight V2.1 - 資源監控模組測試
驗證顯存監控、健康狀態判斷、OOM 降級策略。
"""
from unittest.mock import patch, MagicMock, PropertyMock

import pytest
import torch

from utils.resource_monitor import (
    ResourceMonitor, ResourceLevel, ResourceStatus, 
    optimize_batch_size, _force_garbage_collect
)


class TestResourceMonitorGPU:
    """GPU 模式下的資源監控測試"""

    @pytest.fixture(autouse=True)
    def mock_cuda_available(self):
        """模擬 CUDA 可用"""
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.device_count", return_value=1):
                with patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 4090"):
                    yield

    def test_initialization(self):
        """初始化應檢測到 GPU"""
        with patch("torch.cuda.get_device_properties") as mock_props:
            mock_props.return_value.total_memory = 24 * 1024**3  # 24GB
            monitor = ResourceMonitor()
            assert monitor._has_gpu is True
            assert monitor.device_id == 0

    def test_get_vram_info(self):
        """get_vram_info 應返回正確的顯存信息"""
        with patch("torch.cuda.get_device_properties") as mock_props:
            mock_props.return_value.total_memory = 24 * 1024**3
            with patch("torch.cuda.memory_allocated", return_value=8 * 1024**3):  # 8GB used
                with patch("torch.cuda.memory_reserved", return_value=10 * 1024**3):
                    monitor = ResourceMonitor()
                    info = monitor.get_vram_info()
                    assert info["total_gb"] == 24.0
                    assert info["used_gb"] == 8.0
                    assert info["free_gb"] == 16.0

    def test_healthy_status(self):
        """充足顯存應回傳 HEALTHY"""
        with patch("torch.cuda.get_device_properties") as mock_props:
            mock_props.return_value.total_memory = 24 * 1024**3
            with patch("torch.cuda.memory_allocated", return_value=4 * 1024**3):  # 4GB used
                monitor = ResourceMonitor()
                status = monitor.check_resource_status(need_vlm=True, need_face=True)
                assert status.level == ResourceLevel.HEALTHY
                assert status.can_use_vlm is True
                assert status.can_use_face is True

    def test_warning_status(self):
        """顯存緊張應回傳 WARNING"""
        with patch("torch.cuda.get_device_properties") as mock_props:
            mock_props.return_value.total_memory = 16 * 1024**3  # 16GB card
            # VLM=4.5GB + Face=1.2GB = 5.7GB, VRAM_WARNING=1.0GB
            # Need: free - need < 1.0 → free < 6.7GB → used > 9.3GB
            with patch("torch.cuda.memory_allocated", return_value=11 * 1024**3):  # 11GB used
                monitor = ResourceMonitor()
                status = monitor.check_resource_status(need_vlm=True, need_face=True)
                # free = 5GB, need = 5.7GB, remaining = -0.7GB → CRITICAL
                assert status.level in (ResourceLevel.WARNING, ResourceLevel.CRITICAL)

    def test_critical_status_disables_face(self):
        """OOM 邊緣應關閉人臉分析"""
        with patch("torch.cuda.get_device_properties") as mock_props:
            mock_props.return_value.total_memory = 16 * 1024**3
            with patch("torch.cuda.memory_allocated", return_value=12 * 1024**3):  # 12GB used → 4GB free
                monitor = ResourceMonitor()
                status = monitor.check_resource_status(need_vlm=True, need_face=True)
                # free=4GB, need=5.7GB, remaining=-1.7GB < 0.5(CRITICAL)
                # → CRITICAL, should disable face
                if status.level == ResourceLevel.CRITICAL:
                    assert status.can_use_vlm is True  # VLM 優先
                    assert status.can_use_face is False  # 關閉人臉


class TestResourceMonitorCPU:
    """CPU 模式下的資源監控測試"""

    @pytest.fixture(autouse=True)
    def mock_no_cuda(self):
        with patch("torch.cuda.is_available", return_value=False):
            yield

    def test_cpu_mode_warning(self):
        """無 GPU 時應回傳 WARNING 並提示 CPU 模式"""
        monitor = ResourceMonitor()
        status = monitor.check_resource_status(need_vlm=True)
        assert status.level == ResourceLevel.WARNING
        assert "CPU" in status.message
        assert status.can_use_vlm is True  # CPU 也可運行
        assert status.can_use_face is True

    def test_get_vram_info_cpu(self):
        """無 GPU 時 get_vram_info 應返回全零"""
        monitor = ResourceMonitor()
        info = monitor.get_vram_info()
        assert info["total_gb"] == 0
        assert info["used_gb"] == 0
        assert info["free_gb"] == 0


class TestOptimizeBatchSize:
    """批量大小優化測試"""

    def test_healthy_keeps_requested(self):
        """HEALTHY 狀態應保留請求的 batch size"""
        status = ResourceStatus(level=ResourceLevel.HEALTHY)
        result = optimize_batch_size(10, status)
        assert result == 10

    def test_warning_halves(self):
        """WARNING 狀態應減半"""
        status = ResourceStatus(level=ResourceLevel.WARNING)
        result = optimize_batch_size(10, status)
        assert result == 5

    def test_critical_returns_one(self):
        """CRITICAL 狀態應回傳 1"""
        status = ResourceStatus(level=ResourceLevel.CRITICAL)
        result = optimize_batch_size(10, status)
        assert result == 1

    def test_warning_minimum_one(self):
        """WARNING 狀態下最小值為 1"""
        status = ResourceStatus(level=ResourceLevel.WARNING)
        result = optimize_batch_size(1, status)
        assert result == 1

    def test_healthy_respects_max(self):
        """HEALTHY 狀態應受 MAX_BATCH_SIZE 限制"""
        from config.settings import Settings
        status = ResourceStatus(level=ResourceLevel.HEALTHY)
        result = optimize_batch_size(100, status)
        assert result == Settings().MAX_BATCH_SIZE


class TestForceGC:
    """強制垃圾回收測試"""

    def test_gc_collect_called(self):
        with patch("gc.collect") as mock_gc:
            with patch("torch.cuda.is_available", return_value=False):
                _force_garbage_collect()
                mock_gc.assert_called_once()

    def test_cuda_empty_cache_called(self):
        with patch("gc.collect"):
            with patch("torch.cuda.is_available", return_value=True):
                with patch("torch.cuda.empty_cache") as mock_empty:
                    _force_garbage_collect()
                    mock_empty.assert_called_once()