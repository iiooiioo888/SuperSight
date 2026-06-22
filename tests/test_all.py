"""
SuperSight V2.1 - 全面測試執行腳本
運行方式: python -m pytest tests/test_all.py -v
"""
import sys
from pathlib import Path

# 確保專案根目錄在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

if __name__ == "__main__":
    """
    測試執行入口。
    
    使用方法:
        # 運行所有測試
        python tests/test_all.py
        
        # 運行特定模組測試
        python -m pytest tests/test_config.py -v
        
        # 運行帶覆蓋率報告
        pip install pytest-cov
        python -m pytest tests/ --cov=agents --cov=memory --cov=utils --cov=config -v
        
        # 跳過慢速測試
        python -m pytest tests/ -v -k "not integration"
    """
    test_dir = Path(__file__).parent
    
    # 構建測試發現路徑
    test_files = sorted(test_dir.glob("test_*.py"))
    
    print("=" * 60)
    print("  🔍 SuperSight V2.1 - 全面測試套件")
    print("=" * 60)
    print(f"\n📂 測試目錄: {test_dir}")
    print(f"📋 測試文件數: {len(test_files)}")
    
    for f in test_files:
        if f.name != "test_all.py":
            print(f"   - {f.name}")
    
    print("\n🚀 啟動測試...\n")
    
    # pytest 參數
    args = [
        "-v",                              # verbose
        "--tb=short",                      # 簡短回溯
        "--disable-warnings",              # 隱藏警告
        "-p", "no:cacheprovider",          # 不使用緩存
        str(test_dir),                     # 測試目錄
    ]
    
    # 可選：添加 --failed-first 標誌（如果 pytest-failed-first 已安裝）
    try:
        import pytest_failed_first
        args.append("--failed-first")
    except ImportError:
        pass
    
    # 執行測試
    exit_code = pytest.main(args)
    
    print("\n" + "=" * 60)
    if exit_code == 0:
        print("  ✅ 所有測試通過！")
    else:
        print(f"  ❌ 測試失敗，退出碼: {exit_code}")
    print("=" * 60)
    
    sys.exit(exit_code)