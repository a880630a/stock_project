#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FinBERT API 啟動腳本
提供簡化的API服務啟動方式，包含環境檢查和配置
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse

def check_environment():
    """檢查環境配置"""
    print("🔍 檢查環境配置...")
    
    # 檢查Python版本
    if sys.version_info < (3, 8):
        print("❌ Python版本過低，需要Python 3.8+")
        return False
    
    print(f"✅ Python版本: {sys.version}")
    
    # 檢查HF_TOKEN環境變量
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("❌ 未設置 HF_TOKEN 環境變量")
        print("   請參考 ENVIRONMENT_SETUP.md 設置環境變量")
        return False
    
    print("✅ HF_TOKEN 環境變量已設置")
    
    # 檢查必要文件
    required_files = ['app.py', 'finbert_main.py', 'agents.py', 'requirements.txt']
    for file in required_files:
        if not Path(file).exists():
            print(f"❌ 缺少必要文件: {file}")
            return False
    
    print("✅ 所有必要文件存在")
    
    return True

def install_dependencies():
    """安裝依賴包"""
    print("📦 檢查並安裝依賴包...")
    
    try:
        # 檢查pip
        subprocess.run([sys.executable, '-m', 'pip', '--version'], 
                      check=True, capture_output=True)
        
        # 安裝依賴
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True)
        
        print("✅ 依賴包安裝完成")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 依賴包安裝失敗: {e}")
        return False

def start_api_server(host='0.0.0.0', port=5000, debug=True):
    """啟動API服務器"""
    print(f"🚀 啟動FinBERT API服務器...")
    print(f"   地址: http://{host}:{port}")
    print(f"   除錯模式: {'開啟' if debug else '關閉'}")
    print("   按 Ctrl+C 停止服務器")
    print("-" * 50)
    
    try:
        # 設置環境變量
        env = os.environ.copy()
        env['FLASK_APP'] = 'app.py'
        env['FLASK_ENV'] = 'development' if debug else 'production'
        
        # 啟動Flask應用
        from app import app, init_analyzer
        
        # 初始化分析器
        if not init_analyzer():
            print("❌ FinBERT分析器初始化失敗")
            return False
        
        # 啟動服務器
        app.run(host=host, port=port, debug=debug)
        
    except KeyboardInterrupt:
        print("\n👋 API服務器已停止")
        return True
    except Exception as e:
        print(f"❌ API服務器啟動失敗: {e}")
        return False

def run_api_test():
    """運行API測試"""
    print("🧪 運行API測試...")
    
    try:
        # 導入並運行測試
        from api_test import APITester
        
        tester = APITester()
        success = tester.run_all_tests()
        
        if success:
            print("✅ 所有API測試通過")
        else:
            print("❌ 部分API測試失敗")
        
        return success
        
    except ImportError as e:
        print(f"❌ 無法導入測試模組: {e}")
        return False
    except Exception as e:
        print(f"❌ 測試運行失敗: {e}")
        return False

def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description="FinBERT API 啟動工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
    python start_api.py                    # 啟動API服務器
    python start_api.py --check-env        # 僅檢查環境
    python start_api.py --install-deps     # 僅安裝依賴
    python start_api.py --test             # 僅運行測試
    python start_api.py --host 127.0.0.1   # 指定主機地址
    python start_api.py --port 8080        # 指定端口
        """
    )
    
    parser.add_argument('--host', default='0.0.0.0',
                       help='API服務器主機地址 (預設: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000,
                       help='API服務器端口 (預設: 5000)')
    parser.add_argument('--no-debug', action='store_true',
                       help='關閉除錯模式')
    parser.add_argument('--check-env', action='store_true',
                       help='僅檢查環境配置')
    parser.add_argument('--install-deps', action='store_true',
                       help='僅安裝依賴包')
    parser.add_argument('--test', action='store_true',
                       help='僅運行API測試')
    
    args = parser.parse_args()
    
    print("🚀 FinBERT API 啟動工具")
    print("=" * 50)
    
    # 檢查環境
    if not check_environment():
        print("\n❌ 環境檢查失敗，請修復後重試")
        sys.exit(1)
    
    if args.check_env:
        print("\n✅ 環境檢查完成")
        return
    
    # 安裝依賴
    if args.install_deps or not args.test:
        if not install_dependencies():
            print("\n❌ 依賴安裝失敗")
            sys.exit(1)
    
    if args.install_deps:
        print("\n✅ 依賴安裝完成")
        return
    
    # 運行測試
    if args.test:
        success = run_api_test()
        sys.exit(0 if success else 1)
    
    # 啟動API服務器
    print("\n" + "=" * 50)
    success = start_api_server(
        host=args.host,
        port=args.port,
        debug=not args.no_debug
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
