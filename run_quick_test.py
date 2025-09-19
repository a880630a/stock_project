#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速整合測試腳本
驗證專案核心功能是否正常運行
"""

import os
import sys
import importlib.util
from pathlib import Path

def test_imports():
    """測試核心模組導入"""
    print("🔍 測試核心模組導入...")
    
    try:
        # 測試FinBERT分析器導入
        from finbert_main import FinBERTAnalyzer
        print("✅ FinBERTAnalyzer 導入成功")
        
        # 測試交易代理導入
        from agents import LLMAgent
        print("✅ LLMAgent 導入成功")
        
        # 測試Flask應用導入
        from app import app, init_analyzer
        print("✅ Flask應用 導入成功")
        
        return True
        
    except ImportError as e:
        print(f"❌ 模組導入失敗: {e}")
        return False
    except Exception as e:
        print(f"❌ 導入過程發生錯誤: {e}")
        return False

def test_environment():
    """測試環境配置"""
    print("\n🔍 測試環境配置...")
    
    # 檢查Python版本
    if sys.version_info < (3, 8):
        print(f"❌ Python版本過低: {sys.version}")
        return False
    else:
        print(f"✅ Python版本符合要求: {sys.version}")
    
    # 檢查HF_TOKEN
    hf_token = os.getenv('HF_TOKEN')
    if hf_token:
        print("✅ HF_TOKEN 環境變量已設置")
        # 檢查token格式（不顯示實際值）
        if len(hf_token) > 20 and hf_token.startswith('hf_'):
            print("✅ HF_TOKEN 格式看起來正確")
        else:
            print("⚠️  HF_TOKEN 格式可能不正確")
    else:
        print("❌ HF_TOKEN 環境變量未設置")
        return False
    
    # 檢查必要文件
    required_files = [
        'finbert_main.py',
        'agents.py', 
        'app.py',
        'api_test.py',
        'start_api.py',
        'requirements.txt'
    ]
    
    missing_files = []
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file} 存在")
        else:
            print(f"❌ {file} 不存在")
            missing_files.append(file)
    
    return len(missing_files) == 0

def test_finbert_analyzer():
    """測試FinBERT分析器基本功能"""
    print("\n🔍 測試FinBERT分析器...")
    
    try:
        from finbert_main import FinBERTAnalyzer
        
        # 初始化分析器
        analyzer = FinBERTAnalyzer()
        print("✅ FinBERT分析器初始化成功")
        
        # 測試基本分析（使用簡單文本避免API調用）
        # 這裡只測試方法是否存在，不實際調用API
        if hasattr(analyzer, 'analyze_sentiment'):
            print("✅ analyze_sentiment 方法存在")
        else:
            print("❌ analyze_sentiment 方法不存在")
            return False
            
        if hasattr(analyzer, 'generate_trading_strategy'):
            print("✅ generate_trading_strategy 方法存在")
        else:
            print("❌ generate_trading_strategy 方法不存在")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ FinBERT分析器測試失敗: {e}")
        return False

def test_agents():
    """測試交易代理"""
    print("\n🔍 測試交易代理...")
    
    try:
        from agents import LLMAgent
        
        # 初始化代理（不調用API）
        agent = LLMAgent(role="aggressive")
        print("✅ LLMAgent 初始化成功")
        
        # 檢查基本屬性
        if hasattr(agent, 'role') and agent.role == "aggressive":
            print("✅ 代理角色設置正確")
        else:
            print("❌ 代理角色設置錯誤")
            return False
            
        if hasattr(agent, 'predict'):
            print("✅ predict 方法存在")
        else:
            print("❌ predict 方法不存在")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ 交易代理測試失敗: {e}")
        return False

def test_flask_app():
    """測試Flask應用配置"""
    print("\n🔍 測試Flask應用...")
    
    try:
        from app import app
        
        # 檢查應用配置
        if app:
            print("✅ Flask應用創建成功")
        else:
            print("❌ Flask應用創建失敗")
            return False
            
        # 檢查路由
        routes = [rule.rule for rule in app.url_map.iter_rules()]
        expected_routes = [
            '/api/health',
            '/api/analyze', 
            '/api/analyze-score',
            '/api/trading-strategy',
            '/api/batch-analyze'
        ]
        
        for route in expected_routes:
            if route in routes:
                print(f"✅ 路由 {route} 存在")
            else:
                print(f"❌ 路由 {route} 不存在")
                return False
                
        return True
        
    except Exception as e:
        print(f"❌ Flask應用測試失敗: {e}")
        return False

def main():
    """主測試函數"""
    print("🚀 FinBERT 股票分析專案 - 快速整合測試")
    print("=" * 60)
    
    tests = [
        ("環境配置", test_environment),
        ("模組導入", test_imports),
        ("FinBERT分析器", test_finbert_analyzer),
        ("交易代理", test_agents),
        ("Flask應用", test_flask_app)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}測試異常: {e}")
            results.append((test_name, False))
    
    # 總結報告
    print("\n" + "=" * 60)
    print("📊 快速整合測試總結")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"✅ 通過測試: {passed}/{total}")
    
    print("\n詳細結果:")
    for test_name, result in results:
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"   {test_name}: {status}")
    
    if passed == total:
        print("\n🎉 所有測試通過！專案整合成功。")
        print("\n📋 下一步操作:")
        print("   1. 啟動API服務器: python start_api.py")
        print("   2. 運行API測試: python api_test.py")
        print("   3. 查看整合指南: README_INTEGRATION.md")
    else:
        print(f"\n⚠️  有 {total - passed} 個測試失敗。")
        print("請檢查錯誤信息並修復問題。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
