#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FinBERT 增強功能示範腳本
展示新增的情感評分和交易策略功能

使用方式:
    python finbert_enhanced_demo.py
"""

import os
from finbert_main import FinBERTAnalyzer

def demo_sentiment_scoring():
    """示範情感評分功能"""
    print("🚀 FinBERT 情感評分功能示範")
    print("="*60)
    
    # 初始化分析器
    analyzer = FinBERTAnalyzer()
    
    # 測試案例
    test_cases = [
        "Apple stock surges after earnings report.",
        "Tesla shares plummet due to production delays.",
        "The Federal Reserve will announce interest rates next week.",
        "台積電股價創新高，投資人信心大增",
        "受疫情影響，航空股全面重挫"
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n📝 測試案例 {i}:")
        print(f"輸入: {text}")
        
        result = analyzer.analyze_sentiment_with_score(text, f"Demo_{i}")
        formatted_result = analyzer.format_result(result)
        print(formatted_result)
        print("-" * 50)

def demo_trading_strategy():
    """示範交易策略功能"""
    print("\n\n💼 FinBERT 交易策略功能示範")
    print("="*60)
    
    # 初始化分析器
    analyzer = FinBERTAnalyzer()
    
    # 測試案例：[價格, RSI, 情感評分]
    strategy_cases = [
        ("看漲情境", 100.0, 0.3, 0.8),   # 低RSI + 正面情感 → 應該買入
        ("看跌情境", 100.0, 0.8, -0.6),  # 高RSI + 負面情感 → 應該賣出
        ("中性情境", 100.0, 0.5, 0.1),   # 中等RSI + 中性情感 → 應該持有
        ("極度看漲", 100.0, 0.2, 0.9),   # 極低RSI + 極正面情感 → 強烈買入
        ("極度看跌", 100.0, 0.9, -0.8),  # 極高RSI + 極負面情感 → 強烈賣出
    ]
    
    for i, (scenario, price, rsi, sentiment) in enumerate(strategy_cases, 1):
        print(f"\n📈 測試案例 {i}: {scenario}")
        print(f"市場狀態: 價格={price}, RSI={rsi}, 情感={sentiment}")
        
        result = analyzer.generate_trading_strategy(price, rsi, sentiment, f"Strategy_{i}")
        formatted_result = analyzer.format_result(result)
        print(formatted_result)
        print("-" * 50)

def demo_action_parsing():
    """示範動作解析功能"""
    print("\n\n🔍 動作解析功能示範")
    print("="*60)
    
    analyzer = FinBERTAnalyzer()
    
    # 測試不同的策略回應
    test_responses = [
        "I recommend to BUY this stock due to strong fundamentals.",
        "You should SELL immediately as the market is declining.",
        "HOLD your position and wait for better market conditions.",
        "Strong buy signal based on technical analysis.",
        "Sell recommendation due to overvaluation.",
        "Neutral outlook, maintain current holdings."
    ]
    
    print("測試動作解析邏輯:")
    for i, response in enumerate(test_responses, 1):
        action = analyzer.parse_action(response)
        action_label = analyzer._get_action_label(action)
        print(f"{i}. 回應: {response}")
        print(f"   解析結果: {action_label} (代碼: {action})")
        print()

def main():
    """主示範程式"""
    print("🎯 FinBERT 增強功能完整示範")
    print("="*80)
    print("本示範將展示以下新功能:")
    print("1. 📊 情感評分分析 (數值化情感評分 -1 到 1)")
    print("2. 💼 交易策略生成 (基於價格、RSI、情感的AI建議)")
    print("3. 🔍 動作解析 (將文字建議轉換為數值動作)")
    print("="*80)
    
    try:
        # 示範情感評分
        demo_sentiment_scoring()
        
        # 示範交易策略
        demo_trading_strategy()
        
        # 示範動作解析
        demo_action_parsing()
        
        print("\n✅ 所有示範完成！")
        print("\n💡 使用建議:")
        print("1. 情感評分適合用於新聞情感量化分析")
        print("2. 交易策略適合用於自動化交易決策支援")
        print("3. 動作解析適合用於將AI建議轉換為程式化交易信號")
        
    except Exception as e:
        print(f"\n❌ 示範過程中發生錯誤: {e}")
        print("請確保:")
        print("1. 已安裝所有必要的依賴套件")
        print("2. HuggingFace API 金鑰有效")
        print("3. 網路連接正常")

if __name__ == "__main__":
    main()

