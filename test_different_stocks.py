# -*- coding: utf-8 -*-
"""
測試不同股票的 MCP 整合功能
"""
from agents import LLMAgent

def test_multiple_stocks():
    """測試多支股票的預測"""
    print("🚀 測試多支股票的 MCP 整合")
    print("=" * 50)
    
    stocks = [
        ('2330', '台積電'),
        ('2317', '鴻海'),
        ('2454', '聯發科')
    ]
    
    # 測試積極型代理
    aggressive_agent = LLMAgent(role='aggressive')
    
    for code, name in stocks:
        print(f"\n📈 {name} ({code}) - 積極型代理")
        print("-" * 30)
        
        action, strategy = aggressive_agent.predict(obs=None, stock_code=code)
        action_text = {0: "賣出", 1: "持有", 2: "買入"}[action]
        
        print(f"預測: {action} ({action_text})")
        print(f"策略摘要: {strategy[:80]}...")
    
    print(f"\n🎉 多股票測試完成！")

if __name__ == "__main__":
    test_multiple_stocks()
