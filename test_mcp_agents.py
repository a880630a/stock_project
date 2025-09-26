# -*- coding: utf-8 -*-
"""
MCP 整合 LLMAgent 測試腳本
展示如何使用真實 TWSE 數據
"""
import asyncio
from agents import LLMAgent

async def main():
    print("🚀 MCP 整合 LLMAgent 測試")
    print("=" * 40)
    
    # 創建代理
    agent = LLMAgent(role='aggressive')
    
    print("\n📈 測試 1: 使用真實數據預測台積電")
    print("-" * 30)
    
    # 使用真實 TWSE 數據預測
    action, strategy = await agent.predict(obs=None, stock_code='2330')
    
    # 顯示結果
    action_text = {0: "賣出", 1: "持有", 2: "買入"}[action]
    print(f"✅ 預測結果: {action} ({action_text})")
    print(f"📝 策略: {strategy[:120]}...")
    
    print("\n📚 測試 2: 使用真實數據學習")
    print("-" * 30)
    
    # 使用真實數據學習
    await agent.learn_from_other(other_trajectories=[], stock_code='2330')
    print("✅ 學習完成")
    
    print("\n🎉 測試完成！")
    print("\n💡 您可以:")
    print("- 修改 stock_code 參數測試不同股票")
    print("- 嘗試 role='defensive' 創建保守型代理")
    print("- 支援股票: 2330(台積電), 2317(鴻海), 2454(聯發科)")

if __name__ == "__main__":
    asyncio.run(main())
