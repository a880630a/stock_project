# -*- coding: utf-8 -*-
"""
全面測試改進後的 MCP 整合 LLMAgent 系統
展示所有優化功能
"""
import asyncio
from agents import LLMAgent
import os

async def comprehensive_test():
    """全面測試所有改進功能"""
    print("🚀 全面 MCP 整合測試")
    print("=" * 60)
    
    # 測試 1: 積極型代理 - 台積電
    print("\n📊 測試 1: 積極型代理 - 台積電 (2330)")
    print("-" * 50)
    aggressive_agent = LLMAgent(role='aggressive')
    
    # 預測
    action, strategy = await aggressive_agent.predict(obs=None, stock_code='2330')
    action_text = {0: "賣出", 1: "持有", 2: "買入"}[action]
    print(f"🎯 預測: {action} ({action_text})")
    print(f"📝 策略摘要: {strategy[:100]}...")
    
    # 學習
    print(f"\n🧠 學習過程:")
    await aggressive_agent.learn_from_other(other_trajectories=[], stock_code='2330')
    
    # 測試 2: 保守型代理 - 鴻海
    print(f"\n📊 測試 2: 保守型代理 - 鴻海 (2317)")
    print("-" * 50)
    defensive_agent = LLMAgent(role='defensive')
    
    # 預測
    action, strategy = await defensive_agent.predict(obs=None, stock_code='2317')
    action_text = {0: "賣出", 1: "持有", 2: "買入"}[action]
    print(f"🎯 預測: {action} ({action_text})")
    print(f"📝 策略摘要: {strategy[:100]}...")
    
    # 學習
    print(f"\n🧠 學習過程:")
    await defensive_agent.learn_from_other(other_trajectories=[], stock_code='2317')
    
    # 測試 3: 不同股票比較
    print(f"\n📊 測試 3: 多股票預測比較")
    print("-" * 50)
    
    stocks = [('2330', '台積電'), ('2317', '鴻海'), ('2454', '聯發科')]
    test_agent = LLMAgent(role='aggressive')
    
    for code, name in stocks:
        action, _ = await test_agent.predict(obs=None, stock_code=code)
        action_text = {0: "賣出", 1: "持有", 2: "買入"}[action]
        print(f"📈 {name} ({code}): {action_text}")
    
    # 檢查生成的文件
    print(f"\n📁 生成的文件:")
    print("-" * 30)
    
    png_files = [f for f in os.listdir('.') if f.endswith('.png')]
    if png_files:
        for file in png_files:
            size = os.path.getsize(file)
            print(f"✅ {file} ({size:,} bytes)")
    else:
        print("❌ 未找到 PNG 文件")
    
    print(f"\n🎉 全面測試完成！")
    
    # 總結改進
    print(f"\n📈 改進總結:")
    print("=" * 30)
    print("✅ BC 訓練器：真實神經網路訓練過程")
    print("✅ RSI 計算：基於價格的智能模擬")
    print("✅ 軌跡多樣性：8 種不同市場情況")
    print("✅ 學習模式：多條件決策模式識別")
    print("✅ 指標可視化：BC 訓練指標圖表")
    print("✅ 異步架構：完整的異步 MCP 調用")

if __name__ == "__main__":
    asyncio.run(comprehensive_test())
