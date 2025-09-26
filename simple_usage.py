# -*- coding: utf-8 -*-
"""
簡單使用範例 - 最基本的用法
"""
from agents import LLMAgent

# 創建代理
agent = LLMAgent(role='aggressive')  # 或 'defensive'

# 使用真實數據預測
action, strategy = agent.predict(obs=None, stock_code='2330')
print(f"台積電預測: {action} ({'買入' if action == 2 else '賣出' if action == 0 else '持有'})")

# 使用真實數據學習
agent.learn_from_other(other_trajectories=[], stock_code='2317')
print("學習完成")
