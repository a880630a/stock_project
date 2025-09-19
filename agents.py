#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 驅動股票交易代理類別
整合 FinBERTAnalyzer 作為決策引擎，支援角色化 (aggressive/defensive) 與模仿學習
"""

import os
import sys
import dotenv
import numpy as np
from typing import Tuple, List, Dict, Any
try:
    from imitation.algorithms import bc  # 行為克隆
    from imitation.data import rollout  # 軌跡收集
    from imitation.data.types import Transitions
    import gymnasium as gym
    IMITATION_AVAILABLE = True
except ImportError as e:
    print(f"警告: imitation 庫導入失敗: {e}")
    IMITATION_AVAILABLE = False
from finbert_main import FinBERTAnalyzer  # 匯入你的 FinBERT 類別

class LLMAgent:
    """
    LLM 驅動的股票交易代理，基於 FinBERTAnalyzer 實現角色化決策。
    
    角色:
    - aggressive: 最大化短期獲利，偏好高 sentiment 買入
    - defensive: 最小化風險，偏好低 RSI 賣出/持有
    
    方法:
    - predict: 輸入狀態 [price, RSI, sentiment]，生成動作 (0=sell, 1=hold, 2=buy) + 策略文字
    - learn_from_other: 模仿學習，使用 BC 微調歷史軌跡
    """
    
    def __init__(self, role: str = "aggressive"):
        """
        初始化代理。
        
        Args:
            role: 代理角色 ('aggressive' 或 'defensive')
            api_key: Hugging Face API 金鑰 (若 None，從環境變量讀取)
        """
        dotenv.load_dotenv()
        api_key = os.getenv('HF_TOKEN')
        if role not in ["aggressive", "defensive"]:
            raise ValueError("角色必須為 'aggressive' 或 'defensive'")
        
        self.role = role
        self.analyzer = FinBERTAnalyzer(api_key)  # 整合你的 FinBERTAnalyzer
        self.policy_trajectories = []  # 儲存歷史軌跡，用於模仿學習
        self.action_space = [0, 1, 2]  # sell, hold, buy
    
    def predict(self, obs: List[float]) -> Tuple[int, str]:
        """
        根據狀態生成交易動作與策略。
        
        Args:
            obs: 狀態 [price, RSI, sentiment] (標準化後)
            
        Returns:
            (action: int, strategy: str) - 動作 (0=sell, 1=hold, 2=buy) 與策略文字
        """
        if len(obs) != 3:
            raise ValueError("狀態必須為 [price, RSI, sentiment]")
        
        price, rsi, sentiment = obs
        try:
            # 呼叫 FinBERT 的 generate_trading_strategy (無縫整合)
            result = self.analyzer.generate_trading_strategy(
                price, rsi, sentiment, f"{self.role}_predict"
            )
            
            if result.get('success', False):
                action = result['parsed_action']
                strategy = result['raw_response']
            else:
                # Fallback: 基於角色規則生成默認動作
                action, strategy = self._fallback_predict(price, rsi, sentiment)
            
            # 記錄軌跡 (obs, action) 用於後續學習
            self.policy_trajectories.append({
                'obs': obs.copy(),
                'action': action,
                'strategy': strategy
            })
            
            return action, strategy
        
        except Exception as e:
            print(f"預測錯誤 ({self.role}): {e} - 使用 fallback")
            action, strategy = self._fallback_predict(price, rsi, sentiment)
            return action, strategy
    
    def _fallback_predict(self, price: float, rsi: float, sentiment: float) -> Tuple[int, str]:
        """Fallback 規則基於決策 (無 LLM 時使用)。"""
        if self.role == "aggressive":
            if sentiment > 0.3 or rsi < 0.4:
                return 2, "Fallback: Aggressive buy on high sentiment/low RSI"
            else:
                return 1, "Fallback: Aggressive hold"
        else:  # defensive
            if rsi > 0.6 or sentiment < -0.3:
                return 0, "Fallback: Defensive sell on high RSI/low sentiment"
            else:
                return 1, "Fallback: Defensive hold"
    
    def learn_from_other(self, other_trajectories: List[Dict[str, Any]]) -> None:
        """
        從其他代理學習，使用行為克隆 (BC) 微調策略。
        
        Args:
            other_trajectories: 贏家代理的歷史軌跡 [{'obs': [...], 'action': int, ...}]
        """
        if not IMITATION_AVAILABLE:
            print(f"{self.role} 代理: imitation 庫不可用，跳過學習")
            return
            
        if len(other_trajectories) < 10:  # 至少 10 軌跡才學習
            print(f"{self.role} 代理: 軌跡不足，跳過學習")
            return
        
        try:
            # 收集軌跡數據 (imitation 格式)
            obs_list = np.array([t['obs'] for t in other_trajectories])
            actions_list = np.array([t['action'] for t in other_trajectories])
            
            # 創建 Transitions 對象
            num_transitions = len(obs_list) - 1
            transitions = Transitions(
                obs=obs_list[:-1],  # 當前狀態
                acts=actions_list[:-1],  # 動作
                infos=np.array([{}] * num_transitions),  # 空信息字典數組
                next_obs=obs_list[1:],  # 下一狀態
                dones=np.zeros(num_transitions, dtype=bool)  # 結束標誌
            )
            
            # 簡單訓練 (1 epoch，適合輕量學習)
            trainer = bc.BC(
                observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
                action_space=gym.spaces.Discrete(3),
                demonstrations=transitions,
                batch_size=min(32, len(obs_list) - 1),
                rng=np.random.default_rng(42)  # 添加隨機數生成器
            )
            trainer.train(n_epochs=1)  # 微調步數
            
            print(f"{self.role} 代理: 從 {len(other_trajectories)} 軌跡學習完成")
            
            # 更新本地軌跡 (知識蒸餾)
            self.policy_trajectories.extend(other_trajectories[-5:])  # 保留最新 5 個
            
        except Exception as e:
            print(f"{self.role} 代理學習錯誤: {e} - 維持原策略")

# 測試腳本 (可選，運行 python agents.py 測試)
if __name__ == "__main__":
    
    # 示例測試
    agent_a = LLMAgent(role="aggressive" )
    agent_b = LLMAgent(role="defensive" )
    obs = [100.0, 0.5, 0.8]  # [price, RSI, sentiment]
    
    action_a, strat_a = agent_a.predict(obs)
    action_b, strat_b = agent_b.predict(obs)
    
    print(f"Aggressive Agent: Action {action_a} - {strat_a}")
    print(f"Defensive Agent: Action {action_b} - {strat_b}")
    
    # 模仿測試
    mock_trajectories = [{'obs': [100.0, 0.3, 0.9], 'action': 2}] * 15
    agent_b.learn_from_other(mock_trajectories)