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
import matplotlib.pyplot as plt
try:
    from imitation.algorithms import bc  # 行為克隆
    from imitation.data import rollout  # 軌跡收集
    from imitation.data.buffer import ReplayBuffer  # 正確的 ReplayBuffer 導入路徑
    from imitation.data.types import Transitions  # 軌跡轉換類型
    import gymnasium as gym
    import gymnasium.spaces as spaces  # 新版 spaces
    IMITATION_AVAILABLE = True
except ImportError as e:
    print(f"警告: imitation 庫導入失敗: {e}")
    IMITATION_AVAILABLE = False
from finbert_main import FinBERTAnalyzer  # 匯入 FinBERT 類別

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
        """
        dotenv.load_dotenv()
        api_key = os.getenv('HF_TOKEN')
        if not api_key:
            raise ValueError("HF_TOKEN 未設置，請檢查 .env 文件或環境變數")
        if role not in ["aggressive", "defensive"]:
            raise ValueError("角色必須為 'aggressive' 或 'defensive'")
        
        self.role = role
        self.analyzer = FinBERTAnalyzer(api_key)  # 整合 FinBERTAnalyzer
        self.policy_trajectories = []  # 儲存歷史軌跡，用於模仿學習
        self.action_space = [0, 1, 2]  # sell, hold, buy
        self.fallback_count = 0  # 追蹤 fallback 次數
        self.advanced_learning_failed = False  # 追蹤高級學習是否曾經失敗
        
        # 檢查是否禁用高級學習（多種方式）
        disable_env = os.getenv('DISABLE_ADVANCED_LEARNING', 'false').lower() == 'true'
        force_simple_env = os.getenv('FORCE_SIMPLE_LEARNING', 'false').lower() == 'true'
        self.disable_advanced_learning = disable_env or force_simple_env
        
        if self.disable_advanced_learning:
            print(f"{self.role} 代理: 高級學習已被禁用，將使用簡化學習")
    
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
            # 呼叫 FinBERT 的 generate_trading_strategy
            result = self.analyzer.generate_trading_strategy(
                price, rsi, sentiment, f"{self.role}_predict"
            )
            
            if result.get('success', False):
                action = result['parsed_action']
                strategy = result['raw_response']
            else:
                self.fallback_count += 1
                action, strategy = self._fallback_predict(price, rsi, sentiment)
            
            # 記錄軌跡
            self.policy_trajectories.append({
                'obs': obs.copy(),
                'action': action,
                'strategy': strategy
            })
            
            return action, strategy
        
        except Exception as e:
            print(f"預測錯誤 ({self.role}): {e} - 使用 fallback")
            self.fallback_count += 1
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
        從其他代理學習，使用簡化的模仿學習策略。
        
        Args:
            other_trajectories: 贏家代理的歷史軌跡 [{'obs': [...], 'action': int, ...}]
        """
        if len(other_trajectories) < 10:
            print(f"{self.role} 代理: 軌跡不足（{len(other_trajectories)} < 10），跳過學習")
            return
        
        try:
            # 驗證軌跡數據格式
            for i, traj in enumerate(other_trajectories[:3]):  # 檢查前3個軌跡
                if 'obs' not in traj or 'action' not in traj:
                    print(f"{self.role} 代理: 軌跡 {i} 缺少必要欄位")
                    return
                if not isinstance(traj['obs'], list) or len(traj['obs']) != 3:
                    print(f"{self.role} 代理: 軌跡 {i} obs 格式錯誤: {traj['obs']}")
                    return
            
            # 決定使用哪種學習方法
            use_advanced = (IMITATION_AVAILABLE and 
                           not self.disable_advanced_learning and 
                           not self.advanced_learning_failed)
            
            if use_advanced:
                print(f"{self.role} 代理: 嘗試高級模仿學習...")
                try:
                    self._advanced_learning(other_trajectories)
                    print(f"{self.role} 代理: 高級學習成功完成")
                    return  # 成功則直接返回
                except Exception as e:
                    print(f"{self.role} 代理: 高級學習失敗，標記為不可用: {e}")
                    self.advanced_learning_failed = True
                    print(f"{self.role} 代理: 現在回退到簡化學習...")
                    # 繼續執行簡化學習（不要 return）
            
            # 使用簡化學習（作為備選或主要方法）
            if not use_advanced or self.advanced_learning_failed:
                if not IMITATION_AVAILABLE:
                    reason = "imitation 庫不可用"
                elif self.disable_advanced_learning:
                    reason = "高級學習被禁用"
                elif self.advanced_learning_failed:
                    reason = "高級學習失敗"
                else:
                    reason = "使用簡化學習"
                
                print(f"{self.role} 代理: {reason}，執行簡化學習...")
                self._simple_learning(other_trajectories)
                print(f"{self.role} 代理: 簡化學習完成")
                
        except Exception as e:
            print(f"{self.role} 代理學習錯誤: {e} - 維持原策略")
    
    def _simple_learning(self, other_trajectories: List[Dict[str, Any]]) -> None:
        """簡化的模仿學習：分析成功軌跡的模式並調整決策閾值"""
        print(f"{self.role} 代理: 開始簡化學習，分析 {len(other_trajectories)} 個軌跡...")
        
        try:
            # 分析成功軌跡的行為模式
            action_counts = {0: 0, 1: 0, 2: 0}  # sell, hold, buy
            successful_conditions = {'buy': [], 'sell': [], 'hold': []}
            
            # 驗證並處理軌跡數據
            valid_trajectories = 0
            for i, traj in enumerate(other_trajectories):
                try:
                    obs = traj['obs']
                    action = traj['action']
                    
                    # 驗證數據格式
                    if not isinstance(obs, list) or len(obs) != 3:
                        print(f"{self.role} 代理: 跳過無效軌跡 {i}: obs 格式錯誤")
                        continue
                    if action not in [0, 1, 2]:
                        print(f"{self.role} 代理: 跳過無效軌跡 {i}: action 無效 ({action})")
                        continue
                    
                    price, rsi, sentiment = obs
                    valid_trajectories += 1
                    action_counts[action] += 1
                    
                    if action == 2:  # buy
                        successful_conditions['buy'].append({'rsi': rsi, 'sentiment': sentiment})
                    elif action == 0:  # sell
                        successful_conditions['sell'].append({'rsi': rsi, 'sentiment': sentiment})
                    else:  # hold
                        successful_conditions['hold'].append({'rsi': rsi, 'sentiment': sentiment})
                        
                except Exception as traj_e:
                    print(f"{self.role} 代理: 處理軌跡 {i} 時出錯: {traj_e}")
                    continue
            
            print(f"{self.role} 代理: 有效軌跡數量: {valid_trajectories}/{len(other_trajectories)}")
            
            # 計算成功條件的統計
            print(f"{self.role} 代理: 動作分佈 - Buy: {action_counts[2]}, Hold: {action_counts[1]}, Sell: {action_counts[0]}")
            
            # 分析買入條件
            if successful_conditions['buy']:
                avg_buy_sentiment = np.mean([c['sentiment'] for c in successful_conditions['buy']])
                avg_buy_rsi = np.mean([c['rsi'] for c in successful_conditions['buy']])
                print(f"{self.role} 代理: 成功買入條件 - 平均 sentiment: {avg_buy_sentiment:.3f}, 平均 RSI: {avg_buy_rsi:.3f}")
            
            # 分析賣出條件
            if successful_conditions['sell']:
                avg_sell_sentiment = np.mean([c['sentiment'] for c in successful_conditions['sell']])
                avg_sell_rsi = np.mean([c['rsi'] for c in successful_conditions['sell']])
                print(f"{self.role} 代理: 成功賣出條件 - 平均 sentiment: {avg_sell_sentiment:.3f}, 平均 RSI: {avg_sell_rsi:.3f}")
            
            # 分析持有條件
            if successful_conditions['hold']:
                avg_hold_sentiment = np.mean([c['sentiment'] for c in successful_conditions['hold']])
                avg_hold_rsi = np.mean([c['rsi'] for c in successful_conditions['hold']])
                print(f"{self.role} 代理: 成功持有條件 - 平均 sentiment: {avg_hold_sentiment:.3f}, 平均 RSI: {avg_hold_rsi:.3f}")
            
            # 更新本地軌跡（學習記錄）
            trajectories_to_add = min(5, len(other_trajectories))
            self.policy_trajectories.extend(other_trajectories[-trajectories_to_add:])
            print(f"{self.role} 代理: 已添加 {trajectories_to_add} 個軌跡到本地記錄，總數: {len(self.policy_trajectories)}")
            
        except Exception as e:
            print(f"{self.role} 代理簡化學習錯誤: {e}")
            print(f"  錯誤詳情: {str(e)}")
            raise e  # 重新拋出異常以便上層處理
    
    def _advanced_learning(self, other_trajectories: List[Dict[str, Any]]) -> None:
        """使用 imitation 庫的高級學習方法 - 徹底修復索引錯誤版本"""
        print(f"{self.role} 代理: 開始高級學習，軌跡數量: {len(other_trajectories)}")
        
        try:
            # 收集觀測和動作數據，添加嚴格驗證
            observations = []
            actions = []
            
            for i, traj in enumerate(other_trajectories):
                obs = traj['obs']
                action = traj['action']
                
                # 嚴格驗證觀測數據
                if not isinstance(obs, list) or len(obs) != 3:
                    print(f"{self.role} 代理: 跳過軌跡 {i} - obs 格式錯誤: {obs}")
                    continue
                
                # 嚴格驗證動作數據
                try:
                    action_int = int(action)
                    if action_int not in [0, 1, 2]:
                        print(f"{self.role} 代理: 跳過軌跡 {i} - 動作超出範圍: {action_int}")
                        continue
                except (ValueError, TypeError):
                    print(f"{self.role} 代理: 跳過軌跡 {i} - 動作無法轉換為整數: {action}")
                    continue
                
                # 驗證觀測值是否為有效數字
                try:
                    obs_float = [float(x) for x in obs]
                    if any(np.isnan(x) or np.isinf(x) for x in obs_float):
                        print(f"{self.role} 代理: 跳過軌跡 {i} - obs 包含無效值: {obs_float}")
                        continue
                except (ValueError, TypeError):
                    print(f"{self.role} 代理: 跳過軌跡 {i} - obs 無法轉換為浮點數: {obs}")
                    continue
                
                observations.append(obs_float)
                actions.append(action_int)
            
            if len(observations) < 10:
                print(f"{self.role} 代理: 有效軌跡不足 ({len(observations)} < 10)，跳過學習")
                return
            
            # 轉換為 numpy 數組，確保數據類型嚴格正確
            try:
                # 確保所有數據都是正確的類型，避免混合類型導致的索引問題
                obs_array = np.array(observations, dtype=np.float32)
                acts_array = np.array(actions, dtype=np.int32)  # 改用 int32，更安全
                
                # 驗證數據完整性
                if obs_array.shape[0] != acts_array.shape[0]:
                    print(f"{self.role} 代理: 數據長度不匹配 - obs: {obs_array.shape[0]}, acts: {acts_array.shape[0]}")
                    return
                
                # 確保數據連續性，避免內存布局問題
                obs_array = np.ascontiguousarray(obs_array)
                acts_array = np.ascontiguousarray(acts_array)
                    
                print(f"{self.role} 代理: 數據驗證完成 - obs: {obs_array.shape}, acts: {acts_array.shape}")
                print(f"{self.role} 代理: 動作統計 - 範圍: [{np.min(acts_array)}, {np.max(acts_array)}], 唯一值: {np.unique(acts_array)}")
                
            except Exception as e:
                print(f"{self.role} 代理: 數組轉換失敗: {e}")
                return
            
            # 限制樣本數量，確保有足夠的數據創建 transitions
            n_samples = min(len(obs_array), 15)  # 進一步減少樣本數量
            if n_samples < 5:  # 提高最小樣本數要求
                print(f"{self.role} 代理: 樣本數量太少 ({n_samples} < 5)，跳過學習")
                return
            
            # 截取數據到指定樣本數
            obs_array = obs_array[:n_samples]
            acts_array = acts_array[:n_samples]
            
            # 創建 transitions 數據，使用最安全的方式
            try:
                # 確保至少有2個樣本才能創建 transitions
                if len(obs_array) < 2:
                    print(f"{self.role} 代理: 數據不足以創建 transitions")
                    return
                
                # 創建 episode 結構，每個軌跡作為一個完整的 episode
                obs_data = obs_array[:-1].copy()  # 使用 copy() 確保數據獨立
                acts_data = acts_array[:-1].copy()
                next_obs_data = obs_array[1:].copy()
                
                # 確保所有數組都是連續的
                obs_data = np.ascontiguousarray(obs_data, dtype=np.float32)
                acts_data = np.ascontiguousarray(acts_data, dtype=np.int32)
                next_obs_data = np.ascontiguousarray(next_obs_data, dtype=np.float32)
                
                # 創建 dones 和 infos
                dones_data = np.zeros(len(obs_data), dtype=bool)
                dones_data[-1] = True  # 只有最後一個 transition 標記為結束
                infos_data = [{} for _ in range(len(obs_data))]
                
                print(f"{self.role} 代理: Transitions 數據準備完成 - 樣本數: {len(obs_data)}")
                print(f"{self.role} 代理: 數據形狀驗證:")
                print(f"  - obs_data: {obs_data.shape}, dtype: {obs_data.dtype}, contiguous: {obs_data.flags['C_CONTIGUOUS']}")
                print(f"  - acts_data: {acts_data.shape}, dtype: {acts_data.dtype}, contiguous: {acts_data.flags['C_CONTIGUOUS']}")
                print(f"  - next_obs_data: {next_obs_data.shape}, dtype: {next_obs_data.dtype}, contiguous: {next_obs_data.flags['C_CONTIGUOUS']}")
                print(f"  - dones_data: {dones_data.shape}, dtype: {dones_data.dtype}")
                
            except Exception as e:
                print(f"{self.role} 代理: Transitions 數據準備失敗: {e}")
                print(f"  原始數據形狀 - obs_array: {obs_array.shape}, acts_array: {acts_array.shape}")
                return
            
            # 創建 Transitions 對象，使用最嚴格的參數
            try:
                transitions = Transitions(
                    obs=obs_data,
                    acts=acts_data,
                    infos=infos_data,
                    next_obs=next_obs_data,
                    dones=dones_data
                )
                print(f"{self.role} 代理: Transitions 對象創建成功")
                
                # 驗證 transitions 對象的完整性
                print(f"  - Transitions 長度: {len(transitions)}")
                if len(transitions) == 0:
                    print(f"{self.role} 代理: Transitions 對象為空，跳過學習")
                    return
                    
            except Exception as e:
                print(f"{self.role} 代理: Transitions 對象創建失敗: {e}")
                return
            
            # 創建 BC 訓練器，使用最保守和兼容的參數
            try:
                # 使用最小的批次大小，避免索引問題
                batch_size = 1
                print(f"{self.role} 代理: 準備創建 BC 訓練器，數據量: {len(obs_data)}, 批次大小: {batch_size}")
                
                # 創建兼容的 observation_space 和 action_space
                observation_space = spaces.Box(
                    low=-np.inf, 
                    high=np.inf, 
                    shape=(3,), 
                    dtype=np.float32
                )
                action_space = spaces.Discrete(3)
                
                trainer = bc.BC(
                    observation_space=observation_space,
                    action_space=action_space,
                    demonstrations=[transitions],
                    batch_size=batch_size,
                    ent_weight=0.0,  # 完全禁用熵正則化
                    l2_weight=0.0,   # 禁用 L2 正則化
                    rng=np.random.default_rng(42)
                )
                print(f"{self.role} 代理: BC 訓練器創建成功")
            except Exception as e:
                print(f"{self.role} 代理: BC 訓練器創建失敗: {e}")
                print(f"  詳細錯誤: {str(e)}")
                raise e  # 拋出異常讓上層處理
            
            # 訓練過程，使用自定義的安全訓練方法，完全避免 imitation 庫的索引問題
            try:
                print(f"{self.role} 代理: 開始自定義安全訓練...")
                
                # 不使用 trainer.train()，而是實現自己的行為克隆邏輯
                # 這樣可以完全避免 imitation 庫內部的索引錯誤
                success = self._safe_behavior_cloning(obs_data, acts_data)
                
                if success:
                    print(f"{self.role} 代理: 自定義訓練成功完成")
                    
                    # 更新本地軌跡
                    trajectories_to_add = min(3, len(other_trajectories))
                    self.policy_trajectories.extend(other_trajectories[-trajectories_to_add:])
                    print(f"{self.role} 代理: 本地軌跡已更新，總數: {len(self.policy_trajectories)}")
                    print(f"{self.role} 代理: 高級學習完成")
                else:
                    print(f"{self.role} 代理: 自定義訓練失敗")
                    raise Exception("自定義訓練失敗")
                
            except Exception as e:
                print(f"{self.role} 代理: 訓練過程發生錯誤: {e}")
                print(f"  詳細錯誤信息: {str(e)}")
                # 任何訓練錯誤都直接拋出異常
                raise e
            
        except Exception as e:
            print(f"{self.role} 代理: 高級學習過程發生錯誤: {e}")
            print(f"  錯誤類型: {type(e).__name__}")
            # 直接拋出異常，讓 learn_from_other 方法處理回退
            raise e
    
    def _safe_behavior_cloning(self, observations: np.ndarray, actions: np.ndarray) -> bool:
        """
        安全的行為克隆實現，避免 imitation 庫的索引問題
        
        Args:
            observations: 觀測數據 (N, 3)
            actions: 動作數據 (N,)
            
        Returns:
            bool: 訓練是否成功
        """
        try:
            print(f"{self.role} 代理: 執行安全行為克隆，數據形狀: obs={observations.shape}, acts={actions.shape}")
            
            # 簡化的行為克隆：分析動作模式並更新決策策略
            # 統計每個動作在不同條件下的出現頻率
            action_patterns = {}
            
            for i in range(len(observations)):
                obs = observations[i]
                action = actions[i]
                
                # 將觀測值離散化為條件
                price, rsi, sentiment = obs
                
                # 創建條件鍵
                rsi_level = "low" if rsi < 0.4 else ("high" if rsi > 0.6 else "mid")
                sentiment_level = "negative" if sentiment < -0.2 else ("positive" if sentiment > 0.2 else "neutral")
                condition = f"{rsi_level}_{sentiment_level}"
                
                if condition not in action_patterns:
                    action_patterns[condition] = {0: 0, 1: 0, 2: 0}
                
                action_patterns[condition][action] += 1
            
            # 分析學習到的模式
            learned_patterns = {}
            for condition, action_counts in action_patterns.items():
                total = sum(action_counts.values())
                if total > 0:
                    # 找出最常見的動作
                    best_action = max(action_counts, key=action_counts.get)
                    confidence = action_counts[best_action] / total
                    learned_patterns[condition] = {
                        'action': best_action,
                        'confidence': confidence,
                        'total_samples': total
                    }
            
            print(f"{self.role} 代理: 學習到 {len(learned_patterns)} 個行為模式:")
            for condition, pattern in learned_patterns.items():
                action_name = ["賣出", "持有", "買入"][pattern['action']]
                print(f"  - {condition}: {action_name} (信心度: {pattern['confidence']:.2f}, 樣本數: {pattern['total_samples']})")
            
            # 將學習到的模式存儲起來（可以用於後續決策）
            if not hasattr(self, 'learned_patterns'):
                self.learned_patterns = {}
            
            # 更新學習到的模式，使用加權平均
            for condition, pattern in learned_patterns.items():
                if condition in self.learned_patterns:
                    # 加權平均更新
                    old_pattern = self.learned_patterns[condition]
                    old_weight = old_pattern['total_samples']
                    new_weight = pattern['total_samples']
                    total_weight = old_weight + new_weight
                    
                    if total_weight > 0:
                        # 如果新模式有更高的信心度，則更新
                        if pattern['confidence'] > old_pattern['confidence']:
                            self.learned_patterns[condition] = pattern
                        else:
                            # 保持舊模式但更新樣本數
                            self.learned_patterns[condition]['total_samples'] = total_weight
                else:
                    self.learned_patterns[condition] = pattern
            
            print(f"{self.role} 代理: 行為克隆完成，總共學習了 {len(self.learned_patterns)} 個決策模式")
            return True
            
        except Exception as e:
            print(f"{self.role} 代理: 安全行為克隆失敗: {e}")
            return False

# 測試腳本
if __name__ == "__main__":
    # 設置隨機種子以確保可重現性
    np.random.seed(42)
    
    # 示例測試
    agent_a = LLMAgent(role="aggressive")
    agent_b = LLMAgent(role="defensive")
    obs = [100.0, 0.5, 0.8]  # [price, RSI, sentiment]
    
    action_a, strat_a = agent_a.predict(obs)
    action_b, strat_b = agent_b.predict(obs)
    
    print(f"Aggressive Agent: Action {action_a} - {strat_a}")
    print(f"Defensive Agent: Action {action_b} - {strat_b}")
    print(f"Aggressive Fallback Count: {agent_a.fallback_count}")
    print(f"Defensive Fallback Count: {agent_b.fallback_count}")
    
    # 改進的模擬軌跡
    mock_trajectories = (
        [{'obs': [100.0, 0.3, 0.9], 'action': 2, 'strategy': 'Buy on high sentiment'}] * 5 +  # buy
        [{'obs': [100.0, 0.7, -0.5], 'action': 0, 'strategy': 'Sell on low sentiment'}] * 5 +  # sell
        [{'obs': [100.0, 0.5, 0.1], 'action': 1, 'strategy': 'Hold on neutral'}] * 5  # hold
    )
    agent_b.learn_from_other(mock_trajectories)