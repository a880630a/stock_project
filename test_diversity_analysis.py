# -*- coding: utf-8 -*-
"""
測試軌跡生成多樣性和 BC 學習效果分析
"""
import asyncio
from agents import LLMAgent
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

async def analyze_trajectory_diversity():
    """分析軌跡生成的多樣性"""
    print("🔍 軌跡多樣性分析")
    print("=" * 50)
    
    # 創建代理
    agent = LLMAgent(role='aggressive')
    
    # 生成多組軌跡進行分析
    all_actions = []
    all_rsi_values = []
    all_sentiment_values = []
    
    print("📊 生成軌跡樣本進行分析...")
    
    # 執行多次學習，收集統計數據
    for round_num in range(3):
        print(f"\n第 {round_num + 1} 輪軌跡生成:")
        
        # 重置軌跡
        test_trajectories = []
        
        # 模擬學習過程（不實際訓練，只收集軌跡）
        await agent.learn_from_other(other_trajectories=test_trajectories, stock_code='2330')
        
        # 分析生成的軌跡
        if hasattr(agent, 'policy_trajectories') and agent.policy_trajectories:
            recent_trajectories = agent.policy_trajectories[-10:]  # 最近 10 個軌跡
            
            for traj in recent_trajectories:
                if 'action' in traj and 'obs' in traj:
                    all_actions.append(traj['action'])
                    if len(traj['obs']) >= 3:
                        all_rsi_values.append(traj['obs'][1])
                        all_sentiment_values.append(traj['obs'][2])
    
    # 統計分析
    print(f"\n📈 多樣性統計結果:")
    print(f"總軌跡數: {len(all_actions)}")
    
    if all_actions:
        action_counts = Counter(all_actions)
        print(f"動作分佈:")
        for action, count in sorted(action_counts.items()):
            action_name = {0: "賣出", 1: "持有", 2: "買入"}[action]
            percentage = (count / len(all_actions)) * 100
            print(f"  - {action_name} ({action}): {count} 次 ({percentage:.1f}%)")
        
        print(f"\nRSI 範圍: [{min(all_rsi_values):.3f}, {max(all_rsi_values):.3f}]")
        print(f"情感範圍: [{min(all_sentiment_values):.3f}, {max(all_sentiment_values):.3f}]")
        
        # 計算多樣性指標
        unique_actions = len(set(all_actions))
        diversity_score = unique_actions / 3.0  # 最多 3 種動作
        print(f"動作多樣性分數: {diversity_score:.2f} ({unique_actions}/3 種動作)")
        
        # 生成可視化
        create_diversity_visualization(all_actions, all_rsi_values, all_sentiment_values)
    else:
        print("❌ 未收集到軌跡數據")

def create_diversity_visualization(actions, rsi_values, sentiment_values):
    """創建多樣性可視化圖表"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 子圖 1: 動作分佈
        action_counts = Counter(actions)
        action_labels = {0: "賣出", 1: "持有", 2: "買入"}
        labels = [action_labels.get(action, f"動作{action}") for action in sorted(action_counts.keys())]
        counts = [action_counts[action] for action in sorted(action_counts.keys())]
        colors = ['red', 'orange', 'green'][:len(labels)]
        
        axes[0, 0].pie(counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('動作分佈')
        
        # 子圖 2: RSI 分佈
        axes[0, 1].hist(rsi_values, bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 1].set_title('RSI 分佈')
        axes[0, 1].set_xlabel('RSI 值')
        axes[0, 1].set_ylabel('頻率')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 子圖 3: 情感分佈
        axes[1, 0].hist(sentiment_values, bins=20, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 0].set_title('情感分佈')
        axes[1, 0].set_xlabel('情感值')
        axes[1, 0].set_ylabel('頻率')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 子圖 4: RSI vs 情感散點圖
        action_colors = {0: 'red', 1: 'orange', 2: 'green'}
        for action in set(actions):
            mask = np.array(actions) == action
            if np.any(mask):
                rsi_subset = np.array(rsi_values)[mask]
                sentiment_subset = np.array(sentiment_values)[mask]
                axes[1, 1].scatter(rsi_subset, sentiment_subset, 
                                 c=action_colors[action], 
                                 label=action_labels[action], 
                                 alpha=0.7)
        
        axes[1, 1].set_title('RSI vs 情感 (按動作分色)')
        axes[1, 1].set_xlabel('RSI')
        axes[1, 1].set_ylabel('情感')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('trajectory_diversity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 多樣性分析圖表已保存為 trajectory_diversity_analysis.png")
        
    except Exception as e:
        print(f"❌ 生成多樣性圖表失敗: {e}")

async def test_bc_metrics():
    """測試 BC 指標記錄功能"""
    print(f"\n🧠 BC 指標測試")
    print("=" * 30)
    
    agent = LLMAgent(role='defensive')  # 使用保守型代理測試
    
    # 執行學習並記錄指標
    await agent.learn_from_other(other_trajectories=[], stock_code='2317')
    
    # 檢查是否生成了 BC 指標圖表
    import os
    if os.path.exists('bc_loss_defensive.png'):
        print("✅ BC 指標圖表生成成功: bc_loss_defensive.png")
    else:
        print("⚠️ BC 指標圖表未生成")

async def main():
    """主測試函數"""
    print("🚀 步驟 3.1 優化效果驗證")
    print("=" * 60)
    
    await analyze_trajectory_diversity()
    await test_bc_metrics()
    
    print(f"\n🎉 多樣性分析完成！")
    print(f"\n📊 生成的文件:")
    print(f"- trajectory_diversity_analysis.png (軌跡多樣性分析)")
    print(f"- bc_loss_aggressive.png (積極型代理 BC 指標)")
    print(f"- bc_loss_defensive.png (保守型代理 BC 指標)")

if __name__ == "__main__":
    asyncio.run(main())
