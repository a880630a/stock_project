# 專案執行指南與問題分析報告

## 目錄
1. [環境設置步驟](#環境設置步驟)
2. [執行方式](#執行方式)
3. [發現的問題與錯誤](#發現的問題與錯誤)
4. [改善建議](#改善建議)
5. [下一步開發計畫](#下一步開發計畫)

---

## 環境設置步驟

### 1. 建立虛擬環境（推薦）

```bash
# 進入專案目錄
cd /Users/xian/xian_ws/master_project/stock_project

# 建立虛擬環境
python3 -m venv venv

# 啟動虛擬環境
source venv/bin/activate  # macOS/Linux
```

### 2. 安裝依賴

```bash
pip install -r requirements.txt
```

### 3. 設置環境變數

```bash
# 複製環境變數範例檔案
cp .env.example .env

# 編輯 .env 檔案，填入您的 Hugging Face Token
# 從 https://huggingface.co/settings/tokens 獲取
```

**.env 檔案內容：**
```
HF_TOKEN=hf_您的真實token
```

### 4. 驗證安裝

```bash
# 驗證 Python 版本
python3 --version

# 驗證依賴安裝
pip list | grep -E "huggingface|flask|torch|pandas"
```

---

## 執行方式

### 方式一：Flask API 服務（主要）

```bash
# 啟動 API 服務
python app.py

# 服務會在 http://localhost:5000 啟動
```

**測試 API：**
```bash
# 健康檢查
curl http://localhost:5000/api/health

# 情感分析
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Apple stock surged 15% after earnings"}'

# 交易策略
curl -X POST http://localhost:5000/api/trading-strategy \
  -H "Content-Type: application/json" \
  -d '{"price": 580, "rsi": 0.35, "sentiment": 0.6}'
```

### 方式二：FinBERT 命令列工具

```bash
# 互動模式
python finbert_main.py --interactive

# 分析單一文本
python finbert_main.py --analyze "台積電股價創新高"

# 情感評分
python finbert_main.py --sentiment-score "Market is bullish"

# 交易策略分析
python finbert_main.py --trading-strategy "100,0.5,0.3"

# 增強分析（含解釋）
python finbert_main.py --explain "Tesla shares plummeted"

# 簡化測試
python finbert_main.py --simple-test

# 中英文對比
python finbert_main.py --language-comparison
```

### 方式三：測試腳本

```bash
# MCP 代理測試（需要網路連接）
python test_mcp_agents.py

# 優化功能測試
python test_optimized_functionality.py

# 中文字體測試
python test_chinese_font.py
```

### 方式四：Python 模組整合

```python
import asyncio
from agents import LLMAgent

async def main():
    # 建立代理
    agent = LLMAgent(role='aggressive')

    # 預測交易動作
    action, strategy = await agent.predict(stock_code='2330')
    print(f"動作: {action}, 策略: {strategy}")

asyncio.run(main())
```

---

## 發現的問題與錯誤

### 🔴 嚴重問題

#### 1. 缺少 .env 檔案
**位置：** 專案根目錄
**問題：** `.env` 檔案不存在，導致 `HF_TOKEN` 未設置
**影響：** 所有模組初始化都會失敗

**修復方式：**
```bash
cp .env.example .env
# 編輯 .env，填入真實的 HF_TOKEN
```

#### 2. agents.py 缺少 `random` 導入
**位置：** `agents.py:176`
**問題：** 使用了 `random.choice()` 但未導入 `random` 模組
```python
selected_month = random.choice(monthly_data)  # NameError: name 'random' is not defined
```

**修復方式：** 在檔案開頭添加
```python
import random
```

#### 3. agents.py 主程式區塊使用異步函數錯誤
**位置：** `agents.py:1105-1106`
**問題：** 在 `__main__` 中直接呼叫 `agent.predict(obs)` 而非 `await agent.predict(obs)`
```python
# 錯誤：predict 是 async 函數
action_a, strat_a = agent_a.predict(obs)  # 會返回 coroutine，不是結果
```

**修復方式：**
```python
import asyncio

if __name__ == "__main__":
    async def main():
        agent_a = LLMAgent(role="aggressive")
        obs = [100.0, 0.5, 0.8]
        action_a, strat_a = await agent_a.predict(obs)
        print(f"Action: {action_a}")

    asyncio.run(main())
```

#### 4. twse_mcp_client.py 主程式區塊非異步呼叫
**位置：** `twse_mcp_client.py:606-615`
**問題：** 直接呼叫異步函數而未使用 `asyncio.run()`
```python
# 錯誤：這些是異步函數
stock_data = get_stock_data(test_code)  # 會返回 coroutine
```

**修復方式：**
```python
if __name__ == "__main__":
    import asyncio

    async def main():
        stock_data = await get_stock_data("2330")
        print(f"股票數據: {stock_data}")

    asyncio.run(main())
```

### 🟡 中等問題

#### 5. app.py 敏感資訊洩漏
**位置：** `app.py:35`
**問題：** 直接 print API key
```python
print(api_key)  # 安全風險：不應該印出 API key
```

**修復方式：** 刪除此行或改為
```python
print(f"HF_TOKEN 已設置: {'是' if api_key else '否'}")
```

#### 6. requirements.txt 重複定義
**位置：** `requirements.txt:7` 和 `requirements.txt:24`
**問題：** `numpy>=1.21.0` 重複定義

**修復方式：** 移除重複的行

#### 7. 缺少錯誤處理的邊界情況
**位置：** `finbert_main.py:342-366`
**問題：** `_convert_classification_to_score()` 未處理空列表的邊界情況
```python
if not classification_result:
    return 0.0
# 但如果 classification_result 是空列表 []，max() 會報錯
```

### 🟢 輕微問題

#### 8. 未使用的導入
**位置：** `finbert_main.py:26`
```python
import re  # 在頂部導入，但在方法內又重複導入
```

#### 9. 魔術數字
**位置：** 多處
**問題：** 硬編碼的閾值如 `0.3`, `0.5`, `0.7` 應該定義為常數

#### 10. 日誌級別不一致
**位置：** `twse_mcp_client.py` vs `app.py`
**問題：** 不同模組使用不同的日誌配置方式

---

## 改善建議

### 立即修復（必須）

#### 1. 修復 agents.py 的 random 導入

```python
# agents.py 開頭添加
import random
```

#### 2. 修復 agents.py 主程式區塊

```python
# agents.py 底部修改為
if __name__ == "__main__":
    import asyncio

    async def test_agents():
        np.random.seed(42)

        agent_a = LLMAgent(role="aggressive")
        agent_b = LLMAgent(role="defensive")
        obs = [100.0, 0.5, 0.8]

        action_a, strat_a = await agent_a.predict(obs)
        action_b, strat_b = await agent_b.predict(obs)

        print(f"Aggressive Agent: Action {action_a} - {strat_a}")
        print(f"Defensive Agent: Action {action_b} - {strat_b}")

        # 測試學習
        mock_trajectories = [
            {'obs': [100.0, 0.3, 0.9], 'action': 2},
            {'obs': [100.0, 0.7, -0.5], 'action': 0},
            {'obs': [100.0, 0.5, 0.1], 'action': 1}
        ] * 5
        await agent_b.learn_from_other(mock_trajectories)

    asyncio.run(test_agents())
```

#### 3. 修復 twse_mcp_client.py 主程式區塊

```python
# twse_mcp_client.py 底部修改為
if __name__ == "__main__":
    import asyncio

    async def test_client():
        print("測試 TWSE MCP 客戶端...")
        test_code = "2330"

        print(f"\n=== 測試股票數據 ({test_code}) ===")
        stock_data = await get_stock_data(test_code)
        print(f"股票數據: {stock_data}")

        print(f"\n=== 測試財務洞察 ({test_code}) ===")
        financial_data = await get_financial_insights(test_code)
        print(f"財務洞察: {financial_data}")

        print("\n測試完成！")

    asyncio.run(test_client())
```

#### 4. 移除敏感資訊輸出

```python
# app.py:35 修改為
# print(api_key)  # 刪除此行
print(f"HF_TOKEN 已設置: {'是' if api_key else '否'}")
```

### 建議改進（推薦）

#### 1. 添加常數配置文件

建立 `config.py`：
```python
# config.py
class TradingConfig:
    RSI_OVERSOLD = 0.3
    RSI_OVERBOUGHT = 0.7
    SENTIMENT_POSITIVE_THRESHOLD = 0.3
    SENTIMENT_NEGATIVE_THRESHOLD = -0.3

class ModelConfig:
    FINBERT_MODEL = "ProsusAI/finbert"
    BC_EPOCHS = 10
    BC_BATCH_SIZE = 4
```

#### 2. 統一日誌配置

建立 `logging_config.py`：
```python
import logging

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('app.log')
        ]
    )
```

#### 3. 添加型別提示完整性

```python
from typing import Optional, List, Dict, Any, Tuple

def analyze_sentiment(self, text: str, test_name: str = "") -> Dict[str, Any]:
    ...
```

---

## 下一步開發計畫

### 第一階段：修復錯誤（立即）

- [ ] 建立 `.env` 檔案並設置 `HF_TOKEN`
- [ ] 修復 `agents.py` 的 `random` 導入問題
- [ ] 修復 `agents.py` 和 `twse_mcp_client.py` 的異步呼叫問題
- [ ] 移除 `app.py` 中的敏感資訊輸出
- [ ] 移除 `requirements.txt` 中的重複依賴

### 第二階段：程式碼品質改善（短期）

- [ ] 添加統一的配置管理（config.py）
- [ ] 統一日誌配置
- [ ] 添加完整的型別提示
- [ ] 添加單元測試
- [ ] 添加錯誤邊界處理

### 第三階段：功能擴展（中期）

- [ ] 實作持久化存儲層（SQLite）
- [ ] 添加模型檢查點保存
- [ ] 整合即時新聞 API
- [ ] 實作回測系統
- [ ] 添加更多技術指標（MACD, Bollinger Bands）

### 第四階段：產品化（長期）

- [ ] 建立前端儀表板
- [ ] 實作預警系統
- [ ] 添加用戶認證
- [ ] 部署到雲端服務
- [ ] 實作 CI/CD 流程

---

## 快速開始檢查清單

```bash
# 1. 檢查環境
python3 --version  # 需要 Python 3.8+

# 2. 建立虛擬環境
python3 -m venv venv
source venv/bin/activate

# 3. 安裝依賴
pip install -r requirements.txt

# 4. 設置環境變數
cp .env.example .env
# 編輯 .env，填入 HF_TOKEN

# 5. 驗證設置
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('HF_TOKEN:', 'OK' if os.getenv('HF_TOKEN') else 'Missing')"

# 6. 執行測試
python finbert_main.py --simple-test

# 7. 啟動服務
python app.py
```

---

*文檔生成日期：2025-11-26*
