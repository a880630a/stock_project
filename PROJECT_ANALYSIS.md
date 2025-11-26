# 股票交易智能系統 - 專案分析報告

## 📋 目錄

1. [專案概述](#專案概述)
2. [專案結構](#專案結構)
3. [核心模組分析](#核心模組分析)
4. [系統架構](#系統架構)
5. [數據流程](#數據流程)
6. [API 介面](#api-介面)
7. [ML/AI 組件](#mlai-組件)
8. [依賴項與外部服務](#依賴項與外部服務)
9. [測試架構](#測試架構)
10. [待開發功能建議](#待開發功能建議)

---

## 專案概述

本專案是一個**智能股票交易決策系統**，整合了以下核心技術：

- **FinBERT 情感分析**：使用 Hugging Face 的金融領域 BERT 模型進行文本情感分析
- **LLM 交易代理**：基於行為克隆（Behavioral Cloning）的強化學習代理
- **TWSE MCP 數據整合**：透過 MCP 協議獲取台灣證券交易所即時數據
- **Flask REST API**：提供 Web 服務介面

### 技術亮點

| 特性 | 說明 |
|------|------|
| 即時數據 | 透過 TWSE MCP Server 獲取 12 個月歷史數據 |
| 情感分析 | 使用 ProsusAI/finbert 模型進行金融文本分析 |
| 學習能力 | 支援 BC（行為克隆）和簡單模式學習 |
| 雙角色策略 | 激進型（aggressive）與防守型（defensive）|
| 容錯機制 | 多層級的降級與備援機制 |

---

## 專案結構

```
stock_project/
├── 核心模組 (5 個檔案，共 ~3,835 行程式碼)
│   ├── finbert_main.py         (1,362 行) - FinBERT 情感分析引擎
│   ├── agents.py               (1,118 行) - LLM 交易代理
│   ├── twse_mcp_client.py      (616 行)   - TWSE 數據整合客戶端
│   ├── app.py                  (417 行)   - Flask Web API
│   └── test_optimized_functionality.py (236 行) - 整合測試
│
├── 測試腳本 (3 個檔案)
│   ├── test_mcp_agents.py      (41 行)  - MCP 代理基本測試
│   ├── test_optimized_functionality.py (236 行) - 優化功能測試
│   └── test_chinese_font.py    (45 行)  - 中文字體測試
│
├── 配置文件
│   ├── requirements.txt         - Python 依賴項
│   ├── .env.example            - 環境變數範本
│   ├── .gitignore              - Git 忽略規則
│   └── .claude/settings.local.json - Claude Code 設定
│
├── 文檔
│   ├── README.md               - 主要文檔
│   ├── PROJECT_STRUCTURE.md    - 結構概覽
│   ├── ENVIRONMENT_SETUP.md    - 環境設定指南
│   └── OPTIMIZATION_SUMMARY.md - 性能優化摘要
│
└── 輸出檔案
    ├── bc_loss_aggressive.png  - 激進型訓練曲線
    └── bc_loss_defensive.png   - 防守型訓練曲線
```

---

## 核心模組分析

### 1. FinBERT 情感分析器 (`finbert_main.py`)

**職責**：金融文本情感分析與交易策略生成

#### 主要類別：`FinBERTAnalyzer`

```python
class FinBERTAnalyzer:
    def analyze_sentiment(text)           # 基本情感分類
    def analyze_sentiment_with_score(text) # 數值化情感分數 (-1 到 1)
    def analyze_sentiment_with_explanation(text) # 帶解釋的情感分析
    def generate_trading_strategy(price, rsi, sentiment) # 交易策略生成
```

#### 情感分類對應

| 標籤 | 分數範圍 | 說明 |
|------|----------|------|
| Positive | 0 到 1 | 正面情緒 |
| Negative | -1 到 0 | 負面情緒 |
| Neutral | 0 | 中性情緒 |

---

### 2. LLM 交易代理 (`agents.py`)

**職責**：智能交易決策與行為克隆學習

#### 主要類別：`LLMAgent`

```python
class LLMAgent:
    def __init__(role='aggressive')  # 初始化代理（激進/防守）
    async def predict(obs, stock_code) # 預測交易動作
    async def learn_from_other(trajectories, stock_code) # 從軌跡學習
```

#### 交易動作定義

| 動作碼 | 說明 | 機率分佈 |
|--------|------|----------|
| 0 | 賣出 (Sell) | 30% |
| 1 | 持有 (Hold) | 40% |
| 2 | 買入 (Buy) | 30% |

#### 角色策略差異

| 特性 | 激進型 (Aggressive) | 防守型 (Defensive) |
|------|---------------------|-------------------|
| 風險偏好 | 高風險高報酬 | 低風險穩健 |
| RSI 買入閾值 | < 0.35 | < 0.25 |
| RSI 賣出閾值 | > 0.65 | > 0.75 |
| 情感權重 | 較高 | 較低 |

---

### 3. TWSE MCP 客戶端 (`twse_mcp_client.py`)

**職責**：獲取台灣證券交易所即時市場數據

#### 主要類別：`TWSEMCPClient`

```python
class TWSEMCPClient:
    async def get_monthly_stock_data(code, months=12) # 12個月歷史數據
    async def get_stock_data(code)                    # 整合股票數據
    def calculate_rsi(prices, period=14)              # RSI 計算
```

#### 支援的股票代碼

| 代碼 | 公司名稱 |
|------|----------|
| 2330 | 台積電 (TSMC) |
| 2317 | 華碩 (ASUSTEK) |
| 2454 | 聯發科 (MediaTek) |
| 2412 | 中華電信 |
| 2882 | 國泰金控 |

#### RSI 計算方法

```
1. 計算價格變動百分比 (pct_change)
2. 分離漲幅 (gains) 與跌幅 (losses)
3. 計算 14 日滾動平均
4. RS = avg_gain / avg_loss
5. RSI = 100 - (100 / (1 + RS))
6. 正規化至 0-1 範圍
```

---

### 4. Flask Web API (`app.py`)

**職責**：提供 RESTful API 服務介面

#### 服務配置

- **端口**：5000
- **CORS**：已啟用
- **格式**：JSON

---

## 系統架構

```
┌─────────────────────────────────────────────────────────────┐
│                      客戶端層 (Client Layer)                 │
│              (Web API, CLI, Python 整合)                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                    應用層 (Application Layer)                │
│   ┌────────────────┐    ┌────────────────┐                 │
│   │  app.py        │    │  finbert_main  │                 │
│   │  (Flask API)   │    │  (情感分析)     │                 │
│   └────────────────┘    └────────────────┘                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                   代理層 (Agent Layer)                       │
│   ┌─────────────────────────────────────────────────────┐   │
│   │        agents.py (LLMAgent)                         │   │
│   │  • 交易決策（激進/防守）                              │   │
│   │  • 軌跡管理                                          │   │
│   │  • 學習機制（BC + 簡單學習）                          │   │
│   └──────┬──────────────────────────┬──────────────────┘   │
└──────────┼──────────────────────────┼────────────────────────┘
           │                          │
    ┌──────▼──────┐          ┌────────▼────────┐
    │ FinBERT     │          │ TWSE MCP Client │
    │ Analyzer    │          │ (數據獲取)       │
    └──────────────┘          └────────┬────────┘
                                       │
                        ┌──────────────▼──────────────┐
                        │   TWSE MCP Server           │
                        │   (即時市場數據)             │
                        └─────────────────────────────┘
```

---

## 數據流程

### 預測流程

```
用戶請求 (股票代碼)
    ↓
[TWSE MCP Client]
    ├→ 獲取 12 個月歷史數據
    ├→ 計算 RSI (14 日滾動平均)
    └→ 計算情感分數 (根據月成長率)
    ↓
[LLMAgent.predict()]
    ├→ 觀察狀態: [price, RSI, sentiment]
    ├→ FinBERT 策略生成
    └→ 備援: 規則式決策
    ↓
交易決策輸出
    ├→ action: 0=賣, 1=持有, 2=買
    └→ strategy: 策略說明文字
```

### 學習流程

```
軌跡樣本 (15 條)
    ├→ obs: [price, rsi, sentiment]
    ├→ action: [0, 1, 2]
    └→ 數據驗證
    ↓
[進階學習 - BC 訓練]
    ├→ 專家示範建立
    ├→ Policy Network 訓練
    ├→ Loss 計算與反向傳播
    └→ 熵值監控
    ↓ (失敗時)
[簡單學習 - 模式分析]
    ├→ 成功動作條件提取
    ├→ 狀態-動作映射
    └→ 統計分析
    ↓
策略更新
    └→ 新決策應用
```

---

## API 介面

### REST API 端點

| 端點 | 方法 | 輸入 | 輸出 | 說明 |
|------|------|------|------|------|
| `/api/health` | GET | - | `{status, timestamp, analyzer_ready}` | 健康檢查 |
| `/api/analyze` | POST | `{text}` | `{predictions[], best_prediction, language}` | 情感分類 |
| `/api/analyze-score` | POST | `{text}` | `{sentiment_score, category, label}` | 數值情感 |
| `/api/trading-strategy` | POST | `{price, rsi, sentiment}` | `{recommendation, action_code, analysis}` | 交易策略 |
| `/api/batch-analyze` | POST | `{texts[]}` | `{results[], successful_count}` | 批次分析 |

### 使用範例

```bash
# 健康檢查
curl http://localhost:5000/api/health

# 情感分析
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "台積電營收創新高"}'

# 交易策略
curl -X POST http://localhost:5000/api/trading-strategy \
  -H "Content-Type: application/json" \
  -d '{"price": 580, "rsi": 0.35, "sentiment": 0.6}'
```

### Python 模組介面

```python
# FinBERT 使用
from finbert_main import FinBERTAnalyzer
analyzer = FinBERTAnalyzer(api_key)
result = analyzer.analyze_sentiment("市場表現強勁")
strategy = analyzer.generate_trading_strategy(price=580, rsi=0.35, sentiment=0.6)

# Agent 使用
from agents import LLMAgent
agent = LLMAgent(role='aggressive')
action, strategy = await agent.predict(stock_code='2330')
await agent.learn_from_other(trajectories)

# MCP Client 使用
from twse_mcp_client import TWSEMCPClient
async with TWSEMCPClient() as client:
    data = await client.get_stock_data('2330')
```

---

## ML/AI 組件

### 1. FinBERT 情感分析管線

```
輸入文本
    ↓
[語言檢測] (英文/中文/混合)
    ↓
[Hugging Face Tokenization]
    ↓
[ProsusAI/finbert 推論]
    ├→ Positive (信心度)
    ├→ Negative (信心度)
    └→ Neutral (信心度)
    ↓
[分數轉換: -1 到 1]
    ↓
[策略規則應用]
    ↓
交易建議 (買/持有/賣)
```

### 2. 行為克隆 (BC) 訓練管線

| 參數 | 數值 | 說明 |
|------|------|------|
| 軌跡數量 | 15 | 每次學習 session |
| 動作分佈 | [0.3, 0.4, 0.3] | 賣/持有/買 機率 |
| 訓練熵值 | 0.722 | 目前達成 (最大: 1.585) |
| 正則化 | L2 | 防止過擬合 |
| 數據來源 | 12 個月 | TWSE 真實數據 |

### 3. 備援機制

| 層級 | 主要方案 | 備援方案 |
|------|----------|----------|
| 學習 | BC 進階學習 | 簡單模式學習 |
| 數據 | 12 個月數據 | 單日數據 |
| RSI | MCP 計算 | 模擬 RSI |
| 決策 | LLM 生成 | 規則式決策 |

---

## 依賴項與外部服務

### Python 依賴項

```
# 核心 ML
transformers>=4.21.0        # Hugging Face 模型
torch>=1.12.0               # PyTorch 後端
huggingface_hub>=0.16.0     # HF API 客戶端

# 強化學習
imitation>=0.4.0            # 行為克隆
gymnasium>=0.28.0           # RL 環境

# 數據處理
pandas>=1.5.0               # 數據操作
numpy>=1.21.0               # 數值計算
matplotlib>=3.5.0           # 視覺化

# Web & API
flask>=2.0.0                # Web 框架
flask-cors>=3.0.0           # CORS 支援
requests>=2.25.0            # HTTP 客戶端

# MCP 整合
fastmcp>=2.12.0             # MCP 協議客戶端
tenacity>=9.0.0             # 重試邏輯
```

### 外部服務

| 服務 | 用途 | 設定 |
|------|------|------|
| Hugging Face Inference API | FinBERT 情感分析 | `HF_TOKEN` 環境變數 |
| TWSE MCP Server | 即時股市數據 | `https://TW-Stock-MCP-Server.fastmcp.app/mcp` |

---

## 測試架構

### 測試檔案

| 檔案 | 行數 | 說明 |
|------|------|------|
| `test_mcp_agents.py` | 41 | MCP 代理基本測試 |
| `test_optimized_functionality.py` | 236 | 全面優化測試 |
| `test_chinese_font.py` | 45 | 中文字體配置測試 |

### 測試覆蓋

- ✅ 12 個月數據獲取
- ✅ RSI 計算準確性
- ✅ 月成長率情感計算
- ✅ 15 軌跡生成
- ✅ 動作機率分佈 [0.3, 0.4, 0.3]
- ✅ 熵值改善 (0.722)
- ✅ BC 訓練指標追蹤
- ✅ 備援機制強健性
- ✅ 多語言情感分析
- ✅ 角色決策差異

### 執行測試

```bash
# 基本整合測試
python test_mcp_agents.py

# 全面優化測試
python test_optimized_functionality.py

# 字體配置測試
python test_chinese_font.py
```

---

## 待開發功能建議

### 🔴 高優先級

#### 1. 持久化存儲層
**現狀**：系統目前為無狀態設計，所有數據在程序結束後消失
**建議**：
- 實作 SQLite/PostgreSQL 數據庫存儲
- 保存學習軌跡和模型狀態
- 實現歷史交易記錄查詢

```python
# 建議結構
class StorageLayer:
    def save_trajectories(agent_id, trajectories)
    def load_trajectories(agent_id)
    def save_model_state(agent_id, state)
    def get_trading_history(stock_code, date_range)
```

#### 2. 模型持久化與版本管理
**現狀**：BC 訓練後的模型未被保存
**建議**：
- 實作模型檢查點保存
- 添加模型版本控制
- 支援模型回滾

#### 3. 即時新聞情感分析
**現狀**：情感分數僅從月成長率計算
**建議**：
- 整合財經新聞 API（例如：Yahoo Finance、Google News）
- 即時分析新聞標題情感
- 結合多來源情感加權

---

### 🟡 中優先級

#### 4. 回測系統 (Backtesting)
**說明**：使用歷史數據驗證策略效果
**功能**：
- 選擇歷史時間段
- 模擬交易執行
- 計算績效指標（夏普比率、最大回撤等）
- 生成回測報告

```python
class BacktestEngine:
    def run_backtest(strategy, stock_code, start_date, end_date)
    def calculate_metrics()  # Sharpe, Max Drawdown, Win Rate
    def generate_report()
```

#### 5. 投資組合管理
**說明**：支援多股票組合決策
**功能**：
- 多股票權重分配
- 組合風險評估
- 資產再平衡建議
- 相關性分析

#### 6. 技術指標擴展
**現狀**：僅使用 RSI
**建議新增**：
- MACD（移動平均收斂散度）
- 布林帶 (Bollinger Bands)
- KDJ 指標
- 成交量分析

---

### 🟢 低優先級（長期規劃）

#### 7. 前端儀表板
**說明**：視覺化操作介面
**技術建議**：React/Vue + Chart.js
**功能**：
- 即時股價圖表
- 情感分析視覺化
- 交易信號提醒
- 歷史績效追蹤

#### 8. 預警系統
**說明**：條件觸發通知
**功能**：
- 價格突破預警
- RSI 極值提醒
- 情感大幅變化通知
- 支援 Email/LINE/Telegram

#### 9. 多模型集成
**說明**：整合多種 ML 模型
**建議**：
- 添加 LSTM 價格預測
- 整合其他情感模型（RoBERTa-Financial）
- 實作模型投票機制
- 動態權重調整

#### 10. A/B 測試框架
**說明**：比較不同策略效果
**功能**：
- 並行運行多策略
- 統計顯著性檢驗
- 自動選擇最優策略

---

### 📋 功能開發優先順序建議

| 順序 | 功能 | 預估工作量 | 價值 |
|------|------|------------|------|
| 1 | 持久化存儲層 | 中 | 高 |
| 2 | 模型持久化 | 低 | 高 |
| 3 | 即時新聞情感 | 中 | 高 |
| 4 | 回測系統 | 高 | 高 |
| 5 | 技術指標擴展 | 低 | 中 |
| 6 | 投資組合管理 | 高 | 中 |
| 7 | 前端儀表板 | 高 | 中 |
| 8 | 預警系統 | 中 | 中 |
| 9 | 多模型集成 | 高 | 低 |
| 10 | A/B 測試框架 | 中 | 低 |

---

## 結論

本專案已建立了一個功能完整的智能股票交易決策系統核心架構，具備：

1. **穩固的情感分析能力** - FinBERT 模型整合
2. **智能學習機制** - BC 行為克隆 + 簡單學習備援
3. **即時數據整合** - TWSE MCP 連接
4. **良好的容錯設計** - 多層級備援機制
5. **API 服務就緒** - Flask REST 介面

下一階段的重點應放在**數據持久化**和**回測驗證**，以確保系統在實際應用中的可靠性和策略效果驗證。

---

*文檔生成日期：2025-11-26*
*版本：1.0*
