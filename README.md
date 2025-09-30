# 🚀 智能股票交易系統

一個整合 FinBERT 情感分析和 TWSE MCP 真實數據的智能股票交易代理系統。

## 📋 專案概述

本專案結合了：

-   **FinBERT 情感分析**：基於 Transformer 的金融文本情感分析
-   **TWSE MCP 客戶端**：台灣證券交易所真實數據獲取
-   **LLM 交易代理**：支援 aggressive/defensive 角色的智能交易決策
-   **行為克隆學習**：從歷史軌跡學習最佳交易策略

## 🏗️ 專案結構

```
stock-project/
├── 📁 核心程式
│   ├── finbert_main.py              # FinBERT 情感分析主程式
│   ├── agents.py                    # LLM 交易代理核心邏輯
│   ├── twse_mcp_client.py          # TWSE MCP 客戶端
│   └── app.py                       # Flask Web API
├── 📁 測試程式
│   ├── test_mcp_agents.py          # MCP 代理整合測試
│   └── test_optimized_functionality.py # 優化功能完整測試
├── 📁 配置文件
│   ├── requirements.txt             # Python 依賴
│   └── ENVIRONMENT_SETUP.md         # 環境設置指南
├── 📁 文檔
│   ├── README.md                    # 本文件
│   └── OPTIMIZATION_SUMMARY.md      # 最新優化總結
└── 📁 訓練結果
    ├── bc_loss_aggressive.png       # BC 訓練指標圖表
    └── bc_loss_defensive.png
```

## 🚀 快速開始

### 1. 環境設置

```bash
# 安裝依賴
pip install -r requirements.txt

# 設置環境變數（參考 ENVIRONMENT_SETUP.md）
export HF_TOKEN="your_huggingface_token"
export MCP_URL="https://TW-Stock-MCP-Server.fastmcp.app/mcp"
```

### 2. 基本使用

#### FinBERT 情感分析

```bash
# 基本分析
python finbert_main.py --analyze "Apple stock shows strong performance"

# 增強分析（包含詳細解釋）
python finbert_main.py --explain "市場前景看好，建議買入"

# 情感評分
python finbert_main.py --sentiment-score "The market is volatile"
```

#### 智能交易代理

```python
import asyncio
from agents import LLMAgent

async def main():
    # 創建積極型代理
    agent = LLMAgent(role='aggressive')

    # 使用真實數據預測台積電
    action, strategy = await agent.predict(obs=None, stock_code='2330')
    print(f"預測動作: {action} ({'買入' if action == 2 else '賣出' if action == 0 else '持有'})")
    print(f"策略: {strategy}")

    # 從真實數據學習
    await agent.learn_from_other(other_trajectories=[], stock_code='2330')

asyncio.run(main())
```

#### Web API 服務

```bash
# 啟動 Flask API
python app.py

# API 端點
# POST /api/analyze - 情感分析
# POST /api/explain - 增強分析
# POST /api/score - 情感評分
# POST /api/strategy - 交易策略
```

## 🎯 核心功能

### 1. FinBERT 情感分析

-   **多語言支援**：英文、中文、中英混合
-   **詳細分析**：包含信心度、分析依據、結果解釋
-   **交易策略**：基於價格、RSI、情感的智能決策

### 2. TWSE MCP 整合

-   **真實數據**：獲取台灣證券交易所實時數據
-   **12 月歷史數據**：用於計算真實 RSI 指標
-   **滾動平均 RSI**：使用 pandas 滾動平均提升準確性
-   **月漲幅情感**：基於月漲跌幅計算市場情感（>5% 為 0.7）

### 3. 智能交易代理

-   **角色化決策**：
    -   `aggressive`：積極型，偏好高情感買入
    -   `defensive`：保守型，偏好低 RSI 賣出/持有
-   **行為克隆學習**：從歷史軌跡學習最佳策略
-   **多樣化軌跡**：生成 15 筆多樣化觀察數據，動作概率 [0.3, 0.4, 0.3]

### 4. 優化特性

-   **熵值提升**：解決單一 obs/動作問題，提升預測多樣性
-   **智能回退**：多層回退機制確保系統穩定性
-   **BC 訓練監控**：生成訓練指標圖表監控學習效果

## 📊 測試驗證

### 運行測試

```bash
# MCP 代理整合測試
python test_mcp_agents.py

# 完整優化功能測試
python test_optimized_functionality.py
```

### 測試覆蓋

-   ✅ 12 月數據獲取與真實 RSI 計算
-   ✅ 15 筆軌跡生成與動作概率分佈 [0.3, 0.4, 0.3]
-   ✅ 基於月漲幅的 sentiment 計算（>5% 為 0.7）
-   ✅ 熵值提升與多樣性改善（熵值達到 0.722）
-   ✅ 真實數據整合與預測功能
-   ✅ BC 訓練指標監控

## 🔧 技術架構

### 數據流程

```
TWSE MCP Server → twse_mcp_client.py → agents.py → FinBERT → 交易決策
                                    ↓
                              BC 學習 ← 歷史軌跡
```

### 關鍵技術

-   **FinBERT**：Hugging Face Transformers
-   **MCP 協議**：fastmcp 客戶端
-   **行為克隆**：imitation 庫
-   **數據處理**：pandas, numpy
-   **可視化**：matplotlib
-   **Web API**：Flask + CORS

## 📈 性能指標

-   **軌跡數量**：10 → 15 (+50%)
-   **數據來源**：5 月 → 12 月 (+140%)
-   **RSI 計算**：簡單平均 → 滾動平均（更準確）
-   **熵值提升**：達到 0.722（最大 1.585）
-   **動作分佈**：精確控制 [0.3, 0.4, 0.3]

## 🛠️ 開發指南

### 添加新股票

```python
# 在 twse_mcp_client.py 中支援的股票代碼
supported_stocks = ['2330', '2317', '2454', '2412', '2882']
```

### 自定義交易策略

```python
# 在 agents.py 中修改 _fallback_predict 方法
def _fallback_predict(self, price, rsi, sentiment, stock_code):
    # 自定義決策邏輯
    pass
```

### 擴展 API 端點

```python
# 在 app.py 中添加新端點
@app.route('/api/custom', methods=['POST'])
def custom_analysis():
    # 自定義分析邏輯
    pass
```

## 🔍 故障排除

### 常見問題

1. **MCP 連接失敗**：檢查 `MCP_URL` 環境變數
2. **FinBERT 載入失敗**：檢查 `HF_TOKEN` 設置
3. **依賴安裝問題**：使用 `pip install -r requirements.txt`
4. **多月數據不足**：系統會自動回退到單日數據

### 日誌檢查

```bash
# 查看詳細日誌
export PYTHONPATH=.
python -c "import logging; logging.basicConfig(level=logging.DEBUG)"
```

## 📚 相關文檔

-   [環境設置指南](ENVIRONMENT_SETUP.md)
-   [優化總結報告](OPTIMIZATION_SUMMARY.md)
-   [FinBERT 官方文檔](https://huggingface.co/ProsusAI/finbert)
-   [TWSE MCP Server](https://github.com/your-repo/twse-mcp-server)

## 🤝 貢獻指南

1. Fork 專案
2. 創建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟 Pull Request

## 📄 授權條款

本專案採用 MIT 授權條款 - 詳見 [LICENSE](LICENSE) 文件

## 🙏 致謝

-   [FinBERT](https://huggingface.co/ProsusAI/finbert) - 金融情感分析模型
-   [Hugging Face](https://huggingface.co/) - Transformer 模型平台
-   [TWSE](https://www.twse.com.tw/) - 台灣證券交易所數據
-   [fastmcp](https://github.com/fastmcp/fastmcp) - MCP 協議實現

---

**最後更新**：2025 年 9 月 30 日  
**版本**：v4.0  
**狀態**：✅ 生產就緒
