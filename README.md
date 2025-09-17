# FinBERT 金融情感分析工具

基於 ProsusAI/finbert 模型的完整金融情感分析解決方案，支援中英文文本分析、增強解釋功能和交易策略生成。

## 🎯 專案特色

-   **🔍 統一介面**：單一主程式整合所有功能
-   **🌐 多語言支援**：英文、中文、中英混合文本分析
-   **📊 增強分析**：提供分析依據、文本總結和結果解釋
-   **💼 交易策略**：基於情感分析生成交易建議
-   **🛠️ 多種模式**：互動式、批次處理、測試套件等

## 🚀 快速開始

### 1. 環境準備

```bash
# 安裝依賴
pip install -r requirements.txt

# 設置 HuggingFace API Token
export HF_TOKEN="your_huggingface_token_here"
```

> 詳細環境設置請參考 [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)

### 2. 基本使用

```bash
# 顯示所有可用選項
python finbert_main.py --help

# 互動模式（推薦新手）
python finbert_main.py --interactive

# 快速測試
python finbert_main.py --simple-test

# 分析單一文本
python finbert_main.py --analyze "台積電股價創新高"
```

## 📋 完整功能列表

### 🎮 基本分析功能

```bash
# 基本情感分析
python finbert_main.py --analyze "Apple stock surged 15%"

# 增強分析（包含總結、依據、解釋）
python finbert_main.py --explain "Tesla shares plummeted due to delays"

# 情感評分分析（-1 到 1 數值）
python finbert_main.py --sentiment-score "NVIDIA stock hits new high"
```

### 💼 交易策略功能

```bash
# 交易策略分析（價格,RSI,情感評分）
python finbert_main.py --trading-strategy "100.0,0.3,0.8"
```

### 🧪 測試套件

```bash
# 簡化測試（9個精選案例）
python finbert_main.py --simple-test

# 增強測試（包含情感評分和交易策略）
python finbert_main.py --enhanced-test

# 中英文對比測試
python finbert_main.py --language-comparison
```

### 🎯 互動模式

```bash
python finbert_main.py --interactive
```

互動模式支援特殊指令：

-   `explain:文本` - 增強分析
-   `score:文本` - 情感評分
-   `strategy:價格,RSI,情感` - 交易策略
-   `examples` - 查看範例

## 🔍 增強分析功能

新的增強分析功能提供詳細的分析報告：

### 📄 文本總結

-   自動提取關鍵實體（公司名稱等）
-   識別金融關鍵詞
-   智能總結文本核心內容

### 🔍 分析依據

-   **正面關鍵詞**：上漲、創新高、surge、strong 等
-   **負面關鍵詞**：下跌、重挫、plummet、disappointing 等
-   **中性關鍵詞**：發布、宣布、announce、report 等
-   **語言分析**：中文/英文/中英混合檢測

### 💡 結果解釋

-   **信心度分析**：解釋模型為什麼有這個信心度
-   **競爭分析**：與次高分類的比較
-   **語言風險**：針對非英文文本的特別提醒

### 使用範例

```bash
# 增強分析範例
python finbert_main.py --explain "台積電股價創新高，投資人信心大增，財報表現強勁"
```

**輸出示例**：

```
📝 輸入文本: 台積電股價創新高，投資人信心大增，財報表現強勁
🌐 語言類型: 中文
🎯 分析結果: 🟡 中性 (89.3%)

📄 文本總結: 關鍵詞：財報, 強勁, 投資, 信心, 股價
🔍 分析依據: 正面關鍵詞：信心大增, 創新高, 表現強勁
💡 結果解釋: 模型以非常高的信心度判斷此文本為neutral情感
⚠️  語言提醒: FinBERT主要針對英文訓練，建議翻譯後再分析
```

## 💼 交易策略功能

基於價格、RSI 指標和情感評分生成 AI 交易建議：

```bash
# 看漲情境：低RSI + 正面情感
python finbert_main.py --trading-strategy "100.0,0.3,0.8"

# 看跌情境：高RSI + 負面情感
python finbert_main.py --trading-strategy "100.0,0.8,-0.6"
```

**輸出動作代碼**：

-   `0` = 賣出 (Sell)
-   `1` = 持有 (Hold)
-   `2` = 買入 (Buy)

## 📊 語言處理能力

根據測試結果，不同語言的處理表現：

| 語言類型 | 準確性        | 信心度 | 建議使用 |
| -------- | ------------- | ------ | -------- |
| 🇺🇸 英文  | 高 (94-97%)   | 74-97% | ✅ 推薦  |
| 🇹🇼 中文  | 低 (傾向中性) | 85-92% | ⚠️ 謹慎  |
| 🔀 混合  | 中等          | 90%+   | ⚠️ 謹慎  |

### 💡 使用建議

#### ✅ 英文金融文本

-   **直接使用**：結果準確可信
-   **適用場景**：英文新聞、報告、社交媒體分析

#### ⚠️ 中文金融文本

-   **方案 1**：翻譯成英文後分析（推薦）
-   **方案 2**：結合關鍵詞分析驗證
-   **方案 3**：使用專門的中文金融模型
-   **注意**：FinBERT 對中文文本傾向判斷為「中性」

## 🛠️ 程式整合

### Python API 使用

```python
from finbert_main import FinBERTAnalyzer

# 初始化分析器
analyzer = FinBERTAnalyzer()

# 基本情感分析
result = analyzer.analyze_sentiment("Apple stock surged 15%")
print(analyzer.format_result(result))

# 增強分析
enhanced_result = analyzer.analyze_sentiment_with_explanation("Tesla stock plummeted")
print(analyzer.format_result(enhanced_result))

# 情感評分
score_result = analyzer.analyze_sentiment_with_score("NVIDIA hits new high")
print(f"情感評分: {score_result['sentiment_score']}")

# 交易策略
strategy_result = analyzer.generate_trading_strategy(100.0, 0.3, 0.8)
print(f"建議動作: {strategy_result['action_label']}")
```

### 批次處理範例

```python
texts = [
    "Apple earnings beat expectations",
    "Tesla production delays continue",
    "Fed announces rate decision"
]

for i, text in enumerate(texts):
    result = analyzer.analyze_sentiment_with_explanation(text, f"批次_{i+1}")
    print(f"文本 {i+1}: {analyzer.format_result(result, detailed=False)}")
```

## 📁 專案結構

```
stock-project/
├── finbert_main.py              # 🎯 主程式（統一介面）
├── app.py                       # 🌐 Flask Web 介面
├── requirements.txt             # 📦 依賴列表
├── README.md                    # 📚 主要說明文件
├── README_Enhanced_Analysis.md  # 🔍 增強分析詳細說明
└── ENVIRONMENT_SETUP.md         # ⚙️ 環境設置指南
```

## 📈 效能指標

-   **平均回應時間**：3-5 秒
-   **API 成功率**：100%
-   **支援語言**：英文、中文、中英混合
-   **關鍵詞庫**：涵蓋常見中英文金融詞彙
-   **實體識別**：主要科技和金融公司

## 🔮 應用場景

### 1. 量化交易

```python
# 結合技術指標進行自動化交易決策
news_sentiment = analyzer.analyze_sentiment_with_score(latest_news)
trading_action = analyzer.generate_trading_strategy(
    current_price, rsi_indicator, news_sentiment['sentiment_score']
)
```

### 2. 風險管理

```python
# 基於新聞情感評估市場風險
risk_score = analyzer.analyze_sentiment_with_score(market_news)
if risk_score['sentiment_score'] < -0.5:
    print("高風險警告：市場情緒極度負面")
```

### 3. 投資組合優化

```python
# 批量分析多支股票的情感和策略
for stock in portfolio:
    sentiment = analyzer.analyze_sentiment_with_score(stock.news)
    strategy = analyzer.generate_trading_strategy(
        stock.price, stock.rsi, sentiment['sentiment_score']
    )
    stock.recommended_action = strategy['parsed_action']
```

## ⚠️ 注意事項

1. **API 限制**：請注意 HuggingFace API 的使用限制
2. **網路依賴**：功能需要穩定的網路連接
3. **模型限制**：FinBERT 主要針對英文金融文本訓練
4. **投資建議**：AI 生成的交易策略僅供參考，不構成投資建議

## 📞 技術支援

-   **模型來源**：[ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)
-   **API 文檔**：[HuggingFace Inference API](https://huggingface.co/docs/api-inference/index)
-   **增強分析說明**：查看 [README_Enhanced_Analysis.md](README_Enhanced_Analysis.md)
-   **環境設置**：查看 [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)

## 📄 更新日誌

-   **v3.0**：整合所有功能到統一主程式
-   **v3.0**：新增增強分析功能（總結、依據、解釋）
-   **v3.0**：改進關鍵詞識別和實體提取
-   **v2.0**：新增情感評分、交易策略生成功能
-   **v1.0**：基本情感分析和測試套件

---

**最後更新**：2025 年 9 月 17 日  
**版本**：v3.0  
**作者**：基於 ProsusAI/finbert 擴展開發
