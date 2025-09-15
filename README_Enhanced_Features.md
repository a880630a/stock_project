# FinBERT 增強功能說明

## 新增功能概覽

本次更新為 FinBERT 金融情感分析工具新增了以下功能：

### 1. 📊 情感評分分析 (Sentiment Scoring)

-   **功能**: 將情感分析結果轉換為 -1 到 1 的數值評分
-   **用途**: 量化情感強度，便於程式化處理
-   **評分範圍**:
    -   `-1.0`: 極度負面
    -   `0.0`: 中性
    -   `+1.0`: 極度正面

### 2. 💼 交易策略生成 (Trading Strategy)

-   **功能**: 基於價格、RSI 指標和情感評分生成交易建議
-   **輸入參數**:
    -   `price`: 當前股價
    -   `rsi`: RSI 技術指標 (0-1 範圍)
    -   `sentiment`: 情感評分 (-1 到 1)
-   **輸出**: AI 生成的交易策略建議

### 3. 🔍 動作解析 (Action Parsing)

-   **功能**: 將文字策略建議轉換為數值動作代碼
-   **動作代碼**:
    -   `0`: 賣出 (Sell)
    -   `1`: 持有 (Hold)
    -   `2`: 買入 (Buy)

## 使用方法

### 命令列使用

```bash
# 情感評分分析
python finbert_main.py --sentiment-score "Apple stock surges after earnings"

# 交易策略分析 (格式: 價格,RSI,情感)
python finbert_main.py --trading-strategy "100.0,0.3,0.8"

# 增強版測試 (包含所有新功能)
python finbert_main.py --enhanced-test

# 互動模式 (支援新功能)
python finbert_main.py --interactive
```

### 互動模式特殊指令

在互動模式中，您可以使用以下特殊指令：

```bash
# 情感評分分析
score:您要分析的文本

# 交易策略分析
strategy:100.0,0.5,0.3

# 查看範例
examples
```

### 程式化使用

```python
from finbert_main import FinBERTAnalyzer

# 初始化分析器
analyzer = FinBERTAnalyzer("your_hf_token_here")

# 情感評分分析
result = analyzer.analyze_sentiment_with_score("Apple stock surges")
print(f"情感評分: {result['sentiment_score']}")

# 交易策略生成
strategy_result = analyzer.generate_trading_strategy(
    price=100.0,
    rsi=0.3,
    sentiment=0.8
)
print(f"建議動作: {strategy_result['action_label']}")

# 動作解析
action = analyzer.parse_action("I recommend to buy this stock")
print(f"動作代碼: {action}")  # 輸出: 2 (買入)
```

## 示範腳本

運行完整功能示範：

```bash
python finbert_enhanced_demo.py
```

此腳本將展示：

-   情感評分分析範例
-   交易策略生成範例
-   動作解析邏輯範例

## 技術實現

### Hugging Face InferenceClient 整合

-   使用 `text_generation` API 進行情感評分和策略生成
-   支援自定義提示詞 (prompt) 來引導模型輸出
-   智能解析模型回應，提取數值和動作信息

### 錯誤處理

-   完整的異常捕獲和錯誤回報
-   網路連接失敗的優雅處理
-   無效輸入的友好提示

### 效能優化

-   合理的 API 調用間隔避免限制
-   結果快取機制減少重複請求
-   詳細的回應時間統計

## 應用場景

### 1. 量化交易

```python
# 結合技術指標進行自動化交易決策
news_sentiment = analyzer.analyze_sentiment_with_score(latest_news)
trading_action = analyzer.generate_trading_strategy(
    current_price,
    rsi_indicator,
    news_sentiment['sentiment_score']
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

## 注意事項

1. **API 限制**: 請注意 HuggingFace API 的使用限制
2. **網路依賴**: 功能需要穩定的網路連接
3. **模型限制**: FinBERT 主要針對英文金融文本訓練，中文效果可能有限
4. **投資建議**: AI 生成的交易策略僅供參考，不構成投資建議

## 更新日誌

-   **v2.0**: 新增情感評分、交易策略生成、動作解析功能
-   **v2.0**: 整合 HuggingFace InferenceClient
-   **v2.0**: 增強互動模式和命令列介面
-   **v2.0**: 新增完整的示範腳本和測試套件

