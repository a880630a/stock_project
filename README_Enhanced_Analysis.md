# FinBERT 增強分析功能說明

## 🚀 新功能概述

FinBERT 分析工具現在支持**增強分析功能**，除了基本的情感分析外，還提供：

1. **📄 文本總結** - 智能提取文本核心內容和關鍵信息
2. **🔍 分析依據** - 詳細說明情感判斷的依據和關鍵詞
3. **💡 結果解釋** - 解釋為什麼會得出當前的分析結果

## 🎯 使用方式

### 1. 命令行模式

```bash
# 增強分析指定文本
python finbert_main.py --explain "您的金融文本"

# 範例
python finbert_main.py --explain "Apple stock surged 15% after beating earnings expectations"
python finbert_main.py --explain "台積電股價創新高，投資人信心大增"
```

### 2. 互動模式

```bash
# 啟動互動模式
python finbert_main.py --interactive

# 在互動模式中使用增強分析
explain:您的金融文本

# 範例
explain:Tesla shares plummeted due to production delays
explain:今天蘋果發布了iPhone 17
```

## 📊 輸出格式

增強分析會提供以下詳細信息：

### 基本分析結果

-   📝 輸入文本
-   🌐 語言類型檢測
-   🎯 情感分析結果（正面/負面/中性）
-   📊 信心度評分
-   ⏱️ 分析回應時間

### 詳細預測結果

-   各情感類別的詳細評分
-   信心度百分比顯示

### 📄 文本總結

-   自動提取的關鍵實體（公司名稱等）
-   識別的金融關鍵詞
-   文本核心內容總結

### 🔍 分析依據

-   **正面關鍵詞**：如 "上漲"、"創新高"、"surge"、"strong" 等
-   **負面關鍵詞**：如 "下跌"、"重挫"、"plummet"、"disappointing" 等
-   **中性關鍵詞**：如 "發布"、"宣布"、"announce"、"report" 等
-   **語言類型**：中文/英文/中英混合
-   **文本長度**：字符數統計

### 💡 結果解釋

-   **信心度解釋**：說明為什麼模型有這個信心度
-   **競爭分析**：與次高分類的比較分析
-   **分類確定性**：評估結果的可靠程度
-   **語言風險提醒**：針對非英文文本的特別提醒

## 🔍 功能特色

### 智能關鍵詞識別

支持中英文金融關鍵詞識別：

**中文關鍵詞**：

-   股價、股票、財報、投資、信心大增、創新高、表現強勁
-   台積電、聯發科等台灣公司

**英文關鍵詞**：

-   stock、earnings、surge、plummet、bullish、bearish
-   Apple、Tesla、NVIDIA 等國際公司

### 多語言支持

-   ✅ **英文**：FinBERT 原生支持，分析最準確
-   ✅ **中文**：提供關鍵詞分析和語言風險提醒
-   ✅ **中英混合**：智能識別並提供適當建議

### 實體識別

自動識別文本中的：

-   🏢 公司名稱（Apple/蘋果、Tesla/特斯拉、台積電等）
-   📈 金融指標（股價、財報、EPS 等）
-   💹 市場動作（上漲、下跌、surge、plummet 等）

## 📋 使用範例

### 範例 1：正面英文新聞

```bash
python finbert_main.py --explain "Apple stock surged 15% after beating earnings expectations"
```

**輸出重點**：

-   🎯 分析結果: 🟢 正面 (94.7%)
-   📄 文本總結: 涉及實體：Apple | 關鍵詞：Apple
-   🔍 分析依據: 語言類型：英文
-   💡 結果解釋: 模型以非常高的信心度判斷此文本為 positive 情感

### 範例 2：負面英文新聞

```bash
python finbert_main.py --explain "Tesla shares plummeted 20% following disappointing delivery numbers"
```

**輸出重點**：

-   🎯 分析結果: 🔴 負面 (97.3%)
-   🔍 分析依據: 負面關鍵詞：plummet, disappointing
-   💡 結果解釋: 與次高分類 neutral 差距明顯，分類較為確定

### 範例 3：中文金融新聞

```bash
python finbert_main.py --explain "台積電股價創新高，投資人信心大增，財報表現強勁"
```

**輸出重點**：

-   🎯 分析結果: 🟡 中性 (89.3%)
-   📄 文本總結: 關鍵詞：財報, 強勁, 投資, 信心, 股價
-   🔍 分析依據: 正面關鍵詞：信心大增, 創新高, 表現強勁
-   ⚠️ 語言提醒: FinBERT 主要針對英文訓練，建議翻譯後再分析

## 🎨 與現有功能的兼容性

增強分析功能完全兼容現有的所有功能：

-   ✅ **基本情感分析** (`python finbert_main.py --analyze`)
-   ✅ **情感評分分析** (`python finbert_main.py --sentiment-score`)
-   ✅ **交易策略分析** (`python finbert_main.py --trading-strategy`)
-   ✅ **簡化測試** (`python finbert_main.py --simple-test`)
-   ✅ **增強測試** (`python finbert_main.py --enhanced-test`)
-   ✅ **中英文對比** (`python finbert_main.py --language-comparison`)

## 💡 使用建議

### 1. 語言選擇

-   **英文文本**：直接使用，分析結果最準確
-   **中文文本**：參考關鍵詞分析，考慮翻譯成英文後再次分析
-   **中英混合**：注意語言風險提醒，結合關鍵詞判斷

### 2. 結果解讀

-   **高信心度 (>80%)**：結果較為可靠
-   **中等信心度 (40-80%)**：需要結合關鍵詞分析
-   **低信心度 (<40%)**：建議人工複核或使用其他方法

### 3. 關鍵詞分析

-   關注**正面/負面關鍵詞**的數量和強度
-   結合**實體識別**了解新聞主體
-   參考**語言類型**評估分析可靠性

## 🔧 技術實現

### 核心功能

-   **情感分析**：使用 ProsusAI/finbert 模型
-   **關鍵詞提取**：基於正則表達式的中英文金融詞彙庫
-   **實體識別**：公司名稱和金融術語識別
-   **語言檢測**：中英文字符比例分析

### 性能指標

-   **平均回應時間**：3-5 秒
-   **支持文本長度**：無限制（建議<1000 字符）
-   **關鍵詞庫**：涵蓋常見中英文金融詞彙
-   **實體庫**：包含主要科技和金融公司

---

**📞 如有問題或建議，請參考主要的 README.md 文檔或聯繫開發團隊。**
