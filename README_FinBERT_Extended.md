# FinBERT 擴展功能測試套件

基於 ProsusAI/finbert 模型的綜合金融情感分析工具集，特別針對中英文處理能力進行深度測試和分析。

## 🎯 專案目標

1. **擴展原有功能**: 從簡單的 API 調用擴展為完整的測試套件
2. **中文能力測試**: 深度測試 FinBERT 對中文金融文本的處理能力
3. **語言對比分析**: 比較中英文在相同金融情境下的分析差異
4. **實用工具開發**: 提供互動式工具供日常使用

## 📁 檔案結構

```
stock-project/
├── test_finBert.py                    # 原始簡單測試檔案
├── finbert_simple_test.py             # 簡化版測試腳本 ⭐
├── finbert_comprehensive_test.py      # 綜合測試套件 ⭐
├── finbert_language_comparison.py     # 中英文對比測試 ⭐
├── finbert_interactive_tool.py        # 互動式分析工具 ⭐
├── FinBERT_Analysis_Report.md         # 詳細分析報告 ⭐
└── README_FinBERT_Extended.md         # 本說明文件 ⭐
```

## 🚀 快速開始

### 1. 環境準備

```bash
# 安裝依賴
pip install -r requirements.txt

# 確保包含以下套件
pip install huggingface_hub pandas numpy
```

### 2. 基本測試

```bash
# 執行簡化版測試（推薦新手）
python finbert_simple_test.py

# 執行綜合測試套件
python finbert_comprehensive_test.py

# 執行中英文對比測試
python finbert_language_comparison.py
```

### 3. 互動式使用

```bash
# 啟動互動模式
python finbert_interactive_tool.py

# 批次分析單一文本
python finbert_interactive_tool.py "台積電股價大漲創新高"
```

## 📊 主要功能特色

### 🔍 1. 簡化版測試 (`finbert_simple_test.py`)

**特點**:

-   9 個精選測試案例（英文、中文、中英混合）
-   清晰的結果展示和統計分析
-   自動生成測試報告

**適用場景**: 快速驗證模型基本功能

```python
# 主要測試案例包括
test_cases = [
    ("英文正面", "Apple stock surged 15% after reporting record earnings."),
    ("中文正面", "台積電股價今日大漲8%，創下歷史新高。"),
    ("中英混合", "NVIDIA 輝達公司因AI晶片需求激增，股價創新高。"),
    # ... 更多案例
]
```

### 🔬 2. 綜合測試套件 (`finbert_comprehensive_test.py`)

**特點**:

-   40+ 個測試案例涵蓋各種情境
-   詳細的統計分析和報告生成
-   JSON 和 CSV 格式結果匯出
-   邊界情況測試（短文本、長文本、特殊字符）

**適用場景**: 深度評估模型性能

### 📈 3. 中英文對比測試 (`finbert_language_comparison.py`)

**特點**:

-   相同意思的中英文對照測試
-   一致性分析和準確性比較
-   信心度差異統計
-   詳細的比較報告

**核心發現**:

```
✅ 英文處理: 準確率高，情感分類正確
⚠️ 中文處理: 傾向判斷為中性，需謹慎使用
🔍 建議策略: 中文翻譯成英文後分析
```

### 🛠️ 4. 互動式工具 (`finbert_interactive_tool.py`)

**特點**:

-   即時文本分析
-   支援批次和互動兩種模式
-   智能語言檢測
-   個性化建議提供

**使用範例**:

```bash
# 互動模式
$ python finbert_interactive_tool.py
📝 請輸入要分析的金融文本: 蘋果股價暴漲

# 批次模式
$ python finbert_interactive_tool.py "Tesla stock crashes 20%"
```

## 📋 測試結果摘要

### 🎯 關鍵發現

1. **英文處理優異** ✅

    - 正面情感: 94.6% - 96.2% 信心度
    - 負面情感: 97.4% 信心度
    - 中性情感: 74.1% 信心度

2. **中文處理特殊現象** ⚠️

    - **所有中文文本都被判斷為中性**
    - 信心度範圍: 85.5% - 91.7%
    - 即使明顯的正面/負面文本也被判為中性

3. **語言處理統計**
    ```
    🇺🇸 英文平均信心度: 90.6%
    🇹🇼 中文平均信心度: 89.8%
    🔄 平均回應時間: 0.724秒
    ✅ API 成功率: 100%
    ```

### 💡 實用建議

#### 對於英文金融文本 ✅

-   **直接使用**: 結果準確可信
-   **適用場景**: 英文新聞、報告、社交媒體分析

#### 對於中文金融文本 ⚠️

-   **方案 1**: 翻譯成英文後分析（推薦）
-   **方案 2**: 結合關鍵詞分析
-   **方案 3**: 使用專門的中文金融模型
-   **注意**: 不要完全依賴 FinBERT 的中文結果

## 🔧 進階使用

### 自定義測試案例

```python
from finbert_comprehensive_test import FinBERTTester

# 初始化測試器
tester = FinBERTTester("your_api_key")

# 分析自定義文本
result = tester.analyze_sentiment("您的金融文本", "測試名稱")

# 生成報告
report = tester.generate_report()
print(report)
```

### 批次分析

```python
texts = [
    "Apple earnings beat expectations",
    "台積電營收創新高",
    "市場波動加劇"
]

for i, text in enumerate(texts):
    result = tester.analyze_sentiment(text, f"批次測試_{i+1}")
```

## 📊 效能指標

| 指標       | 英文   | 中文   | 整體   |
| ---------- | ------ | ------ | ------ |
| 平均信心度 | 90.6%  | 89.8%  | 90.2%  |
| 情感準確性 | 高     | 低     | 中等   |
| 回應時間   | 0.7 秒 | 0.7 秒 | 0.7 秒 |
| 推薦使用   | ✅     | ⚠️     | 視情況 |

## 🔮 未來發展方向

1. **模型微調**: 使用中文金融語料進行 Fine-tuning
2. **混合策略**: 開發中英文混合分析管道
3. **實時應用**: 整合到實時金融資訊系統
4. **準確性提升**: 建立反饋機制持續改進

## 📞 技術支援

-   **模型來源**: [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)
-   **API 文檔**: [HuggingFace Inference API](https://huggingface.co/docs/api-inference/index)
-   **問題回報**: 請在專案中提出 Issue

## 📄 授權說明

本專案基於 FinBERT 模型進行擴展開發，遵循相應的開源授權協議。

---

**最後更新**: 2025 年 9 月 11 日  
**版本**: v1.0  
**作者**: 基於原始 test_finBert.py 擴展開發

