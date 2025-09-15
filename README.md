# FinBERT 金融情感分析專案

基於 ProsusAI/finbert 模型的中英文金融情感分析工具集，提供完整的測試和分析功能。

## 🎯 專案簡介

本專案擴展了基本的 FinBERT API 調用，開發了一套完整的金融情感分析工具，特別針對中英文處理能力進行了深度測試和優化。

### ✨ 主要特色

-   🔍 **統一介面**: 整合所有功能的主程式 `finbert_main.py`
-   🌐 **多語言支援**: 支援英文、中文、中英混合文本分析
-   📊 **詳細報告**: 提供信心度、回應時間、語言檢測等詳細資訊
-   🛠️ **多種模式**: 互動式、批次處理、測試套件等多種使用方式
-   📈 **對比分析**: 中英文語言處理能力對比測試

## 🚀 快速開始

### 1. 環境準備

```bash
# 安裝依賴
pip install -r requirements.txt
```

### 2. 基本使用

```bash
# 顯示所有可用選項
python finbert_main.py --help

# 互動模式（推薦）
python finbert_main.py --interactive

# 快速測試
python finbert_main.py --simple-test

# 分析單一文本
python finbert_main.py --analyze "台積電股價創新高"
```

## 📋 功能模式

### 🎮 互動模式

```bash
python finbert_main.py --interactive
```

-   即時輸入文本進行分析
-   智能語言檢測
-   個性化建議提供

### 🧪 快速測試

```bash
python finbert_main.py --simple-test
```

-   執行 9 個精選測試案例
-   涵蓋中英文正面/負面/中性情感
-   自動生成統計報告

### 🔍 語言對比測試

```bash
python finbert_main.py --language-comparison
```

-   相同意思的中英文對照分析
-   一致性和準確性比較
-   語言處理差異統計

### 📝 單文本分析

```bash
python finbert_main.py --analyze "您的金融文本"
```

-   批次處理模式
-   適合腳本整合使用

## 📊 測試結果摘要

### 🔍 關鍵發現

根據我們的測試，FinBERT 模型在處理不同語言時表現如下：

| 語言類型 | 準確性        | 信心度 | 建議使用 |
| -------- | ------------- | ------ | -------- |
| 🇺🇸 英文  | 高 (94-97%)   | 74-97% | ✅ 推薦  |
| 🇹🇼 中文  | 低 (傾向中性) | 85-92% | ⚠️ 謹慎  |
| 🔀 混合  | 中等          | 90%+   | ⚠️ 謹慎  |

### 💡 使用建議

#### ✅ 英文金融文本

-   **直接使用**: 結果準確可信
-   **適用場景**: 英文新聞、報告、社交媒體分析

#### ⚠️ 中文金融文本

-   **方案 1**: 翻譯成英文後分析（推薦）
-   **方案 2**: 結合關鍵詞分析驗證
-   **方案 3**: 使用專門的中文金融模型
-   **注意**: FinBERT 對中文文本傾向判斷為「中性」

## 📁 檔案結構

```
stock-project/
├── finbert_main.py                   # 🎯 主程式（統一介面）
├── test_finBert.py                   # 📝 原始測試檔案
├── finbert_simple_test.py            # 🧪 簡化測試腳本
├── finbert_comprehensive_test.py     # 🔬 綜合測試套件
├── finbert_language_comparison.py    # 📈 語言對比測試
├── finbert_interactive_tool.py       # 🛠️ 互動式工具
├── FinBERT_Analysis_Report.md        # 📋 詳細分析報告
├── README_FinBERT_Extended.md        # 📖 完整功能說明
├── requirements.txt                  # 📦 依賴列表
└── README.md                         # 📚 本文件
```

## 🔧 進階使用

### 自定義 API 金鑰

```bash
python finbert_main.py --api-key "your_api_key" --interactive
```

### 程式整合範例

```python
from finbert_main import FinBERTAnalyzer

# 初始化分析器
analyzer = FinBERTAnalyzer()

# 分析文本
result = analyzer.analyze_sentiment("Apple stock surged 15%")

# 格式化輸出
formatted_result = analyzer.format_result(result)
print(formatted_result)
```

## 📈 效能指標

-   **平均回應時間**: 0.7 秒
-   **API 成功率**: 100%
-   **支援語言**: 英文、中文、中英混合
-   **並發支援**: 單執行緒（避免 API 限制）

## 🔮 未來發展

1. **模型微調**: 使用中文金融語料進行 Fine-tuning
2. **批次處理**: 支援大量文本的批次分析
3. **實時監控**: 整合到金融資訊系統
4. **準確性提升**: 多模型集成策略

## 📞 技術支援

-   **模型來源**: [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)
-   **API 文檔**: [HuggingFace Inference API](https://huggingface.co/docs/api-inference/index)
-   **詳細報告**: 查看 `FinBERT_Analysis_Report.md`

## 📄 使用範例

### 互動模式示範

```
🚀 FinBERT 互動式金融情感分析工具
============================================================
💡 使用說明:
   - 輸入任何金融相關文本進行情感分析
   - 支援中文、英文、中英混合文本
   - 輸入 'quit' 或 'exit' 退出程式

📝 請輸入要分析的金融文本: 蘋果股價暴漲

📝 輸入文本: 蘋果股價暴漲
🌐 語言類型: 中文
🎯 分析結果: 🟡 中性
📊 信心度: 0.890 (89.0%)
⏱️  回應時間: 0.245秒

💡 建議: 檢測到中文文本，FinBERT主要針對英文訓練
   建議結合關鍵詞分析或翻譯成英文後再分析
```

---

**最後更新**: 2025 年 9 月 11 日  
**版本**: v2.0  
**作者**: 基於 ProsusAI/finbert 擴展開發

