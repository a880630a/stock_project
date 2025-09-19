# FinBERT 股票分析專案整合指南

## 📋 專案概述

這是一個基於 FinBERT 的智能股票情感分析和交易策略系統，整合了：

-   **FinBERT 情感分析**: 使用 Hugging Face 的 FinBERT 模型進行金融文本情感分析
-   **LLM 驅動交易代理**: 基於強化學習的股票交易決策代理
-   **RESTful API 服務**: 提供完整的 Web API 接口供前端調用
-   **完整測試套件**: 包含 API 測試和功能驗證

## 🏗️ 專案結構

```
stock-project/
├── finbert_main.py          # FinBERT核心分析器
├── agents.py                # LLM驅動的交易代理
├── app.py                   # Flask API服務器
├── api_test.py              # API測試腳本
├── start_api.py             # API啟動工具
├── requirements.txt         # 依賴包列表
├── ENVIRONMENT_SETUP.md     # 環境設置指南
└── README_INTEGRATION.md    # 整合指南（本文件）
```

## 🚀 快速開始

### 1. 環境準備

```bash
# 1. 確保Python 3.8+
python --version

# 2. 克隆專案（如果需要）
git clone <your-repo-url>
cd stock-project

# 3. 創建虛擬環境（推薦）
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate     # Windows
```

### 2. 安裝依賴

```bash
pip install -r requirements.txt
```

### 3. 配置環境變量

創建`.env`文件：

```bash
# 複製範例文件
cp .env.example .env

# 編輯.env文件，設置您的Hugging Face API Token
# HF_TOKEN=your_actual_huggingface_token_here
```

或設置系統環境變量：

```bash
# Linux/macOS
export HF_TOKEN="your_actual_huggingface_token_here"

# Windows
set HF_TOKEN=your_actual_huggingface_token_here
```

### 4. 啟動 API 服務

使用啟動工具（推薦）：

```bash
python start_api.py
```

或直接啟動：

```bash
python app.py
```

### 5. 測試 API 功能

```bash
# 運行完整API測試
python api_test.py

# 或快速測試
python api_test.py --quick
```

## 🔌 API 端點說明

### 健康檢查

```http
GET /api/health
```

### 基本情感分析

```http
POST /api/analyze
Content-Type: application/json

{
    "text": "Apple stock surged 15% after earnings."
}
```

### 數值化情感評分

```http
POST /api/analyze-score
Content-Type: application/json

{
    "text": "Tesla shares plummeted due to production delays."
}
```

### 交易策略分析

```http
POST /api/trading-strategy
Content-Type: application/json

{
    "price": 100.0,
    "rsi": 0.3,
    "sentiment": 0.8
}
```

### 批次分析

```http
POST /api/batch-analyze
Content-Type: application/json

{
    "texts": [
        "Apple stock surged after earnings.",
        "Tesla shares declined today.",
        "市場保持穩定。"
    ]
}
```

## 🧪 測試指南

### API 測試

```bash
# 完整測試套件
python api_test.py

# 快速測試
python api_test.py --quick

# 指定服務器地址
python api_test.py --url http://localhost:8080
```

### 功能測試

```bash
# FinBERT分析器測試
python finbert_main.py --simple-test

# 互動模式測試
python finbert_main.py --interactive

# 交易代理測試
python agents.py
```

## 🔧 進階配置

### 自定義 API 配置

編輯`app.py`或設置環境變量：

```python
# 在app.py中修改
app.run(host='0.0.0.0', port=8080, debug=False)
```

或使用環境變量：

```bash
export FLASK_HOST=127.0.0.1
export FLASK_PORT=8080
export FLASK_DEBUG=False
```

### 自定義模型配置

在`finbert_main.py`中修改模型設置：

```python
# 使用不同的FinBERT模型
self.client = InferenceClient(model="yiyanghkust/finbert-tone")
```

## 📊 使用範例

### Python 客戶端範例

```python
import requests
import json

# 基本情感分析
def analyze_sentiment(text):
    response = requests.post(
        'http://localhost:5000/api/analyze',
        json={'text': text},
        headers={'Content-Type': 'application/json'}
    )
    return response.json()

# 交易策略分析
def get_trading_strategy(price, rsi, sentiment):
    response = requests.post(
        'http://localhost:5000/api/trading-strategy',
        json={
            'price': price,
            'rsi': rsi,
            'sentiment': sentiment
        },
        headers={'Content-Type': 'application/json'}
    )
    return response.json()

# 使用範例
result = analyze_sentiment("Apple stock surged 15% after earnings.")
print(json.dumps(result, indent=2, ensure_ascii=False))

strategy = get_trading_strategy(100.0, 0.3, 0.8)
print(json.dumps(strategy, indent=2, ensure_ascii=False))
```

### JavaScript/Node.js 客戶端範例

```javascript
const axios = require("axios");

// 基本情感分析
async function analyzeSentiment(text) {
    try {
        const response = await axios.post("http://localhost:5000/api/analyze", {
            text: text,
        });
        return response.data;
    } catch (error) {
        console.error("分析失敗:", error.response?.data || error.message);
    }
}

// 使用範例
analyzeSentiment("Apple stock surged 15% after earnings.").then((result) =>
    console.log(JSON.stringify(result, null, 2))
);
```

## 🔍 故障排除

### 常見問題

1. **"請設置 HF_TOKEN 環境變量"**

    - 確保已正確設置 Hugging Face API Token
    - 檢查.env 文件或系統環境變量

2. **"FinBERT 分析器未初始化"**

    - 檢查 HF_TOKEN 是否有效
    - 確認網路連接正常
    - 檢查 Hugging Face 服務狀態

3. **"imitation 庫導入失敗"**

    - 強化學習功能為可選，不影響基本功能
    - 如需使用，請安裝：`pip install imitation gymnasium`

4. **API 連接失敗**
    - 確認 API 服務器已啟動
    - 檢查防火牆設置
    - 確認端口未被佔用

### 日誌查看

```bash
# 查看API服務器日誌
python start_api.py

# 查看詳細錯誤信息
python app.py
```

## 📈 效能優化

### 生產環境部署

1. **使用 WSGI 服務器**：

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

2. **使用反向代理**（Nginx 配置範例）：

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

3. **Docker 部署**：

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

## 🔒 安全考量

1. **API 金鑰安全**：

    - 絕不將真實 API 金鑰提交到版本控制
    - 定期輪換 API 金鑰
    - 使用環境變量或密鑰管理服務

2. **API 安全**：

    - 在生產環境中添加身份驗證
    - 實施速率限制
    - 使用 HTTPS

3. **輸入驗證**：
    - 驗證所有用戶輸入
    - 防止 SQL 注入和 XSS 攻擊
    - 限制請求大小

## 📚 相關文檔

-   [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) - 環境設置詳細指南
-   [Hugging Face FinBERT](https://huggingface.co/ProsusAI/finbert) - FinBERT 模型文檔
-   [Flask 文檔](https://flask.palletsprojects.com/) - Flask 框架文檔
-   [imitation 文檔](https://imitation.readthedocs.io/) - 強化學習庫文檔

## 🤝 貢獻指南

1. Fork 專案
2. 創建功能分支
3. 提交變更
4. 推送到分支
5. 創建 Pull Request

## 📄 授權

本專案採用 MIT 授權條款。

## 🆘 支援

如有問題或建議，請：

1. 檢查故障排除章節
2. 查看相關文檔
3. 提交 Issue
4. 聯繫維護者

---

**祝您使用愉快！** 🚀
