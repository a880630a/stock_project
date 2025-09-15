# 環境變量設置指南

## 🔐 安全性說明

為了保護您的 API 密鑰安全，本專案已移除所有硬編碼的敏感信息。請按照以下步驟設置環境變量。

## 📋 必要的環境變量

### HF_TOKEN (Hugging Face API Token)

這是使用 Hugging Face 模型所需的 API 密鑰。

## 🛠️ 設置方法

### 方法 1: 使用 .env 文件 (推薦)

1. 在專案根目錄創建 `.env` 文件：

```bash
touch .env
```

2. 編輯 `.env` 文件，添加您的 API 密鑰：

```bash
# Hugging Face API Token
HF_TOKEN=your_actual_huggingface_token_here
```

3. 確保 `.env` 文件不會被提交到 Git（已在 .gitignore 中配置）

### 方法 2: 系統環境變量

#### Linux/macOS:

```bash
# 臨時設置（當前終端會話有效）
export HF_TOKEN="your_actual_huggingface_token_here"

# 永久設置（添加到 ~/.bashrc 或 ~/.zshrc）
echo 'export HF_TOKEN="your_actual_huggingface_token_here"' >> ~/.bashrc
source ~/.bashrc
```

#### Windows:

```cmd
# 臨時設置（當前命令提示符會話有效）
set HF_TOKEN=your_actual_huggingface_token_here

# 永久設置（系統環境變量）
setx HF_TOKEN "your_actual_huggingface_token_here"
```

## 🔑 如何獲取 Hugging Face API Token

1. 訪問 [Hugging Face](https://huggingface.co/)
2. 註冊/登入您的帳戶
3. 前往 Settings > Access Tokens
4. 創建新的 Token（選擇適當的權限）
5. 複製生成的 Token

## ✅ 驗證設置

運行以下命令驗證環境變量是否正確設置：

```bash
# Linux/macOS
echo $HF_TOKEN

# Windows
echo %HF_TOKEN%
```

或者運行 Python 代碼：

```python
import os
print("HF_TOKEN 已設置:" if os.getenv('HF_TOKEN') else "HF_TOKEN 未設置")
```

## 🚀 運行應用

設置完環境變量後，您可以正常運行應用：

```bash
# 運行主程序
python finbert_main.py --interactive

# 運行測試
python finbert_comprehensive_test.py

# 運行語言比較
python finbert_language_comparison.py

# 運行 Flask 應用
python app.py
```

## ⚠️ 注意事項

1. **絕對不要**將真實的 API 密鑰提交到版本控制系統
2. 定期輪換您的 API 密鑰
3. 如果懷疑密鑰洩露，立即在 Hugging Face 平台上撤銷並重新生成
4. 確保 `.env` 文件的權限設置正確（僅您可讀寫）

## 🔧 故障排除

### 常見錯誤及解決方案

1. **"請設置 HF_TOKEN 環境變量"**

    - 確保已正確設置環境變量
    - 重新啟動終端/IDE
    - 檢查變量名稱是否正確（區分大小寫）

2. **"API 認證失敗"**

    - 檢查 Token 是否有效
    - 確認 Token 權限是否足夠
    - 嘗試重新生成 Token

3. **".env 文件未生效"**
    - 確保使用了支持.env 的庫（如 python-dotenv）
    - 檢查.env 文件位置是否正確
    - 確認文件格式無誤（無多餘空格等）

## 📚 相關文檔

-   [Hugging Face API 文檔](https://huggingface.co/docs/api-inference/index)
-   [環境變量最佳實踐](https://12factor.net/config)
