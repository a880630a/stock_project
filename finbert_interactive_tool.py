#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FinBERT 互動式測試工具
讓使用者輸入自定義文本進行即時金融情感分析
"""

import os
import sys
import time
from huggingface_hub import InferenceClient

class FinBERTInteractiveTool:
    def __init__(self, api_key: str):
        """初始化 FinBERT 互動工具"""
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=api_key,
        )
        self.model_name = "ProsusAI/finbert"
        
    def analyze_text(self, text: str) -> dict:
        """分析單一文本的金融情感"""
        try:
            start_time = time.time()
            result = self.client.text_classification(text, model=self.model_name)
            end_time = time.time()
            
            if result and len(result) > 0:
                # 處理預測結果
                predictions = result if isinstance(result, list) else [result]
                best_prediction = max(predictions, key=lambda x: x.get('score', 0))
                
                return {
                    'success': True,
                    'text': text,
                    'best_prediction': best_prediction,
                    'all_predictions': predictions,
                    'response_time': round(end_time - start_time, 3),
                    'language': self._detect_language(text)
                }
            else:
                return {
                    'success': False,
                    'text': text,
                    'error': '無法獲得有效預測結果'
                }
                
        except Exception as e:
            return {
                'success': False,
                'text': text,
                'error': str(e)
            }
    
    def _detect_language(self, text: str) -> str:
        """簡單的語言檢測"""
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        english_chars = sum(1 for char in text if char.isalpha() and ord(char) < 256)
        
        if chinese_chars > len(text) * 0.3:
            if english_chars > 0:
                return "中英混合"
            return "中文"
        elif english_chars > 0:
            return "英文"
        else:
            return "其他"
    
    def format_result(self, result: dict) -> str:
        """格式化輸出結果"""
        if not result['success']:
            return f"❌ 分析失敗: {result['error']}"
        
        text = result['text']
        best_pred = result['best_prediction']
        all_preds = result['all_predictions']
        response_time = result['response_time']
        language = result['language']
        
        # 情感標籤美化
        label_icons = {
            'positive': '🟢 正面',
            'negative': '🔴 負面',
            'neutral': '🟡 中性'
        }
        
        # 主要結果
        label = best_pred.get('label', '').lower()
        score = best_pred.get('score', 0)
        display_label = label_icons.get(label, f"❓ {label}")
        
        output = f"""
📝 輸入文本: {text}
🌐 語言類型: {language}
🎯 分析結果: {display_label}
📊 信心度: {score:.3f} ({score*100:.1f}%)
⏱️  回應時間: {response_time}秒

📋 詳細預測結果:"""
        
        # 所有預測結果
        for pred in sorted(all_preds, key=lambda x: x.get('score', 0), reverse=True):
            pred_label = pred.get('label', '').lower()
            pred_score = pred.get('score', 0)
            pred_display = label_icons.get(pred_label, pred_label)
            output += f"\n   {pred_display}: {pred_score:.3f} ({pred_score*100:.1f}%)"
        
        # 添加建議
        if language in ["中文", "中英混合"]:
            output += f"\n\n💡 建議: 檢測到{language}文本，FinBERT 主要針對英文訓練，建議:"
            output += f"\n   1. 參考英文翻譯結果進行對比"
            output += f"\n   2. 結合關鍵詞分析驗證結果"
            output += f"\n   3. 考慮使用專門的中文金融情感模型"
        
        return output
    
    def run_interactive_mode(self):
        """執行互動模式"""
        print("🚀 FinBERT 互動式金融情感分析工具")
        print("="*60)
        print("💡 使用說明:")
        print("   - 輸入任何金融相關文本進行情感分析")
        print("   - 支援中文、英文、中英混合文本")
        print("   - 輸入 'quit' 或 'exit' 退出程式")
        print("   - 輸入 'examples' 查看測試範例")
        print("="*60)
        
        while True:
            try:
                # 獲取用戶輸入
                user_input = input("\n📝 請輸入要分析的金融文本: ").strip()
                
                # 檢查退出指令
                if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                    print("👋 感謝使用 FinBERT 分析工具！")
                    break
                
                # 顯示範例
                if user_input.lower() in ['examples', 'example', '範例', '例子']:
                    self._show_examples()
                    continue
                
                # 檢查空輸入
                if not user_input:
                    print("⚠️ 請輸入有效的文本內容")
                    continue
                
                # 執行分析
                print("\n🔍 正在分析中...")
                result = self.analyze_text(user_input)
                
                # 顯示結果
                formatted_result = self.format_result(result)
                print(formatted_result)
                print("\n" + "-"*60)
                
            except KeyboardInterrupt:
                print("\n\n👋 程式被中斷，感謝使用！")
                break
            except Exception as e:
                print(f"\n❌ 發生錯誤: {e}")
                continue
    
    def _show_examples(self):
        """顯示測試範例"""
        examples = [
            ("英文正面範例", "Apple stock surged 15% after beating earnings expectations"),
            ("英文負面範例", "Tesla shares plummeted due to production delays"),
            ("英文中性範例", "The Fed will announce interest rates next week"),
            ("中文正面範例", "台積電股價創新高，投資人信心大增"),
            ("中文負面範例", "受疫情影響，航空股全面重挫"),
            ("中文中性範例", "央行宣布維持利率不變"),
            ("中英混合範例", "NVIDIA 因AI需求激增，股價大漲"),
            ("複雜分析範例", "公司Q3營收年增25%，EBITDA margin提升至30%")
        ]
        
        print("\n📚 測試範例:")
        print("="*50)
        for i, (category, example) in enumerate(examples, 1):
            print(f"{i}. {category}")
            print(f"   {example}")
        print("="*50)
        print("💡 您可以複製任一範例進行測試，或輸入自己的文本")

def main():
    """主程式入口"""
    # API 金鑰
    api_key = os.getenv('HF_TOKEN')
    if not api_key:
        print("❌ 錯誤: 請設置 HF_TOKEN 環境變量")
        print("請參考 ENVIRONMENT_SETUP.md 了解如何設置環境變量")
        return
    
    # 檢查命令列參數
    if len(sys.argv) > 1:
        # 批次模式：直接分析命令列參數
        text_to_analyze = " ".join(sys.argv[1:])
        
        print("🚀 FinBERT 批次分析模式")
        print("="*50)
        
        tool = FinBERTInteractiveTool(api_key)
        result = tool.analyze_text(text_to_analyze)
        formatted_result = tool.format_result(result)
        print(formatted_result)
        
    else:
        # 互動模式
        tool = FinBERTInteractiveTool(api_key)
        tool.run_interactive_mode()

if __name__ == "__main__":
    main()

