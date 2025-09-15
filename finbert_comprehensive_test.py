#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FinBERT 綜合測試套件
測試 ProsusAI/finbert 模型在不同語言和場景下的金融情感分析能力
"""

import os
import json
import time
from datetime import datetime
from huggingface_hub import InferenceClient
from typing import List, Dict, Any
import pandas as pd

class FinBERTTester:
    def __init__(self, api_key: str):
        """
        初始化 FinBERT 測試器
        
        Args:
            api_key: Hugging Face API 金鑰
        """
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=api_key,
        )
        self.model_name = "ProsusAI/finbert"
        self.test_results = []
        
    def analyze_sentiment(self, text: str, test_name: str = "") -> Dict[str, Any]:
        """
        分析單一文本的情感
        
        Args:
            text: 要分析的文本
            test_name: 測試案例名稱
            
        Returns:
            包含分析結果的字典
        """
        try:
            start_time = time.time()
            result = self.client.text_classification(text, model=self.model_name)
            end_time = time.time()
            
            # 處理結果格式
            if isinstance(result, list) and len(result) > 0:
                predictions = result
            else:
                predictions = [result] if result else []
            
            analysis_result = {
                "test_name": test_name,
                "input_text": text,
                "predictions": predictions,
                "response_time": round(end_time - start_time, 3),
                "timestamp": datetime.now().isoformat(),
                "language": self._detect_language(text)
            }
            
            self.test_results.append(analysis_result)
            return analysis_result
            
        except Exception as e:
            error_result = {
                "test_name": test_name,
                "input_text": text,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "language": self._detect_language(text)
            }
            self.test_results.append(error_result)
            return error_result
    
    def _detect_language(self, text: str) -> str:
        """簡單的語言檢測"""
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        if chinese_chars > len(text) * 0.3:
            return "Chinese"
        return "English"
    
    def run_english_financial_tests(self):
        """執行英文金融文本測試"""
        print("🔍 執行英文金融情感分析測試...")
        
        english_test_cases = [
            # 正面情感測試
            ("Positive - Stock Growth", "Apple stock surged 15% after reporting record quarterly earnings, exceeding analyst expectations."),
            ("Positive - Company Performance", "The company's revenue increased by 25% year-over-year, demonstrating strong market position."),
            ("Positive - Market Outlook", "Investors are optimistic about the tech sector's growth potential in the coming quarters."),
            ("Positive - Dividend", "The board approved a 10% dividend increase, rewarding shareholders for their continued support."),
            
            # 負面情感測試
            ("Negative - Stock Decline", "Tesla shares plummeted 20% following disappointing delivery numbers and production delays."),
            ("Negative - Company Loss", "The company reported a significant loss of $500 million due to supply chain disruptions."),
            ("Negative - Market Crash", "Market volatility increased as investors panic-sold amid recession fears."),
            ("Negative - Bankruptcy", "The retail chain filed for bankruptcy protection after failing to restructure its debt."),
            
            # 中性情感測試
            ("Neutral - Announcement", "The Federal Reserve will announce its interest rate decision next Wednesday."),
            ("Neutral - Merger", "Company A and Company B are in talks for a potential merger, sources familiar with the matter said."),
            ("Neutral - Earnings Date", "Microsoft is scheduled to report its quarterly earnings on January 25th."),
            ("Neutral - Market Update", "The S&P 500 index closed flat yesterday with mixed trading volumes."),
        ]
        
        for test_name, text in english_test_cases:
            print(f"  測試: {test_name}")
            result = self.analyze_sentiment(text, test_name)
            self._print_result(result)
            time.sleep(1)  # 避免 API 限制
    
    def run_chinese_financial_tests(self):
        """執行中文金融文本測試"""
        print("\n🔍 執行中文金融情感分析測試...")
        
        chinese_test_cases = [
            # 正面情感測試
            ("正面 - 股價上漲", "台積電股價今日大漲8%，創下歷史新高，投資人對其先進製程技術充滿信心。"),
            ("正面 - 營收成長", "該公司第三季營收年增30%，獲利表現超越市場預期，展現強勁成長動能。"),
            ("正面 - 市場前景", "分析師看好電動車產業未來發展，預期相關概念股將持續受到市場青睞。"),
            ("正面 - 股利政策", "董事會宣布提高股利發放，每股配息增加至2.5元，回饋股東支持。"),
            
            # 負面情感測試
            ("負面 - 股價下跌", "受到全球經濟衰退擔憂影響，科技股普遍重挫，台股指數下跌超過3%。"),
            ("負面 - 公司虧損", "由於原物料成本上升和需求疲軟，該公司本季虧損達50億元。"),
            ("負面 - 市場恐慌", "投資人對通膨持續升溫感到憂慮，紛紛拋售持股，市場出現恐慌性賣壓。"),
            ("負面 - 財務危機", "該集團因債務違約問題，股價連續跌停，面臨重整危機。"),
            
            # 中性情感測試
            ("中性 - 政策公告", "央行將於下週召開理監事會議，市場關注利率政策走向。"),
            ("中性 - 併購消息", "據消息人士透露，A公司正與B公司洽談合併事宜，預計下月公布詳細計畫。"),
            ("中性 - 財報發布", "鴻海將於本月底公布第四季財報，法人預估每股盈餘約2.1元。"),
            ("中性 - 市場數據", "今日台股成交量為1,200億元，較昨日略增，大盤呈現震盪整理格局。"),
        ]
        
        for test_name, text in chinese_test_cases:
            print(f"  測試: {test_name}")
            result = self.analyze_sentiment(text, test_name)
            self._print_result(result)
            time.sleep(1)  # 避免 API 限制
    
    def run_mixed_language_tests(self):
        """執行中英混合文本測試"""
        print("\n🔍 執行中英混合文本測試...")
        
        mixed_test_cases = [
            ("混合 - 國際新聞", "Apple 蘋果公司股價上漲5%，受惠於iPhone在中國市場的強勁銷售表現。"),
            ("混合 - 科技股", "NVIDIA 輝達公司因AI晶片需求激增，股價創新高，市值突破1兆美元。"),
            ("混合 - 財經報導", "Tesla 特斯拉在中國上海的Gigafactory產能提升，有助於降低生產成本。"),
            ("混合 - 市場分析", "Fed 聯準會升息政策對台股造成壓力，科技股表現相對疲軟。"),
        ]
        
        for test_name, text in mixed_test_cases:
            print(f"  測試: {test_name}")
            result = self.analyze_sentiment(text, test_name)
            self._print_result(result)
            time.sleep(1)
    
    def run_edge_case_tests(self):
        """執行邊界情況測試"""
        print("\n🔍 執行邊界情況測試...")
        
        edge_cases = [
            ("邊界 - 短文本", "漲"),
            ("邊界 - 長文本", "根據最新的市場研究報告顯示，在全球經濟不確定性持續增加的背景下，投資者對於風險資產的配置策略正在發生根本性的改變，特別是在科技股領域，由於人工智能和機器學習技術的快速發展，相關公司的估值模型需要重新評估，同時考慮到地緣政治風險、通膨壓力、以及央行貨幣政策的變化，市場參與者普遍認為未來六個月內股市將面臨更大的波動性。"),
            ("邊界 - 數字符號", "股價 +15% 📈 營收 $1.2B 💰"),
            ("邊界 - 空白文本", "   "),
            ("邊界 - 特殊字符", "股價@#$%^&*()上漲!!!"),
        ]
        
        for test_name, text in edge_cases:
            print(f"  測試: {test_name}")
            result = self.analyze_sentiment(text, test_name)
            self._print_result(result)
            time.sleep(1)
    
    def _print_result(self, result: Dict[str, Any]):
        """格式化輸出測試結果"""
        if "error" in result:
            print(f"    ❌ 錯誤: {result['error']}")
            return
        
        if "predictions" in result and result["predictions"]:
            predictions = result["predictions"]
            if isinstance(predictions, list) and len(predictions) > 0:
                # 找出最高信心度的預測
                best_prediction = max(predictions, key=lambda x: x.get('score', 0))
                label = best_prediction.get('label', 'Unknown')
                score = best_prediction.get('score', 0)
                
                # 情感標籤對應
                label_map = {
                    'positive': '正面 🟢',
                    'negative': '負面 🔴', 
                    'neutral': '中性 🟡'
                }
                
                display_label = label_map.get(label.lower(), label)
                print(f"    結果: {display_label} (信心度: {score:.3f})")
                print(f"    回應時間: {result['response_time']}秒")
            else:
                print(f"    結果: 無有效預測")
        else:
            print(f"    結果: 無預測結果")
    
    def generate_report(self) -> str:
        """生成測試報告"""
        if not self.test_results:
            return "沒有測試結果可供分析"
        
        # 統計分析
        total_tests = len(self.test_results)
        successful_tests = len([r for r in self.test_results if "error" not in r])
        failed_tests = total_tests - successful_tests
        
        # 語言分布
        language_dist = {}
        sentiment_dist = {'positive': 0, 'negative': 0, 'neutral': 0}
        avg_response_time = 0
        
        for result in self.test_results:
            if "error" not in result:
                lang = result.get('language', 'Unknown')
                language_dist[lang] = language_dist.get(lang, 0) + 1
                
                if 'response_time' in result:
                    avg_response_time += result['response_time']
                
                # 統計情感分布
                if 'predictions' in result and result['predictions']:
                    predictions = result['predictions']
                    if isinstance(predictions, list) and len(predictions) > 0:
                        best_pred = max(predictions, key=lambda x: x.get('score', 0))
                        label = best_pred.get('label', '').lower()
                        if label in sentiment_dist:
                            sentiment_dist[label] += 1
        
        avg_response_time = avg_response_time / successful_tests if successful_tests > 0 else 0
        
        # 生成報告
        report = f"""
📊 FinBERT 測試報告
{'='*50}
📅 測試時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🔢 總測試數: {total_tests}
✅ 成功測試: {successful_tests}
❌ 失敗測試: {failed_tests}
⏱️  平均回應時間: {avg_response_time:.3f}秒

📈 語言分布:
{chr(10).join([f"  {lang}: {count}個測試" for lang, count in language_dist.items()])}

🎯 情感分析結果分布:
  正面情感: {sentiment_dist['positive']}個
  負面情感: {sentiment_dist['negative']}個  
  中性情感: {sentiment_dist['neutral']}個

💡 測試結論:
  - 模型成功率: {(successful_tests/total_tests)*100:.1f}%
  - 中文文本處理能力: {'良好' if language_dist.get('Chinese', 0) > 0 else '待測試'}
  - 平均處理速度: {'快速' if avg_response_time < 2 else '中等' if avg_response_time < 5 else '較慢'}
"""
        return report
    
    def save_results(self, filename: str = "finbert_test_results.json"):
        """儲存測試結果到 JSON 檔案"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, ensure_ascii=False, indent=2)
        print(f"📁 測試結果已儲存至: {filename}")
    
    def export_to_csv(self, filename: str = "finbert_test_results.csv"):
        """匯出測試結果到 CSV 檔案"""
        if not self.test_results:
            print("沒有測試結果可匯出")
            return
        
        # 準備 CSV 資料
        csv_data = []
        for result in self.test_results:
            row = {
                'test_name': result.get('test_name', ''),
                'input_text': result.get('input_text', ''),
                'language': result.get('language', ''),
                'timestamp': result.get('timestamp', ''),
                'response_time': result.get('response_time', ''),
                'error': result.get('error', '')
            }
            
            # 處理預測結果
            if 'predictions' in result and result['predictions']:
                predictions = result['predictions']
                if isinstance(predictions, list) and len(predictions) > 0:
                    best_pred = max(predictions, key=lambda x: x.get('score', 0))
                    row['predicted_label'] = best_pred.get('label', '')
                    row['confidence_score'] = best_pred.get('score', '')
                else:
                    row['predicted_label'] = ''
                    row['confidence_score'] = ''
            else:
                row['predicted_label'] = ''
                row['confidence_score'] = ''
            
            csv_data.append(row)
        
        # 儲存 CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"📊 測試結果已匯出至: {filename}")


def main():
    """主要執行函數"""
    print("🚀 FinBERT 綜合測試套件啟動")
    print("="*50)
    
    # 初始化測試器
    api_key = os.getenv('HF_TOKEN')
    if not api_key:
        print("❌ 錯誤: 請設置 HF_TOKEN 環境變量")
        print("請參考 README.md 了解如何設置環境變量")
        return
    tester = FinBERTTester(api_key)
    
    try:
        # 執行各種測試
        tester.run_english_financial_tests()
        tester.run_chinese_financial_tests()
        tester.run_mixed_language_tests()
        tester.run_edge_case_tests()
        
        # 生成並顯示報告
        print("\n" + "="*50)
        report = tester.generate_report()
        print(report)
        
        # 儲存結果
        tester.save_results()
        tester.export_to_csv()
        
        print("\n✅ 所有測試完成！")
        
    except KeyboardInterrupt:
        print("\n⚠️ 測試被使用者中斷")
    except Exception as e:
        print(f"\n❌ 測試過程中發生錯誤: {e}")
    finally:
        if tester.test_results:
            print(f"📝 已完成 {len(tester.test_results)} 個測試")


if __name__ == "__main__":
    main()

