#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FinBERT 簡化測試腳本
專注於測試中文和英文金融情感分析
"""

import os
import time
from huggingface_hub import InferenceClient

def test_finbert_sentiment():
    """測試 FinBERT 情感分析功能"""
    
    # 初始化客戶端
    api_key = os.getenv('HF_TOKEN')
    if not api_key:
        print("❌ 錯誤: 請設置 HF_TOKEN 環境變量")
        print("請參考 ENVIRONMENT_SETUP.md 了解如何設置環境變量")
        return
    
    client = InferenceClient(
        provider="hf-inference",
        api_key=api_key,
    )
    
    print("🚀 開始測試 FinBERT 金融情感分析模型")
    print("="*60)
    
    # 測試案例
    test_cases = [
        # 英文測試案例
        ("英文正面", "Apple stock surged 15% after reporting record quarterly earnings."),
        ("英文負面", "Tesla shares plummeted 20% following disappointing delivery numbers."),
        ("英文中性", "The Federal Reserve will announce its interest rate decision next week."),
        
        # 中文測試案例  
        ("中文正面", "台積電股價今日大漲8%，創下歷史新高，投資人信心滿滿。"),
        ("中文負面", "受到經濟衰退擔憂影響，科技股普遍重挫，台股大跌3%。"),
        ("中文中性", "央行將於下週召開會議，市場關注利率政策走向。"),
        
        # 中英混合測試
        ("中英混合", "NVIDIA 輝達公司因AI晶片需求激增，股價創新高。"),
        
        # 複雜金融術語測試
        ("複雜英文", "The company's EBITDA margin improved significantly due to operational efficiency gains and cost optimization strategies."),
        ("複雜中文", "該公司本季毛利率提升至35%，主要受惠於產品組合優化和成本控制策略。"),
    ]
    
    results = []
    
    for test_name, text in test_cases:
        print(f"\n📝 測試案例: {test_name}")
        print(f"📄 輸入文本: {text}")
        
        try:
            # 執行情感分析
            start_time = time.time()
            result = client.text_classification(text, model="ProsusAI/finbert")
            end_time = time.time()
            
            # 處理結果
            if isinstance(result, list) and len(result) > 0:
                # 找出最高信心度的預測
                best_prediction = max(result, key=lambda x: x.get('score', 0))
                label = best_prediction.get('label', 'Unknown')
                score = best_prediction.get('score', 0)
                
                # 情感標籤美化
                label_icons = {
                    'positive': '🟢 正面',
                    'negative': '🔴 負面', 
                    'neutral': '🟡 中性'
                }
                
                display_label = label_icons.get(label.lower(), f"❓ {label}")
                response_time = round(end_time - start_time, 3)
                
                print(f"🎯 分析結果: {display_label}")
                print(f"📊 信心度: {score:.3f} ({score*100:.1f}%)")
                print(f"⏱️  回應時間: {response_time}秒")
                
                # 顯示所有預測結果
                if len(result) > 1:
                    print("📋 完整預測結果:")
                    for pred in sorted(result, key=lambda x: x.get('score', 0), reverse=True):
                        pred_label = label_icons.get(pred.get('label', '').lower(), pred.get('label', ''))
                        pred_score = pred.get('score', 0)
                        print(f"   {pred_label}: {pred_score:.3f} ({pred_score*100:.1f}%)")
                
                results.append({
                    'test_name': test_name,
                    'text': text,
                    'label': label,
                    'score': score,
                    'response_time': response_time,
                    'success': True
                })
                
            else:
                print("❌ 無法獲得有效的預測結果")
                results.append({
                    'test_name': test_name,
                    'text': text,
                    'success': False,
                    'error': 'No valid predictions'
                })
                
        except Exception as e:
            print(f"❌ 錯誤: {str(e)}")
            results.append({
                'test_name': test_name,
                'text': text,
                'success': False,
                'error': str(e)
            })
        
        # 避免 API 限制
        time.sleep(1)
    
    # 生成測試總結
    print("\n" + "="*60)
    print("📊 測試總結報告")
    print("="*60)
    
    successful_tests = [r for r in results if r['success']]
    failed_tests = [r for r in results if not r['success']]
    
    print(f"✅ 成功測試: {len(successful_tests)}/{len(results)}")
    print(f"❌ 失敗測試: {len(failed_tests)}")
    
    if successful_tests:
        avg_response_time = sum(r['response_time'] for r in successful_tests) / len(successful_tests)
        print(f"⏱️  平均回應時間: {avg_response_time:.3f}秒")
        
        # 情感分布統計
        sentiment_count = {'positive': 0, 'negative': 0, 'neutral': 0}
        for r in successful_tests:
            label = r['label'].lower()
            if label in sentiment_count:
                sentiment_count[label] += 1
        
        print(f"📈 情感分析結果分布:")
        print(f"   🟢 正面: {sentiment_count['positive']}個")
        print(f"   🔴 負面: {sentiment_count['negative']}個")
        print(f"   🟡 中性: {sentiment_count['neutral']}個")
        
        # 語言處理能力分析
        chinese_tests = [r for r in successful_tests if any('\u4e00' <= char <= '\u9fff' for char in r['text'])]
        english_tests = [r for r in successful_tests if r not in chinese_tests]
        
        print(f"🌐 語言處理能力:")
        print(f"   🇺🇸 英文測試: {len(english_tests)}個成功")
        print(f"   🇹🇼 中文測試: {len(chinese_tests)}個成功")
        
        if chinese_tests:
            avg_chinese_confidence = sum(r['score'] for r in chinese_tests) / len(chinese_tests)
            print(f"   📊 中文平均信心度: {avg_chinese_confidence:.3f} ({avg_chinese_confidence*100:.1f}%)")
        
        if english_tests:
            avg_english_confidence = sum(r['score'] for r in english_tests) / len(english_tests)
            print(f"   📊 英文平均信心度: {avg_english_confidence:.3f} ({avg_english_confidence*100:.1f}%)")
    
    if failed_tests:
        print(f"\n❌ 失敗的測試案例:")
        for r in failed_tests:
            print(f"   {r['test_name']}: {r.get('error', '未知錯誤')}")
    
    print("\n💡 結論與建議:")
    if len(chinese_tests) > 0:
        print("✅ FinBERT 模型能夠處理中文金融文本")
        if chinese_tests:
            avg_chinese_score = sum(r['score'] for r in chinese_tests) / len(chinese_tests)
            if avg_chinese_score > 0.8:
                print("🎯 中文處理品質: 優秀")
            elif avg_chinese_score > 0.6:
                print("🎯 中文處理品質: 良好")
            else:
                print("🎯 中文處理品質: 待改善")
    else:
        print("⚠️ 無法評估中文處理能力")
    
    print("📝 建議: FinBERT 主要針對英文金融文本訓練，中文結果僅供參考")
    print("🔍 如需更準確的中文金融情感分析，建議使用專門的中文金融模型")
    
    return results

if __name__ == "__main__":
    test_finbert_sentiment()

