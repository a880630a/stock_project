#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FinBERT 語言比較測試
比較相同意思的中英文金融文本在 FinBERT 模型下的表現差異
"""

import os
import time
import json
from huggingface_hub import InferenceClient
from datetime import datetime

def compare_languages():
    """比較中英文在相同金融情境下的分析結果"""
    
    api_key = os.getenv('HF_TOKEN')
    if not api_key:
        print("❌ 錯誤: 請設置 HF_TOKEN 環境變量")
        print("請參考 README.md 了解如何設置環境變量")
        return
    
    client = InferenceClient(
        provider="hf-inference",
        api_key=api_key,
    )
    
    print("🔍 FinBERT 中英文金融情感分析比較測試")
    print("="*70)
    
    # 相同意思的中英文對照測試案例
    comparison_cases = [
        {
            "scenario": "股價大漲情境",
            "english": "Apple stock surged 15% after reporting better-than-expected quarterly earnings, boosting investor confidence.",
            "chinese": "蘋果股價在公布超預期季度財報後大漲15%，提振投資人信心。",
            "expected_sentiment": "positive"
        },
        {
            "scenario": "股價重挫情境", 
            "english": "Tesla shares plummeted 20% following disappointing delivery numbers and production delays, causing investor panic.",
            "chinese": "特斯拉股價因交付數據令人失望和生產延遲而重挫20%，引發投資人恐慌。",
            "expected_sentiment": "negative"
        },
        {
            "scenario": "央行政策情境",
            "english": "The Federal Reserve will announce its interest rate decision next Wednesday, with markets awaiting guidance.",
            "chinese": "聯準會將於下週三宣布利率決策，市場等待政策指引。",
            "expected_sentiment": "neutral"
        },
        {
            "scenario": "公司併購情境",
            "english": "Microsoft announced the acquisition of a AI startup for $2 billion, strengthening its artificial intelligence capabilities.",
            "chinese": "微軟宣布以20億美元收購一家AI新創公司，強化人工智慧能力。",
            "expected_sentiment": "positive"
        },
        {
            "scenario": "市場恐慌情境",
            "english": "Global markets crashed amid recession fears, with investors fleeing to safe-haven assets like gold and bonds.",
            "chinese": "全球市場因經濟衰退恐懼而崩跌，投資人逃向黃金和債券等避險資產。",
            "expected_sentiment": "negative"
        },
        {
            "scenario": "財報發布情境",
            "english": "The company will release its quarterly earnings report on Friday, with analysts expecting mixed results.",
            "chinese": "該公司將於週五發布季度財報，分析師預期結果好壞參半。",
            "expected_sentiment": "neutral"
        }
    ]
    
    results = []
    
    for i, case in enumerate(comparison_cases, 1):
        print(f"\n📋 測試案例 {i}: {case['scenario']}")
        print(f"🎯 預期情感: {case['expected_sentiment']}")
        print("-" * 50)
        
        case_result = {
            "scenario": case['scenario'],
            "expected_sentiment": case['expected_sentiment'],
            "english": {},
            "chinese": {}
        }
        
        # 測試英文版本
        print("🇺🇸 英文版本:")
        print(f"   文本: {case['english']}")
        
        try:
            start_time = time.time()
            english_result = client.text_classification(case['english'], model="ProsusAI/finbert")
            end_time = time.time()
            
            if english_result and len(english_result) > 0:
                best_pred = max(english_result, key=lambda x: x.get('score', 0))
                case_result["english"] = {
                    "label": best_pred.get('label', ''),
                    "score": best_pred.get('score', 0),
                    "response_time": round(end_time - start_time, 3),
                    "all_predictions": english_result
                }
                
                label_icon = {'positive': '🟢', 'negative': '🔴', 'neutral': '🟡'}.get(best_pred.get('label', '').lower(), '❓')
                print(f"   結果: {label_icon} {best_pred.get('label', '')} (信心度: {best_pred.get('score', 0):.3f})")
                print(f"   時間: {case_result['english']['response_time']}秒")
                
        except Exception as e:
            print(f"   ❌ 錯誤: {e}")
            case_result["english"] = {"error": str(e)}
        
        time.sleep(1)  # API 限制
        
        # 測試中文版本
        print("🇹🇼 中文版本:")
        print(f"   文本: {case['chinese']}")
        
        try:
            start_time = time.time()
            chinese_result = client.text_classification(case['chinese'], model="ProsusAI/finbert")
            end_time = time.time()
            
            if chinese_result and len(chinese_result) > 0:
                best_pred = max(chinese_result, key=lambda x: x.get('score', 0))
                case_result["chinese"] = {
                    "label": best_pred.get('label', ''),
                    "score": best_pred.get('score', 0),
                    "response_time": round(end_time - start_time, 3),
                    "all_predictions": chinese_result
                }
                
                label_icon = {'positive': '🟢', 'negative': '🔴', 'neutral': '🟡'}.get(best_pred.get('label', '').lower(), '❓')
                print(f"   結果: {label_icon} {best_pred.get('label', '')} (信心度: {best_pred.get('score', 0):.3f})")
                print(f"   時間: {case_result['chinese']['response_time']}秒")
                
        except Exception as e:
            print(f"   ❌ 錯誤: {e}")
            case_result["chinese"] = {"error": str(e)}
        
        # 比較分析
        if "error" not in case_result["english"] and "error" not in case_result["chinese"]:
            print("🔍 比較分析:")
            
            eng_label = case_result["english"]["label"].lower()
            chi_label = case_result["chinese"]["label"].lower()
            eng_score = case_result["english"]["score"]
            chi_score = case_result["chinese"]["score"]
            
            if eng_label == chi_label:
                print(f"   ✅ 情感判斷一致: {eng_label}")
            else:
                print(f"   ⚠️ 情感判斷不一致: 英文={eng_label}, 中文={chi_label}")
            
            score_diff = abs(eng_score - chi_score)
            if score_diff < 0.1:
                print(f"   ✅ 信心度相近 (差異: {score_diff:.3f})")
            else:
                print(f"   ⚠️ 信心度差異較大 (差異: {score_diff:.3f})")
            
            # 準確性檢查
            expected = case['expected_sentiment'].lower()
            eng_correct = eng_label == expected
            chi_correct = chi_label == expected
            
            print(f"   🎯 英文準確性: {'✅' if eng_correct else '❌'}")
            print(f"   🎯 中文準確性: {'✅' if chi_correct else '❌'}")
        
        results.append(case_result)
        time.sleep(1)  # API 限制
    
    # 生成總結報告
    print("\n" + "="*70)
    print("📊 語言比較總結報告")
    print("="*70)
    
    successful_cases = [r for r in results if "error" not in r["english"] and "error" not in r["chinese"]]
    
    if successful_cases:
        print(f"✅ 成功比較案例: {len(successful_cases)}/{len(results)}")
        
        # 一致性分析
        consistent_sentiment = 0
        consistent_accuracy = 0
        english_correct = 0
        chinese_correct = 0
        
        english_scores = []
        chinese_scores = []
        
        for case in successful_cases:
            eng_label = case["english"]["label"].lower()
            chi_label = case["chinese"]["label"].lower()
            expected = case["expected_sentiment"].lower()
            
            if eng_label == chi_label:
                consistent_sentiment += 1
            
            eng_correct_case = eng_label == expected
            chi_correct_case = chi_label == expected
            
            if eng_correct_case:
                english_correct += 1
            if chi_correct_case:
                chinese_correct += 1
            if eng_correct_case == chi_correct_case:
                consistent_accuracy += 1
            
            english_scores.append(case["english"]["score"])
            chinese_scores.append(case["chinese"]["score"])
        
        print(f"\n🎯 一致性分析:")
        print(f"   情感判斷一致率: {consistent_sentiment}/{len(successful_cases)} ({consistent_sentiment/len(successful_cases)*100:.1f}%)")
        print(f"   準確性一致率: {consistent_accuracy}/{len(successful_cases)} ({consistent_accuracy/len(successful_cases)*100:.1f}%)")
        
        print(f"\n📈 準確性分析:")
        print(f"   英文準確率: {english_correct}/{len(successful_cases)} ({english_correct/len(successful_cases)*100:.1f}%)")
        print(f"   中文準確率: {chinese_correct}/{len(successful_cases)} ({chinese_correct/len(successful_cases)*100:.1f}%)")
        
        print(f"\n📊 信心度分析:")
        avg_eng_score = sum(english_scores) / len(english_scores)
        avg_chi_score = sum(chinese_scores) / len(chinese_scores)
        print(f"   英文平均信心度: {avg_eng_score:.3f} ({avg_eng_score*100:.1f}%)")
        print(f"   中文平均信心度: {avg_chi_score:.3f} ({avg_chi_score*100:.1f}%)")
        print(f"   信心度差異: {abs(avg_eng_score - avg_chi_score):.3f}")
        
        # 回應時間比較
        eng_times = [case["english"]["response_time"] for case in successful_cases]
        chi_times = [case["chinese"]["response_time"] for case in successful_cases]
        avg_eng_time = sum(eng_times) / len(eng_times)
        avg_chi_time = sum(chi_times) / len(chi_times)
        
        print(f"\n⏱️ 回應時間分析:")
        print(f"   英文平均時間: {avg_eng_time:.3f}秒")
        print(f"   中文平均時間: {avg_chi_time:.3f}秒")
        print(f"   時間差異: {abs(avg_eng_time - avg_chi_time):.3f}秒")
    
    print(f"\n💡 結論與建議:")
    if successful_cases:
        if consistent_sentiment / len(successful_cases) > 0.7:
            print("✅ FinBERT 在中英文情感判斷上具有良好的一致性")
        else:
            print("⚠️ FinBERT 在中英文情感判斷上存在一定差異")
        
        if english_correct > chinese_correct:
            print("📝 英文處理準確性優於中文，符合模型訓練特性")
        elif chinese_correct > english_correct:
            print("📝 中文處理表現出乎意料地優於英文")
        else:
            print("📝 中英文處理準確性相當")
        
        print("🔍 建議: 對於中文金融文本，建議:")
        print("   1. 將中文翻譯成英文後再分析")
        print("   2. 使用專門的中文金融情感分析模型")
        print("   3. 結合多個模型進行集成預測")
    
    # 儲存詳細結果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"finbert_language_comparison_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump({
            "test_info": {
                "timestamp": datetime.now().isoformat(),
                "model": "ProsusAI/finbert",
                "total_cases": len(results),
                "successful_cases": len(successful_cases)
            },
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n📁 詳細結果已儲存至: {filename}")
    
    return results

if __name__ == "__main__":
    compare_languages()

