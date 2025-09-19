#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FinBERT API 測試腳本
測試所有API端點的功能，確保正常運行
"""

import requests
import json
import time
from datetime import datetime

class APITester:
    """API測試器"""
    
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.test_results = []
    
    def test_health_check(self):
        """測試健康檢查端點"""
        print("🔍 測試健康檢查端點...")
        try:
            response = requests.get(f"{self.base_url}/api/health", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 健康檢查成功")
                print(f"   狀態: {data.get('status')}")
                print(f"   分析器就緒: {data.get('analyzer_ready')}")
                print(f"   時間戳: {data.get('timestamp')}")
                return True
            else:
                print(f"❌ 健康檢查失敗: HTTP {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ 健康檢查連接失敗: {e}")
            return False
    
    def test_basic_sentiment_analysis(self):
        """測試基本情感分析端點"""
        print("\n🔍 測試基本情感分析端點...")
        
        test_cases = [
            {
                "name": "英文正面",
                "text": "Apple stock surged 15% after reporting record quarterly earnings."
            },
            {
                "name": "英文負面", 
                "text": "Tesla shares plummeted 20% following disappointing delivery numbers."
            },
            {
                "name": "中文正面",
                "text": "台積電股價今日大漲8%，創下歷史新高。"
            }
        ]
        
        success_count = 0
        
        for case in test_cases:
            print(f"\n   測試案例: {case['name']}")
            try:
                payload = {"text": case["text"]}
                response = requests.post(
                    f"{self.base_url}/api/analyze",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('success'):
                        print(f"   ✅ 分析成功")
                        print(f"   語言: {data.get('language')}")
                        print(f"   最佳預測: {data.get('best_prediction', {}).get('label')} "
                              f"({data.get('best_prediction', {}).get('confidence')})")
                        print(f"   回應時間: {data.get('response_time')}秒")
                        success_count += 1
                    else:
                        print(f"   ❌ 分析失敗: {data.get('error')}")
                else:
                    print(f"   ❌ HTTP錯誤: {response.status_code}")
                    print(f"   回應: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                print(f"   ❌ 請求失敗: {e}")
            
            time.sleep(1)  # 避免API限制
        
        print(f"\n📊 基本情感分析測試結果: {success_count}/{len(test_cases)} 成功")
        return success_count == len(test_cases)
    
    def test_sentiment_scoring(self):
        """測試情感評分端點"""
        print("\n🔍 測試情感評分端點...")
        
        test_cases = [
            {
                "name": "正面新聞",
                "text": "Apple stock surges after earnings report."
            },
            {
                "name": "負面新聞",
                "text": "Tesla shares plummet due to production delays."
            }
        ]
        
        success_count = 0
        
        for case in test_cases:
            print(f"\n   測試案例: {case['name']}")
            try:
                payload = {"text": case["text"]}
                response = requests.post(
                    f"{self.base_url}/api/analyze-score",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('success'):
                        print(f"   ✅ 評分成功")
                        print(f"   情感評分: {data.get('sentiment_score')}")
                        print(f"   情感類別: {data.get('sentiment_label')} {data.get('sentiment_icon')}")
                        print(f"   回應時間: {data.get('response_time')}秒")
                        success_count += 1
                    else:
                        print(f"   ❌ 評分失敗: {data.get('error')}")
                else:
                    print(f"   ❌ HTTP錯誤: {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                print(f"   ❌ 請求失敗: {e}")
            
            time.sleep(1)
        
        print(f"\n📊 情感評分測試結果: {success_count}/{len(test_cases)} 成功")
        return success_count == len(test_cases)
    
    def test_trading_strategy(self):
        """測試交易策略端點"""
        print("\n🔍 測試交易策略端點...")
        
        test_cases = [
            {
                "name": "看漲情境",
                "price": 100.0,
                "rsi": 0.3,
                "sentiment": 0.8
            },
            {
                "name": "看跌情境", 
                "price": 100.0,
                "rsi": 0.8,
                "sentiment": -0.6
            },
            {
                "name": "中性情境",
                "price": 100.0,
                "rsi": 0.5,
                "sentiment": 0.1
            }
        ]
        
        success_count = 0
        
        for case in test_cases:
            print(f"\n   測試案例: {case['name']}")
            try:
                payload = {
                    "price": case["price"],
                    "rsi": case["rsi"],
                    "sentiment": case["sentiment"]
                }
                response = requests.post(
                    f"{self.base_url}/api/trading-strategy",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('success'):
                        print(f"   ✅ 策略分析成功")
                        rec = data.get('recommendation', {})
                        print(f"   交易建議: {rec.get('label')} {rec.get('icon')} ({rec.get('action')})")
                        print(f"   建議描述: {rec.get('description')}")
                        print(f"   回應時間: {data.get('response_time')}秒")
                        success_count += 1
                    else:
                        print(f"   ❌ 策略分析失敗: {data.get('error')}")
                else:
                    print(f"   ❌ HTTP錯誤: {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                print(f"   ❌ 請求失敗: {e}")
            
            time.sleep(1)
        
        print(f"\n📊 交易策略測試結果: {success_count}/{len(test_cases)} 成功")
        return success_count == len(test_cases)
    
    def test_batch_analysis(self):
        """測試批次分析端點"""
        print("\n🔍 測試批次分析端點...")
        
        try:
            payload = {
                "texts": [
                    "Apple stock surged after earnings.",
                    "Tesla shares declined today.",
                    "市場保持穩定。"
                ]
            }
            response = requests.post(
                f"{self.base_url}/api/batch-analyze",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    print(f"   ✅ 批次分析成功")
                    print(f"   總文本數: {data.get('total_texts')}")
                    print(f"   成功分析數: {data.get('successful_analyses')}")
                    
                    # 顯示結果摘要
                    for result in data.get('results', [])[:3]:  # 只顯示前3個
                        if 'error' not in result:
                            pred = result.get('best_prediction', {})
                            print(f"   文本 {result.get('index')+1}: {pred.get('label')} ({pred.get('confidence')})")
                    
                    return True
                else:
                    print(f"   ❌ 批次分析失敗: {data.get('error')}")
                    return False
            else:
                print(f"   ❌ HTTP錯誤: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ 請求失敗: {e}")
            return False
    
    def test_error_handling(self):
        """測試錯誤處理"""
        print("\n🔍 測試錯誤處理...")
        
        # 測試空文本
        print("   測試空文本...")
        try:
            response = requests.post(
                f"{self.base_url}/api/analyze",
                json={"text": ""},
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            if response.status_code == 400:
                print("   ✅ 空文本錯誤處理正確")
            else:
                print(f"   ❌ 空文本錯誤處理異常: {response.status_code}")
        except:
            print("   ❌ 空文本測試失敗")
        
        # 測試缺少參數
        print("   測試缺少參數...")
        try:
            response = requests.post(
                f"{self.base_url}/api/trading-strategy",
                json={"price": 100.0},  # 缺少 rsi 和 sentiment
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            if response.status_code == 400:
                print("   ✅ 缺少參數錯誤處理正確")
            else:
                print(f"   ❌ 缺少參數錯誤處理異常: {response.status_code}")
        except:
            print("   ❌ 缺少參數測試失敗")
        
        # 測試無效路由
        print("   測試無效路由...")
        try:
            response = requests.get(f"{self.base_url}/api/nonexistent", timeout=10)
            if response.status_code == 404:
                print("   ✅ 無效路由錯誤處理正確")
            else:
                print(f"   ❌ 無效路由錯誤處理異常: {response.status_code}")
        except:
            print("   ❌ 無效路由測試失敗")
    
    def run_all_tests(self):
        """運行所有測試"""
        print("🚀 開始FinBERT API完整測試")
        print("=" * 60)
        
        start_time = time.time()
        
        # 測試序列
        tests = [
            ("健康檢查", self.test_health_check),
            ("基本情感分析", self.test_basic_sentiment_analysis),
            ("情感評分", self.test_sentiment_scoring),
            ("交易策略", self.test_trading_strategy),
            ("批次分析", self.test_batch_analysis),
            ("錯誤處理", self.test_error_handling)
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                result = test_func()
                results.append((test_name, result))
            except Exception as e:
                print(f"❌ {test_name}測試異常: {e}")
                results.append((test_name, False))
        
        # 總結報告
        end_time = time.time()
        print("\n" + "=" * 60)
        print("📊 API測試總結報告")
        print("=" * 60)
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        print(f"✅ 通過測試: {passed}/{total}")
        print(f"⏱️  總測試時間: {end_time - start_time:.2f}秒")
        print(f"📅 測試時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\n詳細結果:")
        for test_name, result in results:
            status = "✅ 通過" if result else "❌ 失敗"
            print(f"   {test_name}: {status}")
        
        if passed == total:
            print("\n🎉 所有API測試通過！API服務運行正常。")
        else:
            print(f"\n⚠️  有 {total - passed} 個測試失敗，請檢查API服務。")
        
        return passed == total

def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description="FinBERT API 測試工具")
    parser.add_argument('--url', default='http://localhost:5000',
                       help='API服務器地址 (預設: http://localhost:5000)')
    parser.add_argument('--quick', action='store_true',
                       help='快速測試模式（僅測試健康檢查和基本功能）')
    
    args = parser.parse_args()
    
    tester = APITester(args.url)
    
    if args.quick:
        print("🏃 快速測試模式")
        health_ok = tester.test_health_check()
        if health_ok:
            basic_ok = tester.test_basic_sentiment_analysis()
            if basic_ok:
                print("\n✅ 快速測試通過！API基本功能正常。")
            else:
                print("\n❌ 快速測試失敗！請檢查API服務。")
        else:
            print("\n❌ 健康檢查失敗！請確認API服務是否運行。")
    else:
        tester.run_all_tests()

if __name__ == "__main__":
    main()
