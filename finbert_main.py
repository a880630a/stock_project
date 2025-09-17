#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FinBERT 金融情感分析 - 主程式
整合所有 FinBERT 功能的統一介面

使用方式:
    python finbert_main.py --help                    # 顯示幫助
    python finbert_main.py --interactive             # 互動模式
    python finbert_main.py --simple-test             # 快速測試
    python finbert_main.py --comprehensive-test      # 綜合測試
    python finbert_main.py --language-comparison     # 中英文對比
    python finbert_main.py --analyze "your text"     # 分析指定文本
"""

import sys
import argparse
import time
import json
import os
from dotenv import load_dotenv
from datetime import datetime
from huggingface_hub import InferenceClient
from typing import List, Dict, Any, Optional
import pandas as pd
import re

class FinBERTAnalyzer:
    """FinBERT 金融情感分析器"""
    
    def __init__(self, api_key: str = None):
        """
        初始化 FinBERT 分析器
        
        Args:
            api_key: HuggingFace API 金鑰，如果為None則從環境變量讀取
        """
        # 載入 .env 檔案
        load_dotenv()
        
        # 設定環境變數
        if api_key is None:
            api_key = os.getenv('HF_TOKEN')
            if not api_key:
                raise ValueError("請設置 HF_TOKEN 環境變量或提供 api_key 參數")
        os.environ["HF_TOKEN"] = api_key
        
        # 初始化 InferenceClient
        self.client = InferenceClient(model="ProsusAI/finbert")
        self.model_name = "ProsusAI/finbert"
        self.test_results = []
        
    def analyze_sentiment(self, text: str, test_name: str = "") -> Dict[str, Any]:
        """
        分析單一文本的金融情感
        
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
                "language": self._detect_language(text),
                "success": True
            }
            
            self.test_results.append(analysis_result)
            return analysis_result
            
        except Exception as e:
            error_result = {
                "test_name": test_name,
                "input_text": text,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "language": self._detect_language(text),
                "success": False
            }
            self.test_results.append(error_result)
            return error_result
    
    def analyze_sentiment_with_score(self, text: str, test_name: str = "") -> Dict[str, Any]:
        """
        分析文本情感並返回數值化評分 (-1 到 1)
        
        Args:
            text: 要分析的文本
            test_name: 測試案例名稱
            
        Returns:
            包含數值化評分的分析結果
        """
        try:
            start_time = time.time()
            
            # 使用 text_generation 進行情感分析
            sentiment_prompt = f"Analyze sentiment of financial news: '{text}'. Score -1 (negative) to 1 (positive). Format: 'Sentiment: [score]'"
            sentiment_response = self.client.text_generation(
                sentiment_prompt, 
                max_new_tokens=20, 
                temperature=0.1
            )
            
            end_time = time.time()
            
            # 解析數值評分
            sentiment_score = self._parse_sentiment_score(sentiment_response)
            
            analysis_result = {
                "test_name": test_name,
                "input_text": text,
                "raw_response": sentiment_response,
                "sentiment_score": sentiment_score,
                "response_time": round(end_time - start_time, 3),
                "timestamp": datetime.now().isoformat(),
                "language": self._detect_language(text),
                "success": True,
                "analysis_type": "sentiment_scoring"
            }
            
            self.test_results.append(analysis_result)
            return analysis_result
            
        except Exception as e:
            error_result = {
                "test_name": test_name,
                "input_text": text,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "language": self._detect_language(text),
                "success": False,
                "analysis_type": "sentiment_scoring"
            }
            self.test_results.append(error_result)
            return error_result
    
    def analyze_sentiment_with_explanation(self, text: str, test_name: str = "") -> Dict[str, Any]:
        """
        分析文本情感並提供詳細解釋、總結和依據
        
        Args:
            text: 要分析的文本
            test_name: 測試案例名稱
            
        Returns:
            包含詳細解釋的分析結果
        """
        try:
            start_time = time.time()
            
            # 1. 首先進行基本情感分析
            basic_result = self.client.text_classification(text, model=self.model_name)
            
            # 2. 使用支持文本生成的模型進行總結和解釋
            # 使用 microsoft/DialoGPT-medium 或其他支持文本生成的模型
            generation_client = InferenceClient(model="microsoft/DialoGPT-medium")
            
            # 3. 生成文本總結（基於關鍵詞分析）
            text_summary = self._generate_text_summary(text)
            
            # 4. 分析關鍵詞和依據
            analysis_explanation = self._generate_analysis_explanation(text, basic_result)
            
            # 5. 結果解釋
            if isinstance(basic_result, list) and len(basic_result) > 0:
                predictions = basic_result
                best_prediction = max(predictions, key=lambda x: x.get('score', 0))
                predicted_label = best_prediction.get('label', 'Unknown').lower()
                confidence = best_prediction.get('score', 0)
                
                result_explanation = self._generate_result_explanation(text, predicted_label, confidence, predictions)
            else:
                predictions = []
                result_explanation = "無法生成結果解釋"
            
            end_time = time.time()
            
            analysis_result = {
                "test_name": test_name,
                "input_text": text,
                "predictions": predictions,
                "text_summary": text_summary,
                "analysis_explanation": analysis_explanation,
                "result_explanation": result_explanation,
                "response_time": round(end_time - start_time, 3),
                "timestamp": datetime.now().isoformat(),
                "language": self._detect_language(text),
                "success": True,
                "analysis_type": "enhanced_sentiment_analysis"
            }
            
            self.test_results.append(analysis_result)
            return analysis_result
            
        except Exception as e:
            error_result = {
                "test_name": test_name,
                "input_text": text,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "language": self._detect_language(text),
                "success": False,
                "analysis_type": "enhanced_sentiment_analysis"
            }
            self.test_results.append(error_result)
            return error_result

    def generate_trading_strategy(self, price: float, rsi: float, sentiment: float, test_name: str = "") -> Dict[str, Any]:
        """
        基於價格、RSI 和情感分析生成交易策略
        
        Args:
            price: 當前股價
            rsi: RSI 指標值 (0-1)
            sentiment: 情感評分 (-1 到 1)
            test_name: 測試案例名稱
            
        Returns:
            包含交易策略建議的分析結果
        """
        try:
            start_time = time.time()
            
            # 構建策略分析提示
            strategy_prompt = (
                f"As aggressive trader, state (price: {price}, RSI: {rsi}, sentiment: {sentiment}), "
                f"suggest buy/sell/hold. Reason: Step 1: Analyze price. Step 2: Assess RSI and sentiment. "
                f"Step 3: Decide."
            )
            
            strategy_response = self.client.text_generation(
                strategy_prompt, 
                max_new_tokens=100, 
                temperature=0.7
            )
            
            end_time = time.time()
            
            # 解析交易動作
            action = self.parse_action(strategy_response)
            
            strategy_result = {
                "test_name": test_name,
                "input_state": {
                    "price": price,
                    "rsi": rsi,
                    "sentiment": sentiment
                },
                "raw_response": strategy_response,
                "parsed_action": action,
                "action_label": self._get_action_label(action),
                "response_time": round(end_time - start_time, 3),
                "timestamp": datetime.now().isoformat(),
                "success": True,
                "analysis_type": "trading_strategy"
            }
            
            self.test_results.append(strategy_result)
            return strategy_result
            
        except Exception as e:
            error_result = {
                "test_name": test_name,
                "input_state": {
                    "price": price,
                    "rsi": rsi,
                    "sentiment": sentiment
                },
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "analysis_type": "trading_strategy"
            }
            self.test_results.append(error_result)
            return error_result
    
    def parse_action(self, response: str) -> int:
        """
        解析交易策略回應中的動作
        
        Args:
            response: 策略回應文本
            
        Returns:
            動作代碼 (0=sell, 1=hold, 2=buy)
        """
        response_lower = response.lower()
        if "buy" in response_lower:
            return 2
        elif "sell" in response_lower:
            return 0
        else:
            return 1  # hold
    
    def _parse_sentiment_score(self, response: str) -> float:
        """
        從回應中解析情感評分
        
        Args:
            response: API 回應文本
            
        Returns:
            情感評分 (-1 到 1)
        """
        try:
            # 尋找數值模式
            import re
            patterns = [
                r'sentiment:\s*([+-]?\d*\.?\d+)',
                r'score:\s*([+-]?\d*\.?\d+)',
                r'([+-]?\d*\.?\d+)',
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, response.lower())
                if matches:
                    score = float(matches[0])
                    # 確保評分在 -1 到 1 範圍內
                    return max(-1.0, min(1.0, score))
            
            # 如果無法解析數值，根據關鍵詞判斷
            response_lower = response.lower()
            if any(word in response_lower for word in ['positive', 'bullish', 'good', 'up']):
                return 0.5
            elif any(word in response_lower for word in ['negative', 'bearish', 'bad', 'down']):
                return -0.5
            else:
                return 0.0
                
        except (ValueError, IndexError):
            return 0.0
    
    def _generate_text_summary(self, text: str) -> str:
        """
        基於關鍵詞分析生成文本總結
        
        Args:
            text: 輸入文本
            
        Returns:
            文本總結
        """
        try:
            # 提取關鍵詞和實體
            keywords = self._extract_keywords(text)
            entities = self._extract_entities(text)
            
            # 生成總結
            if keywords or entities:
                summary_parts = []
                if entities:
                    summary_parts.append(f"涉及實體：{', '.join(entities[:3])}")
                if keywords:
                    summary_parts.append(f"關鍵詞：{', '.join(keywords[:5])}")
                
                summary = f"這是一則關於{' | '.join(summary_parts)}的金融新聞。"
            else:
                summary = f"這是一則金融相關文本，長度為{len(text)}個字符。"
                
            return summary
            
        except Exception as e:
            return f"文本總結生成失敗：{str(e)}"
    
    def _generate_analysis_explanation(self, text: str, predictions: List[Dict]) -> str:
        """
        生成分析依據和關鍵詞解釋
        
        Args:
            text: 輸入文本
            predictions: 預測結果
            
        Returns:
            分析依據說明
        """
        try:
            # 分析關鍵詞
            positive_keywords = self._find_sentiment_keywords(text, 'positive')
            negative_keywords = self._find_sentiment_keywords(text, 'negative')
            neutral_keywords = self._find_sentiment_keywords(text, 'neutral')
            
            explanation_parts = []
            
            # 關鍵詞分析
            if positive_keywords:
                explanation_parts.append(f"正面關鍵詞：{', '.join(positive_keywords)}")
            if negative_keywords:
                explanation_parts.append(f"負面關鍵詞：{', '.join(negative_keywords)}")
            if neutral_keywords:
                explanation_parts.append(f"中性關鍵詞：{', '.join(neutral_keywords)}")
            
            # 語言分析
            language = self._detect_language(text)
            explanation_parts.append(f"語言類型：{language}")
            
            # 文本長度分析
            explanation_parts.append(f"文本長度：{len(text)}字符")
            
            return " | ".join(explanation_parts) if explanation_parts else "無法提取明確的分析依據"
            
        except Exception as e:
            return f"分析依據生成失敗：{str(e)}"
    
    def _generate_result_explanation(self, text: str, predicted_label: str, confidence: float, predictions: List[Dict]) -> str:
        """
        生成結果解釋
        
        Args:
            text: 輸入文本
            predicted_label: 預測標籤
            confidence: 信心度
            predictions: 所有預測結果
            
        Returns:
            結果解釋
        """
        try:
            explanation_parts = []
            
            # 信心度解釋
            if confidence > 0.8:
                confidence_desc = "非常高的信心度"
            elif confidence > 0.6:
                confidence_desc = "較高的信心度"
            elif confidence > 0.4:
                confidence_desc = "中等信心度"
            else:
                confidence_desc = "較低的信心度"
            
            explanation_parts.append(f"模型以{confidence_desc}({confidence:.3f})判斷此文本為{predicted_label}情感")
            
            # 競爭標籤分析
            if len(predictions) > 1:
                sorted_preds = sorted(predictions, key=lambda x: x.get('score', 0), reverse=True)
                if len(sorted_preds) >= 2:
                    second_best = sorted_preds[1]
                    second_label = second_best.get('label', '').lower()
                    second_score = second_best.get('score', 0)
                    
                    score_diff = confidence - second_score
                    if score_diff < 0.2:
                        explanation_parts.append(f"與次高分類{second_label}({second_score:.3f})差距較小，存在一定不確定性")
                    else:
                        explanation_parts.append(f"與次高分類{second_label}({second_score:.3f})差距明顯，分類較為確定")
            
            # 語言相關風險
            language = self._detect_language(text)
            if language in ["中文", "中英混合"]:
                explanation_parts.append(f"注意：FinBERT主要針對英文訓練，對{language}文本的分析可能存在偏差")
            
            return " | ".join(explanation_parts)
            
        except Exception as e:
            return f"結果解釋生成失敗：{str(e)}"
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取關鍵詞"""
        # 簡單的關鍵詞提取邏輯
        import re
        
        # 金融相關關鍵詞模式（中英文）
        financial_patterns = [
            r'(?:股價|股票|股市|股份|stock|share)',
            r'(?:上漲|下跌|大漲|重挫|暴跌|飆升|創新高|創新低|surge|soar|plummet|crash)',
            r'(?:財報|營收|獲利|虧損|EPS|earnings|revenue|profit|loss)',
            r'(?:投資|交易|買入|賣出|持有|investment|trading|buy|sell|hold)',
            r'(?:市場|經濟|金融|銀行|market|economy|financial|bank)',
            r'(?:Apple|蘋果|Tesla|特斯拉|NVIDIA|輝達|台積電|TSMC|聯發科|MediaTek)',
            r'(?:信心|樂觀|悲觀|看好|看空|confidence|optimistic|pessimistic|bullish|bearish)',
            r'(?:強勁|疲軟|穩定|波動|strong|weak|stable|volatile)',
        ]
        
        keywords = []
        for pattern in financial_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            keywords.extend(matches)
        
        return list(set(keywords))[:10]  # 返回前10個不重複的關鍵詞
    
    def _extract_entities(self, text: str) -> List[str]:
        """提取實體（公司名稱等）"""
        import re
        
        # 常見公司名稱模式
        company_patterns = [
            r'\b(?:Apple|蘋果)\b',
            r'\b(?:Tesla|特斯拉)\b', 
            r'\b(?:NVIDIA|輝達)\b',
            r'\b(?:台積電|TSMC)\b',
            r'\b(?:聯發科|MediaTek)\b',
            r'\b(?:Google|谷歌)\b',
            r'\b(?:Microsoft|微軟)\b',
            r'\b(?:Amazon|亞馬遜)\b',
        ]
        
        entities = []
        for pattern in company_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.extend(matches)
        
        return list(set(entities))[:5]  # 返回前5個不重複的實體
    
    def _find_sentiment_keywords(self, text: str, sentiment_type: str) -> List[str]:
        """尋找特定情感的關鍵詞"""
        import re
        
        sentiment_patterns = {
            'positive': [
                r'(?:上漲|大漲|飆升|創新高|突破|強勁|樂觀|看好|利多|信心大增|表現強勁)',
                r'(?:surge|soar|rally|bullish|positive|gain|rise|up|beating|strong|confidence)'
            ],
            'negative': [
                r'(?:下跌|重挫|暴跌|創新低|崩盤|悲觀|看空|利空|虧損|表現疲軟)',
                r'(?:plummet|crash|bearish|negative|loss|decline|down|fall|disappointing|weak)'
            ],
            'neutral': [
                r'(?:持平|穩定|觀望|中性|公布|宣布|會議|決策|發布)',
                r'(?:stable|neutral|announce|meeting|decision|report|release|publish)'
            ]
        }
        
        keywords = []
        patterns = sentiment_patterns.get(sentiment_type, [])
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            keywords.extend(matches)
        
        return list(set(keywords))[:5]  # 返回前5個不重複的關鍵詞

    def _get_action_label(self, action: int) -> str:
        """
        獲取動作標籤
        
        Args:
            action: 動作代碼
            
        Returns:
            動作標籤
        """
        action_labels = {
            0: "賣出 (Sell)",
            1: "持有 (Hold)", 
            2: "買入 (Buy)"
        }
        return action_labels.get(action, "未知動作")
    
    def _detect_language(self, text: str) -> str:
        """檢測文本語言"""
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
    
    def format_result(self, result: Dict[str, Any], detailed: bool = True) -> str:
        """格式化輸出結果"""
        if not result.get('success', False):
            return f"❌ 分析失敗: {result.get('error', '未知錯誤')}"
        
        analysis_type = result.get('analysis_type', 'sentiment_classification')
        
        if analysis_type == 'enhanced_sentiment_analysis':
            return self._format_enhanced_analysis_result(result, detailed)
        elif analysis_type == 'sentiment_scoring':
            return self._format_sentiment_scoring_result(result, detailed)
        elif analysis_type == 'trading_strategy':
            return self._format_trading_strategy_result(result, detailed)
        else:
            return self._format_classification_result(result, detailed)
    
    def _format_classification_result(self, result: Dict[str, Any], detailed: bool = True) -> str:
        """格式化分類結果"""
        predictions = result.get('predictions', [])
        if not predictions:
            return "❌ 無法獲得有效的預測結果"
        
        # 找出最高信心度的預測
        best_prediction = max(predictions, key=lambda x: x.get('score', 0))
        label = best_prediction.get('label', 'Unknown').lower()
        score = best_prediction.get('score', 0)
        
        # 情感標籤美化
        label_icons = {
            'positive': '🟢 正面',
            'negative': '🔴 負面',
            'neutral': '🟡 中性'
        }
        
        display_label = label_icons.get(label, f"❓ {label}")
        
        if detailed:
            output = f"""
📝 輸入文本: {result['input_text']}
🌐 語言類型: {result['language']}
🎯 分析結果: {display_label}
📊 信心度: {score:.3f} ({score*100:.1f}%)
⏱️  回應時間: {result['response_time']}秒

📋 詳細預測結果:"""
            
            for pred in sorted(predictions, key=lambda x: x.get('score', 0), reverse=True):
                pred_label = pred.get('label', '').lower()
                pred_score = pred.get('score', 0)
                pred_display = label_icons.get(pred_label, pred_label)
                output += f"\n   {pred_display}: {pred_score:.3f} ({pred_score*100:.1f}%)"
            
            # 添加建議
            if result['language'] in ["中文", "中英混合"]:
                output += f"\n\n💡 建議: 檢測到{result['language']}文本，FinBERT主要針對英文訓練"
                output += f"\n   建議結合關鍵詞分析或翻譯成英文後再分析"
        else:
            output = f"{display_label} ({score:.3f})"
        
        return output
    
    def _format_sentiment_scoring_result(self, result: Dict[str, Any], detailed: bool = True) -> str:
        """格式化情感評分結果"""
        sentiment_score = result.get('sentiment_score', 0.0)
        
        # 根據評分確定情感類別
        if sentiment_score > 0.3:
            sentiment_icon = "🟢 正面"
        elif sentiment_score < -0.3:
            sentiment_icon = "🔴 負面"
        else:
            sentiment_icon = "🟡 中性"
        
        if detailed:
            output = f"""
📝 輸入文本: {result['input_text']}
🌐 語言類型: {result['language']}
🎯 情感評分: {sentiment_score:.3f} ({sentiment_icon})
📊 評分範圍: -1.0 (極度負面) 到 1.0 (極度正面)
⏱️  回應時間: {result['response_time']}秒
🤖 原始回應: {result.get('raw_response', 'N/A')}"""
            
            # 添加建議
            if result['language'] in ["中文", "中英混合"]:
                output += f"\n\n💡 建議: 檢測到{result['language']}文本，FinBERT主要針對英文訓練"
                output += f"\n   建議結合關鍵詞分析或翻譯成英文後再分析"
        else:
            output = f"{sentiment_icon} ({sentiment_score:.3f})"
        
        return output
    
    def _format_enhanced_analysis_result(self, result: Dict[str, Any], detailed: bool = True) -> str:
        """格式化增強分析結果"""
        predictions = result.get('predictions', [])
        if not predictions:
            return "❌ 無法獲得有效的預測結果"
        
        # 找出最高信心度的預測
        best_prediction = max(predictions, key=lambda x: x.get('score', 0))
        label = best_prediction.get('label', 'Unknown').lower()
        score = best_prediction.get('score', 0)
        
        # 情感標籤美化
        label_icons = {
            'positive': '🟢 正面',
            'negative': '🔴 負面',
            'neutral': '🟡 中性'
        }
        
        display_label = label_icons.get(label, f"❓ {label}")
        
        if detailed:
            output = f"""
📝 輸入文本: {result['input_text']}
🌐 語言類型: {result['language']}
🎯 分析結果: {display_label}
📊 信心度: {score:.3f} ({score*100:.1f}%)
⏱️  回應時間: {result['response_time']}秒

📋 詳細預測結果:"""
            
            for pred in sorted(predictions, key=lambda x: x.get('score', 0), reverse=True):
                pred_label = pred.get('label', '').lower()
                pred_score = pred.get('score', 0)
                pred_display = label_icons.get(pred_label, pred_label)
                output += f"\n   {pred_display}: {pred_score:.3f} ({pred_score*100:.1f}%)"
            
            # 添加文本總結
            text_summary = result.get('text_summary', '').strip()
            if text_summary:
                output += f"\n\n📄 文本總結:\n   {text_summary}"
            
            # 添加分析依據
            analysis_explanation = result.get('analysis_explanation', '').strip()
            if analysis_explanation:
                output += f"\n\n🔍 分析依據:\n   {analysis_explanation.replace(chr(10), chr(10) + '   ')}"
            
            # 添加結果解釋
            result_explanation = result.get('result_explanation', '').strip()
            if result_explanation:
                output += f"\n\n💡 結果解釋:\n   {result_explanation.replace(chr(10), chr(10) + '   ')}"
            
            # 添加建議
            if result['language'] in ["中文", "中英混合"]:
                output += f"\n\n⚠️  語言提醒: 檢測到{result['language']}文本，FinBERT主要針對英文訓練"
                output += f"\n   建議結合關鍵詞分析或翻譯成英文後再分析以獲得更準確的結果"
        else:
            output = f"{display_label} ({score:.3f})"
        
        return output
    
    def _format_trading_strategy_result(self, result: Dict[str, Any], detailed: bool = True) -> str:
        """格式化交易策略結果"""
        input_state = result.get('input_state', {})
        action = result.get('parsed_action', 1)
        action_label = result.get('action_label', '未知動作')
        
        # 動作圖示
        action_icons = {
            0: "🔴 賣出",
            1: "🟡 持有",
            2: "🟢 買入"
        }
        
        action_display = action_icons.get(action, f"❓ {action_label}")
        
        if detailed:
            output = f"""
📊 市場狀態:
   💰 價格: {input_state.get('price', 'N/A')}
   📈 RSI: {input_state.get('rsi', 'N/A')}
   😊 情感: {input_state.get('sentiment', 'N/A')}

🎯 交易建議: {action_display}
📋 策略代碼: {action} (0=賣出, 1=持有, 2=買入)
⏱️  回應時間: {result['response_time']}秒

🤖 AI 分析過程:
{result.get('raw_response', 'N/A')}"""
        else:
            output = f"{action_display} (代碼: {action})"
        
        return output

class FinBERTTestSuite:
    """FinBERT 測試套件"""
    
    def __init__(self, analyzer: FinBERTAnalyzer):
        self.analyzer = analyzer
    
    def run_simple_test(self) -> List[Dict[str, Any]]:
        """執行簡化版測試"""
        print("🚀 執行 FinBERT 簡化版測試")
        print("="*60)
        
        test_cases = [
            ("英文正面", "Apple stock surged 15% after reporting record quarterly earnings."),
            ("英文負面", "Tesla shares plummeted 20% following disappointing delivery numbers."),
            ("英文中性", "The Federal Reserve will announce its interest rate decision next week."),
            ("中文正面", "台積電股價今日大漲8%，創下歷史新高，投資人信心滿滿。"),
            ("中文負面", "受到經濟衰退擔憂影響，科技股普遍重挫，台股大跌3%。"),
            ("中文中性", "央行將於下週召開會議，市場關注利率政策走向。"),
            ("中英混合", "NVIDIA 輝達公司因AI晶片需求激增，股價創新高。"),
            ("複雜英文", "The company's EBITDA margin improved significantly due to operational efficiency gains."),
            ("複雜中文", "該公司本季毛利率提升至35%，主要受惠於產品組合優化。"),
        ]
        
        results = []
        for test_name, text in test_cases:
            print(f"\n📝 測試案例: {test_name}")
            result = self.analyzer.analyze_sentiment(text, test_name)
            formatted_result = self.analyzer.format_result(result)
            print(formatted_result)
            print("-" * 50)
            results.append(result)
            time.sleep(1)  # 避免 API 限制
        
        self._print_summary(results)
        return results
    
    def run_language_comparison(self) -> List[Dict[str, Any]]:
        """執行中英文對比測試"""
        print("🔍 執行 FinBERT 中英文對比測試")
        print("="*70)
        
        comparison_cases = [
            {
                "scenario": "股價大漲情境",
                "english": "Apple stock surged 15% after reporting better-than-expected earnings.",
                "chinese": "蘋果股價在公布超預期財報後大漲15%。",
                "expected_sentiment": "positive"
            },
            {
                "scenario": "股價重挫情境",
                "english": "Tesla shares plummeted 20% following disappointing delivery numbers.",
                "chinese": "特斯拉股價因交付數據令人失望而重挫20%。",
                "expected_sentiment": "negative"
            },
            {
                "scenario": "央行政策情境",
                "english": "The Federal Reserve will announce its interest rate decision next Wednesday.",
                "chinese": "聯準會將於下週三宣布利率決策。",
                "expected_sentiment": "neutral"
            }
        ]
        
        results = []
        for case in comparison_cases:
            print(f"\n📋 測試情境: {case['scenario']}")
            print(f"🎯 預期情感: {case['expected_sentiment']}")
            print("-" * 50)
            
            # 測試英文
            print("🇺🇸 英文版本:")
            eng_result = self.analyzer.analyze_sentiment(case['english'], f"{case['scenario']}_英文")
            print(self.analyzer.format_result(eng_result, detailed=False))
            
            time.sleep(1)
            
            # 測試中文
            print("🇹🇼 中文版本:")
            chi_result = self.analyzer.analyze_sentiment(case['chinese'], f"{case['scenario']}_中文")
            print(self.analyzer.format_result(chi_result, detailed=False))
            
            # 比較分析
            if eng_result.get('success') and chi_result.get('success'):
                eng_predictions = eng_result.get('predictions', [])
                chi_predictions = chi_result.get('predictions', [])
                
                if eng_predictions and chi_predictions:
                    eng_best = max(eng_predictions, key=lambda x: x.get('score', 0))
                    chi_best = max(chi_predictions, key=lambda x: x.get('score', 0))
                    
                    eng_label = eng_best.get('label', '').lower()
                    chi_label = chi_best.get('label', '').lower()
                    
                    if eng_label == chi_label:
                        print("✅ 情感判斷一致")
                    else:
                        print(f"⚠️ 情感判斷不一致: 英文={eng_label}, 中文={chi_label}")
            
            results.extend([eng_result, chi_result])
            print("=" * 50)
            time.sleep(1)
        
        return results
    
    def run_enhanced_test(self) -> List[Dict[str, Any]]:
        """執行增強版測試（包含情感評分和交易策略）"""
        print("🚀 執行 FinBERT 增強版測試")
        print("="*70)
        
        results = []
        
        # 1. 情感評分測試
        print("\n📊 情感評分測試")
        print("-" * 50)
        
        sentiment_test_cases = [
            ("正面新聞", "Apple stock surges after earnings report."),
            ("負面新聞", "Tesla shares plummet due to production delays."),
            ("中性新聞", "The Federal Reserve will announce interest rates next week."),
            ("中文正面", "台積電股價創新高，投資人信心大增"),
            ("中文負面", "受疫情影響，航空股全面重挫")
        ]
        
        for test_name, text in sentiment_test_cases:
            print(f"\n📝 測試案例: {test_name}")
            result = self.analyzer.analyze_sentiment_with_score(text, test_name)
            formatted_result = self.analyzer.format_result(result)
            print(formatted_result)
            print("-" * 30)
            results.append(result)
            time.sleep(1)
        
        # 2. 交易策略測試
        print("\n\n💼 交易策略測試")
        print("-" * 50)
        
        strategy_test_cases = [
            ("看漲情境", 100.0, 0.3, 0.8),  # 低RSI + 正面情感
            ("看跌情境", 100.0, 0.8, -0.6), # 高RSI + 負面情感
            ("中性情境", 100.0, 0.5, 0.1),  # 中等RSI + 中性情感
            ("極度看漲", 100.0, 0.2, 0.9),  # 極低RSI + 極正面情感
            ("極度看跌", 100.0, 0.9, -0.8), # 極高RSI + 極負面情感
        ]
        
        for test_name, price, rsi, sentiment in strategy_test_cases:
            print(f"\n📈 測試案例: {test_name}")
            result = self.analyzer.generate_trading_strategy(price, rsi, sentiment, test_name)
            formatted_result = self.analyzer.format_result(result)
            print(formatted_result)
            print("-" * 30)
            results.append(result)
            time.sleep(1)
        
        self._print_enhanced_summary(results)
        return results
    
    def _print_enhanced_summary(self, results: List[Dict[str, Any]]):
        """列印增強版測試總結"""
        print("\n" + "="*70)
        print("📊 增強版測試總結報告")
        print("="*70)
        
        successful_tests = [r for r in results if r.get('success', False)]
        failed_tests = [r for r in results if not r.get('success', False)]
        
        # 按分析類型分組
        sentiment_scoring_tests = [r for r in successful_tests if r.get('analysis_type') == 'sentiment_scoring']
        trading_strategy_tests = [r for r in successful_tests if r.get('analysis_type') == 'trading_strategy']
        
        print(f"✅ 總成功測試: {len(successful_tests)}/{len(results)}")
        print(f"❌ 總失敗測試: {len(failed_tests)}")
        
        if sentiment_scoring_tests:
            print(f"\n📊 情感評分測試:")
            print(f"   ✅ 成功: {len(sentiment_scoring_tests)}個")
            avg_score_time = sum(r.get('response_time', 0) for r in sentiment_scoring_tests) / len(sentiment_scoring_tests)
            print(f"   ⏱️  平均回應時間: {avg_score_time:.3f}秒")
            
            # 分析情感分佈
            scores = [r.get('sentiment_score', 0) for r in sentiment_scoring_tests]
            positive_count = sum(1 for s in scores if s > 0.3)
            negative_count = sum(1 for s in scores if s < -0.3)
            neutral_count = len(scores) - positive_count - negative_count
            
            print(f"   😊 情感分佈: 正面 {positive_count}個, 負面 {negative_count}個, 中性 {neutral_count}個")
        
        if trading_strategy_tests:
            print(f"\n💼 交易策略測試:")
            print(f"   ✅ 成功: {len(trading_strategy_tests)}個")
            avg_strategy_time = sum(r.get('response_time', 0) for r in trading_strategy_tests) / len(trading_strategy_tests)
            print(f"   ⏱️  平均回應時間: {avg_strategy_time:.3f}秒")
            
            # 分析交易動作分佈
            actions = [r.get('parsed_action', 1) for r in trading_strategy_tests]
            buy_count = sum(1 for a in actions if a == 2)
            sell_count = sum(1 for a in actions if a == 0)
            hold_count = sum(1 for a in actions if a == 1)
            
            print(f"   📈 交易動作分佈: 買入 {buy_count}個, 賣出 {sell_count}個, 持有 {hold_count}個")
        
        if successful_tests:
            avg_response_time = sum(r.get('response_time', 0) for r in successful_tests) / len(successful_tests)
            print(f"\n⏱️  整體平均回應時間: {avg_response_time:.3f}秒")
    
    def _print_summary(self, results: List[Dict[str, Any]]):
        """列印測試總結"""
        print("\n" + "="*60)
        print("📊 測試總結報告")
        print("="*60)
        
        successful_tests = [r for r in results if r.get('success', False)]
        failed_tests = [r for r in results if not r.get('success', False)]
        
        print(f"✅ 成功測試: {len(successful_tests)}/{len(results)}")
        print(f"❌ 失敗測試: {len(failed_tests)}")
        
        if successful_tests:
            avg_response_time = sum(r.get('response_time', 0) for r in successful_tests) / len(successful_tests)
            print(f"⏱️  平均回應時間: {avg_response_time:.3f}秒")
            
            # 語言分析
            chinese_tests = [r for r in successful_tests if '中文' in r.get('language', '')]
            english_tests = [r for r in successful_tests if r.get('language') == '英文']
            
            print(f"🌐 語言處理能力:")
            print(f"   🇺🇸 英文測試: {len(english_tests)}個成功")
            print(f"   🇹🇼 中文測試: {len(chinese_tests)}個成功")

class FinBERTInteractive:
    """FinBERT 互動式介面"""
    
    def __init__(self, analyzer: FinBERTAnalyzer):
        self.analyzer = analyzer
    
    def run_interactive_mode(self):
        """執行互動模式"""
        print("🚀 FinBERT 互動式金融情感分析工具")
        print("="*60)
        print("💡 使用說明:")
        print("   - 輸入任何金融相關文本進行情感分析")
        print("   - 支援中文、英文、中英混合文本")
        print("   - 輸入 'quit' 或 'exit' 退出程式")
        print("   - 輸入 'examples' 查看測試範例")
        print("   - 輸入 'score:文本' 進行情感評分分析")
        print("   - 輸入 'strategy:價格,RSI,情感' 進行交易策略分析")
        print("   - 輸入 'explain:文本' 進行增強分析（包含總結、依據、解釋）")
        print("="*60)
        
        while True:
            try:
                user_input = input("\n📝 請輸入要分析的金融文本: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                    print("👋 感謝使用 FinBERT 分析工具！")
                    break
                
                if user_input.lower() in ['examples', 'example', '範例', '例子']:
                    self._show_examples()
                    continue
                
                if not user_input:
                    print("⚠️ 請輸入有效的文本內容")
                    continue
                
                # 檢查特殊指令
                if user_input.startswith('score:'):
                    # 情感評分分析
                    text = user_input[6:].strip()
                    if text:
                        print("\n🔍 正在進行情感評分分析...")
                        result = self.analyzer.analyze_sentiment_with_score(text)
                        formatted_result = self.analyzer.format_result(result)
                        print(formatted_result)
                    else:
                        print("⚠️ 請提供要分析的文本，格式：score:您的文本")
                
                elif user_input.startswith('strategy:'):
                    # 交易策略分析
                    params = user_input[9:].strip()
                    try:
                        parts = [p.strip() for p in params.split(',')]
                        if len(parts) == 3:
                            price = float(parts[0])
                            rsi = float(parts[1])
                            sentiment = float(parts[2])
                            
                            print("\n🔍 正在進行交易策略分析...")
                            result = self.analyzer.generate_trading_strategy(price, rsi, sentiment)
                            formatted_result = self.analyzer.format_result(result)
                            print(formatted_result)
                        else:
                            print("⚠️ 請提供正確的參數格式：strategy:價格,RSI,情感")
                            print("   例如：strategy:100.0,0.5,0.3")
                    except ValueError:
                        print("⚠️ 參數格式錯誤，請確保價格、RSI、情感都是數字")
                        print("   例如：strategy:100.0,0.5,0.3")
                
                elif user_input.startswith('explain:'):
                    # 增強分析（包含總結、依據、解釋）
                    text = user_input[8:].strip()
                    if text:
                        print("\n🔍 正在進行增強分析（包含總結、依據、解釋）...")
                        result = self.analyzer.analyze_sentiment_with_explanation(text)
                        formatted_result = self.analyzer.format_result(result)
                        print(formatted_result)
                    else:
                        print("⚠️ 請提供要分析的文本，格式：explain:您的文本")
                
                else:
                    # 一般情感分析
                    print("\n🔍 正在分析中...")
                    result = self.analyzer.analyze_sentiment(user_input)
                    formatted_result = self.analyzer.format_result(result)
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
        print("\n📚 測試範例:")
        print("="*60)
        
        print("🔍 一般情感分析範例:")
        sentiment_examples = [
            ("英文正面", "Apple stock surged 15% after beating earnings expectations"),
            ("英文負面", "Tesla shares plummeted due to production delays"),
            ("中文正面", "台積電股價創新高，投資人信心大增"),
            ("中文負面", "受疫情影響，航空股全面重挫"),
            ("中英混合", "NVIDIA 因AI需求激增，股價大漲")
        ]
        
        for i, (category, example) in enumerate(sentiment_examples, 1):
            print(f"   {i}. {category}: {example}")
        
        print("\n📊 情感評分分析範例:")
        print("   score:Apple stock surges after earnings report")
        print("   score:Tesla shares plummet due to production delays")
        print("   score:台積電股價創新高，投資人信心大增")
        
        print("\n💼 交易策略分析範例:")
        print("   strategy:100.0,0.3,0.8    # 價格100, RSI 0.3(低), 情感0.8(正面)")
        print("   strategy:100.0,0.8,-0.6   # 價格100, RSI 0.8(高), 情感-0.6(負面)")
        print("   strategy:100.0,0.5,0.1    # 價格100, RSI 0.5(中), 情感0.1(中性)")
        
        print("\n🔍 增強分析範例（包含總結、依據、解釋）:")
        print("   explain:Apple stock surged 15% after beating earnings expectations")
        print("   explain:Tesla shares plummeted due to production delays")
        print("   explain:今天蘋果發布了iPhone 17")
        print("   explain:台積電股價創新高，投資人信心大增")
        
        print("="*60)

def main():
    """主程式入口"""
    parser = argparse.ArgumentParser(
        description="FinBERT 金融情感分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
    python finbert_main.py --interactive                    # 互動模式
    python finbert_main.py --simple-test                    # 快速測試
    python finbert_main.py --enhanced-test                  # 增強版測試
    python finbert_main.py --language-comparison            # 中英文對比
    python finbert_main.py --analyze "Apple stock surged"   # 分析指定文本
    python finbert_main.py --sentiment-score "Good news"    # 情感評分分析
    python finbert_main.py --trading-strategy "100,0.5,0.3" # 交易策略分析
    python finbert_main.py --explain "Apple stock surged"   # 增強分析（總結+依據+解釋）
        """
    )
    
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='啟動互動模式')
    parser.add_argument('--simple-test', '-s', action='store_true',
                       help='執行簡化版測試')
    parser.add_argument('--enhanced-test', '-e', action='store_true',
                       help='執行增強版測試（包含情感評分和交易策略）')
    parser.add_argument('--language-comparison', '-l', action='store_true',
                       help='執行中英文對比測試')
    parser.add_argument('--analyze', '-a', type=str,
                       help='分析指定的文本')
    parser.add_argument('--sentiment-score', type=str,
                       help='對指定文本進行情感評分分析')
    parser.add_argument('--trading-strategy', type=str,
                       help='進行交易策略分析，格式：價格,RSI,情感 (例如：100.0,0.5,0.3)')
    parser.add_argument('--explain', type=str,
                       help='進行增強分析（包含總結、依據、解釋）')
    parser.add_argument('--api-key', type=str,
                       default=None,
                       help='HuggingFace API 金鑰（如果未提供，將從 HF_TOKEN 環境變量讀取）')
    
    args = parser.parse_args()
    
    # 如果沒有提供任何參數，顯示幫助
    if not any([args.interactive, args.simple_test, args.enhanced_test, args.language_comparison, 
                args.analyze, args.sentiment_score, args.trading_strategy, args.explain]):
        parser.print_help()
        return
    
    # 初始化分析器
    analyzer = FinBERTAnalyzer(args.api_key)
    
    try:
        if args.analyze:
            # 分析指定文本
            print("🚀 FinBERT 單文本分析")
            print("="*50)
            result = analyzer.analyze_sentiment(args.analyze)
            formatted_result = analyzer.format_result(result)
            print(formatted_result)
            
        elif args.interactive:
            # 互動模式
            interactive = FinBERTInteractive(analyzer)
            interactive.run_interactive_mode()
            
        elif args.simple_test:
            # 簡化版測試
            test_suite = FinBERTTestSuite(analyzer)
            test_suite.run_simple_test()
            
        elif args.enhanced_test:
            # 增強版測試
            test_suite = FinBERTTestSuite(analyzer)
            test_suite.run_enhanced_test()
            
        elif args.language_comparison:
            # 中英文對比測試
            test_suite = FinBERTTestSuite(analyzer)
            test_suite.run_language_comparison()
            
        elif args.sentiment_score:
            # 情感評分分析
            print("🚀 FinBERT 情感評分分析")
            print("="*50)
            result = analyzer.analyze_sentiment_with_score(args.sentiment_score)
            formatted_result = analyzer.format_result(result)
            print(formatted_result)
            
        elif args.trading_strategy:
            # 交易策略分析
            try:
                parts = [p.strip() for p in args.trading_strategy.split(',')]
                if len(parts) == 3:
                    price = float(parts[0])
                    rsi = float(parts[1])
                    sentiment = float(parts[2])
                    
                    print("🚀 FinBERT 交易策略分析")
                    print("="*50)
                    result = analyzer.generate_trading_strategy(price, rsi, sentiment)
                    formatted_result = analyzer.format_result(result)
                    print(formatted_result)
                else:
                    print("❌ 參數格式錯誤，請使用格式：價格,RSI,情感")
                    print("   例如：--trading-strategy 100.0,0.5,0.3")
            except ValueError:
                print("❌ 參數格式錯誤，請確保價格、RSI、情感都是數字")
                print("   例如：--trading-strategy 100.0,0.5,0.3")
        
        elif args.explain:
            # 增強分析（包含總結、依據、解釋）
            print("🚀 FinBERT 增強分析（包含總結、依據、解釋）")
            print("="*60)
            result = analyzer.analyze_sentiment_with_explanation(args.explain)
            formatted_result = analyzer.format_result(result)
            print(formatted_result)
            
    except KeyboardInterrupt:
        print("\n⚠️ 程式被使用者中斷")
    except Exception as e:
        print(f"\n❌ 程式執行錯誤: {e}")
    finally:
        if analyzer.test_results:
            print(f"\n📝 本次執行完成 {len(analyzer.test_results)} 個分析")

if __name__ == "__main__":
    main()
