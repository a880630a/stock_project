#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FinBERT 情感分析 Flask API
提供RESTful API接口供前端調用
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
from datetime import datetime
import logging

# 導入FinBERT分析器
from finbert_main import FinBERTAnalyzer

# 配置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化Flask應用
app = Flask(__name__)
CORS(app)  # 允許跨域請求

# 全局分析器實例
analyzer = None

def init_analyzer():
    """初始化FinBERT分析器"""
    global analyzer
    try:
        # 使用預設API金鑰
        api_key = os.getenv('HF_TOKEN')
        if not api_key:
            raise ValueError("請設置 HF_TOKEN 環境變量。請參考 README.md 了解如何設置。")
        analyzer = FinBERTAnalyzer(api_key)
        logger.info("FinBERT分析器初始化成功")
        return True
    except Exception as e:
        logger.error(f"FinBERT分析器初始化失敗: {e}")
        return False

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康檢查端點"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'analyzer_ready': analyzer is not None
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_sentiment():
    """基本情感分析端點"""
    try:
        # 檢查分析器是否就緒
        if analyzer is None:
            return jsonify({
                'success': False,
                'error': 'FinBERT分析器未初始化'
            }), 500
        
        # 獲取請求數據
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': '請提供要分析的文本'
            }), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({
                'success': False,
                'error': '文本內容不能為空'
            }), 400
        
        # 執行情感分析
        result = analyzer.analyze_sentiment(text, "API請求")
        
        # 處理結果
        if result.get('success', False):
            # 格式化預測結果
            predictions = result.get('predictions', [])
            formatted_predictions = []
            
            for pred in predictions:
                formatted_predictions.append({
                    'label': pred.get('label', '').lower(),
                    'score': round(pred.get('score', 0), 4),
                    'confidence': f"{pred.get('score', 0) * 100:.1f}%"
                })
            
            # 找出最高信心度的預測
            best_prediction = None
            if formatted_predictions:
                best_prediction = max(formatted_predictions, key=lambda x: x['score'])
            
            return jsonify({
                'success': True,
                'input_text': text,
                'language': result.get('language', '未知'),
                'predictions': formatted_predictions,
                'best_prediction': best_prediction,
                'response_time': result.get('response_time', 0),
                'timestamp': result.get('timestamp')
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', '分析失敗'),
                'input_text': text
            }), 500
            
    except Exception as e:
        logger.error(f"情感分析錯誤: {e}")
        return jsonify({
            'success': False,
            'error': f'服務器錯誤: {str(e)}'
        }), 500

@app.route('/api/analyze-score', methods=['POST'])
def analyze_sentiment_score():
    """數值化情感評分端點"""
    try:
        # 檢查分析器是否就緒
        if analyzer is None:
            return jsonify({
                'success': False,
                'error': 'FinBERT分析器未初始化'
            }), 500
        
        # 獲取請求數據
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': '請提供要分析的文本'
            }), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({
                'success': False,
                'error': '文本內容不能為空'
            }), 400
        
        # 執行情感評分分析
        result = analyzer.analyze_sentiment_with_score(text, "API評分請求")
        
        # 處理結果
        if result.get('success', False):
            sentiment_score = result.get('sentiment_score', 0.0)
            
            # 根據評分確定情感類別
            if sentiment_score > 0.3:
                sentiment_category = "positive"
                sentiment_icon = "🟢"
                sentiment_label = "正面"
            elif sentiment_score < -0.3:
                sentiment_category = "negative"
                sentiment_icon = "🔴"
                sentiment_label = "負面"
            else:
                sentiment_category = "neutral"
                sentiment_icon = "🟡"
                sentiment_label = "中性"
            
            return jsonify({
                'success': True,
                'input_text': text,
                'language': result.get('language', '未知'),
                'sentiment_score': round(sentiment_score, 4),
                'sentiment_category': sentiment_category,
                'sentiment_label': sentiment_label,
                'sentiment_icon': sentiment_icon,
                'raw_response': result.get('raw_response', ''),
                'response_time': result.get('response_time', 0),
                'timestamp': result.get('timestamp')
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', '分析失敗'),
                'input_text': text
            }), 500
            
    except Exception as e:
        logger.error(f"情感評分錯誤: {e}")
        return jsonify({
            'success': False,
            'error': f'服務器錯誤: {str(e)}'
        }), 500

@app.route('/api/trading-strategy', methods=['POST'])
def trading_strategy():
    """交易策略分析端點"""
    try:
        # 檢查分析器是否就緒
        if analyzer is None:
            return jsonify({
                'success': False,
                'error': 'FinBERT分析器未初始化'
            }), 500
        
        # 獲取請求數據
        data = request.get_json()
        required_fields = ['price', 'rsi', 'sentiment']
        
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'缺少必要參數: {field}'
                }), 400
        
        try:
            price = float(data['price'])
            rsi = float(data['rsi'])
            sentiment = float(data['sentiment'])
        except (ValueError, TypeError):
            return jsonify({
                'success': False,
                'error': '參數格式錯誤，請確保price、rsi、sentiment都是數字'
            }), 400
        
        # 參數驗證
        if rsi < 0 or rsi > 1:
            return jsonify({
                'success': False,
                'error': 'RSI值應在0到1之間'
            }), 400
        
        if sentiment < -1 or sentiment > 1:
            return jsonify({
                'success': False,
                'error': '情感值應在-1到1之間'
            }), 400
        
        # 執行交易策略分析
        result = analyzer.generate_trading_strategy(price, rsi, sentiment, "API策略請求")
        
        # 處理結果
        if result.get('success', False):
            action = result.get('parsed_action', 1)
            action_label = result.get('action_label', '未知動作')
            
            # 動作詳細信息
            action_details = {
                0: {
                    'action': 'sell',
                    'label': '賣出',
                    'icon': '🔴',
                    'description': '建議賣出持股',
                    'color': '#ef4444'
                },
                1: {
                    'action': 'hold',
                    'label': '持有',
                    'icon': '🟡',
                    'description': '建議維持現狀',
                    'color': '#f59e0b'
                },
                2: {
                    'action': 'buy',
                    'label': '買入',
                    'icon': '🟢',
                    'description': '建議買入股票',
                    'color': '#10b981'
                }
            }
            
            action_info = action_details.get(action, action_details[1])
            
            return jsonify({
                'success': True,
                'input_state': {
                    'price': price,
                    'rsi': rsi,
                    'sentiment': sentiment
                },
                'recommendation': {
                    'action': action_info['action'],
                    'action_code': action,
                    'label': action_info['label'],
                    'icon': action_info['icon'],
                    'description': action_info['description'],
                    'color': action_info['color']
                },
                'analysis': result.get('raw_response', ''),
                'response_time': result.get('response_time', 0),
                'timestamp': result.get('timestamp')
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', '策略分析失敗'),
                'input_state': {
                    'price': price,
                    'rsi': rsi,
                    'sentiment': sentiment
                }
            }), 500
            
    except Exception as e:
        logger.error(f"交易策略錯誤: {e}")
        return jsonify({
            'success': False,
            'error': f'服務器錯誤: {str(e)}'
        }), 500

@app.route('/api/batch-analyze', methods=['POST'])
def batch_analyze():
    """批次分析端點"""
    try:
        # 檢查分析器是否就緒
        if analyzer is None:
            return jsonify({
                'success': False,
                'error': 'FinBERT分析器未初始化'
            }), 500
        
        # 獲取請求數據
        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({
                'success': False,
                'error': '請提供要分析的文本列表'
            }), 400
        
        texts = data['texts']
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({
                'success': False,
                'error': '文本列表不能為空'
            }), 400
        
        if len(texts) > 10:  # 限制批次大小
            return jsonify({
                'success': False,
                'error': '批次分析最多支援10個文本'
            }), 400
        
        # 執行批次分析
        results = []
        for i, text in enumerate(texts):
            if not text or not text.strip():
                continue
                
            result = analyzer.analyze_sentiment(text.strip(), f"批次分析_{i+1}")
            
            if result.get('success', False):
                predictions = result.get('predictions', [])
                best_prediction = None
                if predictions:
                    best_prediction = max(predictions, key=lambda x: x.get('score', 0))
                
                results.append({
                    'index': i,
                    'text': text.strip(),
                    'language': result.get('language', '未知'),
                    'best_prediction': {
                        'label': best_prediction.get('label', '').lower() if best_prediction else 'unknown',
                        'score': round(best_prediction.get('score', 0), 4) if best_prediction else 0,
                        'confidence': f"{best_prediction.get('score', 0) * 100:.1f}%" if best_prediction else "0%"
                    } if best_prediction else None,
                    'response_time': result.get('response_time', 0)
                })
            else:
                results.append({
                    'index': i,
                    'text': text.strip(),
                    'error': result.get('error', '分析失敗')
                })
        
        return jsonify({
            'success': True,
            'total_texts': len(texts),
            'successful_analyses': len([r for r in results if 'error' not in r]),
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"批次分析錯誤: {e}")
        return jsonify({
            'success': False,
            'error': f'服務器錯誤: {str(e)}'
        }), 500

@app.errorhandler(404)
def not_found(error):
    """404錯誤處理"""
    return jsonify({
        'success': False,
        'error': '找不到請求的資源'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """500錯誤處理"""
    return jsonify({
        'success': False,
        'error': '內部服務器錯誤'
    }), 500

if __name__ == '__main__':
    # 初始化分析器
    if init_analyzer():
        logger.info("啟動FinBERT情感分析API服務器...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        logger.error("無法初始化FinBERT分析器，服務器啟動失敗")
        sys.exit(1)

