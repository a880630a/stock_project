#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試中文字體顯示
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 配置中文字體支援
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'AR PL UMing CN', 'AR PL UKai CN', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

# 創建測試圖表
fig, ax = plt.subplots(figsize=(10, 6))

# 測試各種中文文字
test_texts = [
    '趨勢: 下降 (-57.8%)',
    '累積平均',
    '總平均=9.957',
    '目標線 (0.6)',
    '最終值: -0.000',
    '✓ PASS',
    '⚠ 超出範圍',
]

for i, text in enumerate(test_texts):
    ax.text(0.1, 0.9 - i*0.1, text, fontsize=14, transform=ax.transAxes)

ax.set_title('中文字體測試 - BC Training Test', fontsize=16, fontweight='bold')
ax.set_xlabel('測試 X 軸', fontsize=12)
ax.set_ylabel('測試 Y 軸', fontsize=12)

plt.tight_layout()
plt.savefig('test_chinese_font.png', dpi=150, bbox_inches='tight')
plt.close()

print("✅ 測試圖表已生成: test_chinese_font.png")

# 打印當前使用的字體
print(f"當前字體配置: {plt.rcParams['font.sans-serif']}")

