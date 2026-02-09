"""
仓位管理和杠杆建议系统

基于信号强度和置信度计算建议的仓位大小和杠杆率
"""

import numpy as np
from typing import Dict, Tuple


class PositionManager:
    """仓位管理器"""
    
    # 信号强度范围：-12 到 +12
    # (6个时间框架 × 2个指标：KDJ金叉/死叉 + MACD金叉/死叉)
    MAX_SIGNAL_STRENGTH = 12
    
    # 置信度范围：0-100
    MAX_CONFIDENCE = 100
    
    def __init__(self, 
                 base_position_size: float = 1.0,
                 max_leverage: float = 10.0,
                 min_leverage: float = 1.0,
                 risk_level: str = 'moderate'):
        """
        初始化仓位管理器
        
        Args:
            base_position_size: 基础仓位大小（1.0 = 100%）
            max_leverage: 最大杠杆率
            min_leverage: 最小杠杆率
            risk_level: 风险等级 ('conservative', 'moderate', 'aggressive')
        """
        self.base_position_size = base_position_size
        self.max_leverage = max_leverage
        self.min_leverage = min_leverage
        self.risk_level = risk_level
        
        # 风险等级配置
        self.risk_configs = {
            'conservative': {
                'position_multiplier': 0.5,  # 保守：仓位减半
                'leverage_multiplier': 0.6,  # 杠杆降低
                'min_confidence': 60,         # 最低置信度要求
                'min_signal': 3,              # 最低信号强度要求
            },
            'moderate': {
                'position_multiplier': 1.0,
                'leverage_multiplier': 1.0,
                'min_confidence': 40,
                'min_signal': 2,
            },
            'aggressive': {
                'position_multiplier': 1.5,  # 激进：仓位增加
                'leverage_multiplier': 1.2,  # 杠杆提高
                'min_confidence': 30,
                'min_signal': 1,
            }
        }
    
    def calculate_position_size(self, 
                                signal_strength: float,
                                confidence: float,
                                prediction_pct: float,
                                vzo_trade_action: str = None) -> Dict[str, float]:
        """
        计算建议仓位大小
        
        Args:
            signal_strength: 信号强度 (-12 到 +12)
            confidence: 置信度 (0-100)
            prediction_pct: 预测涨跌幅百分比
            vzo_trade_action: VZO trade signal ('ENTER', 'EXIT', 'WARNING', 'HOLD', or None)
        
        Returns:
            包含仓位建议的字典
        """
        config = self.risk_configs[self.risk_level]
        
        # 0. VZO EXIT signal overrides everything
        if vzo_trade_action == 'EXIT':
            return {
                'position_size': 0.0,
                'leverage': 1.0,
                'recommendation': 'exit',
                'reason': 'VZO slope signals EXIT - close/reduce position (momentum reversal detected)',
                'risk_score': 0.0,
                'signal_factor': 0.0,
                'confidence_factor': 0.0,
                'magnitude_factor': 0.0,
                'vzo_trade_action': 'EXIT',
            }
        
        # 1. 检查最低要求
        if confidence < config['min_confidence']:
            return {
                'position_size': 0.0,
                'leverage': 1.0,
                'recommendation': 'no_trade',
                'reason': f'置信度过低 ({confidence:.0f}% < {config["min_confidence"]}%)',
                'risk_score': 0.0,
                'vzo_trade_action': vzo_trade_action,
            }
        
        abs_signal = abs(signal_strength)
        if abs_signal < config['min_signal']:
            return {
                'position_size': 0.0,
                'leverage': 1.0,
                'recommendation': 'no_trade',
                'reason': f'信号强度不足 ({abs_signal} < {config["min_signal"]})',
                'risk_score': 0.0,
                'vzo_trade_action': vzo_trade_action,
            }
        
        # 2. 计算信号强度因子 (0-1)
        signal_factor = min(abs_signal / self.MAX_SIGNAL_STRENGTH, 1.0)
        
        # 3. 计算置信度因子 (0-1)
        confidence_factor = confidence / self.MAX_CONFIDENCE
        
        # 4. 计算预测幅度因子（极端预测降低仓位）
        pred_abs = abs(prediction_pct)
        if pred_abs > 5:
            # 预测超过5%，可能是异常值，降低仓位
            magnitude_factor = max(0.5, 1.0 - (pred_abs - 5) * 0.1)
        else:
            magnitude_factor = 1.0
        
        # 5. 综合评分 (0-1)
        risk_score = (signal_factor * 0.4 + confidence_factor * 0.4 + magnitude_factor * 0.2)
        
        # 5a. VZO trade action adjustments
        vzo_adjustment = ''
        if vzo_trade_action == 'WARNING':
            # Reduce position by 50% on WARNING
            risk_score *= 0.5
            vzo_adjustment = ' (reduced 50% due to VZO WARNING)'
        elif vzo_trade_action == 'ENTER':
            # Boost by 20% on confirmed ENTER
            risk_score = min(risk_score * 1.2, 1.0)
            vzo_adjustment = ' (boosted by VZO ENTER confirmation)'
        
        # 6. 计算仓位大小
        position_size = self.base_position_size * config['position_multiplier'] * risk_score
        
        # 限制仓位范围
        position_size = max(0.0, min(position_size, 1.0))  # 0-100%
        
        # 7. 计算杠杆率
        # 基础杠杆：根据风险评分
        base_leverage = self.min_leverage + (self.max_leverage - self.min_leverage) * risk_score
        leverage = base_leverage * config['leverage_multiplier']
        
        # 限制杠杆范围
        leverage = max(self.min_leverage, min(leverage, self.max_leverage))
        
        # 8. 确定建议
        if position_size < 0.1:
            recommendation = 'no_trade'
            reason = '综合评分过低，不建议交易'
        elif risk_score >= 0.7:
            recommendation = 'strong'
            reason = '信号强、置信度高，建议较大仓位' + vzo_adjustment
        elif risk_score >= 0.5:
            recommendation = 'moderate'
            reason = '信号和置信度中等，建议中等仓位' + vzo_adjustment
        else:
            recommendation = 'weak'
            reason = '信号或置信度较低，建议小仓位' + vzo_adjustment
        
        return {
            'position_size': position_size,
            'leverage': leverage,
            'recommendation': recommendation,
            'reason': reason,
            'risk_score': risk_score,
            'signal_factor': signal_factor,
            'confidence_factor': confidence_factor,
            'magnitude_factor': magnitude_factor,
            'vzo_trade_action': vzo_trade_action,
        }
    
    def format_recommendation(self, position_info: Dict[str, float], 
                            direction: str) -> str:
        """
        格式化仓位建议报告
        """
        if position_info['recommendation'] == 'no_trade':
            return f"""❌ <b>不建议交易</b>
原因: {position_info['reason']}"""
        
        # 方向图标
        direction_emoji = '📈' if direction == '看涨' else '📉'
        
        # 建议强度
        strength_map = {
            'strong': '🟢 强烈建议',
            'moderate': '🟡 中等建议',
            'weak': '🟠 谨慎建议'
        }
        strength_text = strength_map.get(position_info['recommendation'], '建议')
        
        # 仓位大小百分比
        position_pct = position_info['position_size'] * 100
        
        # 风险等级
        risk_level_map = {
            'conservative': '保守',
            'moderate': '中等',
            'aggressive': '激进'
        }
        risk_text = risk_level_map.get(self.risk_level, self.risk_level)
        
        return f"""{direction_emoji} <b>交易建议: {strength_text}</b>

💰 <b>建议仓位:</b> {position_pct:.1f}%
⚡ <b>建议杠杆:</b> {position_info['leverage']:.1f}x
📊 <b>风险评分:</b> {position_info['risk_score']*100:.0f}/100

<b>详细分析:</b>
• 信号强度因子: {position_info['signal_factor']*100:.0f}%
• 置信度因子: {position_info['confidence_factor']*100:.0f}%
• 预测幅度因子: {position_info['magnitude_factor']*100:.0f}%

<b>风险等级:</b> {risk_text}
<b>说明:</b> {position_info['reason']}

⚠️ <i>仅供参考，请根据自身风险承受能力调整</i>"""


def explain_signal_strength() -> str:
    """解释信号强度的计算方法"""
    return """
<b>📊 信号强度计算说明</b>

<b>计算公式:</b>
信号强度 = 金叉数量 - 死叉数量

<b>统计范围:</b>
• 时间框架: 5m, 15m, 30m, 1h, 4h, 1d (共6个)
• 技术指标: KDJ 金叉/死叉 + MACD 金叉/死叉 (共2个)
• 最大信号数: 6 × 2 = 12

<b>信号强度范围:</b>
• -12 到 +12
• 正数 = 看涨信号（金叉多于死叉）
• 负数 = 看跌信号（死叉多于金叉）
• 0 = 信号中性

<b>示例:</b>
• 信号强度 = +6: 6个金叉，0个死叉 → 强烈看涨
• 信号强度 = -4: 2个金叉，6个死叉 → 看跌
• 信号强度 = 0: 金叉和死叉数量相等 → 中性
"""

