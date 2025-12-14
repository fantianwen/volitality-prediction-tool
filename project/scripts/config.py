"""
BTC 价格预测服务 - 配置管理

支持从环境变量和 .env 文件加载配置
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


def load_dotenv(env_path: str = None):
    """加载 .env 文件"""
    if env_path is None:
        env_path = Path(__file__).parent / '.env'
    else:
        env_path = Path(env_path)
    
    if not env_path.exists():
        return
    
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                # 移除引号
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                os.environ.setdefault(key, value)


@dataclass
class Config:
    """配置类"""
    
    # Telegram 配置
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    
    # 交易配置
    symbol: str = 'BTCUSDT'
    
    # 模型配置
    model_path: Optional[str] = None
    
    # 服务配置
    prediction_interval_hours: int = 1
    lookforward_periods: int = 20
    
    @classmethod
    def from_env(cls) -> 'Config':
        """从环境变量加载配置"""
        # 先加载 .env 文件
        load_dotenv()
        
        return cls(
            telegram_bot_token=os.getenv('TELEGRAM_BOT_TOKEN'),
            telegram_chat_id=os.getenv('TELEGRAM_CHAT_ID'),
            symbol=os.getenv('SYMBOL', 'BTCUSDT'),
            model_path=os.getenv('MODEL_PATH'),
            prediction_interval_hours=int(os.getenv('PREDICTION_INTERVAL_HOURS', '1')),
            lookforward_periods=int(os.getenv('LOOKFORWARD_PERIODS', '20')),
        )
    
    def validate(self) -> tuple:
        """
        验证配置
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        if self.model_path and not Path(self.model_path).exists():
            errors.append(f"模型文件不存在: {self.model_path}")
        
        # Telegram 配置可选
        if self.telegram_bot_token and not self.telegram_chat_id:
            errors.append("设置了 TELEGRAM_BOT_TOKEN 但未设置 TELEGRAM_CHAT_ID")
        
        if self.telegram_chat_id and not self.telegram_bot_token:
            errors.append("设置了 TELEGRAM_CHAT_ID 但未设置 TELEGRAM_BOT_TOKEN")
        
        return len(errors) == 0, errors
    
    @property
    def telegram_enabled(self) -> bool:
        """是否启用 Telegram"""
        return bool(self.telegram_bot_token and self.telegram_chat_id)


# 默认配置实例
config = Config.from_env()

