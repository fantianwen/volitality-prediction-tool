"""
BTC 价格变动预测 - 模型训练

基于技术指标特征训练预测模型

功能:
1. 加载特征数据
2. 时间序列交叉验证（不随机分割）
3. 训练多种模型 (Random Forest, Gradient Boosting, LSTM, etc.)
4. 评估模型表现 (方向准确率、模拟收益)
5. 保存最佳模型

使用方法:
    python train_model.py --data ../data/BTCUSDT_features_1h_*.csv --output ../models
"""

import pandas as pd
import numpy as np
import argparse
import os
import glob
import pickle
import json
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# 屏蔽 TensorFlow 警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, classification_report, confusion_matrix
)

# TensorFlow/Keras for LSTM
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
    from tensorflow.keras.optimizers import Adam
    LSTM_AVAILABLE = True
    
    # 自定义日志回调
    class TrainingLogger(Callback):
        """训练过程日志记录器"""
        def __init__(self, fold_num=None, total_epochs=50):
            super().__init__()
            self.fold_num = fold_num
            self.total_epochs = total_epochs
            self.start_time = None
            
        def on_train_begin(self, logs=None):
            import time
            self.start_time = time.time()
            fold_str = f"Fold {self.fold_num}" if self.fold_num else "最终模型"
            print(f"    [{fold_str}] 开始训练...")
            
        def on_epoch_end(self, epoch, logs=None):
            import time
            elapsed = time.time() - self.start_time
            loss = logs.get('loss', 0)
            val_loss = logs.get('val_loss', 0)
            
            # 每 5 个 epoch 或最后一个 epoch 打印
            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == self.total_epochs - 1:
                fold_str = f"Fold {self.fold_num}" if self.fold_num else "Final"
                print(f"    [{fold_str}] Epoch {epoch+1}/{self.total_epochs} - "
                      f"loss: {loss:.4f} - val_loss: {val_loss:.4f} - "
                      f"耗时: {elapsed:.1f}s")
        
        def on_train_end(self, logs=None):
            import time
            total_time = time.time() - self.start_time
            fold_str = f"Fold {self.fold_num}" if self.fold_num else "最终模型"
            print(f"    [{fold_str}] 训练完成! 总耗时: {total_time:.1f}s")

except ImportError:
    LSTM_AVAILABLE = False
    print("⚠️ TensorFlow 未安装，LSTM 模型不可用。安装: pip install tensorflow")


class ModelEvaluator:
    """模型评估器"""
    
    @staticmethod
    def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        评估回归模型
        
        Returns:
            包含各项指标的字典
        """
        # 基本回归指标
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # 方向准确率（更重要！）
        direction_accuracy = np.mean(np.sign(y_true) == np.sign(y_pred))
        
        # 模拟交易收益
        # 假设：预测涨就做多，预测跌就做空
        returns = np.where(y_pred > 0, y_true, -y_true)
        total_return = np.sum(returns)
        
        # 夏普比率
        if np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0
        
        # 盈亏比
        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]
        if len(losing_trades) > 0 and np.mean(np.abs(losing_trades)) > 0:
            profit_factor = np.sum(winning_trades) / np.abs(np.sum(losing_trades))
        else:
            profit_factor = np.inf if len(winning_trades) > 0 else 0
        
        # 胜率
        win_rate = len(winning_trades) / len(returns) if len(returns) > 0 else 0
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'direction_accuracy': direction_accuracy,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'profit_factor': profit_factor,
            'win_rate': win_rate,
            'n_trades': len(returns)
        }
    
    @staticmethod
    def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray, 
                                y_prob: Optional[np.ndarray] = None) -> dict:
        """
        评估分类模型
        """
        accuracy = accuracy_score(y_true, y_pred)
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm.tolist(),
        }


class LSTMPredictor:
    """LSTM 价格预测模型"""
    
    def __init__(self, sequence_length: int = 20, task: str = 'regression'):
        """
        Args:
            sequence_length: 输入序列长度
            task: 'regression' 或 'classification'
        """
        self.sequence_length = sequence_length
        self.task = task
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def prepare_lstm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        为 LSTM 准备专门的时序特征
        
        Args:
            df: 原始特征 DataFrame
            
        Returns:
            增强后的特征 DataFrame
        """
        print("  🔧 LSTM 特征工程...")
        
        enhanced_df = df.copy()
        n_original = len(df.columns)
        
        # 获取价格相关列 (用于计算收益率)
        price_cols = [col for col in df.columns if 'close' in col.lower() or 'price' in col.lower()]
        
        # 1. 滞后特征 (Lagged Features) - 前 N 个时间步的值
        lag_cols = ['1h_kdj_k', '1h_kdj_d', '1h_macd', '1h_rsi_14', '1h_volatility']
        lag_cols = [c for c in lag_cols if c in df.columns]
        
        for col in lag_cols:
            for lag in [1, 3, 5]:
                enhanced_df[f'{col}_lag{lag}'] = df[col].shift(lag).fillna(method='bfill')
        
        # 2. 差分特征 (Difference Features) - 变化率
        diff_cols = ['1h_kdj_k', '1h_macd', '1h_rsi_14', '1h_volatility', '1h_vol_ratio_ma20']
        diff_cols = [c for c in diff_cols if c in df.columns]
        
        for col in diff_cols:
            # 一阶差分
            enhanced_df[f'{col}_diff1'] = df[col].diff(1).fillna(0)
            # 5 步差分
            enhanced_df[f'{col}_diff5'] = df[col].diff(5).fillna(0)
        
        # 3. 滚动统计特征 (Rolling Statistics)
        roll_cols = ['1h_kdj_k', '1h_macd_hist', '1h_rsi_14']
        roll_cols = [c for c in roll_cols if c in df.columns]
        
        for col in roll_cols:
            # 滚动均值
            enhanced_df[f'{col}_roll_mean5'] = df[col].rolling(5).mean().fillna(method='bfill')
            enhanced_df[f'{col}_roll_mean10'] = df[col].rolling(10).mean().fillna(method='bfill')
            # 滚动标准差
            enhanced_df[f'{col}_roll_std5'] = df[col].rolling(5).std().fillna(0)
            # 滚动最大/最小
            enhanced_df[f'{col}_roll_max5'] = df[col].rolling(5).max().fillna(method='bfill')
            enhanced_df[f'{col}_roll_min5'] = df[col].rolling(5).min().fillna(method='bfill')
        
        # 4. 动量特征 (Momentum) - 当前值与滚动均值的偏离
        for col in roll_cols:
            roll_mean = df[col].rolling(10).mean().fillna(method='bfill')
            enhanced_df[f'{col}_momentum'] = (df[col] - roll_mean) / (roll_mean.abs() + 1e-8)
        
        # 5. 交叉特征 (Cross Features) - 指标间的关系
        if '1h_kdj_k' in df.columns and '1h_kdj_d' in df.columns:
            enhanced_df['kdj_spread'] = df['1h_kdj_k'] - df['1h_kdj_d']
            enhanced_df['kdj_spread_change'] = enhanced_df['kdj_spread'].diff(1).fillna(0)
        
        if '1h_macd' in df.columns and '1h_macd_signal' in df.columns:
            enhanced_df['macd_spread'] = df['1h_macd'] - df['1h_macd_signal']
            enhanced_df['macd_spread_change'] = enhanced_df['macd_spread'].diff(1).fillna(0)
        
        # 6. RSI 区间特征
        if '1h_rsi_14' in df.columns:
            rsi = df['1h_rsi_14']
            enhanced_df['rsi_zone'] = pd.cut(rsi, bins=[0, 30, 50, 70, 100], labels=[0, 1, 2, 3]).astype(float).fillna(1)
            enhanced_df['rsi_distance_from_50'] = (rsi - 50).abs()
        
        # 7. 波动率变化特征
        if '1h_volatility' in df.columns:
            vol = df['1h_volatility']
            vol_mean = vol.rolling(20).mean().fillna(method='bfill')
            enhanced_df['vol_regime'] = (vol > vol_mean).astype(int)  # 高波动/低波动
            enhanced_df['vol_change_rate'] = vol.pct_change(5).fillna(0).clip(-10, 10)
        
        # 处理无穷值和 NaN
        enhanced_df = enhanced_df.replace([np.inf, -np.inf], 0)
        enhanced_df = enhanced_df.fillna(0)
        
        n_new = len(enhanced_df.columns) - n_original
        print(f"    原始特征: {n_original}, 新增特征: {n_new}, 总计: {len(enhanced_df.columns)}")
        
        return enhanced_df
    
    def prepare_sequences(self, X: np.ndarray, y: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        将数据转换为 LSTM 需要的序列格式
        
        Args:
            X: 特征数组 (n_samples, n_features)
            y: 目标数组 (n_samples,)
            
        Returns:
            X_seq: (n_samples - sequence_length, sequence_length, n_features)
            y_seq: (n_samples - sequence_length,)
        """
        X_seq, y_seq = [], []
        
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            if y is not None:
                y_seq.append(y[i + self.sequence_length])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq) if y is not None else None
        
        return X_seq, y_seq
    
    def build_model(self, input_shape: tuple, n_classes: int = None) -> Sequential:
        """
        构建 LSTM 模型
        
        Args:
            input_shape: (sequence_length, n_features)
            n_classes: 分类任务的类别数
        """
        from tensorflow.keras.regularizers import l2
        
        model = Sequential([
            # 第一层 LSTM (增加正则化)
            LSTM(128, return_sequences=True, input_shape=input_shape,
                 kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            
            # 第二层 LSTM
            LSTM(64, return_sequences=True,
                 kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            
            # 第三层 LSTM
            LSTM(32, return_sequences=False,
                 kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            
            # 全连接层
            Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.2),
            Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.1),
        ])
        
        if self.task == 'regression':
            model.add(Dense(1))
            model.compile(
                optimizer=Adam(learning_rate=0.0005),  # 降低学习率
                loss='huber',  # 使用 Huber loss，对异常值更鲁棒
                metrics=['mae']
            )
        else:
            model.add(Dense(n_classes, activation='softmax'))
            model.compile(
                optimizer=Adam(learning_rate=0.0005),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        
        return model
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              n_splits: int = 5, epochs: int = 50, batch_size: int = 32,
              use_feature_engineering: bool = True) -> dict:
        """
        训练 LSTM 模型
        
        Args:
            X: 特征 DataFrame
            y: 目标 Series
            n_splits: 交叉验证折数
            epochs: 训练轮数
            batch_size: 批次大小
            use_feature_engineering: 是否使用 LSTM 专用特征工程
        """
        if not LSTM_AVAILABLE:
            print("❌ TensorFlow 未安装，跳过 LSTM 训练")
            return {}
        
        # LSTM 专用特征工程
        if use_feature_engineering:
            X = self.prepare_lstm_features(X)
        
        self.feature_names = X.columns.tolist()
        
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        y_values = y.values
        
        # 准备序列数据
        X_seq, y_seq = self.prepare_sequences(X_scaled, y_values)
        
        if len(X_seq) < 100:
            print("❌ 数据量不足以训练 LSTM（需要至少 100 个序列）")
            return {}
        
        print(f"\n{'='*60}")
        print(f"训练模型: LSTM (sequence_length={self.sequence_length})")
        print(f"{'='*60}")
        print(f"  序列数据形状: X={X_seq.shape}, y={y_seq.shape}")
        
        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        fold_metrics = []
        
        # 确定类别数（分类任务）
        n_classes = len(np.unique(y_seq)) if self.task == 'classification' else None
        
        print(f"\n  开始 {n_splits} 折交叉验证...")
        print(f"  每折训练样本: ~{len(X_seq) // (n_splits + 1) * n_splits}")
        print(f"  每折测试样本: ~{len(X_seq) // (n_splits + 1)}")
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_seq)):
            X_train, X_test = X_seq[train_idx], X_seq[test_idx]
            y_train, y_test = y_seq[train_idx], y_seq[test_idx]
            
            print(f"\n  === Fold {fold+1}/{n_splits} ===")
            print(f"  训练集: {len(X_train)}, 测试集: {len(X_test)}")
            
            # 构建新模型（每折重新构建）
            model = self.build_model(
                input_shape=(self.sequence_length, X.shape[1]),
                n_classes=n_classes
            )
            
            # 回调函数（包含日志记录器）
            fold_callbacks = [
                TrainingLogger(fold_num=fold+1, total_epochs=epochs),
                EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=0.00001, verbose=1)
            ]
            
            # 训练
            model.fit(
                X_train, y_train,
                validation_split=0.1,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=fold_callbacks,
                verbose=0
            )
            
            # 预测
            y_pred = model.predict(X_test, verbose=0)
            
            if self.task == 'regression':
                y_pred = y_pred.flatten()
                metrics = ModelEvaluator.evaluate_regression(y_test, y_pred)
                print(f"  Fold {fold+1}: 方向准确率={metrics['direction_accuracy']:.2%}, "
                      f"MAE={metrics['mae']:.4f}, 收益={metrics['total_return']:.2f}%")
            else:
                y_pred_class = np.argmax(y_pred, axis=1)
                metrics = ModelEvaluator.evaluate_classification(y_test, y_pred_class)
                print(f"  Fold {fold+1}: 准确率={metrics['accuracy']:.2%}")
            
            fold_metrics.append(metrics)
        
        # 计算平均指标
        avg_metrics = {}
        for key in fold_metrics[0].keys():
            if key != 'confusion_matrix':
                values = [m[key] for m in fold_metrics]
                avg_metrics[key] = np.mean(values)
                avg_metrics[f'{key}_std'] = np.std(values)
        
        # 使用全部数据重新训练最终模型
        print(f"\n  {'='*50}")
        print(f"  使用全部 {len(X_seq)} 个样本训练最终 LSTM 模型...")
        print(f"  {'='*50}")
        
        self.model = self.build_model(
            input_shape=(self.sequence_length, X.shape[1]),
            n_classes=n_classes
        )
        
        # 最终模型的回调
        final_callbacks = [
            TrainingLogger(fold_num=None, total_epochs=epochs),
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=0.00001, verbose=1)
        ]
        
        self.model.fit(
            X_seq, y_seq,
            validation_split=0.1,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=final_callbacks,
            verbose=0
        )
        
        return avg_metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        X_scaled = self.scaler.transform(X)
        X_seq, _ = self.prepare_sequences(X_scaled)
        
        predictions = self.model.predict(X_seq, verbose=0)
        
        if self.task == 'regression':
            return predictions.flatten()
        else:
            return np.argmax(predictions, axis=1)
    
    def save(self, filepath: str):
        """保存模型"""
        # 保存 Keras 模型
        model_path = filepath.replace('.pkl', '_lstm.keras')
        self.model.save(model_path)
        
        # 保存其他参数
        params = {
            'scaler': self.scaler,
            'sequence_length': self.sequence_length,
            'task': self.task,
            'feature_names': self.feature_names,
            'model_path': model_path
        }
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)
        
        print(f"💾 LSTM 模型已保存到: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'LSTMPredictor':
        """加载模型"""
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        
        predictor = cls(
            sequence_length=params['sequence_length'],
            task=params['task']
        )
        predictor.scaler = params['scaler']
        predictor.feature_names = params['feature_names']
        predictor.model = tf.keras.models.load_model(params['model_path'])
        
        return predictor


class EnsemblePredictor:
    """
    模型融合预测器: GBM 判断方向 + LSTM 预测幅度
    
    策略:
    1. GBM 预测方向（准确率更高，更稳定）
    2. LSTM 预测幅度（捕捉时序模式）
    3. 只有当两者一致且置信度高时才交易
    """
    
    def __init__(self, gbm_weight: float = 0.6, lstm_weight: float = 0.4,
                 confidence_threshold: float = 0.3):
        """
        Args:
            gbm_weight: GBM 预测权重
            lstm_weight: LSTM 预测权重
            confidence_threshold: 交易置信度阈值（预测涨跌幅 > 此值才交易）
        """
        self.gbm_weight = gbm_weight
        self.lstm_weight = lstm_weight
        self.confidence_threshold = confidence_threshold
        
        self.gbm_model = None
        self.lstm_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def train(self, X: pd.DataFrame, y: pd.Series, 
              n_splits: int = 5, lstm_epochs: int = 50,
              lstm_seq_len: int = 20) -> dict:
        """
        训练融合模型
        """
        print("\n" + "="*60)
        print("🔀 训练融合模型 (GBM + LSTM)")
        print("="*60)
        print(f"   GBM 权重: {self.gbm_weight}, LSTM 权重: {self.lstm_weight}")
        print(f"   置信度阈值: {self.confidence_threshold}")
        
        self.feature_names = X.columns.tolist()
        
        # 1. 训练 GBM 模型
        print("\n📈 第一步: 训练 GBM 模型...")
        self.gbm_model = PriceMovementPredictor(
            selected_models=['gbm'],
            model_params={'gbm_estimators': 150, 'gbm_lr': 0.08, 'gbm_depth': 6}
        )
        gbm_results = self.gbm_model.train(X, y, n_splits=n_splits)
        
        # 2. 训练 LSTM 模型
        if LSTM_AVAILABLE:
            print("\n🧠 第二步: 训练 LSTM 模型...")
            self.lstm_model = LSTMPredictor(
                sequence_length=lstm_seq_len,
                task='regression'
            )
            lstm_results = self.lstm_model.train(
                X, y, n_splits=n_splits, 
                epochs=lstm_epochs, batch_size=128,
                use_feature_engineering=True
            )
        else:
            print("\n⚠️ TensorFlow 未安装，只使用 GBM")
            lstm_results = {}
        
        # 3. 评估融合效果（无数据泄露）
        print("\n📊 第三步: 评估融合模型（无数据泄露）...")
        ensemble_results = self._evaluate_ensemble(
            X, y, n_splits, 
            lstm_epochs=lstm_epochs, 
            lstm_seq_len=lstm_seq_len
        )
        
        return {
            'gbm': gbm_results,
            'lstm': lstm_results,
            'ensemble': ensemble_results
        }
    
    def _evaluate_ensemble(self, X: pd.DataFrame, y: pd.Series, 
                          n_splits: int = 5, lstm_epochs: int = 30,
                          lstm_seq_len: int = 20) -> dict:
        """
        评估融合模型效果（无数据泄露版本）
        
        每个 fold 单独训练 GBM 和 LSTM，确保测试数据从未被模型见过
        """
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        fold_metrics = []
        
        print(f"\n  开始 {n_splits} 折交叉验证（无数据泄露）...")
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            print(f"\n  === Ensemble Fold {fold+1}/{n_splits} ===")
            
            X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # 1. 在当前 fold 的训练集上训练 GBM
            print(f"    训练 GBM (训练集: {len(X_train)})...")
            fold_gbm = GradientBoostingRegressor(
                n_estimators=150, learning_rate=0.08, max_depth=6, random_state=42
            )
            gbm_scaler = StandardScaler()
            X_train_scaled = gbm_scaler.fit_transform(X_train)
            X_test_scaled = gbm_scaler.transform(X_test)
            fold_gbm.fit(X_train_scaled, y_train.values)
            gbm_pred = fold_gbm.predict(X_test_scaled)
            
            # 2. 在当前 fold 的训练集上训练 LSTM
            if LSTM_AVAILABLE:
                print(f"    训练 LSTM (训练集: {len(X_train)})...")
                
                # LSTM 特征工程
                fold_lstm_extractor = LSTMPredictor(sequence_length=lstm_seq_len, task='regression')
                X_train_lstm = fold_lstm_extractor.prepare_lstm_features(X_train)
                X_test_lstm = fold_lstm_extractor.prepare_lstm_features(X_test)
                
                # 标准化
                lstm_scaler = StandardScaler()
                X_train_lstm_scaled = lstm_scaler.fit_transform(X_train_lstm)
                X_test_lstm_scaled = lstm_scaler.transform(X_test_lstm)
                
                # 准备序列
                X_train_seq, y_train_seq = fold_lstm_extractor.prepare_sequences(
                    X_train_lstm_scaled, y_train.values
                )
                X_test_seq, _ = fold_lstm_extractor.prepare_sequences(X_test_lstm_scaled)
                
                if len(X_train_seq) > 100 and len(X_test_seq) > 0:
                    # 构建并训练 LSTM
                    from tensorflow.keras.regularizers import l2
                    lstm_model = Sequential([
                        LSTM(64, return_sequences=True, input_shape=(lstm_seq_len, X_train_lstm.shape[1]),
                             kernel_regularizer=l2(0.001)),
                        BatchNormalization(),
                        Dropout(0.3),
                        LSTM(32, return_sequences=False, kernel_regularizer=l2(0.001)),
                        BatchNormalization(),
                        Dropout(0.2),
                        Dense(16, activation='relu'),
                        Dense(1)
                    ])
                    lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='huber', metrics=['mae'])
                    
                    # 训练（静默模式）
                    lstm_model.fit(
                        X_train_seq, y_train_seq,
                        epochs=lstm_epochs, batch_size=64,
                        validation_split=0.1, verbose=0,
                        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
                    )
                    
                    lstm_pred_raw = lstm_model.predict(X_test_seq, verbose=0).flatten()
                    
                    # 对齐预测长度
                    offset = len(gbm_pred) - len(lstm_pred_raw)
                    if offset > 0:
                        gbm_pred_aligned = gbm_pred[offset:]
                        y_test_aligned = y_test.values[offset:]
                        lstm_pred = lstm_pred_raw
                    else:
                        gbm_pred_aligned = gbm_pred
                        y_test_aligned = y_test.values
                        lstm_pred = lstm_pred_raw[:len(gbm_pred)]
                else:
                    gbm_pred_aligned = gbm_pred
                    y_test_aligned = y_test.values
                    lstm_pred = gbm_pred
            else:
                gbm_pred_aligned = gbm_pred
                y_test_aligned = y_test.values
                lstm_pred = gbm_pred
            
            # 融合预测
            ensemble_pred = self._fuse_predictions(gbm_pred_aligned, lstm_pred)
            
            # 应用置信度过滤
            filtered_pred, filtered_true, n_trades = self._apply_confidence_filter(
                ensemble_pred, y_test_aligned
            )
            
            if len(filtered_pred) > 0:
                metrics = ModelEvaluator.evaluate_regression(filtered_true, filtered_pred)
                metrics['n_trades'] = n_trades
                metrics['trade_ratio'] = n_trades / len(ensemble_pred)
            else:
                metrics = {
                    'direction_accuracy': 0.5,
                    'mae': 0,
                    'total_return': 0,
                    'n_trades': 0,
                    'trade_ratio': 0
                }
            
            fold_metrics.append(metrics)
            
            print(f"  Fold {fold+1}: 方向准确率={metrics['direction_accuracy']:.2%}, "
                  f"收益={metrics['total_return']:.2f}%, "
                  f"交易次数={metrics['n_trades']}/{len(ensemble_pred)} ({metrics['trade_ratio']:.1%})")
        
        # 计算平均指标
        avg_metrics = {}
        for key in ['direction_accuracy', 'mae', 'total_return', 'n_trades', 'trade_ratio']:
            if key in fold_metrics[0]:
                values = [m.get(key, 0) for m in fold_metrics]
                avg_metrics[key] = np.mean(values)
                avg_metrics[f'{key}_std'] = np.std(values)
        
        print(f"\n  融合模型平均结果:")
        print(f"    方向准确率: {avg_metrics['direction_accuracy']:.2%} (±{avg_metrics['direction_accuracy_std']:.2%})")
        print(f"    总收益: {avg_metrics['total_return']:.2f}% (±{avg_metrics['total_return_std']:.2f}%)")
        print(f"    交易比例: {avg_metrics['trade_ratio']:.1%}")
        
        return avg_metrics
    
    def _fuse_predictions(self, gbm_pred: np.ndarray, lstm_pred: np.ndarray) -> np.ndarray:
        """
        融合 GBM 和 LSTM 预测
        
        策略:
        - 加权平均
        - 如果方向不一致，降低预测幅度（表示不确定）
        """
        # 加权平均
        fused = self.gbm_weight * gbm_pred + self.lstm_weight * lstm_pred
        
        # 方向一致性检查
        gbm_direction = np.sign(gbm_pred)
        lstm_direction = np.sign(lstm_pred)
        direction_agree = gbm_direction == lstm_direction
        
        # 方向不一致时，降低预测幅度 50%
        fused = np.where(direction_agree, fused, fused * 0.5)
        
        return fused
    
    def _apply_confidence_filter(self, predictions: np.ndarray, 
                                  y_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        应用置信度过滤，只保留高置信度预测
        
        Returns:
            filtered_pred, filtered_true, n_trades
        """
        # 只交易预测幅度超过阈值的信号
        mask = np.abs(predictions) > self.confidence_threshold
        
        filtered_pred = predictions[mask]
        filtered_true = y_true[mask]
        n_trades = np.sum(mask)
        
        return filtered_pred, filtered_true, n_trades
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测
        
        Returns:
            (predictions, confidence) - 预测值和置信度
        """
        gbm_pred = self.gbm_model.predict(X)
        
        if self.lstm_model is not None and len(X) > self.lstm_model.sequence_length:
            # 为 LSTM 准备增强特征
            X_lstm = self.lstm_model.prepare_lstm_features(X.copy())
            X_lstm_scaled = self.lstm_model.scaler.transform(X_lstm)
            X_seq, _ = self.lstm_model.prepare_sequences(X_lstm_scaled)
            lstm_pred = self.lstm_model.model.predict(X_seq, verbose=0).flatten()
            
            offset = len(gbm_pred) - len(lstm_pred)
            if offset > 0:
                gbm_pred = gbm_pred[offset:]
        else:
            lstm_pred = gbm_pred
        
        # 融合预测
        fused = self._fuse_predictions(gbm_pred, lstm_pred)
        
        # 计算置信度 (方向一致性 + 预测幅度)
        gbm_dir = np.sign(gbm_pred)
        lstm_dir = np.sign(lstm_pred)
        direction_agree = (gbm_dir == lstm_dir).astype(float)
        magnitude_confidence = np.abs(fused) / (np.abs(fused).max() + 1e-8)
        
        confidence = 0.5 * direction_agree + 0.5 * magnitude_confidence
        
        return fused, confidence
    
    def save(self, filepath: str):
        """保存融合模型"""
        # 保存 GBM
        gbm_path = filepath.replace('.pkl', '_gbm.pkl')
        self.gbm_model.save(gbm_path)
        
        # 保存 LSTM
        if self.lstm_model is not None:
            lstm_path = filepath.replace('.pkl', '_lstm.pkl')
            self.lstm_model.save(lstm_path)
            lstm_saved = True
        else:
            lstm_path = None
            lstm_saved = False
        
        # 保存融合参数
        params = {
            'gbm_weight': self.gbm_weight,
            'lstm_weight': self.lstm_weight,
            'confidence_threshold': self.confidence_threshold,
            'gbm_path': gbm_path,
            'lstm_path': lstm_path,
            'lstm_saved': lstm_saved,
            'feature_names': self.feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)
        
        print(f"💾 融合模型已保存到: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'EnsemblePredictor':
        """加载融合模型"""
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        
        predictor = cls(
            gbm_weight=params['gbm_weight'],
            lstm_weight=params['lstm_weight'],
            confidence_threshold=params['confidence_threshold']
        )
        predictor.feature_names = params['feature_names']
        
        # 加载 GBM
        predictor.gbm_model = PriceMovementPredictor.load(params['gbm_path'])
        
        # 加载 LSTM
        if params['lstm_saved'] and LSTM_AVAILABLE:
            predictor.lstm_model = LSTMPredictor.load(params['lstm_path'])
        
        return predictor


class PriceMovementPredictor:
    """价格变动预测器 - 回归模型"""
    
    def __init__(self, selected_models: List[str] = None, model_params: dict = None):
        """
        Args:
            selected_models: 要训练的模型列表 ['rf', 'gbm', 'ridge']
            model_params: 模型参数字典
        """
        params = model_params or {}
        
        all_models = {
            'rf': RandomForestRegressor(
                n_estimators=params.get('rf_estimators', 100), 
                max_depth=params.get('rf_depth', 10), 
                min_samples_split=10,
                random_state=42,
                n_jobs=-1
            ),
            'gbm': GradientBoostingRegressor(
                n_estimators=params.get('gbm_estimators', 100), 
                learning_rate=params.get('gbm_lr', 0.1),
                max_depth=params.get('gbm_depth', 5),
                random_state=42
            ),
            'ridge': Ridge(alpha=params.get('ridge_alpha', 1.0))
        }
        
        # 只保留选中的模型
        if selected_models:
            self.models = {k: v for k, v in all_models.items() if k in selected_models}
        else:
            self.models = all_models
        
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        self.feature_importance = None
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              n_splits: int = 5) -> Dict[str, dict]:
        """
        使用时间序列交叉验证训练模型
        
        Args:
            X: 特征 DataFrame
            y: 目标 Series
            n_splits: 交叉验证折数
            
        Returns:
            各模型的评估结果
        """
        self.feature_names = X.columns.tolist()
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"训练模型: {name}")
            print('='*60)
            
            fold_metrics = []
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
                X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                y_train, y_test = y.iloc[train_idx].values, y.iloc[test_idx].values
                
                # 训练
                model.fit(X_train, y_train)
                
                # 预测
                y_pred = model.predict(X_test)
                
                # 评估
                metrics = ModelEvaluator.evaluate_regression(y_test, y_pred)
                fold_metrics.append(metrics)
                
                print(f"  Fold {fold+1}: 方向准确率={metrics['direction_accuracy']:.2%}, "
                      f"MAE={metrics['mae']:.4f}, 收益={metrics['total_return']:.2f}%")
            
            # 计算平均指标
            avg_metrics = {}
            for key in fold_metrics[0].keys():
                if key != 'confusion_matrix':
                    values = [m[key] for m in fold_metrics]
                    avg_metrics[key] = np.mean(values)
                    avg_metrics[f'{key}_std'] = np.std(values)
            
            results[name] = avg_metrics
            
            print(f"\n  平均结果:")
            print(f"    方向准确率: {avg_metrics['direction_accuracy']:.2%} "
                  f"(±{avg_metrics['direction_accuracy_std']:.2%})")
            print(f"    MAE: {avg_metrics['mae']:.4f} (±{avg_metrics['mae_std']:.4f})")
            print(f"    总收益: {avg_metrics['total_return']:.2f}% "
                  f"(±{avg_metrics['total_return_std']:.2f}%)")
            print(f"    夏普比率: {avg_metrics['sharpe_ratio']:.2f}")
            print(f"    胜率: {avg_metrics['win_rate']:.2%}")
        
        # 选择最佳模型（基于方向准确率）
        self.best_model_name = max(results, key=lambda x: results[x]['direction_accuracy'])
        self.best_model = self.models[self.best_model_name]
        
        # 使用全部数据重新训练最佳模型
        print(f"\n🏆 最佳模型: {self.best_model_name}")
        print("   使用全部数据重新训练...")
        self.best_model.fit(X_scaled, y.values)
        
        # 特征重要性
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = dict(zip(
                self.feature_names, 
                self.best_model.feature_importances_
            ))
        
        return results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        X_scaled = self.scaler.transform(X)
        return self.best_model.predict(X_scaled)
    
    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """获取最重要的特征"""
        if self.feature_importance is None:
            return []
        
        sorted_features = sorted(
            self.feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return sorted_features[:n]
    
    def save(self, filepath: str):
        """保存模型"""
        model_data = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"💾 模型已保存到: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'PriceMovementPredictor':
        """加载模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        predictor = cls()
        predictor.best_model = model_data['best_model']
        predictor.best_model_name = model_data['best_model_name']
        predictor.scaler = model_data['scaler']
        predictor.feature_names = model_data['feature_names']
        predictor.feature_importance = model_data['feature_importance']
        
        return predictor


class PriceMovementClassifier:
    """价格变动分类器"""
    
    def __init__(self, selected_models: List[str] = None, model_params: dict = None):
        """
        Args:
            selected_models: 要训练的模型列表 ['rf', 'gbm', 'ridge']
            model_params: 模型参数字典
        """
        params = model_params or {}
        
        all_models = {
            'rf': RandomForestClassifier(
                n_estimators=params.get('rf_estimators', 100), 
                max_depth=params.get('rf_depth', 10),
                min_samples_split=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'gbm': GradientBoostingClassifier(
                n_estimators=params.get('gbm_estimators', 100),
                learning_rate=params.get('gbm_lr', 0.1),
                max_depth=params.get('gbm_depth', 5),
                random_state=42
            ),
            'ridge': LogisticRegression(  # 分类中用 LogisticRegression 代替 Ridge
                C=1.0 / params.get('ridge_alpha', 1.0),  # C = 1/alpha
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            )
        }
        
        # 只保留选中的模型
        if selected_models:
            self.models = {k: v for k, v in all_models.items() if k in selected_models}
        else:
            self.models = all_models
        
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        self.classes = None
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              n_splits: int = 5) -> Dict[str, dict]:
        """训练分类模型"""
        self.feature_names = X.columns.tolist()
        self.classes = np.unique(y)
        
        X_scaled = self.scaler.fit_transform(X)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"训练分类模型: {name}")
            print('='*60)
            
            fold_accuracies = []
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
                X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                y_train, y_test = y.iloc[train_idx].values, y.iloc[test_idx].values
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                fold_accuracies.append(accuracy)
                
                print(f"  Fold {fold+1}: 准确率={accuracy:.2%}")
            
            avg_accuracy = np.mean(fold_accuracies)
            std_accuracy = np.std(fold_accuracies)
            
            results[name] = {
                'accuracy': avg_accuracy,
                'accuracy_std': std_accuracy
            }
            
            print(f"\n  平均准确率: {avg_accuracy:.2%} (±{std_accuracy:.2%})")
        
        # 选择最佳模型
        self.best_model_name = max(results, key=lambda x: results[x]['accuracy'])
        self.best_model = self.models[self.best_model_name]
        
        print(f"\n🏆 最佳分类模型: {self.best_model_name}")
        print("   使用全部数据重新训练...")
        self.best_model.fit(X_scaled, y.values)
        
        return results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测类别"""
        X_scaled = self.scaler.transform(X)
        return self.best_model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测概率"""
        X_scaled = self.scaler.transform(X)
        return self.best_model.predict_proba(X_scaled)
    
    def save(self, filepath: str):
        """保存模型"""
        model_data = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'classes': self.classes
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"💾 分类模型已保存到: {filepath}")


def load_data(data_path: str) -> pd.DataFrame:
    """加载特征数据"""
    # 支持通配符
    if '*' in data_path:
        files = glob.glob(data_path)
        if not files:
            raise FileNotFoundError(f"未找到匹配的文件: {data_path}")
        # 使用最新的文件
        data_path = max(files, key=os.path.getmtime)
    
    print(f"📂 加载数据: {data_path}")
    df = pd.read_csv(data_path)
    print(f"   样本数量: {len(df)}")
    print(f"   特征数量: {len(df.columns)}")
    
    return df


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    准备特征和标签（包含完整的数据清洗）
    
    Returns:
        (X, y_regression, y_classification)
    """
    print("\n🧹 开始数据清洗...")
    
    # 排除非特征列
    exclude_cols = [
        'target_regression', 'target_classification', 'target_direction',
        'base_timestamp', 'timestamp', 'close_price'
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y_reg = df['target_regression'] if 'target_regression' in df.columns else None
    y_cls = df['target_classification'] if 'target_classification' in df.columns else None
    
    # 1. 统计原始数据质量
    n_samples_original = len(X)
    n_nan_original = X.isna().sum().sum()
    n_inf_original = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
    
    print(f"   原始样本数: {n_samples_original}")
    print(f"   原始 NaN 数量: {n_nan_original}")
    print(f"   原始 Inf 数量: {n_inf_original}")
    
    # 2. 处理无穷值 (替换为 NaN，后续统一处理)
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # 3. 删除 NaN 比例过高的列 (>50%)
    nan_ratio = X.isna().sum() / len(X)
    high_nan_cols = nan_ratio[nan_ratio > 0.5].index.tolist()
    if high_nan_cols:
        print(f"   ⚠️ 删除高 NaN 列 ({len(high_nan_cols)}): {high_nan_cols[:5]}...")
        X = X.drop(columns=high_nan_cols)
    
    # 4. 处理缺失值 - 使用中位数填充（比用 0 更合理）
    for col in X.columns:
        if X[col].isna().any():
            median_val = X[col].median()
            if pd.isna(median_val):
                median_val = 0
            X[col] = X[col].fillna(median_val)
    
    # 5. 异常值处理 - 使用 IQR 方法裁剪
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = X[col].quantile(0.01)
        Q3 = X[col].quantile(0.99)
        X[col] = X[col].clip(Q1, Q3)
    
    # 6. 删除标签为空的行
    if y_reg is not None:
        valid_idx = ~y_reg.isna()
        X = X[valid_idx].reset_index(drop=True)
        y_reg = y_reg[valid_idx].reset_index(drop=True)
        if y_cls is not None:
            y_cls = y_cls[valid_idx].reset_index(drop=True)
    
    # 7. 删除常量列（方差为 0 的列没有预测价值）
    constant_cols = X.columns[X.std() == 0].tolist()
    if constant_cols:
        print(f"   ⚠️ 删除常量列 ({len(constant_cols)}): {constant_cols[:5]}...")
        X = X.drop(columns=constant_cols)
    
    # 8. 最终数据质量检查
    n_samples_final = len(X)
    n_nan_final = X.isna().sum().sum()
    
    print(f"\n📊 数据清洗完成:")
    print(f"   清洗后样本数: {n_samples_final} (删除 {n_samples_original - n_samples_final})")
    print(f"   清洗后特征数: {len(X.columns)}")
    print(f"   剩余 NaN 数量: {n_nan_final}")
    
    return X, y_reg, y_cls


def main():
    parser = argparse.ArgumentParser(description='BTC 价格变动预测 - 模型训练')
    parser.add_argument('--data', type=str, required=True, help='特征数据文件路径（支持通配符）')
    parser.add_argument('--output', type=str, default='../models', help='模型保存目录')
    parser.add_argument('--task', type=str, choices=['regression', 'classification', 'both'], 
                        default='both', help='任务类型')
    parser.add_argument('--cv-splits', type=int, default=5, help='交叉验证折数')
    
    # 模型选择
    parser.add_argument('--models', type=str, default='all',
                        help='要训练的模型，逗号分隔。可选: rf,gbm,ridge,lstm,ensemble 或 all (默认: all)')
    
    # 融合模型参数
    parser.add_argument('--ensemble-gbm-weight', type=float, default=0.6, help='融合模型中 GBM 权重')
    parser.add_argument('--ensemble-lstm-weight', type=float, default=0.4, help='融合模型中 LSTM 权重')
    parser.add_argument('--ensemble-threshold', type=float, default=0.3, help='融合模型置信度阈值')
    
    # LSTM 参数
    parser.add_argument('--lstm-seq-len', type=int, default=20, help='LSTM 序列长度')
    parser.add_argument('--lstm-epochs', type=int, default=50, help='LSTM 训练轮数')
    parser.add_argument('--lstm-batch-size', type=int, default=32, help='LSTM 批次大小')
    
    # 传统模型参数
    parser.add_argument('--rf-estimators', type=int, default=100, help='Random Forest 树数量')
    parser.add_argument('--rf-depth', type=int, default=10, help='Random Forest 最大深度')
    parser.add_argument('--gbm-estimators', type=int, default=100, help='GBM 树数量')
    parser.add_argument('--gbm-lr', type=float, default=0.1, help='GBM 学习率')
    parser.add_argument('--gbm-depth', type=int, default=5, help='GBM 最大深度')
    parser.add_argument('--ridge-alpha', type=float, default=1.0, help='Ridge 正则化参数')
    
    args = parser.parse_args()
    
    # 解析模型列表
    if args.models.lower() == 'all':
        selected_models = ['rf', 'gbm', 'ridge', 'lstm']
    else:
        selected_models = [m.strip().lower() for m in args.models.split(',')]
    
    # 验证模型名称
    valid_models = ['rf', 'gbm', 'ridge', 'lstm', 'ensemble']
    for m in selected_models:
        if m not in valid_models:
            print(f"❌ 无效的模型名称: {m}")
            print(f"   可选模型: {', '.join(valid_models)}")
            return
    
    print(f"🎯 选择的模型: {', '.join(selected_models)}")
    
    # 准备模型参数
    model_params = {
        'rf_estimators': args.rf_estimators,
        'rf_depth': args.rf_depth,
        'gbm_estimators': args.gbm_estimators,
        'gbm_lr': args.gbm_lr,
        'gbm_depth': args.gbm_depth,
        'ridge_alpha': args.ridge_alpha,
    }
    
    # 分离传统模型、LSTM 和融合模型
    traditional_models = [m for m in selected_models if m not in ['lstm', 'ensemble']]
    train_lstm = 'lstm' in selected_models
    train_ensemble = 'ensemble' in selected_models
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 加载数据
    df = load_data(args.data)
    
    # 准备特征
    X, y_reg, y_cls = prepare_features(df)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 回归任务 - 传统模型
    if args.task in ['regression', 'both'] and y_reg is not None and traditional_models:
        print("\n" + "="*60)
        print("📈 回归任务: 预测价格变动百分比")
        print("="*60)
        
        predictor = PriceMovementPredictor(
            selected_models=traditional_models,
            model_params=model_params
        )
        results = predictor.train(X, y_reg, n_splits=args.cv_splits)
        
        if predictor.best_model is not None:
            # 打印特征重要性
            print("\n🔍 Top 10 重要特征:")
            for i, (feat, imp) in enumerate(predictor.get_top_features(10), 1):
                print(f"   {i}. {feat}: {imp:.4f}")
            
            # 保存模型
            model_path = os.path.join(args.output, f'regression_model_{timestamp}.pkl')
            predictor.save(model_path)
            
            # 保存结果
            results_path = os.path.join(args.output, f'regression_results_{timestamp}.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"📊 结果已保存到: {results_path}")
    
    # 分类任务 - 传统模型
    if args.task in ['classification', 'both'] and y_cls is not None and traditional_models:
        print("\n" + "="*60)
        print("📊 分类任务: 预测价格变动区间")
        print("="*60)
        
        # 过滤掉 NaN 类别
        valid_cls_idx = ~y_cls.isna()
        X_cls = X[valid_cls_idx]
        y_cls_valid = y_cls[valid_cls_idx].astype(int)
        
        print(f"\n标签分布:")
        labels = {0: '大跌(<-2%)', 1: '小跌(-2%~-0.5%)', 2: '横盘(-0.5%~0.5%)', 
                  3: '小涨(0.5%~2%)', 4: '大涨(>2%)'}
        for label, name in labels.items():
            count = (y_cls_valid == label).sum()
            pct = count / len(y_cls_valid) * 100
            print(f"   {name}: {count} ({pct:.1f}%)")
        
        classifier = PriceMovementClassifier(
            selected_models=traditional_models,
            model_params=model_params
        )
        results = classifier.train(X_cls, y_cls_valid, n_splits=args.cv_splits)
        
        if classifier.best_model is not None:
            # 保存模型
            model_path = os.path.join(args.output, f'classification_model_{timestamp}.pkl')
            classifier.save(model_path)
    
    # LSTM 模型训练
    if train_lstm and LSTM_AVAILABLE:
        print("\n" + "="*60)
        print("🧠 LSTM 模型训练")
        print("="*60)
        
        # LSTM 回归
        if args.task in ['regression', 'both'] and y_reg is not None:
            print("\n📈 LSTM 回归任务:")
            
            lstm_reg = LSTMPredictor(
                sequence_length=args.lstm_seq_len, 
                task='regression'
            )
            lstm_results = lstm_reg.train(
                X, y_reg, 
                n_splits=args.cv_splits,
                epochs=args.lstm_epochs
            )
            
            if lstm_results:
                print(f"\n  平均结果:")
                print(f"    方向准确率: {lstm_results.get('direction_accuracy', 0):.2%} "
                      f"(±{lstm_results.get('direction_accuracy_std', 0):.2%})")
                print(f"    MAE: {lstm_results.get('mae', 0):.4f} "
                      f"(±{lstm_results.get('mae_std', 0):.4f})")
                print(f"    总收益: {lstm_results.get('total_return', 0):.2f}% "
                      f"(±{lstm_results.get('total_return_std', 0):.2f}%)")
                
                # 保存 LSTM 模型
                lstm_path = os.path.join(args.output, f'lstm_regression_{timestamp}.pkl')
                lstm_reg.save(lstm_path)
                
                # 保存结果
                lstm_results_path = os.path.join(args.output, f'lstm_regression_results_{timestamp}.json')
                with open(lstm_results_path, 'w') as f:
                    json.dump(lstm_results, f, indent=2)
        
        # LSTM 分类
        if args.task in ['classification', 'both'] and y_cls is not None:
            print("\n📊 LSTM 分类任务:")
            
            valid_cls_idx = ~y_cls.isna()
            X_cls = X[valid_cls_idx]
            y_cls_valid = y_cls[valid_cls_idx].astype(int)
            
            lstm_cls = LSTMPredictor(
                sequence_length=args.lstm_seq_len, 
                task='classification'
            )
            lstm_cls_results = lstm_cls.train(
                X_cls, y_cls_valid, 
                n_splits=args.cv_splits,
                epochs=args.lstm_epochs
            )
            
            if lstm_cls_results:
                print(f"\n  平均准确率: {lstm_cls_results.get('accuracy', 0):.2%} "
                      f"(±{lstm_cls_results.get('accuracy_std', 0):.2%})")
                
                # 保存 LSTM 分类模型
                lstm_cls_path = os.path.join(args.output, f'lstm_classification_{timestamp}.pkl')
                lstm_cls.save(lstm_cls_path)
    
    elif train_lstm and not LSTM_AVAILABLE:
        print("\n⚠️ 跳过 LSTM 训练：TensorFlow 未安装")
        print("   安装命令: pip install tensorflow")
    
    # 融合模型训练
    if train_ensemble and args.task in ['regression', 'both'] and y_reg is not None:
        print("\n" + "="*60)
        print("🔀 融合模型训练 (GBM + LSTM)")
        print("="*60)
        
        ensemble = EnsemblePredictor(
            gbm_weight=args.ensemble_gbm_weight,
            lstm_weight=args.ensemble_lstm_weight,
            confidence_threshold=args.ensemble_threshold
        )
        
        ensemble_results = ensemble.train(
            X, y_reg,
            n_splits=args.cv_splits,
            lstm_epochs=args.lstm_epochs,
            lstm_seq_len=args.lstm_seq_len
        )
        
        # 保存融合模型
        ensemble_path = os.path.join(args.output, f'ensemble_model_{timestamp}.pkl')
        ensemble.save(ensemble_path)
        
        # 保存结果
        ensemble_results_path = os.path.join(args.output, f'ensemble_results_{timestamp}.json')
        # 转换为可序列化格式
        serializable_results = {}
        for key, value in ensemble_results.items():
            if isinstance(value, dict):
                serializable_results[key] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                             for k, v in value.items()}
            else:
                serializable_results[key] = value
        with open(ensemble_results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"📊 融合模型结果已保存到: {ensemble_results_path}")
    
    print("\n" + "="*60)
    print("✅ 训练完成!")
    print("="*60)


if __name__ == "__main__":
    main()

