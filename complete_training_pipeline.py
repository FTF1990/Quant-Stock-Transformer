"""
完整的股票预测模型训练流程
====================================

完整流程：
1. 从JSON导入选股列表
2. 历史数据获取（yfinance + akshare）
3. 数据预处理和特征工程
4. Stage 1: SST模型训练（双输出：T日 + T+1日）
5. Stage 2: 特征提取（Attention + Encoder + 残差）
6. Stage 3: 时序模型训练（LSTM + GRU + TCN）
7. 模型效果测试和对比

使用方法：
    python complete_training_pipeline.py --stocks_json data/demo.json --target_market CN --target_stock 600519

作者：Quant-Stock-Transformer Team
版本：1.0.0
"""

import json
import pickle
import argparse
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 数据获取库
try:
    import yfinance as yf
except ImportError:
    print("警告: yfinance未安装，无法获取美股/港股/日股数据")

try:
    import akshare as ak
except ImportError:
    print("警告: akshare未安装，无法获取A股数据")

# 导入项目模块
from models.spatial_feature_extractor import SpatialFeatureExtractor
from models.temporal_predictor import (
    LSTMTemporalPredictor,
    GRUTemporalPredictor,
    TCNTemporalPredictor,
    TemporalDataset
)
from models.relationship_extractors import HybridExtractor

warnings.filterwarnings('ignore')

# 设置绘图样式
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


# ============================================================================
# 第一步：数据获取模块
# ============================================================================

class StockDataFetcher:
    """多市场股票数据抓取器"""

    def __init__(self):
        self.data_cache = {}

    def fetch_historical_data(
        self,
        stocks_json: Dict,
        start_date: str,
        end_date: str,
        interval: str = "1d",
        include_market_index: bool = True,
        batch_size: int = 5,
        delay_between_batches: float = 2.0,
        delay_between_stocks: float = 0.5
    ) -> Dict:
        """
        抓取历史数据（智能分批，避免API限流）

        Args:
            stocks_json: 股票JSON字典
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            interval: 数据粒度（"1d"按天，"1h"按小时）
            include_market_index: 是否包含大盘指数
            batch_size: 每批抓取的股票数量
            delay_between_batches: 批次间延迟秒数
            delay_between_stocks: 同批股票间延迟秒数
        """

        print(f"\n{'='*80}")
        print(f"📥 开始抓取历史数据")
        print(f"日期范围: {start_date} 至 {end_date}")
        print(f"数据粒度: {interval}")
        print(f"⏱️  分批配置: 每批{batch_size}支，批次间延迟{delay_between_batches}秒")
        print(f"{'='*80}\n")

        all_data = {}

        for market, stocks in stocks_json.items():
            print(f"\n🔄 正在处理{market}市场 ({len(stocks)}只股票)...")

            market_data = {}

            # 抓取大盘指数
            if include_market_index:
                index_data = self._fetch_market_index(
                    market, start_date, end_date, interval
                )
                if index_data is not None:
                    market_data['_INDEX_'] = index_data
                    print(f"  ✓ 大盘指数数据获取成功 ({len(index_data)}条记录)")
                time.sleep(delay_between_stocks)

            # 分批抓取个股数据
            total_stocks = len(stocks)
            num_batches = (total_stocks + batch_size - 1) // batch_size

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, total_stocks)
                batch_stocks = stocks[start_idx:end_idx]

                print(f"\n  批次 [{batch_idx+1}/{num_batches}]: 抓取第{start_idx+1}-{end_idx}支股票")

                for i, stock in enumerate(batch_stocks, start=start_idx+1):
                    symbol = stock['symbol']
                    try:
                        data = self._fetch_stock_data(
                            market, symbol, start_date, end_date, interval
                        )
                        if data is not None and len(data) > 0:
                            market_data[symbol] = data
                            print(f"    ✓ [{i}/{total_stocks}] {symbol}: {len(data)}条数据")
                        else:
                            print(f"    ✗ [{i}/{total_stocks}] {symbol}: 无数据")
                    except Exception as e:
                        print(f"    ✗ [{i}/{total_stocks}] {symbol}: 失败 ({str(e)[:50]})")

                    if i < total_stocks:
                        time.sleep(delay_between_stocks)

                if batch_idx < num_batches - 1:
                    print(f"  ⏸️  批次完成，等待{delay_between_batches}秒后继续...")
                    time.sleep(delay_between_batches)

            all_data[market] = market_data
            print(f"\n  ✓ {market}市场完成：成功{len(market_data)}支（含指数）")

        self.data_cache = all_data

        print(f"\n{'='*80}")
        print("✓ 所有数据抓取完成！")
        total_success = sum(len(v) for v in all_data.values())
        total_requested = sum(len(v) for v in stocks_json.values()) + len(stocks_json)
        print(f"成功率: {total_success}/{total_requested} ({100*total_success/total_requested:.1f}%)")
        print(f"{'='*80}\n")

        return all_data

    def _fetch_market_index(
        self, market: str, start_date: str, end_date: str, interval: str
    ) -> Optional[pd.DataFrame]:
        """抓取大盘指数"""

        index_symbols = {
            "US": "^GSPC",      # S&P 500
            "CN": "000001",     # 上证指数
            "HK": "^HSI",       # 恒生指数
            "JP": "^N225"       # 日经225
        }

        symbol = index_symbols.get(market)
        if not symbol:
            return None

        try:
            if market == "CN":
                # 方法1: 尝试使用akshare
                try:
                    df = ak.stock_zh_index_daily(symbol=f"sh{symbol}")
                    df['date'] = pd.to_datetime(df['date'])
                    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                    df = df.rename(columns={'close': 'Close', 'open': 'Open',
                                           'high': 'High', 'low': 'Low', 'volume': 'Volume'})
                    df = df.set_index('date')
                    return df
                except Exception as ak_error:
                    # 方法2: akshare失败时，使用yfinance备选方案
                    print(f"  ⚠️  akshare失败，切换到yfinance获取指数...")
                    # 使用上证指数的yfinance符号
                    yahoo_index_symbol = "000001.SS"
                    df = yf.download(yahoo_index_symbol, start=start_date, end=end_date,
                                    interval=interval, progress=False)
                    if len(df) == 0:
                        # 如果上证指数失败，尝试使用^SSEC
                        df = yf.download("^SSEC", start=start_date, end=end_date,
                                        interval=interval, progress=False)
                    return df
            else:
                df = yf.download(symbol, start=start_date, end=end_date,
                                interval=interval, progress=False)
                return df

        except Exception as e:
            print(f"    警告: 大盘指数获取失败 ({e})")
            return None

    def _fetch_stock_data(
        self, market: str, symbol: str, start_date: str, end_date: str, interval: str
    ) -> Optional[pd.DataFrame]:
        """抓取个股数据"""

        try:
            if market == "CN":
                # 方法1: 尝试使用akshare（国内数据更准确）
                try:
                    df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                           start_date=start_date.replace('-', ''),
                                           end_date=end_date.replace('-', ''))
                    df['日期'] = pd.to_datetime(df['日期'])
                    df = df.rename(columns={
                        '日期': 'Date', '收盘': 'Close', '开盘': 'Open',
                        '最高': 'High', '最低': 'Low', '成交量': 'Volume'
                    })
                    df = df.set_index('Date')
                    return df
                except Exception as ak_error:
                    # 方法2: akshare失败时，使用yfinance备选方案（添加交易所后缀）
                    print(f"      ⚠️  akshare失败，切换到yfinance...")

                    # 添加交易所后缀
                    if symbol.startswith('6'):
                        yahoo_symbol = f"{symbol}.SS"  # 上海交易所
                    elif symbol.startswith('0') or symbol.startswith('3'):
                        yahoo_symbol = f"{symbol}.SZ"  # 深圳交易所
                    else:
                        yahoo_symbol = symbol

                    df = yf.download(yahoo_symbol, start=start_date, end=end_date,
                                    interval=interval, progress=False)

                    if len(df) > 0:
                        # yfinance返回的列名已经是英文，可能需要重置索引
                        if df.index.name != 'Date':
                            df.index.name = 'Date'
                        return df
                    else:
                        raise Exception(f"yfinance也未返回数据")
            else:
                if market == "HK" and not symbol.endswith(".HK"):
                    symbol = symbol.zfill(4) + ".HK"
                elif market == "JP" and not symbol.endswith(".T"):
                    if '.' not in symbol:
                        symbol = symbol + ".T"

                df = yf.download(symbol, start=start_date, end=end_date,
                                interval=interval, progress=False)

            return df
        except Exception as e:
            raise Exception(f"数据获取失败: {e}")

    def save_data(self, output_path: str = "historical_data.pkl"):
        """保存数据到pickle文件"""
        with open(output_path, 'wb') as f:
            pickle.dump(self.data_cache, f)
        print(f"✓ 数据已保存到: {output_path}")

    @staticmethod
    def load_data(input_path: str) -> Dict:
        """从pickle文件加载数据"""
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
        print(f"✓ 数据已从 {input_path} 加载")
        return data


# ============================================================================
# 第二步：数据预处理模块
# ============================================================================

class StockDataProcessor:
    """股票数据预处理器"""

    def __init__(self, historical_data: Dict, target_market: str, target_stock: str):
        self.historical_data = historical_data
        self.target_market = target_market
        self.target_stock = target_stock
        self.scaler = StandardScaler()

    def prepare_training_data(
        self,
        use_all_stocks: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List]:
        """
        准备训练数据

        Returns:
            boundary_conditions: [N, num_boundary_sensors] - 边界条件
            targets_T: [N, 1] - T日目标（当日收益率）
            targets_T1: [N, 1] - T+1日目标（次日收益率）
            dates: [N] - 日期索引
        """
        print(f"\n{'='*80}")
        print("🔄 开始数据预处理...")
        print(f"{'='*80}\n")

        # 获取市场数据
        market_data = self.historical_data[self.target_market]

        # 获取大盘指数
        index_df = market_data.get('_INDEX_')

        # 获取目标股票数据
        target_df = market_data[self.target_stock].copy()

        # 计算收益率
        target_df['return'] = target_df['Close'].pct_change()
        target_df['return_next'] = target_df['return'].shift(-1)

        # 移除NaN
        target_df = target_df.dropna()

        print(f"  ✓ 目标股票数据: {len(target_df)}条")

        # 构建边界条件
        if use_all_stocks:
            # 使用所有股票的数据作为边界条件（简化版实现）
            print("  📊 使用目标股票的OHLCV作为边界条件...")
            boundary_features = []

            for i in range(len(target_df) - 1):
                features = [
                    target_df['Open'].iloc[i],
                    target_df['High'].iloc[i],
                    target_df['Low'].iloc[i],
                    target_df['Close'].iloc[i],
                    target_df['Volume'].iloc[i]
                ]

                # 添加大盘指数
                if index_df is not None and i < len(index_df):
                    features.append(index_df['Close'].iloc[i])

                boundary_features.append(features)
        else:
            boundary_features = [[
                target_df['Open'].iloc[i],
                target_df['High'].iloc[i],
                target_df['Low'].iloc[i],
                target_df['Close'].iloc[i],
                target_df['Volume'].iloc[i]
            ] for i in range(len(target_df) - 1)]

        # 提取目标
        targets_T = target_df['return'].values[:-1].reshape(-1, 1)
        targets_T1 = target_df['return_next'].values[:-1].reshape(-1, 1)
        dates = target_df.index[:-1].tolist()

        # 转换为numpy数组
        boundary_features = np.array(boundary_features, dtype=np.float32)
        targets_T = np.array(targets_T, dtype=np.float32)
        targets_T1 = np.array(targets_T1, dtype=np.float32)

        # 标准化边界条件
        boundary_features = self.scaler.fit_transform(boundary_features)

        print(f"  ✓ 边界条件形状: {boundary_features.shape}")
        print(f"  ✓ T日目标形状: {targets_T.shape}")
        print(f"  ✓ T+1日目标形状: {targets_T1.shape}")
        print(f"\n{'='*80}")
        print("✓ 数据预处理完成")
        print(f"{'='*80}\n")

        return boundary_features, targets_T, targets_T1, dates


# ============================================================================
# 第三步：双输出SST模型
# ============================================================================

class DualOutputSST(SpatialFeatureExtractor):
    """双输出SST - 同时预测T日和T+1日收益率"""

    def __init__(self, num_boundary_sensors, num_target_sensors, **kwargs):
        super().__init__(num_boundary_sensors, num_target_sensors, **kwargs)

        # 双输出头
        self.output_projection_T = nn.Linear(self.d_model, num_target_sensors)
        self.output_projection_T1 = nn.Linear(self.d_model, num_target_sensors)

        nn.init.xavier_uniform_(self.output_projection_T.weight)
        nn.init.xavier_uniform_(self.output_projection_T1.weight)

    def forward(self, boundary_conditions):
        """前向传播"""
        batch_size = boundary_conditions.shape[0]

        x = boundary_conditions.unsqueeze(-1)
        x = self.boundary_embedding(x) + self.boundary_position_encoding.unsqueeze(0)
        x = self.transformer(x)
        x_pooled = x.permute(0, 2, 1)
        x_pooled = self.global_pool(x_pooled).squeeze(-1)

        pred_T = self.output_projection_T(x_pooled)
        pred_T1 = self.output_projection_T1(x_pooled)

        return pred_T, pred_T1

    def forward_with_features(self, boundary_conditions, **kwargs):
        """前向传播并返回内部特征"""
        batch_size = boundary_conditions.shape[0]
        features = {}

        x = boundary_conditions.unsqueeze(-1)
        x = self.boundary_embedding(x) + self.boundary_position_encoding.unsqueeze(0)
        features['embeddings'] = x.clone()

        # 获取attention权重
        if kwargs.get('return_attention', True):
            encoder_output = x
            all_attention_weights = []

            for layer in self.transformer.layers:
                encoder_output, attn_weights = self._extract_attention_from_layer(
                    layer, encoder_output
                )
                all_attention_weights.append(attn_weights)

            features['attention_weights'] = torch.stack(all_attention_weights, dim=1)
            features['encoder_output'] = encoder_output
        else:
            encoder_output = self.transformer(x)
            features['encoder_output'] = encoder_output

        x_pooled = encoder_output.permute(0, 2, 1)
        x_pooled = self.global_pool(x_pooled).squeeze(-1)
        features['pooled_features'] = x_pooled

        pred_T = self.output_projection_T(x_pooled)
        pred_T1 = self.output_projection_T1(x_pooled)

        return (pred_T, pred_T1), features

    def _extract_attention_from_layer(self, layer, x):
        """从Transformer层提取attention权重"""
        # 简化实现：直接调用forward
        attn_output = layer.self_attn(x, x, x, need_weights=True)
        if isinstance(attn_output, tuple):
            output, attn_weights = attn_output
        else:
            output = attn_output
            attn_weights = None

        # 残差连接和layer norm
        x = x + layer.dropout1(output)
        x = layer.norm1(x)

        # Feed forward
        ff_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
        x = x + layer.dropout2(ff_output)
        x = layer.norm2(x)

        return x, attn_weights


# ============================================================================
# 第四步：训练器
# ============================================================================

class ModelTrainer:
    """模型训练器"""

    def __init__(self, device='cpu'):
        self.device = device
        self.history = {}

    def train_sst(
        self,
        model,
        X_train, y_T_train, y_T1_train,
        X_val, y_T_val, y_T1_val,
        epochs=50,
        batch_size=32,
        lr=0.001,
        verbose=True
    ):
        """训练双输出SST模型"""

        print(f"\n{'='*80}")
        print("🚀 开始训练SST模型")
        print(f"{'='*80}\n")

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # 转换为tensor
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_T_train_t = torch.FloatTensor(y_T_train).to(self.device)
        y_T1_train_t = torch.FloatTensor(y_T1_train).to(self.device)

        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_T_val_t = torch.FloatTensor(y_T_val).to(self.device)
        y_T1_val_t = torch.FloatTensor(y_T1_val).to(self.device)

        history = {'train_loss': [], 'val_loss': [], 'train_loss_T': [],
                   'train_loss_T1': [], 'val_loss_T': [], 'val_loss_T1': []}

        best_val_loss = float('inf')

        for epoch in range(epochs):
            model.train()

            # 训练
            epoch_loss = 0
            epoch_loss_T = 0
            epoch_loss_T1 = 0
            num_batches = (len(X_train) + batch_size - 1) // batch_size

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(X_train))

                batch_X = X_train_t[start_idx:end_idx]
                batch_y_T = y_T_train_t[start_idx:end_idx]
                batch_y_T1 = y_T1_train_t[start_idx:end_idx]

                optimizer.zero_grad()

                pred_T, pred_T1 = model(batch_X)

                loss_T = criterion(pred_T, batch_y_T)
                loss_T1 = criterion(pred_T1, batch_y_T1)
                loss = loss_T + loss_T1

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_loss_T += loss_T.item()
                epoch_loss_T1 += loss_T1.item()

            epoch_loss /= num_batches
            epoch_loss_T /= num_batches
            epoch_loss_T1 /= num_batches

            # 验证
            model.eval()
            with torch.no_grad():
                val_pred_T, val_pred_T1 = model(X_val_t)
                val_loss_T = criterion(val_pred_T, y_T_val_t).item()
                val_loss_T1 = criterion(val_pred_T1, y_T1_val_t).item()
                val_loss = val_loss_T + val_loss_T1

            history['train_loss'].append(epoch_loss)
            history['train_loss_T'].append(epoch_loss_T)
            history['train_loss_T1'].append(epoch_loss_T1)
            history['val_loss'].append(val_loss)
            history['val_loss_T'].append(val_loss_T)
            history['val_loss_T1'].append(val_loss_T1)

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_sst_model.pth')

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}]")
                print(f"  Train Loss: {epoch_loss:.6f} (T: {epoch_loss_T:.6f}, T+1: {epoch_loss_T1:.6f})")
                print(f"  Val Loss: {val_loss:.6f} (T: {val_loss_T:.6f}, T+1: {val_loss_T1:.6f})")

        print(f"\n{'='*80}")
        print(f"✓ SST训练完成！最佳验证损失: {best_val_loss:.6f}")
        print(f"{'='*80}\n")

        self.history['sst'] = history
        return history

    def train_temporal_model(
        self,
        model,
        train_loader,
        val_loader,
        epochs=100,
        lr=0.001,
        model_name='Temporal',
        verbose=True
    ):
        """训练时序模型"""

        print(f"\n{'='*80}")
        print(f"🚀 开始训练{model_name}模型")
        print(f"{'='*80}\n")

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')

        for epoch in range(epochs):
            # 训练
            model.train()
            train_loss = 0

            for batch_seq, batch_target in train_loader:
                batch_seq = batch_seq.to(self.device)
                batch_target = batch_target.to(self.device)

                optimizer.zero_grad()
                predictions = model(batch_seq)
                loss = criterion(predictions, batch_target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # 验证
            model.eval()
            val_loss = 0

            with torch.no_grad():
                for batch_seq, batch_target in val_loader:
                    batch_seq = batch_seq.to(self.device)
                    batch_target = batch_target.to(self.device)

                    predictions = model(batch_seq)
                    loss = criterion(predictions, batch_target)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'best_{model_name.lower()}_model.pth')

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        print(f"\n{'='*80}")
        print(f"✓ {model_name}训练完成！最佳验证损失: {best_val_loss:.6f}")
        print(f"{'='*80}\n")

        self.history[model_name.lower()] = history
        return history


# ============================================================================
# 第五步：模型评估器
# ============================================================================

class ModelEvaluator:
    """模型评估器"""

    def __init__(self, device='cpu'):
        self.device = device
        self.results = {}

    def evaluate_sst(
        self,
        model,
        X_test,
        y_T_test,
        y_T1_test,
        model_name='SST'
    ) -> Dict:
        """评估SST模型"""

        print(f"\n{'='*80}")
        print(f"📊 评估{model_name}模型")
        print(f"{'='*80}\n")

        model.eval()

        X_test_t = torch.FloatTensor(X_test).to(self.device)

        with torch.no_grad():
            pred_T, pred_T1 = model(X_test_t)
            pred_T = pred_T.cpu().numpy()
            pred_T1 = pred_T1.cpu().numpy()

        # 计算指标
        metrics = {}

        # T日预测指标
        metrics['T_MSE'] = mean_squared_error(y_T_test, pred_T)
        metrics['T_MAE'] = mean_absolute_error(y_T_test, pred_T)
        metrics['T_Direction_Acc'] = self._direction_accuracy(y_T_test, pred_T)

        # T+1日预测指标
        metrics['T1_MSE'] = mean_squared_error(y_T1_test, pred_T1)
        metrics['T1_MAE'] = mean_absolute_error(y_T1_test, pred_T1)
        metrics['T1_Direction_Acc'] = self._direction_accuracy(y_T1_test, pred_T1)

        # 打印结果
        print(f"T日预测:")
        print(f"  MSE: {metrics['T_MSE']:.6f}")
        print(f"  MAE: {metrics['T_MAE']:.6f}")
        print(f"  方向准确率: {metrics['T_Direction_Acc']:.2%}")

        print(f"\nT+1日预测:")
        print(f"  MSE: {metrics['T1_MSE']:.6f}")
        print(f"  MAE: {metrics['T1_MAE']:.6f}")
        print(f"  方向准确率: {metrics['T1_Direction_Acc']:.2%}")

        print(f"\n{'='*80}\n")

        self.results[model_name] = {
            'metrics': metrics,
            'predictions': {'T': pred_T, 'T1': pred_T1},
            'actuals': {'T': y_T_test, 'T1': y_T1_test}
        }

        return metrics

    def evaluate_temporal_model(
        self,
        model,
        test_loader,
        model_name='Temporal'
    ) -> Dict:
        """评估时序模型"""

        print(f"\n{'='*80}")
        print(f"📊 评估{model_name}模型")
        print(f"{'='*80}\n")

        model.eval()

        all_predictions = []
        all_actuals = []

        with torch.no_grad():
            for batch_seq, batch_target in test_loader:
                batch_seq = batch_seq.to(self.device)
                predictions = model(batch_seq)
                all_predictions.append(predictions.cpu().numpy())
                all_actuals.append(batch_target.numpy())

        predictions = np.vstack(all_predictions)
        actuals = np.vstack(all_actuals)

        # 计算指标
        metrics = {}
        metrics['MSE'] = mean_squared_error(actuals, predictions)
        metrics['MAE'] = mean_absolute_error(actuals, predictions)
        metrics['Direction_Acc'] = self._direction_accuracy(actuals, predictions)
        metrics['Sharpe_Ratio'] = self._sharpe_ratio(actuals, predictions)

        # 打印结果
        print(f"  MSE: {metrics['MSE']:.6f}")
        print(f"  MAE: {metrics['MAE']:.6f}")
        print(f"  方向准确率: {metrics['Direction_Acc']:.2%}")
        print(f"  Sharpe比率: {metrics['Sharpe_Ratio']:.4f}")

        print(f"\n{'='*80}\n")

        self.results[model_name] = {
            'metrics': metrics,
            'predictions': predictions,
            'actuals': actuals
        }

        return metrics

    def _direction_accuracy(self, y_true, y_pred):
        """计算方向准确率"""
        true_direction = np.sign(y_true)
        pred_direction = np.sign(y_pred)
        return np.mean(true_direction == pred_direction)

    def _sharpe_ratio(self, y_true, y_pred, risk_free_rate=0.0):
        """计算Sharpe比率"""
        returns = y_pred.flatten()
        excess_returns = returns - risk_free_rate
        if len(excess_returns) == 0 or np.std(excess_returns) == 0:
            return 0.0
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

    def compare_models(self):
        """对比所有模型的性能"""

        print(f"\n{'='*80}")
        print("📊 模型性能对比")
        print(f"{'='*80}\n")

        comparison_data = []

        for model_name, result in self.results.items():
            metrics = result['metrics']
            row = {'Model': model_name}
            row.update(metrics)
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))

        print(f"\n{'='*80}\n")

        return df


# ============================================================================
# 第六步：完整Pipeline
# ============================================================================

class CompletePipeline:
    """完整的训练和评估流程"""

    def __init__(
        self,
        stocks_json_path: str,
        target_market: str,
        target_stock: str,
        device: str = 'cpu'
    ):
        self.stocks_json_path = stocks_json_path
        self.target_market = target_market
        self.target_stock = target_stock
        self.device = device

        # 加载股票列表
        with open(stocks_json_path, 'r', encoding='utf-8') as f:
            self.stocks_json = json.load(f)

        print(f"\n{'='*80}")
        print("🎯 Pipeline配置")
        print(f"{'='*80}")
        print(f"目标市场: {target_market}")
        print(f"目标股票: {target_stock}")
        print(f"设备: {device}")
        print(f"股票列表: {sum(len(v) for v in self.stocks_json.values())}只股票")
        print(f"{'='*80}\n")

    def run(
        self,
        start_date: str = "2020-01-01",
        end_date: str = "2024-12-31",
        fetch_data: bool = True,
        data_path: str = None,
        sst_epochs: int = 50,
        temporal_epochs: int = 100,
        seq_len: int = 60
    ):
        """运行完整流程"""

        # Step 1: 获取历史数据
        if fetch_data:
            fetcher = StockDataFetcher()
            historical_data = fetcher.fetch_historical_data(
                stocks_json=self.stocks_json,
                start_date=start_date,
                end_date=end_date,
                interval="1d",
                include_market_index=True
            )
            fetcher.save_data("historical_data.pkl")
        else:
            if data_path is None:
                data_path = "historical_data.pkl"
            historical_data = StockDataFetcher.load_data(data_path)

        # Step 2: 数据预处理
        processor = StockDataProcessor(
            historical_data=historical_data,
            target_market=self.target_market,
            target_stock=self.target_stock
        )

        X, y_T, y_T1, dates = processor.prepare_training_data()

        # 数据集划分
        train_size = int(0.7 * len(X))
        val_size = int(0.15 * len(X))

        X_train = X[:train_size]
        y_T_train = y_T[:train_size]
        y_T1_train = y_T1[:train_size]

        X_val = X[train_size:train_size+val_size]
        y_T_val = y_T[train_size:train_size+val_size]
        y_T1_val = y_T1[train_size:train_size+val_size]

        X_test = X[train_size+val_size:]
        y_T_test = y_T[train_size+val_size:]
        y_T1_test = y_T1[train_size+val_size:]

        print(f"训练集: {len(X_train)} 样本")
        print(f"验证集: {len(X_val)} 样本")
        print(f"测试集: {len(X_test)} 样本\n")

        # Step 3: 训练SST模型
        num_features = X.shape[1]
        sst_model = DualOutputSST(
            num_boundary_sensors=num_features,
            num_target_sensors=1,
            d_model=128,
            nhead=8,
            num_layers=3,
            dropout=0.1,
            enable_feature_extraction=True
        ).to(self.device)

        print(f"SST模型参数量: {sum(p.numel() for p in sst_model.parameters()):,}\n")

        trainer = ModelTrainer(device=self.device)
        sst_history = trainer.train_sst(
            sst_model,
            X_train, y_T_train, y_T1_train,
            X_val, y_T_val, y_T1_val,
            epochs=sst_epochs,
            batch_size=32,
            lr=0.001
        )

        # Step 4: 评估SST模型
        evaluator = ModelEvaluator(device=self.device)
        sst_metrics = evaluator.evaluate_sst(
            sst_model,
            X_test,
            y_T_test,
            y_T1_test,
            model_name='SST'
        )

        # Step 5: 提取特征
        print(f"\n{'='*80}")
        print("🔍 提取SST内部特征")
        print(f"{'='*80}\n")

        sst_model.eval()
        with torch.no_grad():
            X_all_t = torch.FloatTensor(X).to(self.device)
            (pred_T, pred_T1), features = sst_model.forward_with_features(
                X_all_t,
                return_attention=True,
                return_encoder_output=True
            )

            # 提取特征
            encoder_output = features['encoder_output'].cpu().numpy()
            pooled_features = features['pooled_features'].cpu().numpy()

            # 计算残差
            residual_T = y_T - pred_T.cpu().numpy()
            residual_T1 = y_T1 - pred_T1.cpu().numpy()

        print(f"  ✓ Encoder输出形状: {encoder_output.shape}")
        print(f"  ✓ 池化特征形状: {pooled_features.shape}")
        print(f"  ✓ 残差计算完成")

        # 组合特征用于时序模型
        relationship_features = pooled_features  # 使用池化特征作为关系特征

        # Step 6: 准备时序数据
        print(f"\n{'='*80}")
        print("🔄 准备时序数据")
        print(f"{'='*80}\n")

        # 使用原始特征 + 关系特征
        target_stock_features = torch.FloatTensor(X)
        relationship_features_t = torch.FloatTensor(relationship_features)
        targets = torch.FloatTensor(y_T1)  # 预测T+1日收益

        # 分割数据
        train_dataset = TemporalDataset(
            target_stock_features=target_stock_features[:train_size],
            relationship_features=relationship_features_t[:train_size],
            targets=targets[:train_size],
            seq_len=seq_len
        )

        val_dataset = TemporalDataset(
            target_stock_features=target_stock_features[train_size:train_size+val_size],
            relationship_features=relationship_features_t[train_size:train_size+val_size],
            targets=targets[train_size:train_size+val_size],
            seq_len=seq_len
        )

        test_dataset = TemporalDataset(
            target_stock_features=target_stock_features[train_size+val_size:],
            relationship_features=relationship_features_t[train_size+val_size:],
            targets=targets[train_size+val_size:],
            seq_len=seq_len
        )

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        print(f"  时序训练集: {len(train_dataset)} 样本")
        print(f"  时序验证集: {len(val_dataset)} 样本")
        print(f"  时序测试集: {len(test_dataset)} 样本")

        # Step 7: 训练时序模型
        input_dim = num_features + relationship_features.shape[1]

        # LSTM模型
        lstm_model = LSTMTemporalPredictor(
            input_dim=input_dim,
            hidden_dim=128,
            num_layers=2,
            output_dim=1,
            use_attention=True
        ).to(self.device)

        print(f"\nLSTM模型参数量: {sum(p.numel() for p in lstm_model.parameters()):,}")

        lstm_history = trainer.train_temporal_model(
            lstm_model,
            train_loader,
            val_loader,
            epochs=temporal_epochs,
            lr=0.001,
            model_name='LSTM'
        )

        # GRU模型
        gru_model = GRUTemporalPredictor(
            input_dim=input_dim,
            hidden_dim=128,
            num_layers=2,
            output_dim=1,
            use_attention=True
        ).to(self.device)

        print(f"GRU模型参数量: {sum(p.numel() for p in gru_model.parameters()):,}")

        gru_history = trainer.train_temporal_model(
            gru_model,
            train_loader,
            val_loader,
            epochs=temporal_epochs,
            lr=0.001,
            model_name='GRU'
        )

        # TCN模型
        tcn_model = TCNTemporalPredictor(
            input_dim=input_dim,
            num_channels=[64, 128, 128, 64],
            kernel_size=3,
            output_dim=1
        ).to(self.device)

        print(f"TCN模型参数量: {sum(p.numel() for p in tcn_model.parameters()):,}")

        tcn_history = trainer.train_temporal_model(
            tcn_model,
            train_loader,
            val_loader,
            epochs=temporal_epochs,
            lr=0.001,
            model_name='TCN'
        )

        # Step 8: 评估时序模型
        lstm_metrics = evaluator.evaluate_temporal_model(
            lstm_model, test_loader, model_name='LSTM'
        )

        gru_metrics = evaluator.evaluate_temporal_model(
            gru_model, test_loader, model_name='GRU'
        )

        tcn_metrics = evaluator.evaluate_temporal_model(
            tcn_model, test_loader, model_name='TCN'
        )

        # Step 9: 对比结果
        comparison_df = evaluator.compare_models()

        # Step 10: 保存结果
        print(f"\n{'='*80}")
        print("💾 保存结果")
        print(f"{'='*80}\n")

        results = {
            'sst_metrics': sst_metrics,
            'lstm_metrics': lstm_metrics,
            'gru_metrics': gru_metrics,
            'tcn_metrics': tcn_metrics,
            'comparison': comparison_df,
            'histories': trainer.history
        }

        with open('training_results.pkl', 'wb') as f:
            pickle.dump(results, f)

        print("  ✓ 训练结果已保存到: training_results.pkl")
        print("  ✓ 最佳模型已保存")
        print(f"\n{'='*80}")
        print("✅ 完整流程执行完成！")
        print(f"{'='*80}\n")

        return results


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='完整的股票预测模型训练流程')

    parser.add_argument('--stocks_json', type=str, default='data/demo.json',
                        help='股票列表JSON文件路径')
    parser.add_argument('--target_market', type=str, default='CN',
                        help='目标市场 (US/CN/HK/JP)')
    parser.add_argument('--target_stock', type=str, default='600519',
                        help='目标股票代码')
    parser.add_argument('--start_date', type=str, default='2020-01-01',
                        help='数据开始日期')
    parser.add_argument('--end_date', type=str, default='2024-12-31',
                        help='数据结束日期')
    parser.add_argument('--fetch_data', action='store_true',
                        help='是否重新抓取数据')
    parser.add_argument('--data_path', type=str, default=None,
                        help='已保存的数据文件路径')
    parser.add_argument('--sst_epochs', type=int, default=50,
                        help='SST训练轮数')
    parser.add_argument('--temporal_epochs', type=int, default=100,
                        help='时序模型训练轮数')
    parser.add_argument('--seq_len', type=int, default=60,
                        help='时序窗口长度')
    parser.add_argument('--device', type=str, default='cpu',
                        help='计算设备 (cpu/cuda)')

    args = parser.parse_args()

    # 创建并运行pipeline
    pipeline = CompletePipeline(
        stocks_json_path=args.stocks_json,
        target_market=args.target_market,
        target_stock=args.target_stock,
        device=args.device
    )

    results = pipeline.run(
        start_date=args.start_date,
        end_date=args.end_date,
        fetch_data=args.fetch_data,
        data_path=args.data_path,
        sst_epochs=args.sst_epochs,
        temporal_epochs=args.temporal_epochs,
        seq_len=args.seq_len
    )

    return results


if __name__ == '__main__':
    results = main()
