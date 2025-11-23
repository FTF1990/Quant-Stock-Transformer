"""
Gradio可视化训练Pipeline UI
====================================

功能：
- 分步骤可视化展示完整训练流程
- 实时进度显示
- 数据可视化
- 模型训练曲线
- 性能对比图表

使用方法：
    python gradio_pipeline_ui.py

作者：Quant-Stock-Transformer Team
版本：1.0.0
"""

import gradio as gr
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Tuple, Optional

# 导入pipeline模块
from complete_training_pipeline import (
    StockDataFetcher,
    StockDataProcessor,
    DualOutputSST,
    ModelTrainer,
    ModelEvaluator,
    LSTMTemporalPredictor,
    GRUTemporalPredictor,
    TCNTemporalPredictor,
    TemporalDataset
)
from torch.utils.data import DataLoader

# 设置绘图样式
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 全局状态存储
class PipelineState:
    """存储pipeline执行状态"""
    def __init__(self):
        self.stocks_json = None
        self.historical_data = None
        self.processed_data = None
        self.sst_model = None
        self.lstm_model = None
        self.gru_model = None
        self.tcn_model = None
        self.trainer = None
        self.evaluator = None
        self.results = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

state = PipelineState()


# ============================================================================
# 步骤1：加载股票JSON
# ============================================================================

def load_stocks_json(json_file):
    """加载并显示股票列表"""
    try:
        if json_file is None:
            return "❌ 请上传JSON文件", None, None

        # 读取JSON
        with open(json_file.name, 'r', encoding='utf-8') as f:
            stocks_json = json.load(f)

        state.stocks_json = stocks_json

        # 生成统计信息
        total_stocks = sum(len(v) for v in stocks_json.values())

        stats_text = f"""
## ✅ 股票列表加载成功

**总股票数**: {total_stocks}只

**市场分布**:
"""
        for market, stocks in stocks_json.items():
            stats_text += f"- **{market}市场**: {len(stocks)}只\n"

        # 生成详细表格
        rows = []
        for market, stocks in stocks_json.items():
            for stock in stocks:
                rows.append({
                    '市场': market,
                    '代码': stock['symbol'],
                    '名称': stock['name'],
                    '类别': stock.get('category', 'N/A'),
                    '理由': stock.get('reason', 'N/A')
                })

        df = pd.DataFrame(rows)

        # 生成市场分布饼图
        market_counts = {market: len(stocks) for market, stocks in stocks_json.items()}
        fig = px.pie(
            values=list(market_counts.values()),
            names=list(market_counts.keys()),
            title='股票市场分布'
        )

        return stats_text, df, fig

    except Exception as e:
        return f"❌ 加载失败: {str(e)}", None, None


# ============================================================================
# 步骤2：数据抓取
# ============================================================================

def fetch_historical_data(
    target_market,
    start_date,
    end_date,
    batch_size,
    delay_between_batches,
    progress=gr.Progress()
):
    """抓取历史数据"""
    try:
        if state.stocks_json is None:
            return "❌ 请先加载股票JSON", None

        progress(0, desc="初始化数据抓取...")

        fetcher = StockDataFetcher()

        # 抓取数据
        progress(0.2, desc="开始抓取数据...")
        historical_data = fetcher.fetch_historical_data(
            stocks_json=state.stocks_json,
            start_date=start_date,
            end_date=end_date,
            interval="1d",
            include_market_index=True,
            batch_size=int(batch_size),
            delay_between_batches=float(delay_between_batches)
        )

        state.historical_data = historical_data

        progress(0.8, desc="保存数据...")
        fetcher.save_data("historical_data.pkl")

        progress(1.0, desc="完成！")

        # 生成统计信息
        stats_text = f"""
## ✅ 数据抓取完成

**日期范围**: {start_date} 至 {end_date}
**目标市场**: {target_market}

**数据统计**:
"""

        rows = []
        for market, stocks_data in historical_data.items():
            for symbol, df in stocks_data.items():
                rows.append({
                    '市场': market,
                    '代码': symbol,
                    '数据条数': len(df),
                    '开始日期': df.index[0].strftime('%Y-%m-%d') if len(df) > 0 else 'N/A',
                    '结束日期': df.index[-1].strftime('%Y-%m-%d') if len(df) > 0 else 'N/A'
                })

        df_stats = pd.DataFrame(rows)

        # 检查目标市场的数据
        if target_market in historical_data:
            market_data = historical_data[target_market]
            stats_text += f"\n**{target_market}市场**: 成功获取{len(market_data)}支股票数据\n"
        else:
            stats_text += f"\n⚠️ **{target_market}市场数据未找到**\n"

        return stats_text, df_stats

    except Exception as e:
        return f"❌ 数据抓取失败: {str(e)}", None


# ============================================================================
# 步骤3：数据预处理
# ============================================================================

def preprocess_data(target_market, target_stock, progress=gr.Progress()):
    """数据预处理"""
    try:
        if state.historical_data is None:
            return "❌ 请先抓取历史数据", None, None

        progress(0, desc="开始数据预处理...")

        processor = StockDataProcessor(
            historical_data=state.historical_data,
            target_market=target_market,
            target_stock=target_stock
        )

        progress(0.3, desc="计算特征...")
        X, y_T, y_T1, dates = processor.prepare_training_data()

        progress(0.6, desc="数据集划分...")
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

        # 保存到状态
        state.processed_data = {
            'X_train': X_train, 'y_T_train': y_T_train, 'y_T1_train': y_T1_train,
            'X_val': X_val, 'y_T_val': y_T_val, 'y_T1_val': y_T1_val,
            'X_test': X_test, 'y_T_test': y_T_test, 'y_T1_test': y_T1_test,
            'dates': dates,
            'processor': processor
        }

        progress(1.0, desc="完成！")

        # 生成统计信息
        stats_text = f"""
## ✅ 数据预处理完成

**目标股票**: {target_market} - {target_stock}

**数据集划分**:
- 训练集: {len(X_train)} 样本 (70%)
- 验证集: {len(X_val)} 样本 (15%)
- 测试集: {len(X_test)} 样本 (15%)

**特征维度**: {X.shape[1]}

**目标变量**:
- T日收益率: {y_T.shape}
- T+1日收益率: {y_T1.shape}
"""

        # 绘制收益率分布
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].hist(y_T_train, bins=50, alpha=0.7, edgecolor='black')
        axes[0].set_title('T日收益率分布（训练集）')
        axes[0].set_xlabel('收益率')
        axes[0].set_ylabel('频数')
        axes[0].grid(True, alpha=0.3)

        axes[1].hist(y_T1_train, bins=50, alpha=0.7, edgecolor='black')
        axes[1].set_title('T+1日收益率分布（训练集）')
        axes[1].set_xlabel('收益率')
        axes[1].set_ylabel('频数')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        return stats_text, fig, None

    except Exception as e:
        return f"❌ 预处理失败: {str(e)}", None, None


# ============================================================================
# 步骤4：SST模型训练
# ============================================================================

def train_sst_model(epochs, batch_size, learning_rate, progress=gr.Progress()):
    """训练SST模型"""
    try:
        if state.processed_data is None:
            return "❌ 请先完成数据预处理", None

        progress(0, desc="初始化SST模型...")

        # 获取数据
        data = state.processed_data
        num_features = data['X_train'].shape[1]

        # 创建模型
        sst_model = DualOutputSST(
            num_boundary_sensors=num_features,
            num_target_sensors=1,
            d_model=128,
            nhead=8,
            num_layers=3,
            dropout=0.1,
            enable_feature_extraction=True
        ).to(state.device)

        state.sst_model = sst_model

        # 创建训练器
        if state.trainer is None:
            state.trainer = ModelTrainer(device=state.device)

        progress(0.1, desc="开始训练...")

        # 训练
        history = state.trainer.train_sst(
            sst_model,
            data['X_train'], data['y_T_train'], data['y_T1_train'],
            data['X_val'], data['y_T_val'], data['y_T1_val'],
            epochs=int(epochs),
            batch_size=int(batch_size),
            lr=float(learning_rate),
            verbose=False
        )

        progress(1.0, desc="训练完成！")

        # 生成统计信息
        best_val_loss = min(history['val_loss'])
        final_train_loss = history['train_loss'][-1]

        stats_text = f"""
## ✅ SST模型训练完成

**模型参数**: {sum(p.numel() for p in sst_model.parameters()):,}

**训练配置**:
- Epochs: {epochs}
- Batch Size: {batch_size}
- Learning Rate: {learning_rate}

**训练结果**:
- 最佳验证损失: {best_val_loss:.6f}
- 最终训练损失: {final_train_loss:.6f}
"""

        # 绘制训练曲线
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 总损失
        axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
        axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_title('SST训练损失曲线', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # T和T+1分别的损失
        axes[1].plot(history['train_loss_T'], label='Train Loss (T日)', linewidth=2)
        axes[1].plot(history['train_loss_T1'], label='Train Loss (T+1日)', linewidth=2)
        axes[1].plot(history['val_loss_T'], label='Val Loss (T日)', linewidth=2, linestyle='--')
        axes[1].plot(history['val_loss_T1'], label='Val Loss (T+1日)', linewidth=2, linestyle='--')
        axes[1].set_title('SST分项损失曲线', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        return stats_text, fig

    except Exception as e:
        return f"❌ SST训练失败: {str(e)}", None


# ============================================================================
# 步骤5：特征提取
# ============================================================================

def extract_features(progress=gr.Progress()):
    """提取SST内部特征"""
    try:
        if state.sst_model is None:
            return "❌ 请先训练SST模型", None

        progress(0, desc="开始特征提取...")

        data = state.processed_data

        # 合并所有数据
        X_all = np.vstack([data['X_train'], data['X_val'], data['X_test']])
        y_T_all = np.vstack([data['y_T_train'], data['y_T_val'], data['y_T_test']])
        y_T1_all = np.vstack([data['y_T1_train'], data['y_T1_val'], data['y_T1_test']])

        progress(0.3, desc="提取特征...")

        # 提取特征
        state.sst_model.eval()
        with torch.no_grad():
            X_all_t = torch.FloatTensor(X_all).to(state.device)
            (pred_T, pred_T1), features = state.sst_model.forward_with_features(
                X_all_t,
                return_attention=True,
                return_encoder_output=True
            )

            encoder_output = features['encoder_output'].cpu().numpy()
            pooled_features = features['pooled_features'].cpu().numpy()

            # 计算残差
            residual_T = y_T_all - pred_T.cpu().numpy()
            residual_T1 = y_T1_all - pred_T1.cpu().numpy()

        progress(0.7, desc="准备时序数据...")

        # 保存特征
        state.processed_data['encoder_output'] = encoder_output
        state.processed_data['pooled_features'] = pooled_features
        state.processed_data['residual_T'] = residual_T
        state.processed_data['residual_T1'] = residual_T1

        progress(1.0, desc="完成！")

        # 生成统计信息
        stats_text = f"""
## ✅ 特征提取完成

**提取的特征**:
- Encoder输出: {encoder_output.shape}
- 池化特征: {pooled_features.shape}
- T日残差: {residual_T.shape}
- T+1日残差: {residual_T1.shape}

**特征统计**:
- 池化特征均值: {np.mean(pooled_features):.6f}
- 池化特征标准差: {np.std(pooled_features):.6f}
"""

        # 绘制特征可视化
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 池化特征分布
        axes[0, 0].hist(pooled_features.flatten(), bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('池化特征分布', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('特征值')
        axes[0, 0].set_ylabel('频数')
        axes[0, 0].grid(True, alpha=0.3)

        # 残差分布
        axes[0, 1].hist(residual_T1.flatten(), bins=50, alpha=0.7, edgecolor='black', color='orange')
        axes[0, 1].set_title('T+1日预测残差分布', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('残差')
        axes[0, 1].set_ylabel('频数')
        axes[0, 1].grid(True, alpha=0.3)

        # 池化特征热图（前10维）
        feature_sample = pooled_features[:100, :10]
        im = axes[1, 0].imshow(feature_sample.T, aspect='auto', cmap='viridis')
        axes[1, 0].set_title('池化特征热图（样本×特征）', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('样本')
        axes[1, 0].set_ylabel('特征维度')
        plt.colorbar(im, ax=axes[1, 0])

        # 残差时间序列
        axes[1, 1].plot(residual_T1[:500], alpha=0.7, linewidth=1)
        axes[1, 1].set_title('T+1日残差时间序列（前500样本）', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('样本索引')
        axes[1, 1].set_ylabel('残差')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=1)

        plt.tight_layout()

        return stats_text, fig

    except Exception as e:
        return f"❌ 特征提取失败: {str(e)}", None


# ============================================================================
# 步骤6：时序模型训练
# ============================================================================

def train_temporal_models(
    model_type,
    epochs,
    batch_size,
    learning_rate,
    seq_len,
    progress=gr.Progress()
):
    """训练时序模型"""
    try:
        if 'pooled_features' not in state.processed_data:
            return "❌ 请先提取特征", None

        progress(0, desc=f"初始化{model_type}模型...")

        data = state.processed_data

        # 准备数据
        train_size = len(data['X_train'])
        val_size = len(data['X_val'])

        # 合并所有数据
        X_all = np.vstack([data['X_train'], data['X_val'], data['X_test']])
        y_T1_all = np.vstack([data['y_T1_train'], data['y_T1_val'], data['y_T1_test']])
        pooled_features = data['pooled_features']

        # 创建时序数据集
        target_stock_features = torch.FloatTensor(X_all)
        relationship_features = torch.FloatTensor(pooled_features)
        targets = torch.FloatTensor(y_T1_all)

        train_dataset = TemporalDataset(
            target_stock_features=target_stock_features[:train_size],
            relationship_features=relationship_features[:train_size],
            targets=targets[:train_size],
            seq_len=int(seq_len)
        )

        val_dataset = TemporalDataset(
            target_stock_features=target_stock_features[train_size:train_size+val_size],
            relationship_features=relationship_features[train_size:train_size+val_size],
            targets=targets[train_size:train_size+val_size],
            seq_len=int(seq_len)
        )

        train_loader = DataLoader(train_dataset, batch_size=int(batch_size), shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=int(batch_size), shuffle=False)

        progress(0.1, desc="创建模型...")

        # 创建模型
        input_dim = X_all.shape[1] + pooled_features.shape[1]

        if model_type == 'LSTM':
            model = LSTMTemporalPredictor(
                input_dim=input_dim,
                hidden_dim=128,
                num_layers=2,
                output_dim=1,
                use_attention=True
            ).to(state.device)
            state.lstm_model = model
        elif model_type == 'GRU':
            model = GRUTemporalPredictor(
                input_dim=input_dim,
                hidden_dim=128,
                num_layers=2,
                output_dim=1,
                use_attention=True
            ).to(state.device)
            state.gru_model = model
        elif model_type == 'TCN':
            model = TCNTemporalPredictor(
                input_dim=input_dim,
                num_channels=[64, 128, 128, 64],
                kernel_size=3,
                output_dim=1
            ).to(state.device)
            state.tcn_model = model
        else:
            return f"❌ 未知的模型类型: {model_type}", None

        progress(0.2, desc="开始训练...")

        # 训练
        if state.trainer is None:
            state.trainer = ModelTrainer(device=state.device)

        history = state.trainer.train_temporal_model(
            model,
            train_loader,
            val_loader,
            epochs=int(epochs),
            lr=float(learning_rate),
            model_name=model_type,
            verbose=False
        )

        progress(1.0, desc="训练完成！")

        # 生成统计信息
        best_val_loss = min(history['val_loss'])
        final_train_loss = history['train_loss'][-1]

        stats_text = f"""
## ✅ {model_type}模型训练完成

**模型参数**: {sum(p.numel() for p in model.parameters()):,}

**训练配置**:
- Epochs: {epochs}
- Batch Size: {batch_size}
- Learning Rate: {learning_rate}
- Sequence Length: {seq_len}

**训练结果**:
- 最佳验证损失: {best_val_loss:.6f}
- 最终训练损失: {final_train_loss:.6f}

**数据集**:
- 训练样本: {len(train_dataset)}
- 验证样本: {len(val_dataset)}
"""

        # 绘制训练曲线
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(history['train_loss'], label='Train Loss', linewidth=2)
        ax.plot(history['val_loss'], label='Val Loss', linewidth=2)
        ax.set_title(f'{model_type}训练损失曲线', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        return stats_text, fig

    except Exception as e:
        return f"❌ {model_type}训练失败: {str(e)}", None


# ============================================================================
# 步骤7：模型评估
# ============================================================================

def evaluate_all_models(seq_len, progress=gr.Progress()):
    """评估所有模型"""
    try:
        if state.sst_model is None:
            return "❌ 请先训练模型", None, None

        progress(0, desc="初始化评估...")

        # 创建评估器
        if state.evaluator is None:
            state.evaluator = ModelEvaluator(device=state.device)

        data = state.processed_data

        # 评估SST
        progress(0.2, desc="评估SST模型...")
        sst_metrics = state.evaluator.evaluate_sst(
            state.sst_model,
            data['X_test'],
            data['y_T_test'],
            data['y_T1_test'],
            model_name='SST'
        )

        # 准备时序模型测试数据
        train_size = len(data['X_train'])
        val_size = len(data['X_val'])

        X_all = np.vstack([data['X_train'], data['X_val'], data['X_test']])
        y_T1_all = np.vstack([data['y_T1_train'], data['y_T1_val'], data['y_T1_test']])
        pooled_features = data['pooled_features']

        target_stock_features = torch.FloatTensor(X_all)
        relationship_features = torch.FloatTensor(pooled_features)
        targets = torch.FloatTensor(y_T1_all)

        test_dataset = TemporalDataset(
            target_stock_features=target_stock_features[train_size+val_size:],
            relationship_features=relationship_features[train_size+val_size:],
            targets=targets[train_size+val_size:],
            seq_len=int(seq_len)
        )

        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # 评估时序模型
        if state.lstm_model is not None:
            progress(0.4, desc="评估LSTM模型...")
            lstm_metrics = state.evaluator.evaluate_temporal_model(
                state.lstm_model, test_loader, model_name='LSTM'
            )

        if state.gru_model is not None:
            progress(0.6, desc="评估GRU模型...")
            gru_metrics = state.evaluator.evaluate_temporal_model(
                state.gru_model, test_loader, model_name='GRU'
            )

        if state.tcn_model is not None:
            progress(0.8, desc="评估TCN模型...")
            tcn_metrics = state.evaluator.evaluate_temporal_model(
                state.tcn_model, test_loader, model_name='TCN'
            )

        progress(0.9, desc="生成对比...")

        # 生成对比
        comparison_df = state.evaluator.compare_models()

        progress(1.0, desc="完成！")

        # 生成统计文本
        stats_text = """
## ✅ 模型评估完成

**已评估的模型**:
"""
        for model_name in state.evaluator.results.keys():
            stats_text += f"- {model_name}\n"

        stats_text += "\n详细指标请查看下方对比表格和图表。"

        # 生成对比图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        models = list(state.evaluator.results.keys())

        # 时序模型的指标（排除SST）
        temporal_models = [m for m in models if m != 'SST']

        if len(temporal_models) > 0:
            mse_values = [state.evaluator.results[m]['metrics']['MSE'] for m in temporal_models]
            mae_values = [state.evaluator.results[m]['metrics']['MAE'] for m in temporal_models]
            dir_acc_values = [state.evaluator.results[m]['metrics']['Direction_Acc'] for m in temporal_models]
            sharpe_values = [state.evaluator.results[m]['metrics']['Sharpe_Ratio'] for m in temporal_models]

            # MSE对比
            axes[0, 0].bar(temporal_models, mse_values, alpha=0.7, edgecolor='black')
            axes[0, 0].set_title('MSE对比', fontsize=12, fontweight='bold')
            axes[0, 0].set_ylabel('MSE')
            axes[0, 0].grid(True, alpha=0.3, axis='y')

            # MAE对比
            axes[0, 1].bar(temporal_models, mae_values, alpha=0.7, edgecolor='black', color='orange')
            axes[0, 1].set_title('MAE对比', fontsize=12, fontweight='bold')
            axes[0, 1].set_ylabel('MAE')
            axes[0, 1].grid(True, alpha=0.3, axis='y')

            # Direction Accuracy对比
            axes[1, 0].bar(temporal_models, dir_acc_values, alpha=0.7, edgecolor='black', color='green')
            axes[1, 0].set_title('方向准确率对比', fontsize=12, fontweight='bold')
            axes[1, 0].set_ylabel('准确率')
            axes[1, 0].axhline(y=0.5, color='r', linestyle='--', linewidth=1, label='随机猜测')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3, axis='y')

            # Sharpe Ratio对比
            axes[1, 1].bar(temporal_models, sharpe_values, alpha=0.7, edgecolor='black', color='purple')
            axes[1, 1].set_title('Sharpe比率对比', fontsize=12, fontweight='bold')
            axes[1, 1].set_ylabel('Sharpe Ratio')
            axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        return stats_text, comparison_df, fig

    except Exception as e:
        return f"❌ 评估失败: {str(e)}", None, None


# ============================================================================
# Gradio界面
# ============================================================================

def create_ui():
    """创建Gradio UI"""

    with gr.Blocks(title="股票预测Pipeline可视化", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
# 🚀 股票预测模型训练Pipeline

完整的端到端训练流程可视化界面

**功能**:
- ✅ 选股JSON导入
- ✅ 历史数据获取
- ✅ 数据预处理
- ✅ SST模型训练
- ✅ 特征提取
- ✅ 时序模型训练（LSTM/GRU/TCN）
- ✅ 模型评估对比

---
        """)

        # ========================================================================
        # 步骤1：加载JSON
        # ========================================================================

        with gr.Tab("📋 步骤1: 加载股票JSON"):
            gr.Markdown("### 上传你的股票选择JSON文件")

            with gr.Row():
                json_file = gr.File(
                    label="上传JSON文件",
                    file_types=[".json"],
                    type="filepath"
                )

            load_btn = gr.Button("📥 加载股票列表", variant="primary", size="lg")

            with gr.Row():
                json_stats = gr.Markdown()

            with gr.Row():
                stocks_table = gr.DataFrame(label="股票详细列表")

            with gr.Row():
                market_chart = gr.Plot(label="市场分布")

            load_btn.click(
                fn=load_stocks_json,
                inputs=[json_file],
                outputs=[json_stats, stocks_table, market_chart]
            )

        # ========================================================================
        # 步骤2：数据抓取
        # ========================================================================

        with gr.Tab("📊 步骤2: 数据抓取"):
            gr.Markdown("### 抓取历史股票数据")

            with gr.Row():
                with gr.Column():
                    target_market = gr.Dropdown(
                        choices=['US', 'CN', 'HK', 'JP'],
                        value='CN',
                        label="目标市场"
                    )
                    start_date = gr.Textbox(
                        value="2020-01-01",
                        label="开始日期 (YYYY-MM-DD)"
                    )
                    end_date = gr.Textbox(
                        value="2024-12-31",
                        label="结束日期 (YYYY-MM-DD)"
                    )

                with gr.Column():
                    batch_size = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="批量大小"
                    )
                    delay_between_batches = gr.Slider(
                        minimum=0.5,
                        maximum=5.0,
                        value=2.0,
                        step=0.5,
                        label="批次间延迟（秒）"
                    )

            fetch_btn = gr.Button("📥 开始抓取数据", variant="primary", size="lg")

            with gr.Row():
                fetch_stats = gr.Markdown()

            with gr.Row():
                fetch_table = gr.DataFrame(label="数据抓取统计")

            fetch_btn.click(
                fn=fetch_historical_data,
                inputs=[target_market, start_date, end_date, batch_size, delay_between_batches],
                outputs=[fetch_stats, fetch_table]
            )

        # ========================================================================
        # 步骤3：数据预处理
        # ========================================================================

        with gr.Tab("🔄 步骤3: 数据预处理"):
            gr.Markdown("### 数据预处理和特征工程")

            with gr.Row():
                target_stock = gr.Textbox(
                    value="600519",
                    label="目标股票代码"
                )

            preprocess_btn = gr.Button("🔄 开始预处理", variant="primary", size="lg")

            with gr.Row():
                preprocess_stats = gr.Markdown()

            with gr.Row():
                preprocess_plot = gr.Plot(label="收益率分布")

            preprocess_btn.click(
                fn=preprocess_data,
                inputs=[target_market, target_stock],
                outputs=[preprocess_stats, preprocess_plot, gr.State()]
            )

        # ========================================================================
        # 步骤4：SST训练
        # ========================================================================

        with gr.Tab("🧠 步骤4: SST模型训练"):
            gr.Markdown("### 训练双输出SST模型")

            with gr.Row():
                with gr.Column():
                    sst_epochs = gr.Slider(
                        minimum=10,
                        maximum=200,
                        value=50,
                        step=10,
                        label="训练轮数"
                    )
                    sst_batch_size = gr.Slider(
                        minimum=8,
                        maximum=128,
                        value=32,
                        step=8,
                        label="批量大小"
                    )

                with gr.Column():
                    sst_lr = gr.Slider(
                        minimum=0.0001,
                        maximum=0.01,
                        value=0.001,
                        step=0.0001,
                        label="学习率"
                    )

            sst_train_btn = gr.Button("🚀 开始训练SST", variant="primary", size="lg")

            with gr.Row():
                sst_stats = gr.Markdown()

            with gr.Row():
                sst_plot = gr.Plot(label="训练曲线")

            sst_train_btn.click(
                fn=train_sst_model,
                inputs=[sst_epochs, sst_batch_size, sst_lr],
                outputs=[sst_stats, sst_plot]
            )

        # ========================================================================
        # 步骤5：特征提取
        # ========================================================================

        with gr.Tab("🔍 步骤5: 特征提取"):
            gr.Markdown("### 提取SST内部特征")

            extract_btn = gr.Button("🔍 开始特征提取", variant="primary", size="lg")

            with gr.Row():
                extract_stats = gr.Markdown()

            with gr.Row():
                extract_plot = gr.Plot(label="特征可视化")

            extract_btn.click(
                fn=extract_features,
                inputs=[],
                outputs=[extract_stats, extract_plot]
            )

        # ========================================================================
        # 步骤6：时序模型训练
        # ========================================================================

        with gr.Tab("⏰ 步骤6: 时序模型训练"):
            gr.Markdown("### 训练LSTM/GRU/TCN时序模型")

            with gr.Row():
                with gr.Column():
                    temporal_model_type = gr.Dropdown(
                        choices=['LSTM', 'GRU', 'TCN'],
                        value='LSTM',
                        label="模型类型"
                    )
                    temporal_epochs = gr.Slider(
                        minimum=10,
                        maximum=200,
                        value=100,
                        step=10,
                        label="训练轮数"
                    )
                    temporal_batch_size = gr.Slider(
                        minimum=8,
                        maximum=128,
                        value=32,
                        step=8,
                        label="批量大小"
                    )

                with gr.Column():
                    temporal_lr = gr.Slider(
                        minimum=0.0001,
                        maximum=0.01,
                        value=0.001,
                        step=0.0001,
                        label="学习率"
                    )
                    temporal_seq_len = gr.Slider(
                        minimum=20,
                        maximum=120,
                        value=60,
                        step=10,
                        label="序列长度"
                    )

            temporal_train_btn = gr.Button("🚀 开始训练时序模型", variant="primary", size="lg")

            with gr.Row():
                temporal_stats = gr.Markdown()

            with gr.Row():
                temporal_plot = gr.Plot(label="训练曲线")

            temporal_train_btn.click(
                fn=train_temporal_models,
                inputs=[temporal_model_type, temporal_epochs, temporal_batch_size,
                        temporal_lr, temporal_seq_len],
                outputs=[temporal_stats, temporal_plot]
            )

        # ========================================================================
        # 步骤7：模型评估
        # ========================================================================

        with gr.Tab("📈 步骤7: 模型评估"):
            gr.Markdown("### 评估所有模型并对比性能")

            with gr.Row():
                eval_seq_len = gr.Slider(
                    minimum=20,
                    maximum=120,
                    value=60,
                    step=10,
                    label="序列长度（需与训练时一致）"
                )

            eval_btn = gr.Button("📊 开始评估", variant="primary", size="lg")

            with gr.Row():
                eval_stats = gr.Markdown()

            with gr.Row():
                eval_table = gr.DataFrame(label="模型性能对比表")

            with gr.Row():
                eval_plot = gr.Plot(label="性能对比图")

            eval_btn.click(
                fn=evaluate_all_models,
                inputs=[eval_seq_len],
                outputs=[eval_stats, eval_table, eval_plot]
            )

        # ========================================================================
        # 使用说明
        # ========================================================================

        with gr.Tab("📖 使用说明"):
            gr.Markdown("""
## 📖 使用流程

### 1️⃣ 加载股票JSON
- 上传你的股票选择JSON文件（如`data/demo.json`）
- 查看股票列表和市场分布

### 2️⃣ 数据抓取
- 选择目标市场（US/CN/HK/JP）
- 设置日期范围
- 配置批量抓取参数（避免API限流）
- 点击"开始抓取数据"

### 3️⃣ 数据预处理
- 输入目标股票代码
- 自动计算收益率
- 数据集划分（70% train, 15% val, 15% test）

### 4️⃣ SST模型训练
- 配置训练参数（epochs, batch size, learning rate）
- 训练双输出SST模型
- 查看训练曲线

### 5️⃣ 特征提取
- 从SST模型提取内部特征
- 包括：Encoder输出、Attention权重、池化特征、残差

### 6️⃣ 时序模型训练
- 选择模型类型（LSTM/GRU/TCN）
- 配置训练参数
- 可以训练多个模型进行对比

### 7️⃣ 模型评估
- 评估所有训练的模型
- 查看性能对比表和图表
- 指标：MSE, MAE, Direction Accuracy, Sharpe Ratio

---

## 💡 提示

- **数据抓取**: 建议使用默认的批量参数，避免API限流
- **SST训练**: 50个epoch通常足够，可以先用小epoch数测试
- **时序模型**: LSTM和GRU性能接近，TCN训练更快
- **序列长度**: 60天是常用值，可根据数据特点调整

---

## 🔧 技术细节

**SST模型**:
- 双输出架构（T日 + T+1日）
- Transformer编码器（8 heads, 3 layers）
- 隐藏维度：128

**时序模型**:
- LSTM: 128 hidden, 2 layers, with Attention
- GRU: 128 hidden, 2 layers, with Attention
- TCN: [64, 128, 128, 64] channels

**评估指标**:
- MSE: 均方误差
- MAE: 平均绝对误差
- Direction Accuracy: 方向准确率
- Sharpe Ratio: 风险调整后收益

---

## 📞 帮助

遇到问题？
1. 检查JSON文件格式是否正确
2. 确保网络连接正常（数据抓取需要）
3. 查看终端错误信息
4. 参考README.md文档

---

**Quant-Stock-Transformer Team** | Version 1.0.0
            """)

    return demo


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,              # Set to True for public URL (useful for Colab)
        show_error=True,
        debug=False
    )
