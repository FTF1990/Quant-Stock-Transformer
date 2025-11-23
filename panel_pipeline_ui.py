"""
Panel可视化训练Pipeline UI (Colab优化版)
====================================

功能：
- 分步骤可视化展示完整训练流程
- 实时进度显示
- 数据可视化
- 模型训练曲线
- 性能对比图表

使用方法：
    在Colab中直接运行此文件

作者：Quant-Stock-Transformer Team
版本：2.0.0 (Panel版)
"""

import panel as pn
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
import io
from IPython.display import display, clear_output

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

# 初始化Panel
pn.extension('plotly', 'tabulator', sizing_mode="stretch_width")

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

def load_stocks_json(event):
    """加载并显示股票列表"""
    try:
        json_file = file_input.value
        if json_file is None:
            step1_status.object = "❌ 请上传JSON文件"
            return

        # 读取JSON
        stocks_json = json.loads(json_file.decode('utf-8'))
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

        step1_status.object = stats_text

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
        stocks_table.value = df

        # 生成市场分布饼图
        market_counts = {market: len(stocks) for market, stocks in stocks_json.items()}
        fig = px.pie(
            values=list(market_counts.values()),
            names=list(market_counts.keys()),
            title='股票市场分布'
        )
        market_chart.object = fig

    except Exception as e:
        step1_status.object = f"❌ 加载失败: {str(e)}"


# ============================================================================
# 步骤2：数据抓取
# ============================================================================

def fetch_historical_data(event):
    """抓取历史数据"""
    try:
        if state.stocks_json is None:
            step2_status.object = "❌ 请先加载股票JSON"
            return

        step2_status.object = "⏳ 正在抓取数据..."

        fetcher = StockDataFetcher()

        # 抓取数据
        historical_data = fetcher.fetch_historical_data(
            stocks_json=state.stocks_json,
            start_date=start_date_input.value,
            end_date=end_date_input.value,
            interval="1d",
            include_market_index=True,
            batch_size=int(batch_size_input.value),
            delay_between_batches=float(delay_input.value)
        )

        state.historical_data = historical_data
        fetcher.save_data("historical_data.pkl")

        # 生成统计信息
        stats_text = f"""
## ✅ 数据抓取完成

**日期范围**: {start_date_input.value} 至 {end_date_input.value}
**目标市场**: {target_market_input.value}

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
        fetch_table.value = df_stats

        # 检查目标市场的数据
        target_market = target_market_input.value
        if target_market in historical_data:
            market_data = historical_data[target_market]
            stats_text += f"\n**{target_market}市场**: 成功获取{len(market_data)}支股票数据\n"
        else:
            stats_text += f"\n⚠️ **{target_market}市场数据未找到**\n"

        step2_status.object = stats_text

    except Exception as e:
        step2_status.object = f"❌ 数据抓取失败: {str(e)}"


# ============================================================================
# 步骤3：数据预处理
# ============================================================================

def preprocess_data(event):
    """数据预处理"""
    try:
        if state.historical_data is None:
            step3_status.object = "❌ 请先抓取历史数据"
            return

        step3_status.object = "⏳ 正在预处理数据..."

        processor = StockDataProcessor(
            historical_data=state.historical_data,
            target_market=target_market_input.value,
            target_stock=target_stock_input.value
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

        # 保存到状态
        state.processed_data = {
            'X_train': X_train, 'y_T_train': y_T_train, 'y_T1_train': y_T1_train,
            'X_val': X_val, 'y_T_val': y_T_val, 'y_T1_val': y_T1_val,
            'X_test': X_test, 'y_T_test': y_T_test, 'y_T1_test': y_T1_test,
            'dates': dates,
            'processor': processor
        }

        # 生成统计信息
        stats_text = f"""
## ✅ 数据预处理完成

**目标股票**: {target_market_input.value} - {target_stock_input.value}

**数据集划分**:
- 训练集: {len(X_train)} 样本 (70%)
- 验证集: {len(X_val)} 样本 (15%)
- 测试集: {len(X_test)} 样本 (15%)

**特征维度**: {X.shape[1]}

**目标变量**:
- T日收益率: {y_T.shape}
- T+1日收益率: {y_T1.shape}
"""

        step3_status.object = stats_text

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

        preprocess_plot.object = fig
        plt.close(fig)

    except Exception as e:
        step3_status.object = f"❌ 预处理失败: {str(e)}"


# ============================================================================
# 步骤4：SST模型训练
# ============================================================================

def train_sst_model(event):
    """训练SST模型"""
    try:
        if state.processed_data is None:
            step4_status.object = "❌ 请先完成数据预处理"
            return

        step4_status.object = "⏳ 正在训练SST模型..."

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

        # 训练
        history = state.trainer.train_sst(
            sst_model,
            data['X_train'], data['y_T_train'], data['y_T1_train'],
            data['X_val'], data['y_T_val'], data['y_T1_val'],
            epochs=int(sst_epochs_input.value),
            batch_size=int(sst_batch_size_input.value),
            lr=float(sst_lr_input.value),
            verbose=False
        )

        # 生成统计信息
        best_val_loss = min(history['val_loss'])
        final_train_loss = history['train_loss'][-1]

        stats_text = f"""
## ✅ SST模型训练完成

**模型参数**: {sum(p.numel() for p in sst_model.parameters()):,}

**训练配置**:
- Epochs: {sst_epochs_input.value}
- Batch Size: {sst_batch_size_input.value}
- Learning Rate: {sst_lr_input.value}

**训练结果**:
- 最佳验证损失: {best_val_loss:.6f}
- 最终训练损失: {final_train_loss:.6f}
"""

        step4_status.object = stats_text

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

        sst_plot.object = fig
        plt.close(fig)

    except Exception as e:
        step4_status.object = f"❌ SST训练失败: {str(e)}"


# ============================================================================
# 步骤5：特征提取
# ============================================================================

def extract_features(event):
    """提取SST内部特征"""
    try:
        if state.sst_model is None:
            step5_status.object = "❌ 请先训练SST模型"
            return

        step5_status.object = "⏳ 正在提取特征..."

        data = state.processed_data

        # 合并所有数据
        X_all = np.vstack([data['X_train'], data['X_val'], data['X_test']])
        y_T_all = np.vstack([data['y_T_train'], data['y_T_val'], data['y_T_test']])
        y_T1_all = np.vstack([data['y_T1_train'], data['y_T1_val'], data['y_T1_test']])

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

        # 保存特征
        state.processed_data['encoder_output'] = encoder_output
        state.processed_data['pooled_features'] = pooled_features
        state.processed_data['residual_T'] = residual_T
        state.processed_data['residual_T1'] = residual_T1

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

        step5_status.object = stats_text

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

        extract_plot.object = fig
        plt.close(fig)

    except Exception as e:
        step5_status.object = f"❌ 特征提取失败: {str(e)}"


# ============================================================================
# 步骤6：时序模型训练
# ============================================================================

def train_temporal_models(event):
    """训练时序模型"""
    try:
        if 'pooled_features' not in state.processed_data:
            step6_status.object = "❌ 请先提取特征"
            return

        model_type = temporal_model_type_input.value
        step6_status.object = f"⏳ 正在训练{model_type}模型..."

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

        seq_len = int(temporal_seq_len_input.value)

        train_dataset = TemporalDataset(
            target_stock_features=target_stock_features[:train_size],
            relationship_features=relationship_features[:train_size],
            targets=targets[:train_size],
            seq_len=seq_len
        )

        val_dataset = TemporalDataset(
            target_stock_features=target_stock_features[train_size:train_size+val_size],
            relationship_features=relationship_features[train_size:train_size+val_size],
            targets=targets[train_size:train_size+val_size],
            seq_len=seq_len
        )

        train_loader = DataLoader(train_dataset, batch_size=int(temporal_batch_size_input.value), shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=int(temporal_batch_size_input.value), shuffle=False)

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
            step6_status.object = f"❌ 未知的模型类型: {model_type}"
            return

        # 训练
        if state.trainer is None:
            state.trainer = ModelTrainer(device=state.device)

        history = state.trainer.train_temporal_model(
            model,
            train_loader,
            val_loader,
            epochs=int(temporal_epochs_input.value),
            lr=float(temporal_lr_input.value),
            model_name=model_type,
            verbose=False
        )

        # 生成统计信息
        best_val_loss = min(history['val_loss'])
        final_train_loss = history['train_loss'][-1]

        stats_text = f"""
## ✅ {model_type}模型训练完成

**模型参数**: {sum(p.numel() for p in model.parameters()):,}

**训练配置**:
- Epochs: {temporal_epochs_input.value}
- Batch Size: {temporal_batch_size_input.value}
- Learning Rate: {temporal_lr_input.value}
- Sequence Length: {seq_len}

**训练结果**:
- 最佳验证损失: {best_val_loss:.6f}
- 最终训练损失: {final_train_loss:.6f}

**数据集**:
- 训练样本: {len(train_dataset)}
- 验证样本: {len(val_dataset)}
"""

        step6_status.object = stats_text

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

        temporal_plot.object = fig
        plt.close(fig)

    except Exception as e:
        step6_status.object = f"❌ {model_type}训练失败: {str(e)}"


# ============================================================================
# 步骤7：模型评估
# ============================================================================

def evaluate_all_models(event):
    """评估所有模型"""
    try:
        if state.sst_model is None:
            step7_status.object = "❌ 请先训练模型"
            return

        step7_status.object = "⏳ 正在评估模型..."

        # 创建评估器
        if state.evaluator is None:
            state.evaluator = ModelEvaluator(device=state.device)

        data = state.processed_data

        # 评估SST
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

        seq_len = int(eval_seq_len_input.value)

        test_dataset = TemporalDataset(
            target_stock_features=target_stock_features[train_size+val_size:],
            relationship_features=relationship_features[train_size+val_size:],
            targets=targets[train_size+val_size:],
            seq_len=seq_len
        )

        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # 评估时序模型
        if state.lstm_model is not None:
            lstm_metrics = state.evaluator.evaluate_temporal_model(
                state.lstm_model, test_loader, model_name='LSTM'
            )

        if state.gru_model is not None:
            gru_metrics = state.evaluator.evaluate_temporal_model(
                state.gru_model, test_loader, model_name='GRU'
            )

        if state.tcn_model is not None:
            tcn_metrics = state.evaluator.evaluate_temporal_model(
                state.tcn_model, test_loader, model_name='TCN'
            )

        # 生成对比
        comparison_df = state.evaluator.compare_models()

        # 生成统计文本
        stats_text = """
## ✅ 模型评估完成

**已评估的模型**:
"""
        for model_name in state.evaluator.results.keys():
            stats_text += f"- {model_name}\n"

        stats_text += "\n详细指标请查看下方对比表格和图表。"

        step7_status.object = stats_text
        eval_table.value = comparison_df

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

        eval_plot.object = fig
        plt.close(fig)

    except Exception as e:
        step7_status.object = f"❌ 评估失败: {str(e)}"


# ============================================================================
# 创建UI组件
# ============================================================================

# 步骤1组件
file_input = pn.widgets.FileInput(accept='.json', name='上传JSON文件')
load_btn = pn.widgets.Button(name='📥 加载股票列表', button_type='primary')
step1_status = pn.pane.Markdown("等待上传JSON文件...")
stocks_table = pn.widgets.Tabulator(pd.DataFrame(), width=800, height=300)
market_chart = pn.pane.Plotly()

load_btn.on_click(load_stocks_json)

step1 = pn.Column(
    "## 📋 步骤1: 加载股票JSON",
    pn.Row(file_input, load_btn),
    step1_status,
    stocks_table,
    market_chart
)

# 步骤2组件
target_market_input = pn.widgets.Select(name='目标市场', options=['US', 'CN', 'HK', 'JP'], value='CN')
start_date_input = pn.widgets.TextInput(name='开始日期', value='2020-01-01')
end_date_input = pn.widgets.TextInput(name='结束日期', value='2024-12-31')
batch_size_input = pn.widgets.IntSlider(name='批量大小', start=1, end=10, value=5)
delay_input = pn.widgets.FloatSlider(name='批次间延迟(秒)', start=0.5, end=5.0, value=2.0, step=0.5)
fetch_btn = pn.widgets.Button(name='📥 开始抓取数据', button_type='primary')
step2_status = pn.pane.Markdown("等待开始数据抓取...")
fetch_table = pn.widgets.Tabulator(pd.DataFrame(), width=800, height=300)

fetch_btn.on_click(fetch_historical_data)

step2 = pn.Column(
    "## 📊 步骤2: 数据抓取",
    pn.Row(
        pn.Column(target_market_input, start_date_input, end_date_input),
        pn.Column(batch_size_input, delay_input)
    ),
    fetch_btn,
    step2_status,
    fetch_table
)

# 步骤3组件
target_stock_input = pn.widgets.TextInput(name='目标股票代码', value='600519')
preprocess_btn = pn.widgets.Button(name='🔄 开始预处理', button_type='primary')
step3_status = pn.pane.Markdown("等待开始预处理...")
preprocess_plot = pn.pane.Matplotlib()

preprocess_btn.on_click(preprocess_data)

step3 = pn.Column(
    "## 🔄 步骤3: 数据预处理",
    target_stock_input,
    preprocess_btn,
    step3_status,
    preprocess_plot
)

# 步骤4组件
sst_epochs_input = pn.widgets.IntSlider(name='训练轮数', start=10, end=200, value=50, step=10)
sst_batch_size_input = pn.widgets.IntSlider(name='批量大小', start=8, end=128, value=32, step=8)
sst_lr_input = pn.widgets.FloatInput(name='学习率', value=0.001, step=0.0001)
sst_train_btn = pn.widgets.Button(name='🚀 开始训练SST', button_type='primary')
step4_status = pn.pane.Markdown("等待开始训练...")
sst_plot = pn.pane.Matplotlib()

sst_train_btn.on_click(train_sst_model)

step4 = pn.Column(
    "## 🧠 步骤4: SST模型训练",
    pn.Row(
        pn.Column(sst_epochs_input, sst_batch_size_input),
        pn.Column(sst_lr_input)
    ),
    sst_train_btn,
    step4_status,
    sst_plot
)

# 步骤5组件
extract_btn = pn.widgets.Button(name='🔍 开始特征提取', button_type='primary')
step5_status = pn.pane.Markdown("等待开始特征提取...")
extract_plot = pn.pane.Matplotlib()

extract_btn.on_click(extract_features)

step5 = pn.Column(
    "## 🔍 步骤5: 特征提取",
    extract_btn,
    step5_status,
    extract_plot
)

# 步骤6组件
temporal_model_type_input = pn.widgets.Select(name='模型类型', options=['LSTM', 'GRU', 'TCN'], value='LSTM')
temporal_epochs_input = pn.widgets.IntSlider(name='训练轮数', start=10, end=200, value=100, step=10)
temporal_batch_size_input = pn.widgets.IntSlider(name='批量大小', start=8, end=128, value=32, step=8)
temporal_lr_input = pn.widgets.FloatInput(name='学习率', value=0.001, step=0.0001)
temporal_seq_len_input = pn.widgets.IntSlider(name='序列长度', start=20, end=120, value=60, step=10)
temporal_train_btn = pn.widgets.Button(name='🚀 开始训练时序模型', button_type='primary')
step6_status = pn.pane.Markdown("等待开始训练...")
temporal_plot = pn.pane.Matplotlib()

temporal_train_btn.on_click(train_temporal_models)

step6 = pn.Column(
    "## ⏰ 步骤6: 时序模型训练",
    pn.Row(
        pn.Column(temporal_model_type_input, temporal_epochs_input, temporal_batch_size_input),
        pn.Column(temporal_lr_input, temporal_seq_len_input)
    ),
    temporal_train_btn,
    step6_status,
    temporal_plot
)

# 步骤7组件
eval_seq_len_input = pn.widgets.IntSlider(name='序列长度（需与训练时一致）', start=20, end=120, value=60, step=10)
eval_btn = pn.widgets.Button(name='📊 开始评估', button_type='primary')
step7_status = pn.pane.Markdown("等待开始评估...")
eval_table = pn.widgets.Tabulator(pd.DataFrame(), width=800, height=300)
eval_plot = pn.pane.Matplotlib()

eval_btn.on_click(evaluate_all_models)

step7 = pn.Column(
    "## 📈 步骤7: 模型评估",
    eval_seq_len_input,
    eval_btn,
    step7_status,
    eval_table,
    eval_plot
)

# 使用说明
usage_doc = pn.pane.Markdown("""
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

**Quant-Stock-Transformer Team** | Version 2.0.0 (Panel版)
""")

# 创建Tabs
tabs = pn.Tabs(
    ('步骤1: 加载JSON', step1),
    ('步骤2: 数据抓取', step2),
    ('步骤3: 数据预处理', step3),
    ('步骤4: SST训练', step4),
    ('步骤5: 特征提取', step5),
    ('步骤6: 时序模型训练', step6),
    ('步骤7: 模型评估', step7),
    ('使用说明', usage_doc)
)

# 创建主界面
dashboard = pn.template.MaterialTemplate(
    title='🚀 股票预测模型训练Pipeline (Panel版)',
    sidebar=[
        pn.pane.Markdown("""
## 📊 Pipeline状态

完整的端到端训练流程可视化界面

**设备**: {}

**功能**:
- ✅ 选股JSON导入
- ✅ 历史数据获取
- ✅ 数据预处理
- ✅ SST模型训练
- ✅ 特征提取
- ✅ 时序模型训练
- ✅ 模型评估对比

---

**提示**:
1. 按照步骤顺序执行
2. 每步完成后再进行下一步
3. 可以随时切换Tab查看结果
        """.format(state.device))
    ],
    main=[tabs]
)


# ============================================================================
# 启动函数
# ============================================================================

def launch():
    """启动Panel应用"""

    # 检测是否在Colab环境
    try:
        import google.colab
        IN_COLAB = True
    except:
        IN_COLAB = False

    print("="*80)
    print("🚀 股票预测Pipeline可视化 - Panel UI")
    print("="*80)
    print(f"✅ 设备: {state.device}")
    print(f"✅ 环境: {'Colab' if IN_COLAB else '本地'}")
    print("✅ Panel已初始化")
    print("="*80)

    if IN_COLAB:
        print("\n📱 Colab环境检测到!")
        print("📝 提示: 运行返回的对象会在notebook中直接显示UI")
        print("💡 使用方法:")
        print("   app = launch()")
        print("   app  # 在新cell中运行这行来显示UI\n")
        print("="*80)

    # 返回dashboard以便在Jupyter/Colab中显示
    return dashboard


if __name__ == "__main__":
    # 检测环境
    try:
        import google.colab
        IN_COLAB = True
    except:
        IN_COLAB = False

    if IN_COLAB:
        # Colab中直接显示，不启动服务器
        print("🌐 在Colab中运行，请使用:")
        print("   from panel_pipeline_ui import dashboard")
        print("   dashboard")
    else:
        # 本地环境启动服务器
        print("🌐 在本地环境启动服务器...")
        dashboard.show(port=5006)
