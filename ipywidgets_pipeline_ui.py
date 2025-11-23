"""
IPyWidgets可视化训练Pipeline UI (Colab原生版)
====================================

功能：
- 分步骤可视化展示完整训练流程
- 实时进度显示
- 数据可视化
- 模型训练曲线
- 性能对比图表

特点：
- 使用ipywidgets，Colab原生支持
- 无需服务器，完全客户端渲染
- 零配置，开箱即用

使用方法：
    在Colab中直接运行此文件

作者：Quant-Stock-Transformer Team
版本：3.0.0 (IPyWidgets版)
"""

import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
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

def create_step1_tab():
    """创建步骤1的Tab内容"""

    # 创建组件
    file_upload = widgets.FileUpload(
        accept='.json',
        multiple=False,
        description='上传JSON'
    )

    load_button = widgets.Button(
        description='📥 加载股票列表',
        button_style='primary',
        layout=widgets.Layout(width='200px')
    )

    output_status = widgets.Output()
    output_table = widgets.Output()
    output_chart = widgets.Output()

    def on_load_clicked(b):
        with output_status:
            clear_output()
            try:
                if not file_upload.value:
                    print("❌ 请先上传JSON文件")
                    return

                # 读取上传的文件
                uploaded_file = list(file_upload.value.values())[0]
                content = uploaded_file['content']
                raw_json = json.loads(content.decode('utf-8'))

                # 转换新格式JSON为标准格式
                stocks_json = {}
                rows = []

                # 处理target_stock（单个股票对象）
                if 'target_stock' in raw_json:
                    target = raw_json['target_stock']
                    if isinstance(target, dict) and 'symbol' in target:
                        # 检测市场（根据股票代码）
                        symbol = target['symbol']
                        market = 'CN' if symbol.startswith('6') or symbol.startswith('0') or symbol.startswith('3') else 'US'

                        if market not in stocks_json:
                            stocks_json[market] = []

                        stocks_json[market].append(target)

                        rows.append({
                            '类型': '目标股票',
                            '代码': target['symbol'],
                            '名称': target['name'],
                            '行业': target.get('industry', 'N/A'),
                            '说明': target.get('reason', '主营: ' + ', '.join(target.get('main_business', [])))
                        })

                # 处理related_stocks（嵌套结构）
                if 'related_stocks' in raw_json:
                    related = raw_json['related_stocks']
                    for category, stocks_list in related.items():
                        if isinstance(stocks_list, list):
                            for stock in stocks_list:
                                if isinstance(stock, dict) and 'symbol' in stock:
                                    # 检测市场
                                    symbol = stock['symbol']
                                    market = 'CN' if symbol.startswith('6') or symbol.startswith('0') or symbol.startswith('3') else 'US'

                                    if market not in stocks_json:
                                        stocks_json[market] = []

                                    stocks_json[market].append(stock)

                                    rows.append({
                                        '类型': category,
                                        '代码': stock['symbol'],
                                        '名称': stock['name'],
                                        '行业': stock.get('category', 'N/A'),
                                        '说明': stock.get('reason', 'N/A')
                                    })

                # 如果是旧格式（市场-股票数组），直接使用
                if not stocks_json:
                    stocks_json = raw_json
                    for market, stocks in stocks_json.items():
                        if isinstance(stocks, list):
                            for stock in stocks:
                                rows.append({
                                    '类型': market,
                                    '代码': stock.get('symbol', 'N/A'),
                                    '名称': stock.get('name', 'N/A'),
                                    '行业': stock.get('category', 'N/A'),
                                    '说明': stock.get('reason', 'N/A')
                                })

                state.stocks_json = stocks_json

                # 生成统计信息
                total_stocks = sum(len(v) for v in stocks_json.values())

                print("## ✅ 股票列表加载成功\n")
                print(f"**总股票数**: {total_stocks}只\n")
                print("**市场分布**:")
                for market, stocks in stocks_json.items():
                    print(f"- **{market}市场**: {len(stocks)}只")

                # 显示详细表格
                if rows:
                    df = pd.DataFrame(rows)
                    with output_table:
                        clear_output()
                        display(df)

                # 生成市场分布饼图
                market_counts = {market: len(stocks) for market, stocks in stocks_json.items()}
                fig = px.pie(
                    values=list(market_counts.values()),
                    names=list(market_counts.keys()),
                    title='股票市场分布'
                )

                with output_chart:
                    clear_output()
                    fig.show()

            except Exception as e:
                print(f"❌ 加载失败: {str(e)}")
                import traceback
                traceback.print_exc()

    load_button.on_click(on_load_clicked)

    # 组装界面
    header = widgets.HTML("<h3>📋 步骤1: 加载股票JSON</h3>")

    return widgets.VBox([
        header,
        widgets.HBox([file_upload, load_button]),
        output_status,
        output_table,
        output_chart
    ])


# ============================================================================
# 步骤2：数据抓取与加载
# ============================================================================

def create_step2_tab():
    """创建步骤2的Tab内容"""

    # 创建组件 - 数据抓取
    target_market = widgets.Dropdown(
        options=['US', 'CN', 'HK', 'JP'],
        value='CN',
        description='目标市场:'
    )

    start_date = widgets.Text(
        value='2020-01-01',
        description='开始日期:',
        placeholder='YYYY-MM-DD'
    )

    end_date = widgets.Text(
        value='2024-12-31',
        description='结束日期:',
        placeholder='YYYY-MM-DD'
    )

    # 添加时间粒度选择
    interval = widgets.Dropdown(
        options=[
            ('按天 (1d)', '1d'),
            ('按小时 (1h)', '1h'),
            ('按周 (1wk)', '1wk'),
            ('按月 (1mo)', '1mo')
        ],
        value='1d',
        description='时间粒度:',
        style={'description_width': 'initial'}
    )

    batch_size = widgets.IntSlider(
        value=5,
        min=1,
        max=10,
        description='批量大小:'
    )

    delay = widgets.FloatSlider(
        value=2.0,
        min=0.5,
        max=5.0,
        step=0.5,
        description='批次延迟:'
    )

    fetch_button = widgets.Button(
        description='📥 开始抓取数据',
        button_style='primary',
        layout=widgets.Layout(width='200px')
    )

    # 组件 - 加载已保存数据
    load_csv_dropdown = widgets.Dropdown(
        options=['选择CSV文件...'],
        description='选择文件:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='400px')
    )

    refresh_csv_button = widgets.Button(
        description='🔄 刷新列表',
        button_style='info',
        layout=widgets.Layout(width='120px')
    )

    load_csv_button = widgets.Button(
        description='📂 加载选中数据',
        button_style='success',
        layout=widgets.Layout(width='200px')
    )

    output_status = widgets.Output()
    output_table = widgets.Output()

    def save_data_to_csv(historical_data, target_market_name):
        """保存数据到CSV文件"""
        try:
            import os
            os.makedirs('data', exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            saved_files = []

            for market, stocks_data in historical_data.items():
                for symbol, df in stocks_data.items():
                    if len(df) > 0:
                        # 文件名格式: market_symbol_startdate_enddate_timestamp.csv
                        start_str = df.index[0].strftime('%Y%m%d')
                        end_str = df.index[-1].strftime('%Y%m%d')
                        filename = f"data/{market}_{symbol}_{start_str}_{end_str}_{timestamp}.csv"

                        # 保存CSV
                        df.to_csv(filename)
                        saved_files.append(filename)

            return saved_files
        except Exception as e:
            raise Exception(f"CSV保存失败: {str(e)}")

    def refresh_csv_list():
        """刷新CSV文件列表"""
        try:
            import os
            import glob

            csv_files = glob.glob('data/*.csv')
            if csv_files:
                # 按修改时间倒序排序
                csv_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                options = ['选择CSV文件...'] + csv_files
            else:
                options = ['选择CSV文件... (data文件夹为空)']

            load_csv_dropdown.options = options
        except Exception as e:
            print(f"刷新列表失败: {str(e)}")

    def on_fetch_clicked(b):
        with output_status:
            clear_output()
            try:
                if state.stocks_json is None:
                    print("❌ 请先加载股票JSON")
                    return

                print("⏳ 正在抓取数据...")
                print(f"📊 时间粒度: {interval.value}")

                fetcher = StockDataFetcher()

                historical_data = fetcher.fetch_historical_data(
                    stocks_json=state.stocks_json,
                    start_date=start_date.value,
                    end_date=end_date.value,
                    interval=interval.value,
                    include_market_index=True,
                    batch_size=int(batch_size.value),
                    delay_between_batches=float(delay.value)
                )

                state.historical_data = historical_data

                # 保存为pickle
                fetcher.save_data("historical_data.pkl")

                # 保存为CSV
                print("\n💾 正在保存CSV文件...")
                saved_files = save_data_to_csv(historical_data, target_market.value)
                print(f"✅ 已保存 {len(saved_files)} 个CSV文件到data文件夹")

                print("\n## ✅ 数据抓取完成\n")
                print(f"**日期范围**: {start_date.value} 至 {end_date.value}")
                print(f"**时间粒度**: {interval.value}")
                print(f"**目标市场**: {target_market.value}\n")

                # 生成统计表格
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

                with output_table:
                    clear_output()
                    display(df_stats)

                if target_market.value in historical_data:
                    market_data = historical_data[target_market.value]
                    print(f"\n**{target_market.value}市场**: 成功获取{len(market_data)}支股票数据")

                # 刷新CSV列表
                refresh_csv_list()

            except Exception as e:
                print(f"❌ 数据抓取失败: {str(e)}")
                import traceback
                traceback.print_exc()

    def on_refresh_csv_clicked(b):
        with output_status:
            clear_output()
            print("🔄 正在刷新CSV文件列表...")
            refresh_csv_list()
            print(f"✅ 找到 {len(load_csv_dropdown.options) - 1} 个CSV文件")

    def on_load_csv_clicked(b):
        with output_status:
            clear_output()
            try:
                selected_file = load_csv_dropdown.value
                if selected_file.startswith('选择CSV文件'):
                    print("❌ 请先选择一个CSV文件")
                    return

                print(f"⏳ 正在加载: {selected_file}")

                # 从文件名解析信息
                import os
                basename = os.path.basename(selected_file)
                parts = basename.replace('.csv', '').split('_')

                if len(parts) >= 2:
                    market = parts[0]
                    symbol = parts[1]

                    # 读取CSV
                    df = pd.read_csv(selected_file, index_col=0, parse_dates=True)

                    # 初始化historical_data结构
                    if state.historical_data is None:
                        state.historical_data = {}

                    if market not in state.historical_data:
                        state.historical_data[market] = {}

                    state.historical_data[market][symbol] = df

                    print(f"✅ 成功加载: {market} - {symbol}")
                    print(f"📊 数据条数: {len(df)}")
                    print(f"📅 日期范围: {df.index[0]} 至 {df.index[-1]}")

                    # 显示数据预览
                    with output_table:
                        clear_output()
                        print(f"\n数据预览 ({symbol}):")
                        display(df.tail(10))

                else:
                    print("❌ CSV文件名格式不正确")

            except Exception as e:
                print(f"❌ 加载失败: {str(e)}")
                import traceback
                traceback.print_exc()

    fetch_button.on_click(on_fetch_clicked)
    refresh_csv_button.on_click(on_refresh_csv_clicked)
    load_csv_button.on_click(on_load_csv_clicked)

    # 初始化时刷新CSV列表
    refresh_csv_list()

    # 组装界面
    header = widgets.HTML("<h3>📊 步骤2: 数据抓取与加载</h3>")

    fetch_section = widgets.VBox([
        widgets.HTML("<h4>📥 方式1: 在线抓取数据</h4>"),
        widgets.HBox([target_market, start_date, end_date]),
        widgets.HBox([interval, batch_size, delay]),
        fetch_button
    ])

    load_section = widgets.VBox([
        widgets.HTML("<h4>📂 方式2: 加载已保存数据</h4>"),
        widgets.HBox([load_csv_dropdown, refresh_csv_button]),
        load_csv_button
    ])

    return widgets.VBox([
        header,
        fetch_section,
        widgets.HTML("<hr>"),
        load_section,
        output_status,
        output_table
    ])


# ============================================================================
# 步骤3：数据预处理
# ============================================================================

def create_step3_tab():
    """创建步骤3的Tab内容"""

    # 创建组件
    target_stock = widgets.Text(
        value='600519',
        description='目标股票:',
        placeholder='输入股票代码'
    )

    target_market = widgets.Dropdown(
        options=['US', 'CN', 'HK', 'JP'],
        value='CN',
        description='目标市场:'
    )

    preprocess_button = widgets.Button(
        description='🔄 开始预处理',
        button_style='primary',
        layout=widgets.Layout(width='200px')
    )

    output_status = widgets.Output()
    output_plot = widgets.Output()

    def on_preprocess_clicked(b):
        with output_status:
            clear_output()
            try:
                if state.historical_data is None:
                    print("❌ 请先抓取历史数据")
                    return

                print("⏳ 正在预处理数据...")

                processor = StockDataProcessor(
                    historical_data=state.historical_data,
                    target_market=target_market.value,
                    target_stock=target_stock.value
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

                print("## ✅ 数据预处理完成\n")
                print(f"**目标股票**: {target_market.value} - {target_stock.value}\n")
                print("**数据集划分**:")
                print(f"- 训练集: {len(X_train)} 样本 (70%)")
                print(f"- 验证集: {len(X_val)} 样本 (15%)")
                print(f"- 测试集: {len(X_test)} 样本 (15%)\n")
                print(f"**特征维度**: {X.shape[1]}")

                # 绘制收益率分布
                with output_plot:
                    clear_output()
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
                    plt.show()

            except Exception as e:
                print(f"❌ 预处理失败: {str(e)}")

    preprocess_button.on_click(on_preprocess_clicked)

    # 组装界面
    header = widgets.HTML("<h3>🔄 步骤3: 数据预处理</h3>")

    return widgets.VBox([
        header,
        widgets.HBox([target_market, target_stock]),
        preprocess_button,
        output_status,
        output_plot
    ])


# ============================================================================
# 步骤4：SST模型训练
# ============================================================================

def create_step4_tab():
    """创建步骤4的Tab内容"""

    # 创建组件
    sst_epochs = widgets.IntSlider(
        value=50,
        min=10,
        max=200,
        step=10,
        description='训练轮数:'
    )

    sst_batch_size = widgets.IntSlider(
        value=32,
        min=8,
        max=128,
        step=8,
        description='批量大小:'
    )

    sst_lr = widgets.FloatText(
        value=0.001,
        description='学习率:',
        step=0.0001
    )

    train_button = widgets.Button(
        description='🚀 开始训练SST',
        button_style='primary',
        layout=widgets.Layout(width='200px')
    )

    output_status = widgets.Output()
    output_plot = widgets.Output()

    def on_train_clicked(b):
        with output_status:
            clear_output()
            try:
                if state.processed_data is None:
                    print("❌ 请先完成数据预处理")
                    return

                print("⏳ 正在训练SST模型...")

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

                if state.trainer is None:
                    state.trainer = ModelTrainer(device=state.device)

                # 训练
                history = state.trainer.train_sst(
                    sst_model,
                    data['X_train'], data['y_T_train'], data['y_T1_train'],
                    data['X_val'], data['y_T_val'], data['y_T1_val'],
                    epochs=int(sst_epochs.value),
                    batch_size=int(sst_batch_size.value),
                    lr=float(sst_lr.value),
                    verbose=True
                )

                best_val_loss = min(history['val_loss'])
                final_train_loss = history['train_loss'][-1]

                print("\n## ✅ SST模型训练完成\n")
                print(f"**模型参数**: {sum(p.numel() for p in sst_model.parameters()):,}\n")
                print("**训练配置**:")
                print(f"- Epochs: {sst_epochs.value}")
                print(f"- Batch Size: {sst_batch_size.value}")
                print(f"- Learning Rate: {sst_lr.value}\n")
                print("**训练结果**:")
                print(f"- 最佳验证损失: {best_val_loss:.6f}")
                print(f"- 最终训练损失: {final_train_loss:.6f}")

                # 绘制训练曲线
                with output_plot:
                    clear_output()
                    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
                    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
                    axes[0].set_title('SST训练损失曲线', fontsize=14, fontweight='bold')
                    axes[0].set_xlabel('Epoch')
                    axes[0].set_ylabel('Loss')
                    axes[0].legend()
                    axes[0].grid(True, alpha=0.3)

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
                    plt.show()

            except Exception as e:
                print(f"❌ SST训练失败: {str(e)}")
                import traceback
                traceback.print_exc()

    train_button.on_click(on_train_clicked)

    # 组装界面
    header = widgets.HTML("<h3>🧠 步骤4: SST模型训练</h3>")

    return widgets.VBox([
        header,
        widgets.HBox([sst_epochs, sst_batch_size, sst_lr]),
        train_button,
        output_status,
        output_plot
    ])


# ============================================================================
# 创建说明Tab
# ============================================================================

def create_help_tab():
    """创建使用说明Tab"""

    help_html = """
    <div style="padding: 20px;">
        <h2>📖 使用流程</h2>

        <h3>1️⃣ 加载股票JSON</h3>
        <ul>
            <li>上传你的股票选择JSON文件（如data/demo.json）</li>
            <li>查看股票列表和市场分布</li>
        </ul>

        <h3>2️⃣ 数据抓取</h3>
        <ul>
            <li>选择目标市场（US/CN/HK/JP）</li>
            <li>设置日期范围</li>
            <li>配置批量抓取参数（避免API限流）</li>
            <li>点击"开始抓取数据"</li>
        </ul>

        <h3>3️⃣ 数据预处理</h3>
        <ul>
            <li>输入目标股票代码</li>
            <li>自动计算收益率</li>
            <li>数据集划分（70% train, 15% val, 15% test）</li>
        </ul>

        <h3>4️⃣ SST模型训练</h3>
        <ul>
            <li>配置训练参数（epochs, batch size, learning rate）</li>
            <li>训练双输出SST模型</li>
            <li>查看训练曲线</li>
        </ul>

        <hr>

        <h2>💡 提示</h2>
        <ul>
            <li><strong>数据抓取</strong>: 建议使用默认的批量参数，避免API限流</li>
            <li><strong>SST训练</strong>: 50个epoch通常足够，可以先用小epoch数测试</li>
            <li><strong>设备</strong>: 当前使用 <code>{}</code></li>
        </ul>

        <hr>

        <h2>🔧 技术细节</h2>
        <h3>SST模型</h3>
        <ul>
            <li>双输出架构（T日 + T+1日）</li>
            <li>Transformer编码器（8 heads, 3 layers）</li>
            <li>隐藏维度：128</li>
        </ul>

        <hr>

        <p><strong>Quant-Stock-Transformer Team</strong> | Version 3.0.0 (IPyWidgets版)</p>
    </div>
    """.format(state.device)

    return widgets.HTML(help_html)


# ============================================================================
# 主函数
# ============================================================================

def create_ui():
    """创建完整的UI"""

    # 创建所有Tab
    tab1 = create_step1_tab()
    tab2 = create_step2_tab()
    tab3 = create_step3_tab()
    tab4 = create_step4_tab()
    help_tab = create_help_tab()

    # 创建Tab控件
    tabs = widgets.Tab()
    tabs.children = [tab1, tab2, tab3, tab4, help_tab]
    tabs.set_title(0, '📋 步骤1: 加载JSON')
    tabs.set_title(1, '📊 步骤2: 数据抓取')
    tabs.set_title(2, '🔄 步骤3: 数据预处理')
    tabs.set_title(3, '🧠 步骤4: SST训练')
    tabs.set_title(4, '📖 使用说明')

    # 创建标题
    title = widgets.HTML("""
        <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 20px;">
            <h1 style="color: white; margin: 0; text-align: center;">
                🚀 股票预测模型训练Pipeline (IPyWidgets版)
            </h1>
            <p style="color: white; margin: 10px 0 0 0; text-align: center;">
                完整的端到端训练流程可视化界面 | Colab原生支持 | 无需服务器
            </p>
        </div>
    """)

    # 状态信息
    status_html = widgets.HTML(f"""
        <div style="background: #f0f0f0;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;">
            <strong>📊 系统状态</strong><br>
            设备: <code>{state.device}</code><br>
            版本: <code>3.0.0 (IPyWidgets)</code>
        </div>
    """)

    # 组装完整界面
    ui = widgets.VBox([
        title,
        status_html,
        tabs
    ])

    return ui


def launch():
    """启动应用"""

    # 检测环境
    try:
        import google.colab
        IN_COLAB = True
    except:
        IN_COLAB = False

    print("="*80)
    print("🚀 股票预测Pipeline可视化 - IPyWidgets UI")
    print("="*80)
    print(f"✅ 设备: {state.device}")
    print(f"✅ 环境: {'Colab' if IN_COLAB else 'Jupyter'}")
    print("✅ IPyWidgets已初始化")
    print("✅ 无需服务器，完全客户端渲染")
    print("="*80)

    # 创建并返回UI
    ui = create_ui()
    return ui


if __name__ == "__main__":
    ui = launch()
    display(ui)
