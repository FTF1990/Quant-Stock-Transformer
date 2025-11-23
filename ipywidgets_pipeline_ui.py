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
# 步骤4：SST模型训练（重新设计版）
# ============================================================================

def create_step4_tab():
    """创建步骤4的Tab内容 - 完整的SST模型训练界面"""

    import os
    import glob

    # ========== 辅助函数 ==========
    def get_saved_models():
        """获取已保存的模型列表"""
        model_dir = 'saved_models/sst_models'
        if not os.path.exists(model_dir):
            return []
        models = glob.glob(f"{model_dir}/*.pth")
        return sorted(models, key=lambda x: os.path.getmtime(x), reverse=True)

    # ========== 左侧：参数设置 ==========

    # 🏗️ 模型架构参数
    arch_header = widgets.HTML("<h4>🏗️ 模型架构参数</h4>")

    d_model = widgets.IntSlider(
        value=256, min=32, max=1280, step=32,
        description='d_model:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )

    nhead = widgets.IntSlider(
        value=16, min=2, max=80, step=2,
        description='注意力头数:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )

    num_layers = widgets.IntSlider(
        value=6, min=1, max=30, step=1,
        description='Transformer层数:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )

    dropout = widgets.FloatSlider(
        value=0.1, min=0.0, max=0.5, step=0.05,
        description='Dropout:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )

    # 🎯 训练参数
    train_header = widgets.HTML("<h4>🎯 训练参数</h4>")

    epochs = widgets.IntSlider(
        value=50, min=10, max=250, step=10,
        description='训练轮数:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )

    batch_size = widgets.IntSlider(
        value=512, min=16, max=2560, step=16,
        description='批量大小:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )

    learning_rate = widgets.FloatText(
        value=0.00003,
        description='学习率:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )

    weight_decay = widgets.FloatText(
        value=1e-5,
        description='权重衰减:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )

    # ⚙️ 优化器设置
    optimizer_header = widgets.HTML("<h4>⚙️ 优化器设置</h4>")

    grad_clip_norm = widgets.FloatSlider(
        value=1.0, min=0.1, max=5.0, step=0.1,
        description='梯度裁剪:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )

    scheduler_patience = widgets.IntSlider(
        value=8, min=1, max=15, step=1,
        description='LR衰减耐心:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )

    scheduler_factor = widgets.FloatSlider(
        value=0.5, min=0.1, max=0.9, step=0.1,
        description='LR衰减因子:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )

    # 🔀 数据划分
    data_split_header = widgets.HTML("<h4>🔀 数据划分</h4>")

    test_size = widgets.FloatSlider(
        value=0.15, min=0.1, max=0.3, step=0.05,
        description='测试集比例:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )

    val_size = widgets.FloatSlider(
        value=0.15, min=0.1, max=0.3, step=0.05,
        description='验证集比例:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )

    # 💾 模型保存/加载
    save_header = widgets.HTML("<h4>💾 模型管理</h4>")

    model_name = widgets.Text(
        value='sst_model',
        description='模型名称:',
        placeholder='输入模型名称',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )

    saved_models_dropdown = widgets.Dropdown(
        options=get_saved_models(),
        description='已保存模型:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )

    refresh_models_btn = widgets.Button(
        description='🔄 刷新',
        button_style='info',
        layout=widgets.Layout(width='100px')
    )

    load_model_btn = widgets.Button(
        description='📂 加载模型',
        button_style='warning',
        layout=widgets.Layout(width='150px')
    )

    # 🚀 训练按钮
    train_button = widgets.Button(
        description='▶️ 开始训练SST',
        button_style='success',
        layout=widgets.Layout(width='200px', height='50px')
    )

    stop_button = widgets.Button(
        description='⏹️ 停止训练',
        button_style='danger',
        layout=widgets.Layout(width='200px', height='50px')
    )

    # ========== 右侧：训练日志与可视化 ==========

    log_header = widgets.HTML("<h4>📊 训练日志</h4>")

    output_log = widgets.Textarea(
        value='',
        placeholder='训练日志将在此显示...',
        layout=widgets.Layout(width='100%', height='500px'),
        disabled=True
    )

    output_plot = widgets.Output()

    # ========== 事件处理 ==========

    # 训练状态
    training_state = {'stop_requested': False}

    def on_refresh_models(b):
        """刷新模型列表"""
        saved_models_dropdown.options = get_saved_models()

    def on_load_model(b):
        """加载已保存的模型"""
        with output_log:
            try:
                selected_model = saved_models_dropdown.value
                if not selected_model:
                    print("❌ 请先选择一个模型")
                    return

                if not os.path.exists(selected_model):
                    print(f"❌ 模型文件不存在: {selected_model}")
                    return

                # 加载模型
                checkpoint = torch.load(selected_model, map_location=state.device)

                # 重建模型
                if state.processed_data is None:
                    print("❌ 请先完成数据预处理")
                    return

                num_features = state.processed_data['X_train'].shape[1]

                model_config = checkpoint.get('model_config', {})
                sst_model = DualOutputSST(
                    num_boundary_sensors=num_features,
                    num_target_sensors=1,
                    d_model=model_config.get('d_model', 256),
                    nhead=model_config.get('nhead', 16),
                    num_layers=model_config.get('num_layers', 6),
                    dropout=model_config.get('dropout', 0.1),
                    enable_feature_extraction=True
                ).to(state.device)

                sst_model.load_state_dict(checkpoint['model_state_dict'])
                state.sst_model = sst_model

                print(f"✅ 模型加载成功: {selected_model}")
                print(f"  模型配置: d_model={model_config.get('d_model')}, "
                      f"nhead={model_config.get('nhead')}, "
                      f"num_layers={model_config.get('num_layers')}")

            except Exception as e:
                print(f"❌ 加载模型失败: {str(e)}")
                import traceback
                traceback.print_exc()

    def on_train_clicked(b):
        """开始训练"""
        training_state['stop_requested'] = False
        output_log.value = ''

        log_buffer = []

        def log(msg):
            log_buffer.append(msg)
            output_log.value = '\n'.join(log_buffer)

        try:
            if state.processed_data is None:
                log("❌ 请先完成数据预处理")
                return

            log("="*80)
            log("🚀 开始训练SST模型")
            log("="*80)
            log("")

            # 获取数据
            data = state.processed_data
            X_full = np.vstack([data['X_train'], data['X_val'], data['X_test']])
            y_T_full = np.vstack([data['y_T_train'], data['y_T_val'], data['y_T_test']])
            y_T1_full = np.vstack([data['y_T1_train'], data['y_T1_val'], data['y_T1_test']])

            # 重新划分数据集
            total_samples = len(X_full)
            test_samples = int(total_samples * test_size.value)
            val_samples = int(total_samples * val_size.value)
            train_samples = total_samples - test_samples - val_samples

            X_train = X_full[:train_samples]
            y_T_train = y_T_full[:train_samples]
            y_T1_train = y_T1_full[:train_samples]

            X_val = X_full[train_samples:train_samples+val_samples]
            y_T_val = y_T_full[train_samples:train_samples+val_samples]
            y_T1_val = y_T1_full[train_samples:train_samples+val_samples]

            X_test = X_full[train_samples+val_samples:]
            y_T_test = y_T_full[train_samples+val_samples:]
            y_T1_test = y_T1_full[train_samples+val_samples:]

            log(f"📊 数据集划分:")
            log(f"  - 训练集: {len(X_train)} 样本 ({len(X_train)/total_samples*100:.1f}%)")
            log(f"  - 验证集: {len(X_val)} 样本 ({len(X_val)/total_samples*100:.1f}%)")
            log(f"  - 测试集: {len(X_test)} 样本 ({len(X_test)/total_samples*100:.1f}%)")
            log("")

            # 创建模型
            num_features = X_train.shape[1]

            log(f"🏗️ 创建SST模型:")
            log(f"  - 边界传感器数量: {num_features}")
            log(f"  - 目标传感器数量: 1")
            log(f"  - d_model: {d_model.value}")
            log(f"  - nhead: {nhead.value}")
            log(f"  - num_layers: {num_layers.value}")
            log(f"  - dropout: {dropout.value}")
            log("")

            sst_model = DualOutputSST(
                num_boundary_sensors=num_features,
                num_target_sensors=1,
                d_model=d_model.value,
                nhead=nhead.value,
                num_layers=num_layers.value,
                dropout=dropout.value,
                enable_feature_extraction=True
            ).to(state.device)

            total_params = sum(p.numel() for p in sst_model.parameters())
            trainable_params = sum(p.numel() for p in sst_model.parameters() if p.requires_grad)

            log(f"  ✓ 模型参数总量: {total_params:,}")
            log(f"  ✓ 可训练参数: {trainable_params:,}")
            log("")

            # 训练配置
            log(f"🎯 训练配置:")
            log(f"  - Epochs: {epochs.value}")
            log(f"  - Batch Size: {batch_size.value}")
            log(f"  - Learning Rate: {learning_rate.value}")
            log(f"  - Weight Decay: {weight_decay.value}")
            log(f"  - Gradient Clipping: {grad_clip_norm.value}")
            log(f"  - Scheduler Patience: {scheduler_patience.value}")
            log(f"  - Scheduler Factor: {scheduler_factor.value}")
            log("")

            log("⏳ 开始训练...")
            log("")

            # 初始化训练器
            if state.trainer is None:
                state.trainer = ModelTrainer(device=state.device)

            # 训练
            history = state.trainer.train_sst(
                sst_model,
                X_train, y_T_train, y_T1_train,
                X_val, y_T_val, y_T1_val,
                epochs=epochs.value,
                batch_size=batch_size.value,
                lr=learning_rate.value,
                verbose=True
            )

            state.sst_model = sst_model

            # 训练结果
            best_val_loss = min(history['val_loss'])
            final_train_loss = history['train_loss'][-1]

            log("")
            log("="*80)
            log("✅ SST模型训练完成")
            log("="*80)
            log("")
            log(f"📊 训练结果:")
            log(f"  - 最佳验证损失: {best_val_loss:.6f}")
            log(f"  - 最终训练损失: {final_train_loss:.6f}")
            log("")

            # 保存模型
            save_dir = 'saved_models/sst_models'
            os.makedirs(save_dir, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = f"{save_dir}/{model_name.value}_{timestamp}.pth"

            torch.save({
                'model_state_dict': sst_model.state_dict(),
                'model_config': {
                    'd_model': d_model.value,
                    'nhead': nhead.value,
                    'num_layers': num_layers.value,
                    'dropout': dropout.value,
                    'num_boundary_sensors': num_features,
                    'num_target_sensors': 1
                },
                'training_config': {
                    'epochs': epochs.value,
                    'batch_size': batch_size.value,
                    'learning_rate': learning_rate.value,
                    'weight_decay': weight_decay.value
                },
                'history': history,
                'best_val_loss': best_val_loss
            }, save_path)

            log(f"💾 模型已保存: {save_path}")

            # 绘制训练曲线
            with output_plot:
                clear_output()

                fig, axes = plt.subplots(2, 2, figsize=(16, 12))

                # 总体损失
                axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2, color='#2E86AB')
                axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2, color='#A23B72')
                axes[0, 0].set_title('整体损失曲线', fontsize=14, fontweight='bold')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)

                # T日损失
                axes[0, 1].plot(history['train_loss_T'], label='Train Loss (T日)', linewidth=2, color='#06A77D')
                axes[0, 1].plot(history['val_loss_T'], label='Val Loss (T日)', linewidth=2, color='#F18F01', linestyle='--')
                axes[0, 1].set_title('T日预测损失', fontsize=14, fontweight='bold')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Loss')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)

                # T+1日损失
                axes[1, 0].plot(history['train_loss_T1'], label='Train Loss (T+1日)', linewidth=2, color='#C73E1D')
                axes[1, 0].plot(history['val_loss_T1'], label='Val Loss (T+1日)', linewidth=2, color='#6A4C93', linestyle='--')
                axes[1, 0].set_title('T+1日预测损失', fontsize=14, fontweight='bold')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Loss')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)

                # 学习曲线对比
                epochs_range = range(1, len(history['train_loss']) + 1)
                axes[1, 1].plot(epochs_range, history['train_loss'], label='Train', linewidth=2, color='#2E86AB')
                axes[1, 1].plot(epochs_range, history['val_loss'], label='Validation', linewidth=2, color='#A23B72')
                axes[1, 1].axhline(y=best_val_loss, color='r', linestyle=':', label=f'Best Val: {best_val_loss:.6f}')
                axes[1, 1].set_title('学习曲线', fontsize=14, fontweight='bold')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Loss')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)

                plt.tight_layout()
                plt.show()

        except Exception as e:
            log("")
            log(f"❌ 训练失败: {str(e)}")
            import traceback
            log(traceback.format_exc())

    def on_stop_clicked(b):
        """停止训练"""
        training_state['stop_requested'] = True
        output_log.value += "\n\n⏹️ 用户请求停止训练..."

    # 绑定事件
    refresh_models_btn.on_click(on_refresh_models)
    load_model_btn.on_click(on_load_model)
    train_button.on_click(on_train_clicked)
    stop_button.on_click(on_stop_clicked)

    # ========== 组装界面 ==========

    header = widgets.HTML("<h3>🧠 步骤4: SST模型训练（完整版）</h3>")

    # 左侧控制面板
    left_panel = widgets.VBox([
        arch_header,
        d_model,
        nhead,
        num_layers,
        dropout,
        train_header,
        epochs,
        batch_size,
        learning_rate,
        weight_decay,
        optimizer_header,
        grad_clip_norm,
        scheduler_patience,
        scheduler_factor,
        data_split_header,
        test_size,
        val_size,
        save_header,
        model_name,
        widgets.HBox([saved_models_dropdown]),
        widgets.HBox([refresh_models_btn, load_model_btn]),
        widgets.HTML("<br>"),
        widgets.HBox([train_button, stop_button])
    ], layout=widgets.Layout(width='500px', padding='10px'))

    # 右侧日志和可视化
    right_panel = widgets.VBox([
        log_header,
        output_log,
        widgets.HTML("<h4>📈 训练可视化</h4>"),
        output_plot
    ], layout=widgets.Layout(width='calc(100% - 520px)', padding='10px'))

    # 整体布局
    main_layout = widgets.HBox([left_panel, right_panel])

    return widgets.VBox([
        header,
        main_layout
    ])


# ============================================================================
# 步骤5：SST模型推理与预测对比
# ============================================================================

def create_step5_tab():
    """创建步骤5的Tab内容 - SST模型推理与预测对比"""

    import os
    import glob
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    # ========== 辅助函数 ==========
    def get_saved_models():
        """获取已保存的SST模型列表"""
        model_dir = 'saved_models/sst_models'
        if not os.path.exists(model_dir):
            return []
        models = glob.glob(f"{model_dir}/*.pth")
        return sorted(models, key=lambda x: os.path.getmtime(x), reverse=True)

    def compute_r2_safe(y_true, y_pred):
        """安全计算R²分数"""
        try:
            # 对每个输出维度分别计算R²，然后取平均
            if len(y_true.shape) == 1:
                y_true = y_true.reshape(-1, 1)
            if len(y_pred.shape) == 1:
                y_pred = y_pred.reshape(-1, 1)

            r2_scores = []
            for i in range(y_true.shape[1]):
                r2 = r2_score(y_true[:, i], y_pred[:, i])
                r2_scores.append(r2)
            return np.mean(r2_scores), r2_scores
        except:
            return 0.0, [0.0] * y_true.shape[1]

    # ========== 左侧：控制面板 ==========

    # 📂 模型选择
    model_header = widgets.HTML("<h4>📂 模型选择</h4>")

    saved_models_dropdown = widgets.Dropdown(
        options=get_saved_models(),
        description='选择模型:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='450px')
    )

    refresh_models_btn = widgets.Button(
        description='🔄 刷新',
        button_style='info',
        layout=widgets.Layout(width='100px')
    )

    load_model_btn = widgets.Button(
        description='📂 加载模型',
        button_style='success',
        layout=widgets.Layout(width='150px')
    )

    model_info_output = widgets.Output()

    # 📊 数据集选择
    dataset_header = widgets.HTML("<h4>📊 数据集选择</h4>")

    dataset_selector = widgets.Dropdown(
        options=['训练集 (Train)', '验证集 (Val)', '测试集 (Test)', '全部数据 (All)'],
        value='测试集 (Test)',
        description='数据集:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='450px')
    )

    # 📏 时间范围选择
    range_header = widgets.HTML("<h4>📏 时间范围选择</h4>")

    start_idx = widgets.IntText(
        value=0,
        description='起始索引:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='250px')
    )

    end_idx = widgets.IntText(
        value=100,
        description='结束索引:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='250px')
    )

    use_full_range = widgets.Checkbox(
        value=False,
        description='使用全部数据',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='150px')
    )

    # 🚀 推理按钮
    inference_button = widgets.Button(
        description='▶️ 开始推理',
        button_style='success',
        layout=widgets.Layout(width='200px', height='50px')
    )

    # 💾 导出结果
    export_header = widgets.HTML("<h4>💾 导出结果</h4>")

    export_csv_btn = widgets.Button(
        description='📊 导出CSV',
        button_style='primary',
        layout=widgets.Layout(width='150px')
    )

    export_json_btn = widgets.Button(
        description='📋 导出JSON',
        button_style='primary',
        layout=widgets.Layout(width='150px')
    )

    export_png_btn = widgets.Button(
        description='🖼️ 导出图表',
        button_style='primary',
        layout=widgets.Layout(width='150px')
    )

    # ========== 右侧：结果显示 ==========

    status_header = widgets.HTML("<h4>📊 推理结果</h4>")

    output_status = widgets.Textarea(
        value='',
        placeholder='推理结果将在此显示...',
        layout=widgets.Layout(width='100%', height='400px'),
        disabled=True
    )

    output_plot = widgets.Output()

    # ========== 存储推理结果 ==========
    inference_results = {
        'y_true': None,
        'y_pred_T': None,
        'y_pred_T1': None,
        'dataset_name': None,
        'model_name': None,
        'figure': None
    }

    # ========== 事件处理 ==========

    def on_refresh_models(b):
        """刷新模型列表"""
        saved_models_dropdown.options = get_saved_models()

    def on_load_model(b):
        """加载已保存的模型"""
        with model_info_output:
            clear_output()
            try:
                selected_model = saved_models_dropdown.value
                if not selected_model:
                    print("❌ 请先选择一个模型")
                    return

                if not os.path.exists(selected_model):
                    print(f"❌ 模型文件不存在: {selected_model}")
                    return

                # 加载模型
                checkpoint = torch.load(selected_model, map_location=state.device)

                # 检查数据是否已准备
                if state.processed_data is None:
                    print("❌ 请先完成数据预处理（步骤3）")
                    return

                num_features = state.processed_data['X_train'].shape[1]

                model_config = checkpoint.get('model_config', {})
                sst_model = DualOutputSST(
                    num_boundary_sensors=num_features,
                    num_target_sensors=1,
                    d_model=model_config.get('d_model', 256),
                    nhead=model_config.get('nhead', 16),
                    num_layers=model_config.get('num_layers', 6),
                    dropout=model_config.get('dropout', 0.1),
                    enable_feature_extraction=True
                ).to(state.device)

                sst_model.load_state_dict(checkpoint['model_state_dict'])
                sst_model.eval()
                state.sst_model = sst_model

                print(f"✅ 模型加载成功: {os.path.basename(selected_model)}")
                print(f"\n📊 模型配置:")
                print(f"  - d_model: {model_config.get('d_model')}")
                print(f"  - nhead: {model_config.get('nhead')}")
                print(f"  - num_layers: {model_config.get('num_layers')}")
                print(f"  - dropout: {model_config.get('dropout')}")

                if 'training_config' in checkpoint:
                    train_config = checkpoint['training_config']
                    print(f"\n🎯 训练配置:")
                    print(f"  - epochs: {train_config.get('epochs')}")
                    print(f"  - batch_size: {train_config.get('batch_size')}")
                    print(f"  - learning_rate: {train_config.get('learning_rate')}")

                if 'best_val_loss' in checkpoint:
                    print(f"\n📈 最佳验证损失: {checkpoint['best_val_loss']:.6f}")

            except Exception as e:
                print(f"❌ 加载模型失败: {str(e)}")
                import traceback
                traceback.print_exc()

    def on_inference_clicked(b):
        """执行推理并生成对比可视化"""
        log_buffer = []

        def log(msg):
            log_buffer.append(msg)
            output_status.value = '\n'.join(log_buffer)

        try:
            if state.sst_model is None:
                log("❌ 请先加载模型")
                return

            if state.processed_data is None:
                log("❌ 请先完成数据预处理（步骤3）")
                return

            log("=" * 80)
            log("🚀 开始模型推理")
            log("=" * 80)
            log("")

            # 选择数据集
            dataset_name = dataset_selector.value
            data = state.processed_data

            if '训练集' in dataset_name:
                X = data['X_train']
                y_T = data['y_T_train']
                y_T1 = data['y_T1_train']
                dataset_type = 'Train'
            elif '验证集' in dataset_name:
                X = data['X_val']
                y_T = data['y_T_val']
                y_T1 = data['y_T1_val']
                dataset_type = 'Val'
            elif '测试集' in dataset_name:
                X = data['X_test']
                y_T = data['y_T_test']
                y_T1 = data['y_T1_test']
                dataset_type = 'Test'
            else:  # 全部数据
                X = np.vstack([data['X_train'], data['X_val'], data['X_test']])
                y_T = np.vstack([data['y_T_train'], data['y_T_val'], data['y_T1_test']])
                y_T1 = np.vstack([data['y_T1_train'], data['y_T1_val'], data['y_T1_test']])
                dataset_type = 'All'

            log(f"📊 数据集: {dataset_name}")
            log(f"📏 总样本数: {len(X):,}")

            # 时间范围选择
            if use_full_range.value:
                start = 0
                end = len(X)
            else:
                start = max(0, start_idx.value)
                end = min(len(X), end_idx.value)

            X_seg = X[start:end]
            y_T_seg = y_T[start:end]
            y_T1_seg = y_T1[start:end]

            log(f"📏 推理范围: [{start}, {end})")
            log(f"📈 样本数: {len(X_seg):,}")
            log("")

            # 执行推理
            log("⏳ 正在执行推理...")
            state.sst_model.eval()

            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_seg).to(state.device)
                pred_T, pred_T1 = state.sst_model(X_tensor)

                y_pred_T = pred_T.cpu().numpy()
                y_pred_T1 = pred_T1.cpu().numpy()

            log("✅ 推理完成")
            log("")

            # 计算性能指标
            log("📊 计算性能指标...")
            log("")

            # T日预测性能
            mae_T = mean_absolute_error(y_T_seg, y_pred_T)
            mse_T = mean_squared_error(y_T_seg, y_pred_T)
            rmse_T = np.sqrt(mse_T)
            r2_T, r2_T_per = compute_r2_safe(y_T_seg, y_pred_T)

            # T+1日预测性能
            mae_T1 = mean_absolute_error(y_T1_seg, y_pred_T1)
            mse_T1 = mean_squared_error(y_T1_seg, y_pred_T1)
            rmse_T1 = np.sqrt(mse_T1)
            r2_T1, r2_T1_per = compute_r2_safe(y_T1_seg, y_pred_T1)

            # 显示结果
            log("=" * 80)
            log("📈 T日预测性能")
            log("=" * 80)
            log(f"  MAE:  {mae_T:.6f}")
            log(f"  MSE:  {mse_T:.6f}")
            log(f"  RMSE: {rmse_T:.6f}")
            log(f"  R²:   {r2_T:.4f}")
            log("")

            log("=" * 80)
            log("📈 T+1日预测性能")
            log("=" * 80)
            log(f"  MAE:  {mae_T1:.6f}")
            log(f"  MSE:  {mse_T1:.6f}")
            log(f"  RMSE: {rmse_T1:.6f}")
            log(f"  R²:   {r2_T1:.4f}")
            log("")

            # 方向准确率
            direction_correct_T = np.sum(np.sign(y_T_seg) == np.sign(y_pred_T)) / len(y_T_seg) * 100
            direction_correct_T1 = np.sum(np.sign(y_T1_seg) == np.sign(y_pred_T1)) / len(y_T1_seg) * 100

            log("=" * 80)
            log("🎯 方向准确率")
            log("=" * 80)
            log(f"  T日:   {direction_correct_T:.2f}%")
            log(f"  T+1日: {direction_correct_T1:.2f}%")
            log("")

            # 保存结果
            inference_results['y_true_T'] = y_T_seg
            inference_results['y_true_T1'] = y_T1_seg
            inference_results['y_pred_T'] = y_pred_T
            inference_results['y_pred_T1'] = y_pred_T1
            inference_results['dataset_name'] = dataset_type
            inference_results['model_name'] = os.path.basename(saved_models_dropdown.value)
            inference_results['start_idx'] = start
            inference_results['end_idx'] = end
            inference_results['metrics'] = {
                'T': {'mae': mae_T, 'mse': mse_T, 'rmse': rmse_T, 'r2': r2_T, 'direction_acc': direction_correct_T},
                'T1': {'mae': mae_T1, 'mse': mse_T1, 'rmse': rmse_T1, 'r2': r2_T1, 'direction_acc': direction_correct_T1}
            }

            log("✅ 性能指标计算完成")
            log("")

            # 生成可视化
            log("🎨 生成可视化图表...")
            with output_plot:
                clear_output()

                # 创建6个子图：时间序列(T, T+1)、散点图(T, T+1)、残差分布(T, T+1)
                fig = plt.figure(figsize=(20, 12))
                gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

                plot_samples = min(500, len(y_T_seg))

                # ============ T日预测 ============

                # 1. T日时间序列对比
                ax1 = fig.add_subplot(gs[0, 0])
                ax1.plot(y_T_seg[:plot_samples], label='实际值', alpha=0.8, linewidth=2, color='green')
                ax1.plot(y_pred_T[:plot_samples], label='预测值', alpha=0.7, linewidth=1.5, color='blue', linestyle='--')
                ax1.set_title(f'T日预测对比 (R²={r2_T:.4f})', fontsize=14, fontweight='bold')
                ax1.set_xlabel('样本索引', fontsize=11)
                ax1.set_ylabel('收益率', fontsize=11)
                ax1.legend(fontsize=10)
                ax1.grid(True, alpha=0.3)

                # 2. T日散点图
                ax2 = fig.add_subplot(gs[1, 0])
                ax2.scatter(y_T_seg, y_pred_T, alpha=0.5, s=10, color='blue')

                # 添加对角线
                min_val = min(y_T_seg.min(), y_pred_T.min())
                max_val = max(y_T_seg.max(), y_pred_T.max())
                ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测')

                ax2.set_title(f'T日预测散点图 (R²={r2_T:.4f})', fontsize=14, fontweight='bold')
                ax2.set_xlabel('实际值', fontsize=11)
                ax2.set_ylabel('预测值', fontsize=11)
                ax2.legend(fontsize=10)
                ax2.grid(True, alpha=0.3)

                # 3. T日残差分布
                ax3 = fig.add_subplot(gs[2, 0])
                residuals_T = y_T_seg - y_pred_T
                ax3.hist(residuals_T, bins=50, alpha=0.7, color='blue', edgecolor='black')
                ax3.axvline(x=0, color='r', linestyle='--', linewidth=2, label='零残差')
                ax3.set_title(f'T日残差分布 (均值={residuals_T.mean():.6f})', fontsize=14, fontweight='bold')
                ax3.set_xlabel('残差', fontsize=11)
                ax3.set_ylabel('频数', fontsize=11)
                ax3.legend(fontsize=10)
                ax3.grid(True, alpha=0.3)

                # ============ T+1日预测 ============

                # 4. T+1日时间序列对比
                ax4 = fig.add_subplot(gs[0, 1])
                ax4.plot(y_T1_seg[:plot_samples], label='实际值', alpha=0.8, linewidth=2, color='green')
                ax4.plot(y_pred_T1[:plot_samples], label='预测值', alpha=0.7, linewidth=1.5, color='orange', linestyle='--')
                ax4.set_title(f'T+1日预测对比 (R²={r2_T1:.4f})', fontsize=14, fontweight='bold')
                ax4.set_xlabel('样本索引', fontsize=11)
                ax4.set_ylabel('收益率', fontsize=11)
                ax4.legend(fontsize=10)
                ax4.grid(True, alpha=0.3)

                # 5. T+1日散点图
                ax5 = fig.add_subplot(gs[1, 1])
                ax5.scatter(y_T1_seg, y_pred_T1, alpha=0.5, s=10, color='orange')

                # 添加对角线
                min_val = min(y_T1_seg.min(), y_pred_T1.min())
                max_val = max(y_T1_seg.max(), y_pred_T1.max())
                ax5.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测')

                ax5.set_title(f'T+1日预测散点图 (R²={r2_T1:.4f})', fontsize=14, fontweight='bold')
                ax5.set_xlabel('实际值', fontsize=11)
                ax5.set_ylabel('预测值', fontsize=11)
                ax5.legend(fontsize=10)
                ax5.grid(True, alpha=0.3)

                # 6. T+1日残差分布
                ax6 = fig.add_subplot(gs[2, 1])
                residuals_T1 = y_T1_seg - y_pred_T1
                ax6.hist(residuals_T1, bins=50, alpha=0.7, color='orange', edgecolor='black')
                ax6.axvline(x=0, color='r', linestyle='--', linewidth=2, label='零残差')
                ax6.set_title(f'T+1日残差分布 (均值={residuals_T1.mean():.6f})', fontsize=14, fontweight='bold')
                ax6.set_xlabel('残差', fontsize=11)
                ax6.set_ylabel('频数', fontsize=11)
                ax6.legend(fontsize=10)
                ax6.grid(True, alpha=0.3)

                # 总标题
                fig.suptitle(f'SST模型推理结果对比 - {dataset_type}数据集',
                           fontsize=16, fontweight='bold', y=0.995)

                plt.tight_layout()
                plt.show()

                inference_results['figure'] = fig

            log("✅ 可视化完成")

        except Exception as e:
            log("")
            log(f"❌ 推理失败: {str(e)}")
            import traceback
            log(traceback.format_exc())

    def on_export_csv(b):
        """导出CSV结果"""
        try:
            if inference_results['y_true_T'] is None:
                print("❌ 请先执行推理")
                return

            os.makedirs('saved_models/inference_results', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            model_name = inference_results['model_name'].replace('.pth', '')
            dataset_name = inference_results['dataset_name']

            csv_filename = f'saved_models/inference_results/{model_name}_{dataset_name}_{timestamp}.csv'

            # 构建CSV数据
            csv_data = {
                'sample_index': np.arange(inference_results['start_idx'], inference_results['end_idx']),
                'y_T_true': inference_results['y_true_T'].flatten(),
                'y_T_pred': inference_results['y_pred_T'].flatten(),
                'y_T_residual': (inference_results['y_true_T'] - inference_results['y_pred_T']).flatten(),
                'y_T1_true': inference_results['y_true_T1'].flatten(),
                'y_T1_pred': inference_results['y_pred_T1'].flatten(),
                'y_T1_residual': (inference_results['y_true_T1'] - inference_results['y_pred_T1']).flatten()
            }

            df_export = pd.DataFrame(csv_data)
            df_export.to_csv(csv_filename, index=False)

            print(f"✅ CSV已保存: {csv_filename}")

        except Exception as e:
            print(f"❌ 导出CSV失败: {str(e)}")

    def on_export_json(b):
        """导出JSON性能指标"""
        try:
            if inference_results['y_true_T'] is None:
                print("❌ 请先执行推理")
                return

            os.makedirs('saved_models/inference_results', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            model_name = inference_results['model_name'].replace('.pth', '')
            dataset_name = inference_results['dataset_name']

            json_filename = f'saved_models/inference_results/{model_name}_{dataset_name}_metrics_{timestamp}.json'

            # 构建JSON数据
            json_data = {
                'model_name': inference_results['model_name'],
                'dataset': inference_results['dataset_name'],
                'sample_range': {
                    'start': int(inference_results['start_idx']),
                    'end': int(inference_results['end_idx']),
                    'count': int(inference_results['end_idx'] - inference_results['start_idx'])
                },
                'metrics': {
                    'T_day': {
                        'MAE': float(inference_results['metrics']['T']['mae']),
                        'MSE': float(inference_results['metrics']['T']['mse']),
                        'RMSE': float(inference_results['metrics']['T']['rmse']),
                        'R2': float(inference_results['metrics']['T']['r2']),
                        'Direction_Accuracy': float(inference_results['metrics']['T']['direction_acc'])
                    },
                    'T1_day': {
                        'MAE': float(inference_results['metrics']['T1']['mae']),
                        'MSE': float(inference_results['metrics']['T1']['mse']),
                        'RMSE': float(inference_results['metrics']['T1']['rmse']),
                        'R2': float(inference_results['metrics']['T1']['r2']),
                        'Direction_Accuracy': float(inference_results['metrics']['T1']['direction_acc'])
                    }
                },
                'timestamp': timestamp
            }

            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

            print(f"✅ JSON已保存: {json_filename}")

        except Exception as e:
            print(f"❌ 导出JSON失败: {str(e)}")

    def on_export_png(b):
        """导出PNG图表"""
        try:
            if inference_results['figure'] is None:
                print("❌ 请先执行推理")
                return

            os.makedirs('saved_models/inference_results', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            model_name = inference_results['model_name'].replace('.pth', '')
            dataset_name = inference_results['dataset_name']

            png_filename = f'saved_models/inference_results/{model_name}_{dataset_name}_{timestamp}.png'

            inference_results['figure'].savefig(png_filename, dpi=150, bbox_inches='tight')

            print(f"✅ 图表已保存: {png_filename}")

        except Exception as e:
            print(f"❌ 导出图表失败: {str(e)}")

    # 绑定事件
    refresh_models_btn.on_click(on_refresh_models)
    load_model_btn.on_click(on_load_model)
    inference_button.on_click(on_inference_clicked)
    export_csv_btn.on_click(on_export_csv)
    export_json_btn.on_click(on_export_json)
    export_png_btn.on_click(on_export_png)

    # ========== 组装界面 ==========

    header = widgets.HTML("<h3>🔮 步骤5: SST模型推理与预测对比</h3>")

    # 左侧控制面板
    left_panel = widgets.VBox([
        model_header,
        widgets.HBox([saved_models_dropdown]),
        widgets.HBox([refresh_models_btn, load_model_btn]),
        model_info_output,
        dataset_header,
        dataset_selector,
        range_header,
        widgets.HBox([start_idx, end_idx]),
        use_full_range,
        widgets.HTML("<br>"),
        inference_button,
        widgets.HTML("<br>"),
        export_header,
        widgets.HBox([export_csv_btn, export_json_btn, export_png_btn])
    ], layout=widgets.Layout(width='500px', padding='10px'))

    # 右侧结果显示
    right_panel = widgets.VBox([
        status_header,
        output_status,
        widgets.HTML("<h4>📈 可视化对比</h4>"),
        output_plot
    ], layout=widgets.Layout(width='calc(100% - 520px)', padding='10px'))

    # 整体布局
    main_layout = widgets.HBox([left_panel, right_panel])

    return widgets.VBox([
        header,
        main_layout
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
    tab5 = create_step5_tab()
    help_tab = create_help_tab()

    # 创建Tab控件
    tabs = widgets.Tab()
    tabs.children = [tab1, tab2, tab3, tab4, tab5, help_tab]
    tabs.set_title(0, '📋 步骤1: 加载JSON')
    tabs.set_title(1, '📊 步骤2: 数据抓取')
    tabs.set_title(2, '🔄 步骤3: 数据预处理')
    tabs.set_title(3, '🧠 步骤4: SST训练')
    tabs.set_title(4, '🔮 步骤5: 模型推理')
    tabs.set_title(5, '📖 使用说明')

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
