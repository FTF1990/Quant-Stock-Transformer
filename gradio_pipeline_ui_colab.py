"""
Gradio Pipeline UI - Colab版本
====================================

专为Google Colab优化的版本

使用方法（在Colab中）：
1. 上传项目文件到Colab
2. 安装依赖
3. 运行此脚本

"""

# 首先检查并安装依赖
import subprocess
import sys

def install_dependencies():
    """安装必要的依赖"""
    packages = [
        'gradio',
        'plotly',
        'torch',
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'akshare',
        'yfinance'
    ]

    print("正在检查并安装依赖...")
    for package in packages:
        try:
            __import__(package)
            print(f"✓ {package} 已安装")
        except ImportError:
            print(f"⚠ 正在安装 {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
            print(f"✓ {package} 安装完成")

# 安装依赖
install_dependencies()

# 导入必要的库
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
import warnings
import os

warnings.filterwarnings('ignore')

# 检查是否在Colab环境
try:
    from google.colab import files
    IN_COLAB = True
    print("✓ 检测到Colab环境")
except ImportError:
    IN_COLAB = False
    print("✓ 本地环境")

# 设置工作目录
if IN_COLAB:
    # 如果在Colab，可能需要设置工作目录
    # 如果你已经clone了repo，取消下面的注释并修改路径
    # os.chdir('/content/Quant-Stock-Transformer')
    pass

# 尝试导入项目模块（如果失败，使用内联定义）
try:
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
    MODULES_AVAILABLE = True
    print("✓ 成功导入pipeline模块")
except ImportError as e:
    MODULES_AVAILABLE = False
    print(f"⚠ 警告: 无法导入pipeline模块: {e}")
    print("⚠ UI将以受限模式运行，某些功能可能不可用")

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
        print(f"✓ 使用设备: {self.device}")

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
        if hasattr(json_file, 'name'):
            file_path = json_file.name
        else:
            file_path = json_file

        with open(file_path, 'r', encoding='utf-8') as f:
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
                    '理由': stock.get('reason', 'N/A')[:50] + '...'  # 截断长文本
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
        return f"❌ 加载失败: {str(e)}\n\n请确保JSON格式正确", None, None


# ============================================================================
# 步骤2：数据抓取（简化版）
# ============================================================================

def fetch_historical_data(
    target_market,
    start_date,
    end_date,
    batch_size,
    delay_between_batches,
    progress=gr.Progress()
):
    """抓取历史数据（简化版，用于演示）"""
    try:
        if state.stocks_json is None:
            return "❌ 请先加载股票JSON", None

        if not MODULES_AVAILABLE:
            return "❌ Pipeline模块未加载，此功能不可用\n\n请确保complete_training_pipeline.py在同一目录", None

        progress(0, desc="初始化数据抓取...")

        fetcher = StockDataFetcher()

        progress(0.2, desc="开始抓取数据...")

        # 简化版：只抓取目标市场的数据
        target_stocks = {target_market: state.stocks_json.get(target_market, [])}

        historical_data = fetcher.fetch_historical_data(
            stocks_json=target_stocks,
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
                if len(df) > 0:
                    rows.append({
                        '市场': market,
                        '代码': symbol,
                        '数据条数': len(df),
                        '开始日期': df.index[0].strftime('%Y-%m-%d'),
                        '结束日期': df.index[-1].strftime('%Y-%m-%d')
                    })

        df_stats = pd.DataFrame(rows) if rows else pd.DataFrame()

        if target_market in historical_data:
            market_data = historical_data[target_market]
            stats_text += f"\n**{target_market}市场**: 成功获取{len(market_data)}支股票数据\n"

        return stats_text, df_stats

    except Exception as e:
        import traceback
        error_msg = f"❌ 数据抓取失败: {str(e)}\n\n详细错误:\n{traceback.format_exc()}"
        return error_msg, None


# ============================================================================
# 简化的UI界面（用于测试）
# ============================================================================

def create_simple_ui():
    """创建简化的UI用于测试"""

    with gr.Blocks(title="股票预测Pipeline - Colab版", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
# 🚀 股票预测模型训练Pipeline (Colab版)

**状态检查**:
- ✅ Gradio已加载
- {} Pipeline模块
- ✅ 设备: {}

---
        """.format(
            "✅" if MODULES_AVAILABLE else "⚠️",
            state.device
        ))

        # ========================================================================
        # 步骤1：加载JSON
        # ========================================================================

        with gr.Tab("📋 步骤1: 加载股票JSON"):
            gr.Markdown("### 上传你的股票选择JSON文件")

            gr.Markdown("""
**提示**:
- 你可以使用 `data/demo.json` 作为示例
- 或在左侧文件浏览器中上传你自己的JSON文件
            """)

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

            if not MODULES_AVAILABLE:
                gr.Markdown("""
⚠️ **警告**: Pipeline模块未加载，数据抓取功能不可用

**解决方法**:
1. 确保 `complete_training_pipeline.py` 在同一目录
2. 重新运行cell
                """)

            with gr.Row():
                with gr.Column():
                    target_market = gr.Dropdown(
                        choices=['US', 'CN', 'HK', 'JP'],
                        value='CN',
                        label="目标市场"
                    )
                    start_date = gr.Textbox(
                        value="2023-01-01",
                        label="开始日期 (YYYY-MM-DD)"
                    )
                    end_date = gr.Textbox(
                        value="2024-01-01",
                        label="结束日期 (YYYY-MM-DD)"
                    )

                with gr.Column():
                    batch_size = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        label="批量大小（Colab建议用小值）"
                    )
                    delay_between_batches = gr.Slider(
                        minimum=1.0,
                        maximum=5.0,
                        value=3.0,
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
        # 使用说明
        # ========================================================================

        with gr.Tab("📖 使用说明"):
            gr.Markdown("""
## 📖 Colab使用指南

### 🚀 快速开始

1. **上传项目文件**
   ```python
   # 方法1: 从GitHub克隆
   !git clone https://github.com/FTF1990/Quant-Stock-Transformer.git
   %cd Quant-Stock-Transformer

   # 方法2: 手动上传文件
   # 使用左侧文件浏览器上传必要文件
   ```

2. **安装依赖**
   ```python
   !pip install -q gradio plotly torch pandas numpy scikit-learn matplotlib seaborn akshare yfinance
   ```

3. **运行UI**
   ```python
   !python gradio_pipeline_ui_colab.py
   ```

### ⚙️ Colab环境配置

**检查GPU**:
```python
import torch
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

**设置运行时**:
- 菜单 -> 运行时 -> 更改运行时类型
- 硬件加速器: GPU (T4)

### 📁 文件结构

确保以下文件在同一目录：
```
├── complete_training_pipeline.py  # 主pipeline
├── gradio_pipeline_ui_colab.py   # 本文件
├── models/                        # 模型模块
│   ├── spatial_feature_extractor.py
│   ├── temporal_predictor.py
│   └── relationship_extractors.py
└── data/
    └── demo.json                  # 示例数据
```

### 🐛 常见问题

**问题1: 无法导入模块**
```
解决: 确保所有.py文件都已上传到Colab
```

**问题2: 数据抓取失败**
```
解决:
- 减小batch_size（建议2-3）
- 增加delay（建议3-5秒）
- 缩短日期范围（测试时用1-3个月）
```

**问题3: 内存不足**
```
解决:
- 使用更少的股票
- 缩短日期范围
- 重启运行时释放内存
```

### 💡 Colab优化建议

1. **使用GPU**: 菜单 -> 运行时 -> 更改运行时类型 -> GPU
2. **挂载Google Drive**: 保存训练结果
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. **定期保存**: Colab会话有时间限制，定期保存模型
4. **小规模测试**: 先用少量数据测试，确认无误后再全量训练

### 📞 获取帮助

- GitHub Issues
- 查看 UI_USAGE.md
- 查看 PIPELINE_FLOW_CONFIRMATION.md

---

**🚧 当前版本为Colab测试版 | 完整功能请使用本地环境 🚧**
            """)

    return demo


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 启动Gradio Pipeline UI (Colab版)")
    print("="*80)
    print(f"Colab环境: {IN_COLAB}")
    print(f"Pipeline模块: {'✅ 已加载' if MODULES_AVAILABLE else '⚠️ 未加载（部分功能不可用）'}")
    print(f"设备: {state.device}")
    print("="*80 + "\n")

    demo = create_simple_ui()

    # Colab专用配置
    demo.launch(
        share=True,              # ✅ 设置为True以获取公开链接
        debug=True,              # 启用调试模式
        show_error=True,         # 显示详细错误
        server_name="0.0.0.0",   # 允许外部访问
        inline=False,            # Colab中设置为False
        quiet=False              # 显示详细日志
    )
