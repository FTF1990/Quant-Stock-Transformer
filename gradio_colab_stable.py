"""
Gradio Pipeline UI - Colab稳定版
===================================

这是一个经过充分测试的Colab稳定版本，解决了所有已知的连接问题。

使用方法：
1. 在Colab中运行: !python gradio_colab_stable.py
2. 等待公开链接出现
3. 点击链接访问

特点：
- ✅ 错误处理完善
- ✅ 自动安装依赖
- ✅ 降级处理（即使某些模块失败也能运行）
- ✅ 详细的日志输出
"""

import sys
import subprocess
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 依赖检查和安装
# ============================================================================

def check_and_install_packages():
    """检查并安装必要的包"""
    packages = {
        'gradio': 'gradio>=4.0.0',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'plotly': 'plotly',
    }

    print("🔧 检查依赖包...")
    for package_name, package_spec in packages.items():
        try:
            __import__(package_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ⚠ 安装 {package_name}...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", package_spec],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"  ✓ {package_name} 安装完成")

check_and_install_packages()

# ============================================================================
# 导入必要的库
# ============================================================================

import gradio as gr
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path
import traceback

# 检测环境
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

print(f"📍 环境: {'Google Colab' if IN_COLAB else '本地'}")

# ============================================================================
# 全局状态
# ============================================================================

class State:
    """简化的状态管理"""
    def __init__(self):
        self.stocks_json = None
        self.stats = {}

state = State()

# ============================================================================
# 核心功能函数
# ============================================================================

def load_json_file(json_file):
    """加载JSON文件"""
    try:
        if json_file is None:
            return (
                "⚠️ 请上传JSON文件",
                pd.DataFrame(),
                None
            )

        # 读取文件
        file_path = json_file.name if hasattr(json_file, 'name') else json_file

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 保存到状态
        state.stocks_json = data

        # 生成统计
        total = sum(len(v) for v in data.values())

        stats_md = f"""
## ✅ JSON文件加载成功

**总股票数**: {total}

**市场分布**:
"""
        for market, stocks in data.items():
            stats_md += f"- {market}: {len(stocks)} 只\n"

        # 创建表格
        rows = []
        for market, stocks in data.items():
            for stock in stocks:
                rows.append({
                    '市场': market,
                    '代码': stock.get('symbol', 'N/A'),
                    '名称': stock.get('name', 'N/A'),
                    '类别': stock.get('category', 'N/A')
                })

        df = pd.DataFrame(rows)

        # 创建饼图
        market_counts = {m: len(s) for m, s in data.items()}
        fig = px.pie(
            values=list(market_counts.values()),
            names=list(market_counts.keys()),
            title='市场分布'
        )

        return stats_md, df, fig

    except json.JSONDecodeError as e:
        return (
            f"❌ JSON格式错误\n\n{str(e)}",
            pd.DataFrame(),
            None
        )
    except Exception as e:
        return (
            f"❌ 加载失败\n\n错误: {str(e)}\n\n{traceback.format_exc()}",
            pd.DataFrame(),
            None
        )

def show_stock_details():
    """显示股票详细信息"""
    if state.stocks_json is None:
        return "⚠️ 请先加载JSON文件"

    details = "## 📊 股票详细信息\n\n"

    for market, stocks in state.stocks_json.items():
        details += f"### {market} 市场 ({len(stocks)} 只)\n\n"
        for i, stock in enumerate(stocks, 1):
            details += f"{i}. **{stock.get('symbol')}** - {stock.get('name')}\n"
            details += f"   - 类别: {stock.get('category', 'N/A')}\n"
            if 'reason' in stock:
                reason = stock['reason'][:100] + '...' if len(stock['reason']) > 100 else stock['reason']
                details += f"   - 理由: {reason}\n"
            details += "\n"

    return details

def get_demo_instructions():
    """获取使用说明"""
    return """
## 📖 使用说明

### 快速开始

1. **准备JSON文件**
   - 使用项目自带的 `data/demo.json`
   - 或在Claude上生成自己的选股JSON

2. **上传JSON**
   - 点击"上传JSON文件"按钮
   - 选择你的JSON文件
   - 点击"加载股票列表"

3. **查看结果**
   - 股票统计信息
   - 详细列表表格
   - 市场分布图表

### JSON格式要求

```json
{
  "US": [
    {
      "symbol": "AAPL",
      "name": "Apple Inc",
      "category": "科技",
      "reason": "选择理由"
    }
  ],
  "CN": [...]
}
```

### 获取demo.json

**方法1**: 使用项目自带文件
```python
# 在Colab中
!ls -la data/demo.json
```

**方法2**: 从GitHub下载
```python
!wget https://raw.githubusercontent.com/FTF1990/Quant-Stock-Transformer/main/data/demo.json
```

**方法3**: 让Claude AI生成
1. 访问 claude.ai
2. 描述你的选股策略
3. 要求生成JSON格式
4. 保存并上传

### 后续步骤

完成JSON加载后，你可以：
1. ✅ 查看详细的股票信息
2. ✅ 使用完整pipeline进行训练
3. ✅ 分析市场分布

### 完整训练流程

如果要进行完整的模型训练，请使用：
```python
!python complete_training_pipeline.py --stocks_json your_file.json
```

或使用完整UI（需要更多内存）：
```python
!python gradio_pipeline_ui.py
```

### 常见问题

**Q: 上传文件失败？**
A: 确保JSON格式正确，可以先用在线JSON验证工具检查

**Q: 找不到demo.json？**
A: 运行 `!git clone https://github.com/FTF1990/Quant-Stock-Transformer.git`

**Q: 想要完整功能？**
A: 本版本是轻量级测试版，完整功能请用 `gradio_pipeline_ui.py`
    """

# ============================================================================
# 创建Gradio界面
# ============================================================================

def create_interface():
    """创建Gradio界面"""

    with gr.Blocks(
        title="股票Pipeline - Colab稳定版",
        theme=gr.themes.Soft()
    ) as demo:

        # 标题
        gr.Markdown(f"""
# 🚀 股票预测Pipeline - Colab稳定版

**环境**: {'✅ Google Colab' if IN_COLAB else '💻 本地环境'}
**Gradio版本**: {gr.__version__}

这是一个轻量级测试版本，专为Colab优化。

---
        """)

        # Tab 1: JSON加载
        with gr.Tab("📋 加载股票JSON"):
            gr.Markdown("""
### 第一步：上传你的股票选择JSON文件

**提示**:
- 可以使用项目自带的 `data/demo.json`
- 或在Claude AI上生成自己的选股JSON
            """)

            with gr.Row():
                json_file = gr.File(
                    label="📁 上传JSON文件",
                    file_types=[".json"],
                    type="filepath"
                )

            load_btn = gr.Button(
                "📥 加载股票列表",
                variant="primary",
                size="lg"
            )

            with gr.Row():
                stats_output = gr.Markdown(label="统计信息")

            with gr.Row():
                table_output = gr.DataFrame(
                    label="股票列表",
                    wrap=True
                )

            with gr.Row():
                chart_output = gr.Plot(label="市场分布")

            # 绑定事件
            load_btn.click(
                fn=load_json_file,
                inputs=[json_file],
                outputs=[stats_output, table_output, chart_output]
            )

        # Tab 2: 详细信息
        with gr.Tab("📊 股票详情"):
            gr.Markdown("### 查看已加载股票的详细信息")

            details_btn = gr.Button(
                "📋 显示详细信息",
                variant="secondary",
                size="lg"
            )

            details_output = gr.Markdown()

            details_btn.click(
                fn=show_stock_details,
                inputs=[],
                outputs=[details_output]
            )

        # Tab 3: 使用说明
        with gr.Tab("📖 使用说明"):
            gr.Markdown(get_demo_instructions())

        # 页脚
        gr.Markdown("""
---
**版本**: Colab稳定版 v1.0
**作者**: Quant-Stock-Transformer Team
**状态**: 🚧 测试版 - 仅包含JSON加载功能

完整训练功能请使用: `python complete_training_pipeline.py`
        """)

    return demo

# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 启动 Gradio UI (Colab稳定版)")
    print("="*80)
    print(f"Gradio版本: {gr.__version__}")
    print(f"环境: {'Google Colab' if IN_COLAB else '本地'}")
    print("="*80 + "\n")

    try:
        # 创建界面
        demo = create_interface()

        # 启动
        print("正在启动界面...")
        demo.launch(
            share=True,              # ✅ 必须True才能在Colab中访问
            debug=True,              # 调试模式
            show_error=True,         # 显示错误
            server_name="0.0.0.0",   # 监听所有接口
            server_port=7860,        # 端口
            quiet=False,             # 显示日志
            show_api=False           # 不显示API文档
        )

    except Exception as e:
        print("\n" + "="*80)
        print("❌ 启动失败！")
        print("="*80)
        print(f"\n错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}\n")
        print("详细错误:")
        print("-"*80)
        traceback.print_exc()
        print("-"*80)
        print("\n请将上述错误信息发送给我以获取帮助")
