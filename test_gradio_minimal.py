"""
Gradio UI 最小测试版本 - 用于诊断Colab问题
"""

import sys
import subprocess

print("="*80)
print("🔍 Gradio Colab 诊断工具")
print("="*80)

# Step 1: 检查并安装gradio
print("\n[1/5] 检查Gradio安装...")
try:
    import gradio as gr
    print(f"✓ Gradio 已安装 (版本: {gr.__version__})")
except ImportError:
    print("⚠ Gradio 未安装，正在安装...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gradio"])
    import gradio as gr
    print(f"✓ Gradio 安装完成 (版本: {gr.__version__})")

# Step 2: 检查其他必要的包
print("\n[2/5] 检查依赖包...")
required_packages = ['pandas', 'numpy', 'matplotlib']
for package in required_packages:
    try:
        __import__(package)
        print(f"✓ {package} 已安装")
    except ImportError:
        print(f"⚠ 正在安装 {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
        print(f"✓ {package} 安装完成")

# Step 3: 检查环境
print("\n[3/5] 检查环境...")
try:
    from google.colab import drive
    print("✓ Google Colab 环境检测成功")
    IN_COLAB = True
except ImportError:
    print("✓ 本地环境")
    IN_COLAB = False

# Step 4: 创建最简单的测试界面
print("\n[4/5] 创建测试界面...")

import gradio as gr
import pandas as pd
import json

def test_function(text):
    """测试函数"""
    return f"✅ 收到输入: {text}\n\n系统正常工作！"

def load_demo_json():
    """尝试加载demo.json"""
    try:
        with open('data/demo.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        total = sum(len(v) for v in data.values())
        return f"✅ 成功加载demo.json\n\n总计 {total} 只股票"
    except FileNotFoundError:
        return "⚠ demo.json 未找到\n\n请确保文件在 data/demo.json"
    except Exception as e:
        return f"❌ 加载失败: {str(e)}"

# 创建简化界面
with gr.Blocks(title="Colab连接测试", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
# 🧪 Gradio Colab 连接测试

如果你能看到这个界面，说明Gradio已经成功启动！

## 测试步骤：
1. 在下方输入框输入任意文字
2. 点击"测试"按钮
3. 如果看到返回结果，说明一切正常

---
    """)

    with gr.Tab("基础测试"):
        gr.Markdown("### 测试1: 基础功能")

        with gr.Row():
            test_input = gr.Textbox(
                label="输入测试文字",
                placeholder="输入任意内容...",
                value="Hello Gradio!"
            )

        test_btn = gr.Button("🧪 测试", variant="primary")
        test_output = gr.Textbox(label="输出结果", lines=3)

        test_btn.click(fn=test_function, inputs=[test_input], outputs=[test_output])

        gr.Markdown("---")
        gr.Markdown("### 测试2: 加载demo.json")

        json_btn = gr.Button("📋 加载demo.json", variant="secondary")
        json_output = gr.Textbox(label="加载结果", lines=5)

        json_btn.click(fn=load_demo_json, inputs=[], outputs=[json_output])

    with gr.Tab("环境信息"):
        gr.Markdown(f"""
## 环境信息

- **Gradio版本**: {gr.__version__}
- **Python版本**: {sys.version.split()[0]}
- **Colab环境**: {'是' if IN_COLAB else '否'}
- **当前目录**: 运行 `!pwd` 查看

## 如果这个界面可以正常访问：

说明Gradio本身工作正常，问题可能在于：
1. ✅ 完整UI代码有bug
2. ✅ 某些import失败
3. ✅ 内存不足

## 下一步：

如果这个测试界面能正常访问，请在Colab中运行：

```python
# 查看完整UI的错误日志
!python gradio_pipeline_ui_colab.py 2>&1 | tail -50
```

把错误信息告诉我，我会帮你修复。
        """)

print("✓ 界面创建完成")

# Step 5: 启动界面
print("\n[5/5] 启动Gradio界面...")
print("="*80)
print("🚀 正在启动...")
print("="*80)

if __name__ == "__main__":
    try:
        demo.launch(
            share=True,           # 生成公开链接
            debug=True,           # 调试模式
            show_error=True,      # 显示错误
            server_name="0.0.0.0",
            quiet=False           # 显示详细日志
        )
    except Exception as e:
        print("\n" + "="*80)
        print("❌ 启动失败！")
        print("="*80)
        print(f"错误: {str(e)}")
        print("\n请将上述错误信息截图发送给我")
        import traceback
        traceback.print_exc()
