#!/usr/bin/env python3
"""
Colab专用启动脚本 - 独立启动Gradio应用

如果主文件启动有问题，使用这个脚本：
!python launch_colab.py
"""

import os
import sys

print("="*80)
print("🚀 Colab Gradio 启动器")
print("="*80)

# 确保在正确的目录
project_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_dir)
sys.path.insert(0, project_dir)

print(f"📁 工作目录: {os.getcwd()}")

try:
    # 导入主应用
    print("\n📦 导入Gradio应用模块...")
    import gradio_residual_tft_app as app

    print("✅ 模块导入成功")

    # 检查函数是否存在
    if not hasattr(app, 'create_unified_interface'):
        print("❌ 错误: create_unified_interface 函数不存在")
        sys.exit(1)

    print("✅ create_unified_interface 函数已找到")

    # 创建界面
    print("\n🏗️  创建Gradio界面...")
    demo = app.create_unified_interface()
    print("✅ 界面创建成功")

    # 启动
    print("\n🌐 启动Gradio服务器（公网链接）...")
    print("="*80)

    demo.launch(
        share=True,          # 生成公网链接（Colab必需）
        debug=True,          # 调试模式
        show_error=True,     # 显示错误
        inline=False,        # 使用独立窗口
        quiet=False          # 显示详细日志
    )

except ImportError as e:
    print(f"\n❌ 导入错误: {e}")
    print("\n请确保已安装所有依赖:")
    print("  !pip install gradio torch pandas numpy scikit-learn matplotlib seaborn")
    sys.exit(1)

except Exception as e:
    print(f"\n❌ 启动失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
