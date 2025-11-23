"""
Colab启动脚本 - Panel Pipeline UI
=================================

在Google Colab中使用此脚本启动Panel UI

使用方法:
    1. 在Colab notebook中运行:
       !pip install panel

    2. 然后在一个cell中运行:
       from run_panel_ui_colab import launch_panel_ui
       launch_panel_ui()

    3. UI将直接显示在notebook中
"""

import panel as pn

def launch_panel_ui():
    """在Colab中启动Panel UI"""

    # 确保Panel扩展已加载
    pn.extension('plotly', 'tabulator', sizing_mode="stretch_width")

    # 导入主应用
    from panel_pipeline_ui import dashboard

    # 在notebook中显示
    print("="*80)
    print("🚀 Panel UI 启动中...")
    print("="*80)
    print("✅ UI将在下方显示")
    print("📝 如果看不到UI,请确保已安装: pip install panel")
    print("="*80)

    # 直接返回dashboard,它会在notebook中渲染
    return dashboard.servable()


if __name__ == "__main__":
    # 如果在Colab中直接运行此文件
    app = launch_panel_ui()
    display(app)
