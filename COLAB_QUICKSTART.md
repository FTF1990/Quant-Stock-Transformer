# 🚀 Colab快速启动指南

## 📱 一键复制粘贴版本

直接在Google Colab中创建新的notebook,然后按顺序运行以下cells:

---

### Cell 1: 安装依赖
```python
!pip install panel plotly jupyter_bokeh -q
print("✅ Panel安装完成!")
```

---

### Cell 2: 克隆仓库
```python
!git clone https://github.com/FTF1990/Quant-Stock-Transformer.git
%cd Quant-Stock-Transformer
print("✅ 仓库克隆完成!")
```

---

### Cell 3: 安装项目依赖
```python
!pip install -r requirements.txt -q
print("✅ 项目依赖安装完成!")
```

---

### Cell 4: 启动Panel UI ⭐
```python
# 导入必要的库
import panel as pn

# 初始化Panel扩展
pn.extension('plotly', 'tabulator', sizing_mode="stretch_width")

# 导入UI应用
from panel_pipeline_ui import launch

# 启动应用并显示
print("🚀 正在启动Panel UI...")
app = launch()

# 直接显示UI (在notebook中内联渲染)
app
```

**重要**: 最后一行 `app` 会直接在notebook中显示UI界面，不会启动服务器！

---

## 🎯 使用步骤

UI启动后,按照以下顺序使用:

1. **Tab 1** - 上传JSON文件 (如 `data/demo.json`)
2. **Tab 2** - 配置并抓取数据
3. **Tab 3** - 数据预处理
4. **Tab 4** - 训练SST模型
5. **Tab 5** - 提取特征
6. **Tab 6** - 训练时序模型 (LSTM/GRU/TCN)
7. **Tab 7** - 评估并对比所有模型

---

## 💡 快速测试版本

如果只想快速测试UI是否正常工作,可以用这个简化版本:

```python
# 简化版 - 只启动UI查看界面
# 确保先安装: !pip install panel plotly jupyter_bokeh -q

import panel as pn
pn.extension('plotly', 'tabulator')

from panel_pipeline_ui import dashboard

# 直接显示 (不需要.servable()，直接运行对象即可)
dashboard
```

---

## 📊 如何上传文件到Colab

### 方法1: 从本地上传
```python
from google.colab import files
uploaded = files.upload()
```

### 方法2: 从Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')

# 然后在UI中使用文件路径
# 例如: /content/drive/MyDrive/demo.json
```

### 方法3: 直接下载示例文件
```python
!wget https://raw.githubusercontent.com/FTF1990/Quant-Stock-Transformer/main/data/demo.json
```

---

## 🔍 验证安装

运行此代码验证所有依赖都已正确安装:

```python
import sys
print("Python版本:", sys.version)

# 检查关键库
libraries = ['panel', 'plotly', 'torch', 'pandas', 'numpy', 'matplotlib']

for lib in libraries:
    try:
        __import__(lib)
        print(f"✅ {lib}")
    except ImportError:
        print(f"❌ {lib} - 需要安装!")
```

---

## 🎨 UI预览

启动后你会看到:

```
================================================================================
🚀 股票预测Pipeline可视化 - Panel UI
================================================================================
✅ 设备: cuda (或 cpu)
✅ Panel已初始化
================================================================================
```

然后下方会显示完整的交互式UI,包含:
- 🎯 侧边栏: 显示状态和功能列表
- 📑 主区域: 7个Tab页面,每个对应一个步骤
- 🎛️ 控件: 按钮、滑块、输入框等交互组件
- 📊 可视化: 图表和表格直接显示

---

## ⚡ 性能优化建议

### 1. 使用GPU
确保Colab使用GPU:
- 菜单: `Runtime` → `Change runtime type` → `GPU`

### 2. 减小训练规模(测试时)
```python
# 在训练时使用较小的参数
- Epochs: 10-20 (而不是50-100)
- Batch Size: 16-32
- 序列长度: 30-40 (而不是60)
```

### 3. 保存检查点
```python
# 训练完成后保存模型
import torch
torch.save(state.sst_model.state_dict(), 'sst_model.pth')

# 恢复模型
state.sst_model.load_state_dict(torch.load('sst_model.pth'))
```

---

## 🐛 常见问题解决

### 问题: localhost拒绝连接 / 服务器无法访问

**原因**: Panel试图启动本地服务器，但Colab是云端环境，无法访问localhost。

**解决方案**: ⭐ **不要使用 `.servable()` 或 `.show()`**，直接运行对象：

```python
# ❌ 错误方式
app = launch()
app.servable()  # 这会尝试启动服务器

# ✅ 正确方式
app = launch()
app  # 直接运行对象，在notebook中内联显示
```

或者使用最简单的方式：

```python
import panel as pn
pn.extension('plotly', 'tabulator')

from panel_pipeline_ui import dashboard
dashboard  # 直接显示，不启动服务器
```

---

### 问题: UI不显示

```python
# 解决方案1: 重新初始化Panel
import panel as pn
pn.extension('plotly', 'tabulator', sizing_mode="stretch_width")

# 解决方案2: 清空输出后重新运行
from IPython.display import clear_output
clear_output()

# 解决方案3: 确保在cell的最后一行返回对象
from panel_pipeline_ui import dashboard
dashboard  # 必须是cell的最后一行，且没有分号

# 解决方案4: 重启runtime
# 菜单: Runtime → Restart runtime
```

### 问题: 找不到模块

```python
# 解决方案: 确认工作目录
import os
print("当前目录:", os.getcwd())

# 应该显示: /content/Quant-Stock-Transformer
# 如果不是,运行:
%cd /content/Quant-Stock-Transformer
```

### 问题: 内存不足

```python
# 解决方案: 清理内存
import gc
import torch

torch.cuda.empty_cache()
gc.collect()

# 或使用更小的batch size和模型
```

---

## 📱 完整示例Notebook

创建新的Colab notebook并按顺序运行:

```python
# ============================================================
# Cell 1: 环境设置
# ============================================================
!pip install panel plotly jupyter_bokeh -q

# ============================================================
# Cell 2: 克隆项目
# ============================================================
!git clone https://github.com/FTF1990/Quant-Stock-Transformer.git
%cd Quant-Stock-Transformer

# ============================================================
# Cell 3: 安装依赖
# ============================================================
!pip install -r requirements.txt -q

# ============================================================
# Cell 4: 下载示例数据 (可选)
# ============================================================
# 如果你没有自己的数据,可以使用示例数据
!wget https://raw.githubusercontent.com/FTF1990/Quant-Stock-Transformer/main/data/demo.json -O demo.json

# ============================================================
# Cell 5: 启动Panel UI
# ============================================================
import panel as pn
pn.extension('plotly', 'tabulator', sizing_mode="stretch_width")

from panel_pipeline_ui import launch

print("🚀 启动中...")
app = launch()

# ============================================================
# Cell 6: 显示UI (在新cell中运行)
# ============================================================
# 直接运行app对象，UI会在下方显示
app

# ============================================================
# 现在你可以在上方的UI中进行所有操作!
# ============================================================
```

---

## 🎓 学习资源

- **Panel文档**: https://panel.holoviz.org/
- **完整使用指南**: 查看 `PANEL_UI_GUIDE.md`
- **项目README**: 查看主README了解算法细节

---

## 🆘 获取帮助

如果遇到问题:
1. 查看 `PANEL_UI_GUIDE.md` 的故障排除部分
2. 在GitHub上提Issue: https://github.com/FTF1990/Quant-Stock-Transformer/issues
3. 确保使用最新版本: `git pull origin main`

---

**祝使用愉快! 🎉**

Version: 2.0.0 | Updated: 2025-11-23
