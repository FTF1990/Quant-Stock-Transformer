# ⚡ Colab 一键启动 - 超简单版本

## 🎯 复制这段代码到Colab的一个cell中运行即可！

```python
# ============================================================
# 一键启动 Panel UI - 完整版本
# ============================================================

# 第1步: 安装依赖 (包括Colab必需的jupyter_bokeh)
print("📦 安装依赖中...")
!pip install panel plotly jupyter_bokeh -q 2>&1 | grep -v "already satisfied" || true

# 第2步: 克隆项目
print("\n📥 克隆项目中...")
import os
if not os.path.exists('Quant-Stock-Transformer'):
    !git clone https://github.com/FTF1990/Quant-Stock-Transformer.git

# 第3步: 切换到项目目录
print("\n📂 切换到项目目录...")
os.chdir('/content/Quant-Stock-Transformer')
print(f"✅ 当前目录: {os.getcwd()}")

# 第4步: 安装项目依赖
print("\n📦 安装项目依赖中...")
!pip install -r requirements.txt -q 2>&1 | grep -v "already satisfied" || true

# 第5步: 启动Panel UI
print("\n🚀 启动Panel UI...")
print("="*80)

import panel as pn
pn.extension('plotly', 'tabulator', sizing_mode="stretch_width")

from panel_pipeline_ui import dashboard

print("✅ UI已准备就绪!")
print("📝 UI将在下方显示")
print("="*80)

# 显示UI - 使用display()确保在Colab中正确显示
from IPython.display import display
display(dashboard)
```

---

## ✨ 就这么简单！

复制上面的代码 → 粘贴到Colab → 运行 → UI出现！

---

## 🎨 你会看到什么

运行后，下方会出现完整的UI界面：

```
📦 安装依赖中...
✅ 已安装 panel

📥 克隆项目中...
✅ 项目已克隆

📦 安装项目依赖中...
✅ 依赖已安装

🚀 启动Panel UI...
================================================================================
✅ UI已准备就绪!
📝 UI将在下方显示
================================================================================

[下方显示完整的Panel UI界面，包含7个Tab]
```

---

## 📋 使用流程

1. **Tab 1** - 上传你的股票JSON文件
2. **Tab 2** - 配置日期范围并抓取数据
3. **Tab 3** - 输入目标股票代码并预处理
4. **Tab 4** - 训练SST模型
5. **Tab 5** - 提取内部特征
6. **Tab 6** - 训练LSTM/GRU/TCN时序模型
7. **Tab 7** - 评估并对比所有模型性能

---

## 💡 测试用示例数据

如果你没有JSON文件，可以先下载示例数据：

```python
# 在UI上方添加一个cell运行这个
!wget https://raw.githubusercontent.com/FTF1990/Quant-Stock-Transformer/main/data/demo.json -O demo.json
print("✅ 示例数据已下载到: demo.json")
```

然后在UI的Tab 1中上传 `demo.json`

---

## 🔧 如果遇到问题

### 问题1: Cell运行很久没反应

**解决**: 这是正常的，首次安装依赖需要1-2分钟，请耐心等待

### 问题2: UI不显示

**解决**: 确保代码块的最后一行是 `dashboard` (没有分号!)

### 问题3: 显示 "localhost拒绝连接"

**解决**: 使用上面的代码，它已经修复了这个问题（使用内联显示而不是服务器）

### 问题4: 导入错误

**解决**:
```python
# 确认当前目录
import os
print(os.getcwd())  # 应该显示 /content/Quant-Stock-Transformer

# 如果不是，运行:
os.chdir('/content/Quant-Stock-Transformer')
```

---

## 🚀 更简洁的版本（如果项目已克隆）

如果你已经克隆过项目，下次只需运行这个：

```python
import os
os.chdir('/content/Quant-Stock-Transformer')

import panel as pn
pn.extension('plotly', 'tabulator', sizing_mode="stretch_width")

from panel_pipeline_ui import dashboard
from IPython.display import display
display(dashboard)
```

---

## 📱 终极简化版（3行代码）

```python
%cd /content/Quant-Stock-Transformer
!pip install panel plotly jupyter_bokeh -q
from panel_pipeline_ui import dashboard; import panel as pn; from IPython.display import display; pn.extension('plotly', 'tabulator'); display(dashboard)
```

但建议使用第一个完整版本，因为它有更好的错误处理和提示信息。

---

**就这么简单！享受使用吧！🎉**

Version: 2.0.1 | Updated: 2025-11-23
