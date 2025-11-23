# Panel UI 使用指南 - Colab版

## 🎯 为什么使用Panel而不是Gradio?

Panel是专为Jupyter/Colab环境设计的UI工具,具有以下优势:

✅ **原生Colab支持** - 直接在notebook中渲染,无需外部链接
✅ **更稳定** - 不依赖网络隧道或公共URL
✅ **功能强大** - 支持复杂的Tab布局和交互组件
✅ **无需配置** - 不需要处理端口、share链接等问题

---

## 🚀 在Colab中快速开始

### 步骤1: 安装依赖

在Colab的一个cell中运行:

```python
!pip install panel
```

### 步骤2: 克隆仓库(如果还没有)

```python
!git clone https://github.com/FTF1990/Quant-Stock-Transformer.git
%cd Quant-Stock-Transformer
```

### 步骤3: 启动Panel UI

在新的cell中运行:

```python
from panel_pipeline_ui import launch

# 启动UI
app = launch()

# 显示UI
app.servable()
```

UI将直接显示在notebook的输出中!

---

## 📋 完整Colab示例

这是一个完整的Colab notebook示例:

```python
# Cell 1: 安装依赖
!pip install panel plotly

# Cell 2: 克隆仓库并进入目录
!git clone https://github.com/FTF1990/Quant-Stock-Transformer.git
%cd Quant-Stock-Transformer

# Cell 3: 安装项目依赖
!pip install -r requirements.txt

# Cell 4: 启动Panel UI
from panel_pipeline_ui import launch

app = launch()
app.servable()
```

---

## 🎨 UI功能说明

Panel UI提供了7个主要步骤的Tab页面:

### Tab 1: 加载股票JSON
- 📤 上传JSON文件
- 📊 查看股票列表
- 📈 市场分布饼图

### Tab 2: 数据抓取
- 🌍 选择目标市场
- 📅 设置日期范围
- ⚙️ 配置批量参数
- 📥 抓取历史数据

### Tab 3: 数据预处理
- 🎯 输入目标股票代码
- 🔄 自动特征计算
- 📊 收益率分布可视化

### Tab 4: SST模型训练
- 🧠 配置训练参数
- 🚀 训练双输出SST模型
- 📈 实时训练曲线

### Tab 5: 特征提取
- 🔍 提取SST内部特征
- 📊 特征分布可视化
- 📉 残差分析

### Tab 6: 时序模型训练
- 🤖 选择模型类型(LSTM/GRU/TCN)
- ⚙️ 配置训练参数
- 📈 训练损失曲线

### Tab 7: 模型评估
- 📊 评估所有模型
- 📈 性能对比图表
- 📋 详细指标表格

---

## 💡 使用技巧

### 1. 分步执行
按照Tab的顺序依次执行,每步完成后再进行下一步。

### 2. 保存进度
训练过程中的数据会自动保存在全局状态中,可以随时在不同Tab间切换查看。

### 3. 调整参数
所有参数都可以通过滑块和输入框轻松调整。

### 4. 查看结果
图表会直接显示在UI中,也可以右键保存。

---

## 🔧 故障排除

### 问题1: UI不显示

**解决方案:**
```python
# 确保Panel扩展已加载
import panel as pn
pn.extension('plotly', 'tabulator')

# 重新导入并启动
from panel_pipeline_ui import launch
app = launch()
app.servable()
```

### 问题2: 找不到模块

**解决方案:**
```python
# 确保在正确的目录
%cd /content/Quant-Stock-Transformer

# 确保已安装所有依赖
!pip install -r requirements.txt
```

### 问题3: 图表不显示

**解决方案:**
```python
# 重新加载Panel扩展
import panel as pn
pn.extension('plotly', 'tabulator', sizing_mode="stretch_width")
```

---

## 🆚 Panel vs Gradio对比

| 特性 | Panel | Gradio |
|------|-------|--------|
| Colab原生支持 | ✅ 完美 | ⚠️ 需要公共链接 |
| 网络依赖 | ✅ 无需外部网络 | ❌ 需要稳定连接 |
| 显示方式 | ✅ 直接在notebook中 | ⚠️ 需要点击外部链接 |
| 稳定性 | ✅ 非常稳定 | ⚠️ 可能连接失败 |
| 配置复杂度 | ✅ 简单 | ⚠️ 需要配置share/port |
| Tab支持 | ✅ 原生支持 | ✅ 支持 |
| 组件丰富度 | ✅ 非常丰富 | ✅ 丰富 |

---

## 📚 更多资源

- **Panel官方文档**: https://panel.holoviz.org/
- **Panel示例画廊**: https://panel.holoviz.org/gallery/index.html
- **项目GitHub**: https://github.com/FTF1990/Quant-Stock-Transformer

---

## 🙋 常见问题

**Q: Panel比Gradio快吗?**
A: 在Colab中,Panel通常更快,因为它不需要建立外部连接。

**Q: 可以在本地环境使用吗?**
A: 可以! 运行`python panel_pipeline_ui.py`即可启动。

**Q: 数据会保存吗?**
A: 训练过程中的数据会保存在内存和本地文件(如historical_data.pkl)中。

**Q: 可以导出结果吗?**
A: 可以! 右键点击图表即可保存,表格数据可以通过代码访问state对象获取。

---

**版本**: 2.0.0 (Panel版)
**更新日期**: 2025-11-23
**作者**: Quant-Stock-Transformer Team
