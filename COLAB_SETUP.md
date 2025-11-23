# 🌐 Google Colab 使用指南

## 🚀 快速开始

### 方法1: 使用Colab专用版本（推荐）

#### Step 1: 克隆项目

在Colab notebook的第一个cell中运行：

```python
# 克隆项目
!git clone https://github.com/FTF1990/Quant-Stock-Transformer.git
%cd Quant-Stock-Transformer

# 列出文件确认
!ls -la
```

#### Step 2: 安装依赖

```python
# 安装必要的包
!pip install -q gradio plotly akshare yfinance
```

#### Step 3: 运行Colab专用UI

```python
# 运行Colab优化版本
!python gradio_pipeline_ui_colab.py
```

这个版本会：
- ✅ 自动检测Colab环境
- ✅ 自动安装缺失的依赖
- ✅ 设置 `share=True` 生成公开链接
- ✅ 简化界面，减少资源占用
- ✅ 提供详细的错误信息

---

### 方法2: 使用完整版本

如果你想使用完整的7步UI：

```python
# 克隆项目
!git clone https://github.com/FTF1990/Quant-Stock-Transformer.git
%cd Quant-Stock-Transformer

# 安装依赖
!pip install -r requirements.txt

# 运行完整UI
!python gradio_pipeline_ui.py
```

**注意**: 完整版本需要更多内存，可能在免费Colab上运行较慢。

---

## ⚙️ Colab环境配置

### 1. 启用GPU加速

1. 点击菜单: **运行时** → **更改运行时类型**
2. 硬件加速器: 选择 **GPU (T4)**
3. 点击 **保存**

验证GPU：
```python
import torch
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### 2. 挂载Google Drive（可选）

保存训练结果到Google Drive：

```python
from google.colab import drive
drive.mount('/content/drive')

# 创建保存目录
!mkdir -p /content/drive/MyDrive/stock_models
```

---

## 📁 文件上传

### 上传demo.json

如果你有自己的选股JSON文件：

```python
from google.colab import files

# 上传文件
uploaded = files.upload()

# 移动到data目录
!mkdir -p data
!mv *.json data/
```

或者使用左侧文件浏览器拖拽上传。

---

## 🐛 常见问题排查

### 问题1: 无法加载UI，显示"Error"

**原因**: 可能是share链接生成问题

**解决方案**:
```python
# 方法1: 使用Colab专用版本
!python gradio_pipeline_ui_colab.py

# 方法2: 检查gradio版本
!pip install --upgrade gradio

# 方法3: 手动设置share参数
# 编辑代码，确保 share=True
```

### 问题2: ModuleNotFoundError

**原因**: 缺少依赖包

**解决方案**:
```python
# 安装缺失的包
!pip install gradio plotly akshare yfinance torch pandas numpy scikit-learn

# 或安装所有依赖
!pip install -r requirements.txt
```

### 问题3: ImportError: cannot import 'complete_training_pipeline'

**原因**: 文件不在同一目录

**解决方案**:
```python
# 确认文件存在
!ls -la *.py

# 确认在正确目录
!pwd

# 如果需要，切换到项目目录
%cd /content/Quant-Stock-Transformer
```

### 问题4: 数据抓取失败

**原因**: API限流或网络问题

**解决方案**:
```python
# 在UI中调整参数：
# - 批量大小: 2-3（不要太大）
# - 批次延迟: 3-5秒（增加延迟）
# - 日期范围: 缩短为1-3个月进行测试

# 或者使用缓存数据
# 上传之前抓取好的 historical_data.pkl
```

### 问题5: 运行时断开连接

**原因**: Colab免费版有时间限制

**解决方案**:
```python
# 1. 定期保存模型
# 在训练过程中会自动保存best_*.pth

# 2. 下载重要文件
from google.colab import files
files.download('best_sst_model.pth')
files.download('training_results.pkl')

# 3. 使用Google Drive保存
# 见上面"挂载Google Drive"部分
```

### 问题6: 内存不足（OOM）

**原因**: Colab免费版RAM有限（约12GB）

**解决方案**:
```python
# 1. 减少数据量
# - 使用更少的股票
# - 缩短日期范围

# 2. 减小batch size
# 在UI中将batch size设为8或16

# 3. 使用更小的模型
# 在UI中减少epochs

# 4. 释放内存
import gc
gc.collect()

# 5. 重启运行时
# 菜单 -> 运行时 -> 重启运行时
```

---

## 💡 Colab优化建议

### 1. 使用小规模数据测试

第一次运行时：
- 使用2-3个月的数据
- 选择5-10只股票
- SST训练: 20 epochs
- 时序训练: 30 epochs

测试成功后再扩大规模。

### 2. 分步骤运行

不要一次运行所有7个步骤，而是：
1. 先加载JSON，确认正确
2. 抓取少量数据测试
3. 预处理并查看数据
4. 训练SST（小epochs）
5. 确认无误后再训练时序模型

### 3. 保存中间结果

```python
# 数据抓取后立即下载
from google.colab import files
files.download('historical_data.pkl')

# 或保存到Drive
!cp historical_data.pkl /content/drive/MyDrive/stock_models/
```

### 4. 监控资源使用

```python
# 查看RAM使用
!free -h

# 查看GPU使用
!nvidia-smi
```

---

## 📊 完整Colab Notebook示例

创建一个新的Colab notebook，复制以下代码：

```python
# ==================== Cell 1: 环境设置 ====================
!git clone https://github.com/FTF1990/Quant-Stock-Transformer.git
%cd Quant-Stock-Transformer
!pip install -q gradio plotly akshare yfinance

# ==================== Cell 2: 检查环境 ====================
import torch
print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")

# ==================== Cell 3: 启动UI ====================
!python gradio_pipeline_ui_colab.py

# 等待UI加载，然后点击生成的公开链接
```

---

## 🔗 获取公开链接

运行UI后，你会看到类似的输出：

```
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://xxx.gradio.live
```

**点击 `https://xxx.gradio.live` 链接** 即可在新标签页中打开UI。

这个链接：
- ✅ 可以分享给他人
- ✅ 在手机上也能访问
- ⚠️ 72小时后会过期
- ⚠️ 关闭Colab session后会失效

---

## 📦 使用demo.json

项目已包含demo.json示例文件：

```python
# 查看demo.json
!cat data/demo.json

# 在UI中：
# 1. 进入"步骤1: 加载JSON"
# 2. 点击"上传JSON文件"
# 3. 选择 data/demo.json
# 4. 点击"加载股票列表"
```

---

## 🎯 推荐工作流程（Colab）

### 快速测试版（5-10分钟）

1. **克隆项目** (30秒)
   ```python
   !git clone https://github.com/FTF1990/Quant-Stock-Transformer.git
   %cd Quant-Stock-Transformer
   ```

2. **安装依赖** (1分钟)
   ```python
   !pip install -q gradio plotly akshare yfinance
   ```

3. **运行Colab UI** (30秒)
   ```python
   !python gradio_pipeline_ui_colab.py
   ```

4. **在UI中操作**:
   - 上传 data/demo.json
   - 数据抓取: 使用CN市场，2023-01-01 to 2023-03-31（3个月）
   - 其他步骤按提示操作

### 完整训练版（30-60分钟）

使用完整的gradio_pipeline_ui.py，所有7个步骤，正常参数。

---

## ⚡ 性能对比

| 环境 | RAM | GPU | 训练速度 | 推荐用途 |
|------|-----|-----|----------|----------|
| Colab免费版 | 12GB | T4 | 中等 | 测试、小规模训练 |
| Colab Pro | 25GB | A100 | 快 | 中大规模训练 |
| Colab Pro+ | 50GB | A100 | 很快 | 大规模训练 |
| 本地(GPU) | 自定义 | 自定义 | 取决于硬件 | 生产环境 |

---

## 📞 获取帮助

- **GitHub Issues**: 报告bug
- **UI_USAGE.md**: 详细UI使用说明
- **PIPELINE_FLOW_CONFIRMATION.md**: 流程验证文档
- **README.md**: 项目整体介绍

---

## ✅ 检查清单

在运行前确认：

- [ ] 已克隆项目到Colab
- [ ] 已安装gradio、plotly等依赖
- [ ] 已切换到项目目录
- [ ] (可选) 已启用GPU
- [ ] (可选) 已挂载Google Drive

运行后确认：

- [ ] 看到公开URL链接
- [ ] 能够访问UI界面
- [ ] 能够上传JSON文件
- [ ] 步骤1正常工作

---

**🎉 祝你在Colab上训练愉快！**

**注意**: Colab环境会定期断开，记得保存重要结果！
