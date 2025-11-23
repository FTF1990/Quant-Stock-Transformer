# 🔧 Colab故障排查指南

## ❌ 问题：ERR_CONNECTION_CLOSED

如果你看到这个错误，说明Gradio进程没有成功启动或立即崩溃了。

---

## ✅ 解决方案（按顺序尝试）

### 🔍 方案1: 基础连接测试（30秒）

运行这个最小化测试来诊断问题：

```python
# Cell 1: 克隆项目
!git clone https://github.com/FTF1990/Quant-Stock-Transformer.git
%cd Quant-Stock-Transformer

# Cell 2: 运行诊断工具
!python test_gradio_minimal.py
```

**预期结果**:
- 如果看到公开URL并能访问 → Gradio本身正常，问题在完整UI代码
- 如果仍然失败 → 继续下一步

---

### 🛡️ 方案2: 使用稳定版（推荐）

使用经过充分测试的稳定版本：

```python
# Cell 1: 准备环境
!git clone https://github.com/FTF1990/Quant-Stock-Transformer.git
%cd Quant-Stock-Transformer

# Cell 2: 运行稳定版
!python gradio_colab_stable.py
```

**特点**:
- ✅ 完善的错误处理
- ✅ 自动安装依赖
- ✅ 轻量级（仅JSON加载功能）
- ✅ 详细的错误日志

---

### 🔧 方案3: 手动诊断

#### 步骤1: 检查Gradio版本

```python
import gradio as gr
print(gr.__version__)

# 如果版本 < 4.0，升级
!pip install --upgrade gradio
```

#### 步骤2: 检查导入

```python
# 测试导入
try:
    from complete_training_pipeline import StockDataFetcher
    print("✓ Pipeline模块导入成功")
except Exception as e:
    print(f"✗ 导入失败: {e}")
    # 这是正常的，意味着需要运行简化版
```

#### 步骤3: 查看错误日志

```python
# 运行并捕获所有输出
!python gradio_pipeline_ui_colab.py 2>&1 | tee gradio_log.txt

# 查看最后50行日志
!tail -50 gradio_log.txt
```

把日志内容发给我，我可以帮你分析。

---

### 🚀 方案4: 全新开始（干净环境）

如果以上都不行，从头开始：

```python
# Cell 1: 重置环境
# 菜单 -> 运行时 -> 重启运行时

# Cell 2: 全新安装
!rm -rf Quant-Stock-Transformer
!git clone https://github.com/FTF1990/Quant-Stock-Transformer.git
%cd Quant-Stock-Transformer

# Cell 3: 安装依赖
!pip install -q --upgrade gradio plotly pandas numpy matplotlib

# Cell 4: 运行稳定版
!python gradio_colab_stable.py
```

---

## 📋 完整的Colab Notebook模板

复制以下内容到新的Colab notebook：

### Cell 1: 环境准备
```python
# 克隆项目
!git clone https://github.com/FTF1990/Quant-Stock-Transformer.git
%cd Quant-Stock-Transformer

# 确认文件存在
!ls -la *.py | head -5
```

### Cell 2: 检查环境
```python
import sys
import torch

print(f"Python版本: {sys.version.split()[0]}")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Cell 3: 选择运行版本

**选项A: 诊断测试**
```python
!python test_gradio_minimal.py
```

**选项B: 稳定版（推荐）**
```python
!python gradio_colab_stable.py
```

**选项C: Colab专用版**
```python
!python gradio_pipeline_ui_colab.py
```

**选项D: 完整版**
```python
!pip install -r requirements.txt
!python gradio_pipeline_ui.py
```

### Cell 4: 如果失败，查看日志
```python
# 查看详细错误
!python gradio_colab_stable.py 2>&1 | tail -100
```

---

## 🐛 常见错误及解决方案

### 错误1: ImportError: cannot import name 'xxx'

**原因**: 缺少依赖或版本不兼容

**解决**:
```python
# 重新安装所有依赖
!pip install --force-reinstall -q gradio pandas numpy matplotlib plotly
```

### 错误2: ModuleNotFoundError: No module named 'complete_training_pipeline'

**原因**: 不在项目目录，或文件不存在

**解决**:
```python
# 检查当前目录
!pwd
!ls *.py

# 切换到正确目录
%cd /content/Quant-Stock-Transformer
```

### 错误3: gradio.exceptions.Error: Network error

**原因**: Gradio网络问题

**解决**:
```python
# 方法1: 重启运行时后重试

# 方法2: 使用代理（如果在限制地区）
# 这需要额外配置

# 方法3: 等待几分钟后重试
# 有时是gradio服务器暂时问题
```

### 错误4: OSError: [Errno 98] Address already in use

**原因**: 端口被占用

**解决**:
```python
# 重启运行时
# 菜单 -> 运行时 -> 重启运行时
```

### 错误5: Out of memory

**原因**: Colab RAM不足

**解决**:
```python
# 使用轻量级版本
!python gradio_colab_stable.py  # 而不是完整版

# 或释放内存
import gc
gc.collect()
```

---

## 📊 版本对比

| 文件名 | 大小 | 功能 | 内存占用 | 推荐场景 |
|--------|------|------|----------|----------|
| `test_gradio_minimal.py` | 最小 | 基础测试 | 极低 | 诊断问题 |
| `gradio_colab_stable.py` | 小 | JSON加载 | 低 | 快速测试 |
| `gradio_pipeline_ui_colab.py` | 中 | 2个核心步骤 | 中 | Colab训练 |
| `gradio_pipeline_ui.py` | 大 | 7个完整步骤 | 高 | 完整功能 |

---

## 🎯 快速决策树

```
能否访问gradio.live链接？
├─ 否 → 运行 test_gradio_minimal.py
│   ├─ 仍然失败 → 检查网络/防火墙/VPN
│   └─ 成功 → 问题在完整UI，改用 gradio_colab_stable.py
│
└─ 是，但功能不全 → 正常，使用 gradio_pipeline_ui.py 获取完整功能
```

---

## 💡 最佳实践

### 1. 始终从测试开始

```python
# 第一次使用，先测试
!python test_gradio_minimal.py

# 确认可用后，再用完整版
!python gradio_pipeline_ui.py
```

### 2. 保存重要信息

```python
# 运行前保存notebook状态
# 菜单 -> 文件 -> 在云端硬盘中保存副本

# 挂载Drive保存结果
from google.colab import drive
drive.mount('/content/drive')
```

### 3. 定期重启运行时

```python
# Colab会话时间长后可能不稳定
# 建议每2-3小时重启一次
# 菜单 -> 运行时 -> 重启运行时
```

---

## 🆘 还是不行？

### 收集以下信息发给我：

1. **Gradio版本**
```python
import gradio as gr
print(gr.__version__)
```

2. **Python版本**
```python
import sys
print(sys.version)
```

3. **运行的命令**
```
你运行的完整命令
```

4. **错误日志**
```python
!python gradio_colab_stable.py 2>&1 | tail -100
```

5. **文件列表**
```python
!ls -la *.py
```

把以上信息发给我，我会帮你诊断具体问题！

---

## 📞 获取支持

- GitHub Issues: https://github.com/FTF1990/Quant-Stock-Transformer/issues
- 文档: COLAB_SETUP.md
- UI说明: UI_USAGE.md

---

**最后更新**: 2024
**状态**: ✅ 持续更新中
