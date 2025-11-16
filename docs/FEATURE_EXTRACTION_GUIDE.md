# SST内部特征提取完全指南

## 🎯 核心问题

**问题**：通过SST模型的常规推理（`forward()`），只能得到T日和T+1日的预测输出，如何提取内部特征（attention weights、encoder output）？

**答案**：需要使用专门的方法，不是常规推理！

---

## 📊 两种推理模式对比

### 模式1️⃣：常规推理（仅用于预测）

```python
# 标准的forward调用
predictions = model(boundary_conditions)

# 输出：
# predictions: [batch, num_target_sensors]
#
# 问题：无法获取中间特征！
```

**特点**：
- ✅ 快速、高效
- ✅ 用于训练和生产推理
- ❌ 只返回最终预测
- ❌ 无法获取attention weights
- ❌ 无法获取encoder output

---

### 模式2️⃣：特征提取推理（用于分析和增强）

```python
# 使用SpatialFeatureExtractor的专门方法
predictions, features = model.forward_with_features(
    boundary_conditions,
    return_attention=True,
    return_encoder_output=True
)

# 输出：
# predictions: [batch, num_target_sensors] - 预测结果
# features: dict - 包含所有中间特征
#   {
#       'embeddings': [batch, num_sensors, d_model],
#       'encoder_output': [batch, num_sensors, d_model],
#       'attention_weights': [batch, num_layers, num_heads, num_sensors, num_sensors],
#       'pooled_features': [batch, d_model]
#   }
```

**特点**：
- ✅ 返回所有中间特征
- ✅ 可以分析模型内部机制
- ✅ 用于构建增强模型（如Stage3 LSTM）
- ⚠️ 计算稍慢（需要额外的张量操作）

---

## 🔬 技术实现原理

### 为什么常规`forward()`无法返回中间特征？

PyTorch的标准`nn.TransformerEncoder`设计如下：

```python
# PyTorch源码（简化）
class TransformerEncoder(nn.Module):
    def forward(self, src):
        output = src
        for layer in self.layers:
            output = layer(output)  # ← 这里调用了attention，但没有返回权重
        return output  # ← 只返回最终输出
```

**核心问题**：
- `TransformerEncoderLayer.forward()` 内部调用了 `self.self_attn()`
- 但它**不返回**attention weights（尽管内部计算了）
- 标准forward只返回transformer的输出张量

---

### `SpatialFeatureExtractor`的解决方案

通过**手动逐层执行**transformer，在每一层手动调用`self_attn()`并设置`need_weights=True`：

```python
def _forward_transformer_with_attention(self, x):
    attention_weights_list = []

    # 逐层执行（而不是调用self.transformer(x)）
    for layer in self.transformer.layers:
        residual = x

        # 手动调用MultiheadAttention，并要求返回权重
        attn_output, attn_weights = layer.self_attn(
            x, x, x,
            need_weights=True,              # ← 关键！
            average_attn_weights=False      # ← 返回每个head的权重
        )

        attention_weights_list.append(attn_weights)

        # 手动执行剩余的操作（dropout、residual、norm、FFN）
        x = residual + layer.dropout1(attn_output)
        x = layer.norm1(x)
        residual = x
        ff_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
        x = residual + layer.dropout2(ff_output)
        x = layer.norm2(x)

    # 合并所有层的attention
    attention_weights = torch.stack(attention_weights_list, dim=1)

    return x, attention_weights
```

**关键技术点**：
1. **绕过标准forward**：不调用`self.transformer(x)`
2. **手动逐层执行**：直接访问`self.transformer.layers`
3. **显式请求权重**：`need_weights=True, average_attn_weights=False`
4. **手动实现residual connection和normalization**：复现TransformerEncoderLayer的完整逻辑

---

## 🔧 完整使用流程

### Step 1: 创建支持特征提取的模型

```python
from models.spatial_feature_extractor import SpatialFeatureExtractor

# 创建模型
model = SpatialFeatureExtractor(
    num_boundary_sensors=23,  # 20个股票 + 3个指数
    num_target_sensors=1,     # 预测1个目标
    d_model=128,
    nhead=8,
    num_layers=3,
    enable_feature_extraction=True  # ← 启用特征提取
)

# 加载训练好的权重
model.load_state_dict(torch.load('sst_model.pth'))
model.eval()
```

### Step 2: 准备输入数据

```python
import torch

# 边界条件：大盘、板块、龙头
boundary_conditions = torch.tensor([
    # [指数1, 指数2, 指数3, 股票1, 股票2, ..., 股票20]
    [0.01, 0.02, 0.015, 0.005, -0.003, ...]  # T日的数据
], dtype=torch.float32)

# shape: [1, 23]
```

### Step 3: 提取完整特征

```python
with torch.no_grad():
    predictions, features = model.forward_with_features(
        boundary_conditions,
        return_attention=True,
        return_encoder_output=True
    )

# 检查输出
print(f"预测: {predictions.shape}")  # [1, 1]
print(f"Embeddings: {features['embeddings'].shape}")  # [1, 23, 128]
print(f"Encoder Output: {features['encoder_output'].shape}")  # [1, 23, 128]
print(f"Attention: {features['attention_weights'].shape}")  # [1, 3, 8, 23, 23]
print(f"Pooled: {features['pooled_features'].shape}")  # [1, 128]
```

### Step 4: 快速提取单一特征

如果只需要某一类特征（更高效）：

```python
# 只提取encoder output
encoder_output = model.get_encoder_output(boundary_conditions)
# shape: [1, 23, 128]

# 只提取attention weights
attention_weights = model.get_attention_weights(boundary_conditions)
# shape: [1, 3, 8, 23, 23]
```

---

## 🎯 双输出问题：T日 vs T+1日

### 问题描述

理论框架要求：
- **输出1（T日）**：同时刻预测（纯空间响应）
- **输出2（T+1日）**：次日预测（空间+时序）

但当前`StaticSensorTransformer`只有**单输出头**：

```python
# 当前实现（单输出）
self.output_projection = nn.Linear(d_model, num_target_sensors)
```

### 解决方案：扩展为双输出头

需要修改模型架构：

```python
class DualOutputSST(SpatialFeatureExtractor):
    """双输出SST：同时预测T日和T+1日"""

    def __init__(self, num_boundary_sensors, num_target_sensors, **kwargs):
        super().__init__(num_boundary_sensors, num_target_sensors, **kwargs)

        # 替换单一输出层为双输出头
        self.output_projection_T = nn.Linear(self.d_model, num_target_sensors)
        self.output_projection_T1 = nn.Linear(self.d_model, num_target_sensors)

    def forward(self, boundary_conditions):
        """标准forward：返回双输出"""
        batch_size = boundary_conditions.shape[0]

        # 1. Embed
        x = boundary_conditions.unsqueeze(-1)
        x = self.boundary_embedding(x) + self.boundary_position_encoding.unsqueeze(0)

        # 2. Transform
        x = self.transformer(x)

        # 3. Pool
        x_pooled = x.permute(0, 2, 1)
        x_pooled = self.global_pool(x_pooled).squeeze(-1)

        # 4. 双输出
        pred_T = self.output_projection_T(x_pooled)    # T日预测
        pred_T1 = self.output_projection_T1(x_pooled)  # T+1日预测

        return pred_T, pred_T1

    def forward_with_features(self, boundary_conditions, **kwargs):
        """带特征的双输出forward"""
        batch_size = boundary_conditions.shape[0]
        features = {}

        # 1. Embed
        x = boundary_conditions.unsqueeze(-1)
        x = self.boundary_embedding(x) + self.boundary_position_encoding.unsqueeze(0)
        features['embeddings'] = x.clone()

        # 2. Transform (with attention)
        if kwargs.get('return_attention', True):
            encoder_output, attention_weights = self._forward_transformer_with_attention(x)
            features['attention_weights'] = attention_weights
        else:
            encoder_output = self.transformer(x)

        features['encoder_output'] = encoder_output

        # 3. Pool
        x_pooled = encoder_output.permute(0, 2, 1)
        x_pooled = self.global_pool(x_pooled).squeeze(-1)
        features['pooled_features'] = x_pooled

        # 4. 双输出
        pred_T = self.output_projection_T(x_pooled)
        pred_T1 = self.output_projection_T1(x_pooled)

        return (pred_T, pred_T1), features
```

### 使用双输出模型

```python
# 训练时
pred_T, pred_T1 = model(boundary_conditions)
loss = criterion(pred_T, target_T) + criterion(pred_T1, target_T1)

# 特征提取时
(pred_T, pred_T1), features = model.forward_with_features(
    boundary_conditions,
    return_attention=True,
    return_encoder_output=True
)

# 计算残差
residual_T = target_T - pred_T      # 空间残差
residual_T1 = target_T1 - pred_T1   # 时空残差
```

---

## 📦 完整数据流（60天历史）

```python
import torch
import numpy as np

# 假设有60天历史数据
num_days = 60
batch_size = 1
num_sensors = 23
d_model = 128

# 存储所有天的特征
all_features = {
    'attention': [],
    'encoder_output': [],
    'residual_T': [],
    'residual_T1': []
}

# 逐天提取
for day in range(num_days):
    # 当天的边界条件
    boundary_conditions = historical_data[day]  # shape: [1, 23]

    # 获取真实值
    target_T = true_values_T[day]    # shape: [1, 1]
    target_T1 = true_values_T1[day]  # shape: [1, 1]

    # 提取特征
    with torch.no_grad():
        (pred_T, pred_T1), features = model.forward_with_features(
            boundary_conditions,
            return_attention=True,
            return_encoder_output=True
        )

    # 计算残差
    residual_T = target_T - pred_T
    residual_T1 = target_T1 - pred_T1

    # 保存
    all_features['attention'].append(features['attention_weights'])
    all_features['encoder_output'].append(features['encoder_output'])
    all_features['residual_T'].append(residual_T)
    all_features['residual_T1'].append(residual_T1)

# 合并成序列
attention_sequence = torch.cat(all_features['attention'], dim=0)
# shape: [60, num_layers, num_heads, num_sensors, num_sensors]

encoder_sequence = torch.cat(all_features['encoder_output'], dim=0)
# shape: [60, num_sensors, d_model]

residual_T_sequence = torch.cat(all_features['residual_T'], dim=0)
# shape: [60, 1]

residual_T1_sequence = torch.cat(all_features['residual_T1'], dim=0)
# shape: [60, 1]
```

---

## 🧮 降维后构建LSTM输入

```python
from models.relationship_extractors import AttentionBasedExtractor, EmbeddingBasedExtractor

# Step 1: 创建特征提取器（降维）
attention_extractor = AttentionBasedExtractor(
    num_sensors=23,
    output_dim=10,  # 9600维 → 10维
    method='graph_features'
)

embedding_extractor = EmbeddingBasedExtractor(
    d_model=128,
    output_dim=32,  # 2560维 → 32维
    pooling_method='autoencoder'
)

# Step 2: 逐天降维
lstm_input_sequence = []

for day in range(60):
    # 提取attention特征（10维）
    attn_features = attention_extractor(
        all_features['attention'][day],
        target_stock_idx=0
    )  # shape: [1, 10]

    # 提取encoder特征（32维）
    enc_features = embedding_extractor(
        encoder_output=all_features['encoder_output'][day],
        target_stock_idx=0
    )  # shape: [1, 32]

    # 残差特征（2维）
    res_features = torch.cat([
        all_features['residual_T'][day],
        all_features['residual_T1'][day]
    ], dim=-1)  # shape: [1, 2]

    # 合并（10+32+2=44维）
    day_features = torch.cat([attn_features, enc_features, res_features], dim=-1)
    lstm_input_sequence.append(day_features)

# Step 3: 构建LSTM输入
lstm_input = torch.cat(lstm_input_sequence, dim=0)
# shape: [60, 44]

# 添加batch维度
lstm_input = lstm_input.unsqueeze(0)
# shape: [1, 60, 44]
```

---

## ✅ 核心要点总结

| 维度 | 常规推理 | 特征提取推理 |
|------|----------|--------------|
| **方法** | `model(x)` | `model.forward_with_features(x)` |
| **返回** | 仅预测 | 预测 + 中间特征 |
| **Attention** | ❌ | ✅ |
| **Encoder Output** | ❌ | ✅ |
| **用途** | 训练/生产 | 分析/增强 |
| **速度** | 快 | 稍慢 |

**关键理解**：
1. 常规`forward()`调用`nn.TransformerEncoder`，它不返回attention权重
2. 需要手动逐层执行，并在`self_attn()`调用时设置`need_weights=True`
3. `SpatialFeatureExtractor`已经实现了这个逻辑
4. 双输出需要扩展模型（添加两个输出头）

---

## 📚 参考代码位置

- **SpatialFeatureExtractor**: `/home/user/Quant-Stock-Transformer/models/spatial_feature_extractor.py`
- **关键方法**:
  - `forward_with_features()`: 第79-130行
  - `_forward_transformer_with_attention()`: 第132-184行
  - `get_encoder_output()`: 第236-248行
  - `get_attention_weights()`: 第250-262行
