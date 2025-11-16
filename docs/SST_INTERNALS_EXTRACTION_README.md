# SST内部特征提取 - 快速开始

## 📌 核心问题解答

**Q: 通过SST的常规推理（forward），我只能得到T日和T+1日的预测输出，如何获取内部特征（attention weights、encoder output）？**

**A: 需要使用专门的特征提取方法，不是常规的forward推理！**

---

## 🔑 关键理解

### 常规推理 ❌
```python
predictions = model(boundary_conditions)
# 只返回预测，无法获取中间特征
```

### 特征提取推理 ✅
```python
predictions, features = model.forward_with_features(
    boundary_conditions,
    return_attention=True,
    return_encoder_output=True
)
# 返回预测 + 所有中间特征
```

---

## 📂 文件说明

| 文件 | 说明 |
|------|------|
| `FEATURE_EXTRACTION_GUIDE.md` | 完整技术文档（理论+实现） |
| `../examples/extract_sst_internals_demo.py` | 可运行的完整示例 |
| `../models/spatial_feature_extractor.py` | 核心模型实现 |

---

## 🚀 快速开始

### Step 1: 理解两种模式

```python
from models.spatial_feature_extractor import SpatialFeatureExtractor

# 创建模型
model = SpatialFeatureExtractor(
    num_boundary_sensors=23,
    num_target_sensors=1,
    d_model=128,
    nhead=8,
    num_layers=3,
    enable_feature_extraction=True  # ← 启用特征提取
)

# 加载权重
model.load_state_dict(torch.load('sst_model.pth'))
model.eval()

# 准备输入
boundary_conditions = torch.randn(1, 23)  # [batch=1, num_sensors=23]
```

### Step 2: 提取完整特征

```python
with torch.no_grad():
    predictions, features = model.forward_with_features(
        boundary_conditions,
        return_attention=True,
        return_encoder_output=True
    )

# 检查输出
print(f"预测: {predictions.shape}")  # [1, 1]
print(f"\n中间特征:")
for key, value in features.items():
    print(f"  {key}: {value.shape}")

# 输出：
#   embeddings: torch.Size([1, 23, 128])
#   encoder_output: torch.Size([1, 23, 128])
#   attention_weights: torch.Size([1, 3, 8, 23, 23])
#   pooled_features: torch.Size([1, 128])
```

### Step 3: 快速提取单一特征

```python
# 只提取encoder output（更快）
encoder_output = model.get_encoder_output(boundary_conditions)
# shape: [1, 23, 128]

# 只提取attention weights
attention_weights = model.get_attention_weights(boundary_conditions)
# shape: [1, 3, 8, 23, 23]
```

---

## 🎯 双输出SST（T日 + T+1日）

当前的`SpatialFeatureExtractor`只有单输出。如需双输出，参考 `examples/extract_sst_internals_demo.py` 中的 `DualOutputSST` 实现：

```python
class DualOutputSST(SpatialFeatureExtractor):
    def __init__(self, num_boundary_sensors, num_target_sensors, **kwargs):
        super().__init__(num_boundary_sensors, num_target_sensors, **kwargs)

        # 双输出头
        self.output_projection_T = nn.Linear(self.d_model, num_target_sensors)
        self.output_projection_T1 = nn.Linear(self.d_model, num_target_sensors)

    def forward(self, boundary_conditions):
        # ... (省略embedding和transformer)

        # 双输出
        pred_T = self.output_projection_T(x_pooled)    # T日预测
        pred_T1 = self.output_projection_T1(x_pooled)  # T+1日预测

        return pred_T, pred_T1
```

使用：
```python
# 训练
pred_T, pred_T1 = model(boundary_conditions)
loss = criterion(pred_T, target_T) + criterion(pred_T1, target_T1)

# 特征提取
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

## 📊 构建60天历史序列

```python
# 存储60天的特征
all_features = {
    'attention_weights': [],
    'encoder_output': [],
    'residual_T': [],
    'residual_T1': []
}

# 逐天提取
for day in range(60):
    bc = historical_data[day:day+1]

    (pred_T, pred_T1), features = model.forward_with_features(
        bc, return_attention=True, return_encoder_output=True
    )

    # 计算残差
    residual_T = true_values_T[day:day+1] - pred_T
    residual_T1 = true_values_T1[day:day+1] - pred_T1

    # 保存
    all_features['attention_weights'].append(features['attention_weights'])
    all_features['encoder_output'].append(features['encoder_output'])
    all_features['residual_T'].append(residual_T)
    all_features['residual_T1'].append(residual_T1)

# 合并
attention_seq = torch.cat(all_features['attention_weights'], dim=0)
# shape: [60, num_layers, num_heads, num_sensors, num_sensors]

encoder_seq = torch.cat(all_features['encoder_output'], dim=0)
# shape: [60, num_sensors, d_model]

residual_T_seq = torch.cat(all_features['residual_T'], dim=0)
residual_T1_seq = torch.cat(all_features['residual_T1'], dim=0)
# shape: [60, 1]
```

---

## 🧮 降维并构建LSTM输入

```python
# 创建降维器（参考demo中的实现）
attention_extractor = SimpleAttentionExtractor(output_dim=10)
encoder_extractor = SimpleEncoderExtractor(d_model=128, output_dim=32)

# 逐天降维
lstm_input_list = []

for day in range(60):
    # 提取attention特征（10维）
    attn_feat = attention_extractor(
        all_features['attention_weights'][day],
        target_stock_idx=5
    )

    # 提取encoder特征（32维）
    enc_feat = encoder_extractor(
        all_features['encoder_output'][day],
        target_stock_idx=5
    )

    # 残差特征（2维）
    res_feat = torch.cat([
        all_features['residual_T'][day],
        all_features['residual_T1'][day]
    ], dim=-1)

    # 合并（10+32+2=44维）
    day_feat = torch.cat([attn_feat, enc_feat, res_feat], dim=-1)
    lstm_input_list.append(day_feat)

# 构建LSTM输入
lstm_input = torch.cat(lstm_input_list, dim=0)  # [60, 44]
lstm_input = lstm_input.unsqueeze(0)  # [1, 60, 44]

# 现在可以输入LSTM了！
lstm = nn.LSTM(input_size=44, hidden_size=64, num_layers=2)
output, (h_n, c_n) = lstm(lstm_input)
```

---

## 🔬 技术原理（简要）

### 为什么常规forward无法返回中间特征？

PyTorch的`nn.TransformerEncoder`只返回最终输出，不返回attention权重：

```python
# PyTorch源码（简化）
def forward(self, src):
    output = src
    for layer in self.layers:
        output = layer(output)  # ← 内部计算了attention，但没返回
    return output  # ← 只返回输出
```

### SpatialFeatureExtractor的解决方案

手动逐层执行，显式请求attention权重：

```python
def _forward_transformer_with_attention(self, x):
    attention_weights_list = []

    for layer in self.transformer.layers:
        # 手动调用attention，并要求返回权重
        attn_output, attn_weights = layer.self_attn(
            x, x, x,
            need_weights=True,              # ← 关键！
            average_attn_weights=False      # ← 返回每个head
        )

        attention_weights_list.append(attn_weights)

        # 手动执行residual、norm、FFN...
        x = residual + layer.dropout1(attn_output)
        x = layer.norm1(x)
        # ... (省略FFN部分)

    return x, torch.stack(attention_weights_list, dim=1)
```

详细技术说明请参考 `FEATURE_EXTRACTION_GUIDE.md`。

---

## 🏃 运行演示

```bash
# 确保安装了PyTorch
pip install torch

# 运行完整演示
python examples/extract_sst_internals_demo.py
```

演示输出：
```
================================================================================
SST内部特征提取完整演示
================================================================================

Step 1: 创建双输出SST模型
--------------------------------------------------------------------------------
模型参数量: 339,073

Step 2: 准备历史数据（60天）
--------------------------------------------------------------------------------
历史数据形状: torch.Size([60, 23])
真实值T形状: torch.Size([60, 1])
真实值T+1形状: torch.Size([60, 1])

Step 3: 对比两种推理模式
--------------------------------------------------------------------------------
【模式1】常规推理（仅用于预测）
  输入: torch.Size([1, 23])
  输出 pred_T: torch.Size([1, 1])
  输出 pred_T1: torch.Size([1, 1])
  ✗ 无法获取中间特征

【模式2】特征提取推理（用于分析和增强）
  输入: torch.Size([1, 23])
  输出 pred_T: torch.Size([1, 1])
  输出 pred_T1: torch.Size([1, 1])
  ✓ 中间特征:
    - embeddings: torch.Size([1, 23, 128])
    - attention_weights: torch.Size([1, 3, 8, 23, 23])
    - encoder_output: torch.Size([1, 23, 128])
    - pooled_features: torch.Size([1, 128])

  验证: 两种模式的预测是否一致?
    pred_T差异: 3.91e-06
    pred_T1差异: 3.91e-06

... (更多输出)

【LSTM输入】
  - 形状: torch.Size([1, 60, 44])
  - 说明: 60个时间步，每步44维压缩特征

✓ 演示完成！
```

---

## ✅ 核心要点

| 维度 | 常规推理 | 特征提取推理 |
|------|----------|--------------|
| **方法** | `model(x)` | `model.forward_with_features(x)` |
| **返回** | 仅预测 | 预测 + 中间特征 |
| **Attention** | ❌ | ✅ |
| **Encoder Output** | ❌ | ✅ |
| **用途** | 训练/生产 | 分析/增强 |

---

## 📚 参考

- **完整技术文档**: `docs/FEATURE_EXTRACTION_GUIDE.md`
- **可运行示例**: `examples/extract_sst_internals_demo.py`
- **核心模型**: `models/spatial_feature_extractor.py`
- **特征提取器**: `models/relationship_extractors.py`

---

## 💬 常见问题

**Q1: 为什么需要双输出（T日和T+1日）？**

A: 理论框架要求：
- T日预测 = 纯空间响应（只依赖空间关系）
- T+1日预测 = 空间响应 + 时序演化
- 差值 = 纯时序成分

这样可以分离空间和时序效应。

**Q2: 降维一定要用这些方法吗？**

A: 不一定。示例中的降维方法（SimpleAttentionExtractor等）只是参考实现。你可以：
- 使用PCA降维
- 使用Autoencoder
- 使用因子模型
- 或直接用原始特征（如果LSTM能处理）

**Q3: LSTM的输入一定要44维吗？**

A: 不一定。44维是示例中的配置（10注意力+32编码器+2残差）。你可以根据需要调整每部分的维度。

**Q4: 能不能只用残差，不用attention和encoder？**

A: 可以，但会丢失关系信息。理论框架强调：
- Attention捕捉"谁影响谁"
- Encoder捕捉"上下文嵌入"
- 残差捕捉"系统性偏差"

三者结合效果更好。

---

## 📝 下一步

1. **理解理论**: 阅读 `FEATURE_EXTRACTION_GUIDE.md`
2. **运行示例**: 执行 `extract_sst_internals_demo.py`
3. **实现Stage3**: 构建LSTM增强模型
4. **实验验证**: 对比SST vs SST+LSTM的性能

祝你成功！🎉
