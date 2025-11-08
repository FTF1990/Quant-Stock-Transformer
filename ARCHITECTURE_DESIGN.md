# 三阶段混合架构设计方案

## 架构概述

### Stage1: 空间特征提取器 (Cross-Sectional Feature Extractor)
**目标**: 提取股票间的关系特征，学习市场结构

**输入**:
- 多股票横截面数据: `[batch_size, num_stocks, num_features]`
- 每个时间点t，所有股票的特征快照
- 包含: OHLCV, 技术指标, 大盘指数

**处理流程**:
```
多股票数据 → Embedding → Positional Encoding
           → Multi-Head Attention (学习股票间关系)
           → Transformer Layers
           → 提取关系特征
```

**输出**:
1. **关系Embedding**: `[batch_size, num_stocks, d_model]` - 每只股票的关系向量
2. **Attention权重**: `[batch_size, num_heads, num_stocks, num_stocks]` - 股票间影响力矩阵
3. **聚合特征**: `[batch_size, relationship_dim]` - 压缩的市场关系特征
4. **Baseline预测**: 初步的价格/收益预测

**关键设计**:
- 使用Pooling提取目标股票的关系表示
- Attention可视化：哪些股票影响目标股票
- 输出维度可配置 (建议16-64维)

---

### Stage2: 残差提升 (Residual Boost) - 可选
**目标**: 改进Stage1预测精度

**输入**: Stage1的预测残差
**输出**: 残差修正
**结果**: Stage1 + Stage2 = 更好的Baseline

---

### Stage3: 时序预测器 (Temporal Predictor)
**目标**: 利用时间序列模型，结合关系特征做最终预测

**模型选择**:

#### 方案A: TFT (Temporal Fusion Transformer)
**优点**:
- 多horizon预测
- 可解释性强 (variable importance)
- 处理静态/动态协变量

**缺点**:
- 资源占用大 (长序列时)

**适用场景**:
- 中短序列 (lookback < 60天)
- 特征数适中 (< 50维)

#### 方案B: LSTM/GRU
**优点**:
- 轻量级，训练快
- 适合长序列

**缺点**:
- 可解释性较弱

**适用场景**:
- 长序列 (lookback > 60天)
- 资源受限

#### 方案C: Temporal CNN (TCN)
**优点**:
- 并行计算，速度快
- 可控感受野

**适用场景**:
- 需要快速推理

**输入特征组合**:
```python
# 目标股票的时序特征
target_stock_features = [
    'close', 'volume', 'MA5', 'MA20', 'RSI', ...  # 原始技术指标
]

# 从Stage1提取的关系特征 (每个时间步)
relationship_features = [
    'market_embedding_0',      # 市场关系向量
    'market_embedding_1',
    ...,
    'market_embedding_15',
    'attention_to_index',       # 与大盘的attention权重
    'attention_to_sector',      # 与板块的attention权重
]

# 时间特征
temporal_features = [
    'day_of_week', 'day_of_month', 'month', 'is_month_end', ...
]

# 合并输入
input_shape = [batch_size, sequence_length,
               len(target_stock_features) + len(relationship_features) + len(temporal_features)]
```

**输出**:
- 最终的价格/收益率预测
- 置信区间 (如果使用TFT)

---

## 数据流示例

### 时间点t的处理流程

```python
# 1. Stage1: 提取t时刻的市场关系
all_stocks_data_t = [stock1_features_t, stock2_features_t, ..., index_features_t]
relationship_embedding_t = stage1_model.extract_features(all_stocks_data_t)
# Output: [16-dim vector] 包含市场结构信息

# 2. 构建Stage3输入序列
# 对于目标股票 (如000001)，收集历史60天的数据
target_sequence = []
for i in range(t-60, t):
    target_stock_features_i = get_stock_features(stock='000001', time=i)
    relationship_features_i = stage1_model.extract_features(all_stocks_data_i)
    combined_features_i = concat(target_stock_features_i, relationship_features_i)
    target_sequence.append(combined_features_i)

# 3. Stage3: 时序预测
prediction = stage3_model(target_sequence)
```

---

## 关系特征提取的三种方案

### 方案1: 使用Attention权重作为关系特征
**描述**: 提取目标股票对其他股票/大盘的注意力分数

**优点**:
- 直观，可解释
- 计算开销小

**实现**:
```python
class RelationshipExtractorV1:
    def extract(self, model, data, target_stock_idx):
        # 获取attention权重 [batch, heads, stocks, stocks]
        attention_weights = model.get_attention_weights(data)

        # 提取目标股票的attention分布
        target_attention = attention_weights[:, :, target_stock_idx, :]
        # [batch, heads, num_stocks]

        # 平均所有heads
        avg_attention = target_attention.mean(dim=1)
        # [batch, num_stocks]

        # 分组聚合
        attention_to_index = avg_attention[:, index_indices].mean(dim=1)
        attention_to_sector = avg_attention[:, sector_indices].mean(dim=1)

        return concat([avg_attention, attention_to_index, attention_to_sector])
```

**输出维度**: num_stocks + 2 (假设20只股票 → 22维)

---

### 方案2: 使用Transformer中间层Embedding
**描述**: 提取目标股票经过Transformer编码后的向量

**优点**:
- 信息丰富，包含非线性关系
- 维度可控 (d_model)

**实现**:
```python
class RelationshipExtractorV2:
    def extract(self, model, data, target_stock_idx):
        # 获取encoder输出 [batch, num_stocks, d_model]
        encoder_output = model.get_encoder_output(data)

        # 提取目标股票的embedding
        target_embedding = encoder_output[:, target_stock_idx, :]
        # [batch, d_model] 例如 [batch, 128]

        # 可选: 降维到更小的维度
        compressed = self.projection_layer(target_embedding)
        # [batch, 16] 或 [batch, 32]

        return compressed
```

**输出维度**: 16-64维 (可配置)

---

### 方案3: 对比学习的关系特征 (高级)
**描述**: 训练专门的关系特征提取器

**优点**:
- 学习更好的关系表示
- 可以加入领域知识 (板块、行业)

**实现**:
```python
class RelationshipExtractorV3:
    """使用对比学习训练关系编码器"""

    def __init__(self):
        self.encoder = TransformerEncoder()
        self.projection = nn.Linear(d_model, relationship_dim)

    def contrastive_loss(self, anchor, positive, negative):
        """
        anchor: 目标股票
        positive: 同板块/相关股票
        negative: 不相关股票
        """
        # 拉近相关股票的表示，推远不相关股票
        ...

    def extract(self, data, target_stock_idx):
        embeddings = self.encoder(data)
        target_emb = embeddings[:, target_stock_idx, :]

        # 计算与其他股票的相似度
        similarities = cosine_similarity(target_emb, embeddings)

        # 投影到低维空间
        relationship_features = self.projection(target_emb)

        return concat([relationship_features, similarities])
```

**输出维度**: relationship_dim + num_stocks

---

## 资源占用对比

### TFT资源消耗分析

**原始方案 (不使用Stage1)**:
```
输入特征 = 20股票 × 30指标 = 600维
序列长度 = 90天
→ 内存占用: ~2GB (单批次)
→ 训练时间: ~10分钟/epoch
```

**新方案 (使用Stage1关系特征)**:
```
输入特征 = 1目标股票 × 30指标 + 16维关系特征 = 46维
序列长度 = 90天
→ 内存占用: ~200MB (单批次)
→ 训练时间: ~1分钟/epoch
```

**资源节省**: ~90% 内存, ~90% 训练时间

---

## 推荐配置

### 小规模 (5-10只股票, 日线数据)
- **Stage1**: d_model=64, nhead=4, num_layers=2
- **关系特征**: 方案2, 输出16维
- **Stage3**: GRU/LSTM, hidden=64, layers=2
- **序列长度**: 60天
- **预测horizon**: 1-5天

### 中等规模 (10-30只股票, 日线数据)
- **Stage1**: d_model=128, nhead=8, num_layers=3
- **关系特征**: 方案2, 输出32维
- **Stage3**: TFT, hidden=64, attention_heads=4
- **序列长度**: 90天
- **预测horizon**: 1-10天

### 大规模 (30+只股票, 分钟线数据)
- **Stage1**: d_model=256, nhead=8, num_layers=4
- **关系特征**: 方案1+方案2组合, 输出64维
- **Stage3**: TCN或轻量LSTM
- **序列长度**: 根据内存调整
- **预测horizon**: 根据需求

---

## 训练策略

### 阶段1: 预训练Stage1
```python
# 训练目标: 预测下一时刻的价格/收益
# 使用现有的两阶段训练 (Base + Residual Boost)
python gradio_sensor_transformer_app.py
# → 得到 stage1_model.pth, stage2_model.pth
```

### 阶段2: 提取关系特征
```python
# 对整个训练集，提取每个时间点的关系特征
# 保存为新的特征列
relationship_features = extract_relationship_features(
    stage1_model,
    all_stocks_data,
    target_stock='000001'
)
# → 生成 data_with_relationships.csv
```

### 阶段3: 训练时序模型
```python
# 使用 原始特征 + 关系特征 训练TFT/LSTM
tft_model.fit(
    target_stock_features + relationship_features,
    labels
)
# → 得到 stage3_tft_model.pth
```

### 推理流程
```python
# 1. 获取实时市场数据
current_market_data = fetch_all_stocks()

# 2. Stage1提取关系特征
relationship_emb = stage1_model.extract_features(current_market_data)

# 3. 组合特征序列 (历史60天)
input_sequence = build_sequence(
    target_stock_history,
    relationship_history  # 需要保存历史关系特征
)

# 4. Stage3预测
prediction = stage3_model(input_sequence)
```

---

## 可选优化

### 1. 多任务学习
在Stage1同时训练:
- 任务1: 预测价格
- 任务2: 预测股票相关性
- 任务3: 板块分类

### 2. 动态关系建模
根据市场状态调整关系特征权重:
```python
# 牛市: 板块轮动明显，增加板块关系权重
# 熊市: 跟随大盘，增加指数关系权重
market_regime = detect_market_regime()
relationship_weights = adjust_weights(market_regime)
```

### 3. 增量更新
每日只更新关系特征，无需重新训练Stage1:
```python
# 快速推理模式
with torch.no_grad():
    relationship_features = stage1_model.extract_features(today_data)
```

---

## 文件结构

```
models/
├── spatial_feature_extractor.py    # Stage1: 关系特征提取
├── relationship_extractors.py      # 三种关系特征提取方案
├── temporal_predictor_tft.py       # Stage3: TFT实现
├── temporal_predictor_lstm.py      # Stage3: LSTM实现
└── temporal_predictor_tcn.py       # Stage3: TCN实现

src/
├── feature_engineering.py          # 合并原始+关系特征
├── stage1_trainer.py               # Stage1训练器
├── stage3_trainer.py               # Stage3训练器
└── pipeline.py                     # 完整训练流程

notebooks/
├── 01_stage1_training.ipynb        # Stage1训练教程
├── 02_relationship_extraction.ipynb # 关系特征提取
└── 03_stage3_training.ipynb        # Stage3训练教程
```

---

## 总结

### 核心优势
1. ✅ **资源高效**: 降维后大幅减少TFT输入维度
2. ✅ **信息丰富**: 关系特征捕获市场结构
3. ✅ **模块化**: 三阶段独立训练，灵活组合
4. ✅ **可解释**: Attention权重可视化股票影响力

### 下一步
1. 实现关系特征提取器 (推荐从方案2开始)
2. 修改现有Stage1，添加特征提取接口
3. 实现轻量级Stage3 (先用LSTM验证可行性)
4. 集成TFT (如果资源允许)
