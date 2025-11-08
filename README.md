# Quant-Stock-Transformer

> ⚠️ **🚧 Under Active Development | 开发中 🚧**
> This is an experimental quantitative stock prediction framework. Code and documentation are being actively developed and refined.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

---

**A novel three-stage quantitative stock prediction framework that achieves resource savings through spatial-temporal separation.**

基于空间-时序分离的量化股票预测框架，实现算力资源节省。

---

## 🎯 Core Idea | 核心思路

### The Problem | 问题

Traditional approach: Directly use TFT to process all stocks' time-series data
```
100 stocks × 30 features × 90 days = 270,000 data points
→ Memory: ~2GB, Training: ~10 min/epoch
→ Resource intensive! 资源密集！
```

### Our Solution | 我们的方案

**Separate spatial (cross-stock) and temporal modeling:**

```
┌─────────────────────────────────────────────────────────┐
│ Stage1: Spatial Feature Extractor (Transformer)         │
│  Input:  Multi-stock cross-section (100 stocks)         │
│  Learn:  Stock relationships, sector effects, index     │
│  Output: Relationship features (32-dim) ← Dimension     │
│          reduction! 降维！                               │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ Stage3: Temporal Predictor (LSTM/GRU/TCN)               │
│  Input:  Target stock + relationship features           │
│          (30 + 32 = 62 dims)                            │
│  Learn:  Temporal dynamics, trends                      │
│  Output: Final prediction                               │
└─────────────────────────────────────────────────────────┘

Result: 62-dim × 90 days instead of 3000-dim × 90 days
→ Memory: ~200MB (save 90%), Training: ~1 min/epoch (10x faster)
```

---

## 💡 Why This Works | 为什么有效

1. **Dimension Reduction | 降维**
   - From 3000 dims (all stocks) → 32 dims (relationship features)
   - Compression ratio: ~1%

2. **Information Preservation | 保留信息**
   - Relationship features capture market structure
   - Attention mechanism learns "who affects whom"

3. **Model Specialization | 模型专精**
   - Transformer: Excellent at spatial relationships
   - LSTM/GRU: Excellent at temporal sequences
   - Each does what it's best at | 术业有专攻

---

## 📊 Architecture | 架构

### Stage1: Cross-Stock Relationship Learning
```python
# At time t, snapshot of all stocks (cross-section)
Input: [Stock1_features, Stock2_features, ..., Stock100_features, Index_features]
       Shape: [batch, 3090-dim]  # 103 stocks × 30 features

↓ Transformer (Spatial attention)

Output: Relationship embedding for target stock
        Shape: [batch, 32-dim]
```

**What does it learn? | 学什么？**
- Which stocks influence the target stock?
- How strong is the index correlation?
- Sector rotation signals?

### Stage3: Temporal Prediction
```python
# Combine target stock features + relationship features
for each day in [Day1, Day2, ..., Day60]:
    features[day] = concat([
        target_stock_features[day],  # 30-dim
        relationship_features[day]    # 32-dim (from Stage1)
    ])  # Total: 62-dim

↓ LSTM/GRU/TCN

Output: Future return prediction
```

---


**🚧 Status: Under Active Development | 积极开发中**

We're actively refining the code and documentation. Expect frequent updates!
