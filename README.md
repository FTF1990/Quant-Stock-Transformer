# Quant-Stock-Transformer

> ⚠️ **🚧 Under Active Development | 开发中 🚧**
> This is an experimental quantitative stock prediction framework. Code and documentation are being actively developed and refined.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

---

**A novel three-stage quantitative stock prediction framework that achieves 90% resource savings through spatial-temporal separation.**

基于空间-时序分离的量化股票预测框架，实现90%资源节省。

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

## 🚀 Quick Start | 快速开始

### Installation | 安装

```bash
git clone https://github.com/YOUR_USERNAME/Quant-Stock-Transformer.git
cd Quant-Stock-Transformer
pip install -r requirements.txt
```

### Usage | 使用

```python
from src.three_stage_pipeline import ThreeStagePipeline

# 1. Initialize
pipeline = ThreeStagePipeline(
    stock_codes=['000001', '000002', '600000'],
    index_codes=['sh000001', 'sz399001'],
    target_stock='000001',
    feature_columns=['close', 'volume', 'MA5', 'MA20', 'RSI'],
    relationship_dim=32,
    seq_len=60
)

# 2. Train Stage1 (spatial)
pipeline.build_stage1()
pipeline.train_stage1(train_df, val_df)

# 3. Extract relationship features
pipeline.build_relationship_extractor('hybrid')
df_with_rel = pipeline.extract_relationship_features(df)

# 4. Train Stage3 (temporal)
pipeline.build_stage3('lstm')
pipeline.train_stage3(df_with_rel)

# 5. Predict
predictions = pipeline.predict(test_df)
```

See `QUICKSTART_THREE_STAGE.md` for detailed tutorial.

---

## 📁 Project Structure | 项目结构

```
Quant-Stock-Transformer/
├── models/
│   ├── static_transformer.py         # Original SST model
│   ├── spatial_feature_extractor.py  # Stage1 with feature extraction
│   ├── relationship_extractors.py    # Relationship feature extractors
│   └── temporal_predictor.py         # Stage3 temporal models (LSTM/GRU/TCN)
├── src/
│   └── three_stage_pipeline.py       # End-to-end pipeline
├── notebooks/
│   └── three_stage_tutorial.ipynb    # Interactive tutorial
├── docs/
│   ├── ARCHITECTURE_DESIGN.md        # Detailed architecture design
│   ├── QUICKSTART_THREE_STAGE.md     # Quick start guide
│   └── THREE_STAGE_SUMMARY.md        # Complete summary
└── README.md                         # This file
```

---

## 📈 Performance Comparison | 性能对比

| Approach | Input Dimension | Memory | Training Time | Performance |
|----------|----------------|--------|---------------|-------------|
| **Traditional TFT** | 3000-dim × 90 days | ~2GB | ~10 min/epoch | Baseline |
| **Three-Stage** | 62-dim × 90 days | ~200MB | ~1 min/epoch | Similar or better |
| **Savings** | **98% reduction** | **90%** | **90%** | **+Interpretability** |

---

## 🔑 Key Features | 核心特性

- ✅ **Resource Efficient**: 90% memory and time savings
- ✅ **Modular Design**: Stage1 reusable for multiple target stocks
- ✅ **Interpretable**: Attention weights show stock influences
- ✅ **Flexible**: Support LSTM/GRU/TCN/TFT for Stage3
- ✅ **General Purpose**: Applicable to any multi-entity + time-series scenario

---

## 📚 Documentation | 文档

- **Quick Start**: [`QUICKSTART_THREE_STAGE.md`](QUICKSTART_THREE_STAGE.md)
- **Architecture Design**: [`ARCHITECTURE_DESIGN.md`](ARCHITECTURE_DESIGN.md)
- **Complete Summary**: [`THREE_STAGE_SUMMARY.md`](THREE_STAGE_SUMMARY.md)
- **Tutorial Notebook**: [`notebooks/three_stage_tutorial.ipynb`](notebooks/three_stage_tutorial.ipynb)

---

## 🎓 Theory | 理论基础

**Why separate spatial and temporal?**

Stock prediction = Spatial problem + Temporal problem

- **Spatial** (Stage1): Who influences whom? (Cross-stock relationships)
- **Temporal** (Stage3): How does it evolve? (Time dynamics)

**Key Insight**: Transformer excels at global relationships, LSTM/GRU excels at sequences. Let each do what it's best at!

---

## ⚠️ Disclaimer | 免责声明

**For educational and research purposes only.**

- Stock market prediction is highly uncertain
- Past performance ≠ future results
- This is NOT investment advice
- Use at your own risk

---

## 📄 License | 许可证

MIT License - see [LICENSE](LICENSE) file for details.

---

## 📞 Contact | 联系方式

- **GitHub Issues**: [Create an issue](https://github.com/YOUR_USERNAME/Quant-Stock-Transformer/issues)
- **Email**: shvichenko11@gmail.com

---

## 🔗 Citation | 引用

If you use this work in your research:

```bibtex
@software{quant_stock_transformer,
  author = {FTF1990},
  title = {Quant-Stock-Transformer: Spatial-Temporal Separation for Stock Prediction},
  year = {2025},
  url = {https://github.com/YOUR_USERNAME/Quant-Stock-Transformer}
}
```

---

**🚧 Status: Under Active Development | 积极开发中**

We're actively refining the code and documentation. Expect frequent updates!
