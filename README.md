# Quant-Stock-Transformer

> ⚠️ **🚧 Under Active Development | 开发测试中 🚧**
>
> This is an experimental quantitative stock prediction framework. Code and documentation are being actively developed and refined.

---

## 📖 Overview | 项目简介

**Quant-Stock-Transformer** is a novel three-stage quantitative stock prediction framework that achieves resource savings through spatial-temporal separation.

基于空间-时序分离的三阶段量化股票预测框架，通过分离空间关系建模和时序演化建模，实现算力资源的高效利用。

### 🎯 Key Innovation | 核心创新

1. **Spatial-Temporal Separation | 空间-时序分离**
   - Stage 1: Static Sensor Transformer (SST) for spatial relationships
   - Stage 2: Internal feature extraction (attention + encoder + residuals)
   - Stage 3: Temporal models for time-series enhancement

2. **AI-Powered Stock Selection | AI驱动的股票选择**
   - LLM-based intelligent stock correlation analysis
   - Support for multiple markets (US, CN, HK, JP)
   - Automatic industry chain analysis

3. **Multi-Model Comparison | 多模型对比**
   - SST (baseline)
   - SST + iTransformer
   - SST + LSTM
   - SST + GRU

---

## 🚀 Quick Start | 快速开始

### 📦 Installation | 安装

```bash
# Clone the repository
git clone https://github.com/FTF1990/Quant-Stock-Transformer.git
cd Quant-Stock-Transformer

# Install dependencies
pip install -r requirements.txt
```

### 🎮 Usage | 使用方法

#### Option 1: Google Colab (Recommended | 推荐)

1. **Stock Selection Agent | 股票选择智能体**
   - Open `notebooks/stock_analysis_agent.ipynb` in Google Colab
   - Configure your LLM (Google AI / OpenAI / DeepSeek)
   - Run cells to generate stock list and fetch historical data

2. **Model Training Pipeline | 模型训练流程**
   - Open `notebooks/model_training_pipeline.ipynb` in Google Colab
   - Load the data from Step 1
   - Train and evaluate models

#### Option 2: Local Environment | 本地环境

```python
# Example: Using the SST model
from models.spatial_feature_extractor import SpatialFeatureExtractor

# Create model
model = SpatialFeatureExtractor(
    num_boundary_sensors=23,
    num_target_sensors=1,
    d_model=128,
    nhead=8,
    num_layers=3
)

# Extract internal features
predictions, features = model.forward_with_features(
    boundary_conditions,
    return_attention=True,
    return_encoder_output=True
)
```

---

## 📚 Documentation | 文档

### Core Documentation | 核心文档

- **[Feature Extraction Guide](docs/FEATURE_EXTRACTION_GUIDE.md)** - Complete technical guide for extracting SST internal features
- **[SST Internals README](docs/SST_INTERNALS_EXTRACTION_README.md)** - Quick start for feature extraction
- **[Demo Notebook](docs/sst_feature_extraction_demo.md)** - Complete demonstration of the system

### Technical Papers | 技术文档

- **Three-Stage Framework** - Detailed explanation of the spatial-temporal separation approach
- **SST Architecture** - Sensor Sequence Transformer design
- **Feature Engineering** - Attention weights and encoder output analysis

---

## 🗂️ Project Structure | 项目结构

```
Quant-Stock-Transformer/
├── models/                          # Core model implementations
│   ├── static_transformer.py        # SST base model
│   ├── spatial_feature_extractor.py # SST with feature extraction
│   ├── relationship_extractors.py   # Feature extractors
│   └── temporal_predictor.py        # Temporal models
├── notebooks/                       # Jupyter/Colab notebooks
│   ├── stock_analysis_agent.ipynb   # 🤖 AI stock selection agent
│   └── model_training_pipeline.ipynb # 🚀 Complete training pipeline
├── examples/                        # Example scripts
│   └── extract_sst_internals_demo.py # Feature extraction demo
├── docs/                            # Documentation
│   ├── FEATURE_EXTRACTION_GUIDE.md
│   └── SST_INTERNALS_EXTRACTION_README.md
└── README.md                        # This file
```

---

## 🤖 AI Stock Analysis Agent | 智能股票分析

### Features | 功能特性

- ✅ **Multi-LLM Support** - Google AI (Gemini), OpenAI, DeepSeek, Custom APIs
- ✅ **Multi-Market Coverage** - US, China A-shares, Hong Kong, Japan
- ✅ **Industry Chain Analysis** - Upstream/downstream/competitors/correlations
- ✅ **Automatic Data Fetching** - Historical data (hourly/daily) with market indices
- ✅ **Configurable Minimums** - Set minimum stocks per market

### Usage Example | 使用示例

```python
from notebooks.stock_analysis_agent import StockAnalysisAgent, LLMConfig

# Configure LLM
llm_config = LLMConfig(provider="google")  # or "openai", "deepseek"

# Create agent
agent = StockAnalysisAgent(llm_config)

# Analyze industry
result = agent.analyze_industry(
    industry="半导体",  # Semiconductor
    markets=["US", "CN", "HK", "JP"],
    min_stocks_per_market={"US": 8, "CN": 10, "HK": 5, "JP": 5}
)

# Save results
agent.save_results(
    json_path="selected_stocks.json",
    report_path="analysis_report.md"
)
```

---

## 📈 Data Fetching | 数据获取

### Supported Data Sources | 支持的数据源

- **A-shares (CN)**: AkShare - Open source, no API key required
- **US/HK/JP**: yfinance - Free Yahoo Finance API

### Features | 功能

- ✅ Hourly or daily data
- ✅ Market indices included
- ✅ OHLCV + Volume
- ✅ Automatic data cleaning
- ✅ Pickle format for fast loading

```python
from notebooks.stock_analysis_agent import StockDataFetcher

fetcher = StockDataFetcher()

historical_data = fetcher.fetch_historical_data(
    stocks_json=selected_stocks,
    start_date="2020-01-01",
    end_date="2024-12-31",
    interval="1d",  # "1h" for hourly
    include_market_index=True
)

fetcher.save_data("historical_data.pkl")
```

---

## 🧠 Model Training | 模型训练

### Stage 1: SST Training | SST训练

Train a dual-output SST that predicts both T-day and T+1-day returns:

```python
from examples.extract_sst_internals_demo import DualOutputSST

model = DualOutputSST(
    num_boundary_sensors=23,
    num_target_sensors=1,
    d_model=128,
    nhead=8,
    num_layers=3
)

# Train
pred_T, pred_T1 = model(boundary_conditions)
loss = criterion(pred_T, target_T) + criterion(pred_T1, target_T1)
```

### Stage 2: Feature Extraction | 特征提取

Extract internal features from trained SST:

```python
(pred_T, pred_T1), features = model.forward_with_features(
    boundary_conditions,
    return_attention=True,
    return_encoder_output=True
)

# features contains:
# - attention_weights: [batch, num_layers, num_heads, 23, 23]
# - encoder_output: [batch, 23, 128]
# - embeddings: [batch, 23, 128]
# - pooled_features: [batch, 128]

# Calculate residuals
residual_T = target_T - pred_T
residual_T1 = target_T1 - pred_T1
```

### Stage 3: Temporal Enhancement | 时序增强

Train temporal models using extracted features:

```python
# Prepare LSTM input from extracted features
lstm_input = build_sequence_features(
    attention_features,   # 10-dim
    encoder_features,     # 32-dim
    residual_features     # 2-dim
)  # Result: [batch, sequence_length, 44]

# Train LSTM
lstm = nn.LSTM(input_size=44, hidden_size=64, num_layers=2)
output, (h_n, c_n) = lstm(lstm_input)
```

---

## 📊 Model Evaluation | 模型评估

### Metrics | 评估指标

- **MSE** - Mean Squared Error
- **MAE** - Mean Absolute Error
- **Direction Accuracy** - Prediction direction correctness
- **Sharpe Ratio** - Risk-adjusted returns
- **Max Drawdown** - Maximum loss from peak

### Comparison | 模型对比

| Model | MSE | MAE | Direction Acc | Sharpe | Status |
|-------|-----|-----|---------------|--------|--------|
| SST (baseline) | - | - | - | - | ✅ Implemented |
| SST + iTransformer | - | - | - | - | 🚧 In Progress |
| SST + LSTM | - | - | - | - | 🚧 In Progress |
| SST + GRU | - | - | - | - | 🚧 In Progress |

*Note: Metrics will be updated after testing phase*

---

## 🛠️ Development Status | 开发状态

### ✅ Completed | 已完成

- [x] SST base model implementation
- [x] Spatial feature extractor with attention/encoder extraction
- [x] Dual-output SST (T and T+1 predictions)
- [x] Feature extraction demo
- [x] AI stock analysis agent
- [x] Multi-market data fetcher
- [x] Comprehensive documentation

### 🚧 In Progress | 进行中

- [ ] Complete training pipeline notebook
- [ ] Temporal models (iTransformer, LSTM, GRU)
- [ ] Feature dimension reduction
- [ ] Model evaluation and comparison
- [ ] Backtesting framework

### 📋 Planned | 计划中

- [ ] Real-time prediction API
- [ ] Web interface
- [ ] More temporal models (Informer, Autoformer)
- [ ] Ensemble methods
- [ ] Risk management module

---

## 🤝 Contributing | 贡献

Contributions are welcome! Please feel free to submit a Pull Request.

欢迎贡献！请随时提交Pull Request。

### Development Guidelines | 开发指南

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License | 许可证

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Contact | 联系方式

- **Issues**: [GitHub Issues](https://github.com/FTF1990/Quant-Stock-Transformer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/FTF1990/Quant-Stock-Transformer/discussions)

---

## 🙏 Acknowledgments | 致谢

- PyTorch team for the excellent deep learning framework
- AkShare for providing free A-share data access
- yfinance for Yahoo Finance data API
- Google AI, OpenAI, DeepSeek for LLM APIs

---

## ⚠️ Disclaimer | 免责声明

**This project is for research and educational purposes only. Not financial advice.**

**本项目仅供研究和教育目的使用，不构成投资建议。**

- Past performance does not guarantee future results
- Stock trading involves substantial risk of loss
- Always do your own research before investing
- The authors are not responsible for any financial losses

---

## 📈 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=FTF1990/Quant-Stock-Transformer&type=Date)](https://star-history.com/#FTF1990/Quant-Stock-Transformer&Date)

---

**Made with ❤️ by the Quant-Stock-Transformer Team**

**🚧 Active Development - Stay Tuned for Updates! | 积极开发中 - 敬请期待更新！🚧**
