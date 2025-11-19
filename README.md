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
   - Stage 1: Dual-output SST for spatial relationships (T-day & T+1-day)
   - Stage 2: Internal feature extraction (attention + encoder + residuals)
   - Stage 3: Temporal models for time-series enhancement (LSTM/GRU/TCN)

2. **Complete End-to-End Pipeline | 完整端到端流程**
   - Stock selection JSON import
   - Intelligent multi-market data fetching with batching
   - Automated preprocessing and feature engineering
   - Multi-model training and comparison
   - Comprehensive evaluation metrics

3. **Dual Usage Modes | 双使用模式**
   - **CLI**: Full-featured command-line pipeline
   - **UI**: Gradio-based visual interface with 7-step workflow

4. **Multi-Model Comparison | 多模型对比**
   - SST (baseline with dual outputs)
   - SST + LSTM (with Attention)
   - SST + GRU (lightweight)
   - SST + TCN (temporal convolution)

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

#### ✨ Option 1: Gradio Visual UI (Recommended | 推荐)

Launch the interactive web interface with 7-step visual workflow:

```bash
python gradio_pipeline_ui.py
```

Then open your browser at `http://localhost:7860`

**7-Step Workflow**:
1. 📋 **Load Stock JSON** - Upload your stock selection file
2. 📊 **Fetch Data** - Intelligent batch data fetching (US/CN/HK/JP markets)
3. 🔄 **Preprocess** - Calculate returns and split datasets
4. 🧠 **Train SST** - Dual-output Transformer model
5. 🔍 **Extract Features** - Attention weights, encoder outputs, residuals
6. ⏰ **Train Temporal** - LSTM/GRU/TCN models (choose any)
7. 📈 **Evaluate** - Compare all models with metrics and charts

**Features**:
- Real-time progress tracking
- Interactive parameter configuration
- Rich visualizations (training curves, feature distributions, performance comparisons)
- No command-line required

See **[UI_USAGE.md](UI_USAGE.md)** for detailed usage guide.

---

#### 🖥️ Option 2: Command-Line Pipeline

Run the complete training pipeline programmatically:

```bash
# Basic usage
python complete_training_pipeline.py \
    --stocks_json data/demo.json \
    --target_market CN \
    --target_stock 600519

# Full parameters
python complete_training_pipeline.py \
    --stocks_json data/demo.json \
    --target_market CN \
    --target_stock 600519 \
    --start_date 2020-01-01 \
    --end_date 2024-12-31 \
    --fetch_data \
    --sst_epochs 50 \
    --temporal_epochs 100 \
    --seq_len 60 \
    --device cuda
```

**Key Parameters**:
- `--stocks_json`: Path to stock selection JSON
- `--target_market`: Target market (US/CN/HK/JP)
- `--target_stock`: Stock symbol to predict
- `--fetch_data`: Re-fetch historical data (vs. using cache)
- `--sst_epochs`: SST training epochs (default: 50)
- `--temporal_epochs`: Temporal model epochs (default: 100)
- `--device`: cpu or cuda

See **[PIPELINE_FLOW_CONFIRMATION.md](PIPELINE_FLOW_CONFIRMATION.md)** for complete flow verification.

---

#### 📋 Option 3: Python API

Use individual components in your code:

```python
from complete_training_pipeline import (
    StockDataFetcher,
    StockDataProcessor,
    DualOutputSST,
    ModelTrainer,
    ModelEvaluator
)

# Fetch data
fetcher = StockDataFetcher()
historical_data = fetcher.fetch_historical_data(
    stocks_json=your_stocks,
    start_date="2020-01-01",
    end_date="2024-12-31"
)

# Preprocess
processor = StockDataProcessor(
    historical_data=historical_data,
    target_market="CN",
    target_stock="600519"
)
X, y_T, y_T1, dates = processor.prepare_training_data()

# Train SST
sst_model = DualOutputSST(
    num_boundary_sensors=X.shape[1],
    num_target_sensors=1,
    d_model=128,
    nhead=8,
    num_layers=3
)

trainer = ModelTrainer(device='cuda')
history = trainer.train_sst(sst_model, X_train, y_T_train, y_T1_train, ...)

# Evaluate
evaluator = ModelEvaluator(device='cuda')
metrics = evaluator.evaluate_sst(sst_model, X_test, y_T_test, y_T1_test)
```

---

## 📚 Documentation | 文档

### Core Documentation | 核心文档

- **[UI Usage Guide](UI_USAGE.md)** - Complete 7-step visual UI guide
- **[Pipeline Flow Confirmation](PIPELINE_FLOW_CONFIRMATION.md)** - End-to-end flow verification
- **[Feature Extraction Guide](docs/FEATURE_EXTRACTION_GUIDE.md)** - Technical guide for SST features
- **[SST Internals README](docs/SST_INTERNALS_EXTRACTION_README.md)** - Quick start for feature extraction

### Example Data | 示例数据

- **[data/demo.json](data/demo.json)** - Sample stock selection (28 stocks across 4 markets)

---

## 🗂️ Project Structure | 项目结构

```
Quant-Stock-Transformer/
├── models/                          # Core model implementations
│   ├── static_transformer.py        # SST base model
│   ├── spatial_feature_extractor.py # SST with feature extraction
│   ├── relationship_extractors.py   # Attention/embedding extractors
│   └── temporal_predictor.py        # LSTM/GRU/TCN temporal models
├── data/
│   ├── demo.json                    # 📋 Sample stock selection (28 stocks)
│   └── default_signals_config.json  # Signal configuration
├── complete_training_pipeline.py    # 🚀 Complete CLI training pipeline (1139 lines)
├── gradio_pipeline_ui.py            # 🎨 Gradio visual UI (1173 lines)
├── notebooks/                       # Jupyter/Colab notebooks
│   ├── stock_analysis_agent.ipynb   # 🤖 LLM-based stock analysis (optional)
│   └── model_training_pipeline.ipynb # Model training reference
├── examples/                        # Example scripts
│   └── extract_sst_internals_demo.py # Feature extraction demo
├── docs/                            # Documentation
│   ├── FEATURE_EXTRACTION_GUIDE.md
│   ├── SST_INTERNALS_EXTRACTION_README.md
│   └── sst_feature_extraction_demo.md
├── UI_USAGE.md                      # 📖 Gradio UI usage guide
├── PIPELINE_FLOW_CONFIRMATION.md    # ✅ Flow verification doc
└── README.md                        # This file
```

---

## 📊 Stock Selection | 股票选择

### Using Claude AI Agent (Recommended | 推荐)

Generate your stock selection JSON using Claude AI:

1. Open Claude (claude.ai)
2. Describe your stock selection strategy
3. Ask Claude to generate a JSON file in the required format
4. Save the JSON and use it with the pipeline

**Required JSON Format**:
```json
{
  "US": [
    {"symbol": "NVDA", "name": "NVIDIA", "reason": "...", "category": "..."}
  ],
  "CN": [
    {"symbol": "600519", "name": "贵州茅台", "reason": "...", "category": "..."}
  ],
  "HK": [...],
  "JP": [...]
}
```

### Using Demo Data | 使用示例数据

Start with the provided demo.json:

```bash
# 28 stocks across 4 markets
data/demo.json
  ├── US: 8 stocks (NVDA, AMD, INTC, TSM, ASML, QCOM, AVGO, MU)
  ├── CN: 10 stocks (贵州茅台, 招商银行, etc.)
  ├── HK: 5 stocks (腾讯, 阿里巴巴, etc.)
  └── JP: 5 stocks (Sony, 京瓷, etc.)
```

### Optional: LLM-Powered Analysis | LLM驱动分析（可选）

For advanced users, use the notebook-based stock analysis agent:

- `notebooks/stock_analysis_agent.ipynb` - Industry chain analysis with LLM
- Supports: Google AI (Gemini), OpenAI, DeepSeek
- Multi-market coverage: US, CN, HK, JP
- Automatic data fetching

---

## 📈 Data Fetching | 数据获取

### Intelligent Batch Fetching | 智能分批抓取

**Features**:
- ✅ **Multi-Source Support**
  - A-shares (CN): AkShare (free, no API key)
  - US/HK/JP: yfinance (free Yahoo Finance API)
- ✅ **Smart Batching** - Avoid API rate limits
  - Configurable batch size (default: 5 stocks/batch)
  - Configurable delays (default: 2s between batches)
- ✅ **Auto-Retry** - Handles network errors gracefully
- ✅ **Progress Tracking** - Real-time progress display
- ✅ **Market Indices** - Includes S&P 500, 上证指数, 恒生指数, 日经225

**Example**:
```python
from complete_training_pipeline import StockDataFetcher

fetcher = StockDataFetcher()
historical_data = fetcher.fetch_historical_data(
    stocks_json=my_stocks,
    start_date="2020-01-01",
    end_date="2024-12-31",
    interval="1d",                    # "1h" for hourly data
    include_market_index=True,
    batch_size=5,                     # 5 stocks per batch
    delay_between_batches=2.0,        # 2 seconds between batches
    delay_between_stocks=0.5          # 0.5 seconds between stocks
)

fetcher.save_data("historical_data.pkl")
```

**Data Fields**:
- Open, High, Low, Close
- Volume
- Date index

---

## 🧠 Model Training | 模型训练

### Complete 3-Stage Pipeline | 完整三阶段流程

**Stage 1: Dual-Output SST**
- Simultaneously predicts T-day and T+1-day returns
- Transformer encoder (8 heads, 3 layers, 128 hidden dim)
- Global average pooling
- Dual output heads

```python
from complete_training_pipeline import DualOutputSST

model = DualOutputSST(
    num_boundary_sensors=num_features,
    num_target_sensors=1,
    d_model=128,
    nhead=8,
    num_layers=3,
    enable_feature_extraction=True
)

# Returns both T and T+1 predictions
pred_T, pred_T1 = model(boundary_conditions)
```

**Stage 2: Feature Extraction**
- Encoder outputs: [batch, sensors, 128]
- Attention weights: [batch, layers, heads, sensors, sensors]
- Pooled features: [batch, 128]
- Residuals: actual - predicted

```python
# Extract features
(pred_T, pred_T1), features = model.forward_with_features(
    boundary_conditions,
    return_attention=True,
    return_encoder_output=True
)

encoder_output = features['encoder_output']
attention_weights = features['attention_weights']
pooled_features = features['pooled_features']

# Calculate residuals
residual_T = target_T - pred_T
residual_T1 = target_T1 - pred_T1
```

**Stage 3: Temporal Models**

Train time-series models using SST features:

```python
from complete_training_pipeline import (
    LSTMTemporalPredictor,
    GRUTemporalPredictor,
    TCNTemporalPredictor
)

# LSTM with Attention
lstm_model = LSTMTemporalPredictor(
    input_dim=num_features + relationship_dim,
    hidden_dim=128,
    num_layers=2,
    output_dim=1,
    use_attention=True
)

# GRU (lightweight)
gru_model = GRUTemporalPredictor(
    input_dim=num_features + relationship_dim,
    hidden_dim=128,
    num_layers=2,
    output_dim=1
)

# TCN (parallel)
tcn_model = TCNTemporalPredictor(
    input_dim=num_features + relationship_dim,
    num_channels=[64, 128, 128, 64],
    output_dim=1
)
```

---

## 📊 Model Evaluation | 模型评估

### Metrics | 评估指标

- ✅ **MSE** (Mean Squared Error) - Lower is better
- ✅ **MAE** (Mean Absolute Error) - Lower is better
- ✅ **Direction Accuracy** - Percentage of correct up/down predictions
- ✅ **Sharpe Ratio** - Risk-adjusted returns (annualized)

### Model Comparison | 模型对比

| Model | Status | Parameters | Features |
|-------|--------|------------|----------|
| SST (baseline) | ✅ Implemented | ~500K | Dual outputs (T + T+1) |
| SST + LSTM | ✅ Implemented | ~600K | Attention mechanism |
| SST + GRU | ✅ Implemented | ~550K | Lightweight version |
| SST + TCN | ✅ Implemented | ~580K | Parallel computation |

**Evaluation Output**:
```
Model    MSE       MAE       Direction_Acc  Sharpe_Ratio
SST      0.001234  0.025678  52.34%         0.4521
LSTM     0.001156  0.024532  54.56%         0.5234
GRU      0.001189  0.024789  53.89%         0.5123
TCN      0.001201  0.025012  53.12%         0.4987
```

*Note: Example metrics - actual values depend on data and training*

---

## 🎨 Gradio UI Features | UI功能特性

### Visual Training Pipeline | 可视化训练流程

**7-Step Interactive Workflow**:

1. **📋 Load JSON** - Upload & visualize stock lists
   - Stock count statistics
   - Market distribution pie chart
   - Detailed stock table

2. **📊 Fetch Data** - Intelligent batch data fetching
   - Date range configuration
   - Batch size & delay settings
   - Real-time progress bar
   - Data statistics table

3. **🔄 Preprocess** - Data preparation
   - Return calculation (T & T+1)
   - Dataset split (70/15/15)
   - Return distribution plots

4. **🧠 Train SST** - Transformer training
   - Epoch/batch/LR sliders
   - Real-time training curves
   - Loss breakdown (T vs T+1)

5. **🔍 Extract Features** - Feature visualization
   - Feature distribution plots
   - Residual analysis
   - Feature heatmaps

6. **⏰ Train Temporal** - Time-series models
   - Model type selector (LSTM/GRU/TCN)
   - Sequence length configuration
   - Training curve display

7. **📈 Evaluate** - Performance comparison
   - Metrics comparison table
   - Performance bar charts
   - Best model highlighting

**Visualizations**:
- Training loss curves
- Feature distributions
- Performance comparison charts
- Market distribution plots
- Return histograms

---

## 🛠️ Development Status | 开发状态

### ✅ Completed | 已完成

- [x] SST base model with dual outputs (T + T+1)
- [x] Spatial feature extractor with attention/encoder extraction
- [x] Complete training pipeline (CLI)
- [x] Gradio visual UI (7-step workflow)
- [x] Temporal models (LSTM, GRU, TCN)
- [x] Multi-market data fetcher with smart batching
- [x] Model evaluation and comparison
- [x] Comprehensive documentation
- [x] Demo stock selection (28 stocks)

### 🚧 In Progress | 进行中

- [ ] Advanced feature engineering
- [ ] Hyperparameter optimization
- [ ] Backtesting framework
- [ ] Model ensemble methods

### 📋 Planned | 计划中

- [ ] Real-time prediction API
- [ ] More temporal models (Informer, Autoformer)
- [ ] Risk management module
- [ ] Portfolio optimization
- [ ] Multi-target prediction (volume, volatility)

---

## 💡 Usage Tips | 使用技巧

### For Beginners | 新手建议

1. Start with the Gradio UI (`python gradio_pipeline_ui.py`)
2. Use the demo.json file for initial testing
3. Try small epochs first (SST: 20, Temporal: 30)
4. Use CPU for testing, GPU for production training

### For Advanced Users | 进阶用户

1. Generate custom stock selections with Claude AI
2. Experiment with hyperparameters
3. Try different markets and date ranges
4. Analyze feature importance from SST
5. Implement custom temporal models

### Performance Optimization | 性能优化

**Training Speed**:
- Use GPU (`--device cuda`)
- Increase batch size (if memory allows)
- Use GRU instead of LSTM for faster training
- Use TCN for fastest inference

**Data Fetching**:
- Use cached data (`historical_data.pkl`) when possible
- Adjust batch size and delays based on network
- Fetch data overnight for large stock lists

---

## 🐛 Troubleshooting | 常见问题

### Data Fetching Issues

**Problem**: API rate limit errors
**Solution**: Reduce batch size, increase delays

**Problem**: Stock symbol not found
**Solution**: Check symbol format (US: AAPL, CN: 600519, HK: 00700, JP: 6758.T)

### Training Issues

**Problem**: Out of memory
**Solution**: Reduce batch size, use smaller model, reduce sequence length

**Problem**: Slow training
**Solution**: Use GPU, increase batch size, reduce epochs for testing

### Model Performance

**Problem**: Low accuracy
**Solution**: More training epochs, different hyperparameters, more data, better stock selection

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
- Gradio team for the amazing UI framework
- Claude AI for intelligent code assistance

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
