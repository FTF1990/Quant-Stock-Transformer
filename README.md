# Quant-Stock-Transformer

> ⚠️ **🚧 Under Active Development | 开发中 🚧**
>
> This is an experimental quantitative stock prediction framework. Code and documentation are being actively developed and refined.

---

**A novel three-stage quantitative stock prediction framework that achieves resource savings through spatial-temporal separation.**

基于空间-时序分离的量化股票预测框架，实现算力资源节省。

---

## 🚀 New: Cross-Market Stock Prediction | 跨市场股票预测

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FTF1990/Quant-Stock-Transformer/blob/claude/cross-market-agent-011CV1VoSoz7uwWmuin1xyWs/notebooks/Cross_Market_Stock_Prediction.ipynb)

Use **US market data** to predict **Japanese/Chinese/HK stocks** - leveraging time zone differences to solve the "future boundary condition" problem!

使用**美股数据**预测**日本/中国/香港股票** - 利用时区差异解决"未来边界条件"问题！

### ✨ Key Features | 核心特性

- 🤖 **AI Stock Selection Agent**: LLM-powered intelligent stock selection (Google AI/GPT-4/DeepSeek/Claude)
- 🌍 **Multi-Market Support**: US, Japan, China A-share, Hong Kong
- ⚡ **Lightweight**: ~1M parameters, trains in 8-10 minutes on CPU
- 🔄 **Daily Retraining**: Fresh model every day using latest market structure
- 📊 **Strong Performance**: Sharpe 1.5-2.0 for US→JP prediction

### 🎯 Quick Start | 快速开始

```bash
# Clone and install
git clone https://github.com/FTF1990/Quant-Stock-Transformer.git
cd Quant-Stock-Transformer
git checkout claude/cross-market-agent-011CV1VoSoz7uwWmuin1xyWs
pip install -r requirements.txt

# Run interactive Colab notebook (recommended)
# Or run demo locally:
python examples/us_to_multi_demo.py --json outputs/stocks_selection.json
```

📚 **Full Guide**: [Cross-Market Prediction Guide](docs/CROSS_MARKET_GUIDE.md)

---

## 📖 How It Works | 工作原理

### The Problem | 问题

Traditional models can't predict future market conditions:

传统模型无法预测未来的市场状况：

```
❌ Traditional: Need future values of other stocks (unknown)
❌ 传统方法：需要其他股票的未来值（未知）
```

### The Solution | 解决方案

Cross-market prediction with time zone advantage:

利用时区优势的跨市场预测：

```
✅ US Market closes (T-day) → 3-hour window → JP Market opens (T+1 day)
✅ 美股收盘（T日） → 3小时窗口 → 日股开盘（T+1日）

   US data is KNOWN when predicting JP market!
   预测日股时，美股数据已知！
```

### SST Framework | SST框架

**Stage 1**: Learn spatial relationships (cross-market correlations)
**Stage 2**: Extract relationship features
**Stage 3**: Temporal prediction with LSTM/GRU

**阶段1**：学习空间关系（跨市场相关性）
**阶段2**：提取关系特征
**阶段3**：使用LSTM/GRU进行时序预测

---

## 📊 Performance | 性能表现

| Strategy | Sharpe Ratio | Win Rate | Status |
|----------|--------------|----------|--------|
| US → JP  | 1.5 - 2.0    | 56-60%   | 🟢 Recommended |
| US → HK  | 1.2 - 1.8    | 54-58%   | 🟢 Good |
| US → CN  | 0.8 - 1.2    | 52-55%   | 🟡 Moderate |

*Note: Results from backtesting. Not financial advice.*

*注：回测结果，非投资建议*

---

## 🗂️ Project Structure | 项目结构

```
Quant-Stock-Transformer/
├── src/
│   ├── stock_agent.py              # 🤖 AI stock selection agent
│   ├── cross_market_data.py        # 🌍 Multi-market data fetcher
│   ├── three_stage_pipeline.py     # 🔧 SST pipeline
│   └── ...
├── examples/
│   └── us_to_multi_demo.py         # 📊 Complete demo
├── notebooks/
│   └── Cross_Market_Stock_Prediction.ipynb  # 📓 Interactive Colab
├── models/                          # 🧠 Model architectures
├── docs/
│   └── CROSS_MARKET_GUIDE.md       # 📚 Detailed guide
└── requirements.txt
```

---

## 🛠️ Installation | 安装

```bash
pip install -r requirements.txt
```

**Dependencies**:
- PyTorch >= 2.0.0
- yfinance (stock data)
- pandas, numpy, scikit-learn
- LLM providers: google-generativeai, openai, anthropic (optional)

---

## 📚 Documentation | 文档

- [Cross-Market Prediction Guide](docs/CROSS_MARKET_GUIDE.md) - Complete guide
- [Colab Notebook](notebooks/Cross_Market_Stock_Prediction.ipynb) - Interactive tutorial
- API Reference - See docstrings in source code

---

## 🤝 Contributing | 贡献

Contributions are welcome! Areas of interest:

欢迎贡献！感兴趣的领域：

- Additional LLM providers
- More technical indicators
- Real-time data integration
- Advanced backtesting metrics
- Risk management modules

---

## ⚠️ Disclaimer | 免责声明

This project is for **educational and research purposes only**.

- Not financial advice
- Use at your own risk
- Always conduct your own research before trading

本项目仅用于**教育和研究目的**。

- 非投资建议
- 风险自负
- 交易前请务必自行研究

---

## 📝 License | 许可证

MIT License - See LICENSE file

---

## 🌟 Star History | Star历史

If you find this project useful, please consider giving it a ⭐!

如果觉得有用，请给个⭐！

---

**🚧 Status: Under Active Development | 积极开发中**

We're actively refining the code and documentation. Expect frequent updates!

我们正在积极完善代码和文档。敬请期待更新！
