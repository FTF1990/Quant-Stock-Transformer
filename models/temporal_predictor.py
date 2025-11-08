"""
Stage3: 时序预测器 (Temporal Predictor)

多种时序模型实现:
1. LSTM/GRU - 轻量级，适合长序列
2. TCN (Temporal Convolutional Network) - 快速，并行计算
3. TFT接口 - 使用外部库实现

结合Stage1提取的关系特征进行预测
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
import math


# ============================================================================
# 方案1: LSTM-based Temporal Predictor
# ============================================================================

class LSTMTemporalPredictor(nn.Module):
    """
    基于LSTM的时序预测器

    优点:
    - 轻量级，训练快
    - 适合长序列
    - 资源占用小

    输入特征组合:
    - 目标股票的历史技术指标
    - Stage1提取的关系特征 (每个时间步)
    - 时间特征 (可选)
    """

    def __init__(
        self,
        input_dim: int,                    # 总输入特征维度
        hidden_dim: int = 128,             # LSTM隐藏层维度
        num_layers: int = 2,               # LSTM层数
        output_dim: int = 1,               # 输出维度 (如预测未来1天收益)
        dropout: float = 0.2,
        bidirectional: bool = False,       # 是否使用双向LSTM
        use_attention: bool = True         # 是否使用attention pooling
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        self.use_attention = use_attention

        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Attention层 (可选)
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        if use_attention:
            self.attention = TemporalAttention(lstm_output_dim)

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim] 输入序列
               包含: 目标股票特征 + 关系特征 + 时间特征

        Returns:
            predictions: [batch, output_dim] 预测结果
        """
        batch_size, seq_len, _ = x.shape

        # 1. LSTM编码
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: [batch, seq_len, hidden_dim * num_directions]

        # 2. 时间聚合
        if self.use_attention:
            # Attention pooling
            pooled, attn_weights = self.attention(lstm_out)
            # pooled: [batch, hidden_dim * num_directions]
        else:
            # 使用最后一个时间步
            if self.bidirectional:
                # 双向LSTM: 拼接forward和backward的最后隐藏状态
                pooled = torch.cat([h_n[-2], h_n[-1]], dim=1)
            else:
                pooled = h_n[-1]

        # 3. 输出预测
        predictions = self.fc(pooled)

        if return_attention and self.use_attention:
            return predictions, attn_weights
        else:
            return predictions


class TemporalAttention(nn.Module):
    """时间attention层，自动学习哪些时间步更重要"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention_weights = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_output: [batch, seq_len, hidden_dim]

        Returns:
            context: [batch, hidden_dim] 加权后的输出
            attn_weights: [batch, seq_len] attention权重
        """
        # 计算attention scores
        scores = self.attention_weights(lstm_output)  # [batch, seq_len, 1]
        attn_weights = torch.softmax(scores, dim=1)   # [batch, seq_len, 1]

        # 加权求和
        context = torch.sum(lstm_output * attn_weights, dim=1)  # [batch, hidden_dim]

        return context, attn_weights.squeeze(-1)


# ============================================================================
# 方案2: GRU-based Temporal Predictor (更轻量)
# ============================================================================

class GRUTemporalPredictor(nn.Module):
    """
    基于GRU的时序预测器 (比LSTM更轻量)

    GRU相比LSTM:
    - 参数更少 (更快训练)
    - 性能相近
    - 适合资源受限场景
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2,
        use_attention: bool = True
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.use_attention = use_attention

        # GRU
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Attention
        if use_attention:
            self.attention = TemporalAttention(hidden_dim)

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim]

        Returns:
            predictions: [batch, output_dim]
        """
        # GRU编码
        gru_out, h_n = self.gru(x)

        # 聚合
        if self.use_attention:
            pooled, _ = self.attention(gru_out)
        else:
            pooled = h_n[-1]

        # 预测
        predictions = self.fc(pooled)

        return predictions


# ============================================================================
# 方案3: TCN (Temporal Convolutional Network)
# ============================================================================

class TCNTemporalPredictor(nn.Module):
    """
    基于时序卷积网络的预测器

    优点:
    - 并行计算 (比RNN快)
    - 可控的感受野
    - 适合GPU加速

    核心: 因果卷积 + 膨胀卷积
    """

    def __init__(
        self,
        input_dim: int,
        num_channels: list = [64, 128, 128, 64],  # 每层的通道数
        kernel_size: int = 3,
        dropout: float = 0.2,
        output_dim: int = 1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_channels = num_channels
        self.output_dim = output_dim

        # 输入投影
        self.input_projection = nn.Conv1d(
            input_dim,
            num_channels[0],
            kernel_size=1
        )

        # TCN层
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_channels[i-1] if i > 0 else num_channels[0]
            out_channels = num_channels[i]

            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    dropout=dropout
                )
            )

        self.network = nn.Sequential(*layers)

        # 输出层
        self.fc = nn.Linear(num_channels[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim]

        Returns:
            predictions: [batch, output_dim]
        """
        # 转换为卷积格式 [batch, channels, seq_len]
        x = x.permute(0, 2, 1)

        # 输入投影
        x = self.input_projection(x)

        # TCN
        x = self.network(x)

        # 取最后一个时间步
        x = x[:, :, -1]  # [batch, channels]

        # 输出
        predictions = self.fc(x)

        return predictions


class TemporalBlock(nn.Module):
    """TCN的基本模块: 因果卷积 + 残差连接"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        dropout: float
    ):
        super().__init__()

        # 因果卷积 (padding只在左侧)
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)  # 移除右侧padding
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        # 残差连接
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """移除卷积右侧的padding"""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


# ============================================================================
# 数据准备工具
# ============================================================================

class TemporalDataset(torch.utils.data.Dataset):
    """
    时序数据集

    将原始特征 + 关系特征组合成时序序列
    """

    def __init__(
        self,
        target_stock_features: torch.Tensor,    # [num_samples, num_features]
        relationship_features: torch.Tensor,    # [num_samples, relationship_dim]
        targets: torch.Tensor,                  # [num_samples, 1]
        seq_len: int = 60,                      # 序列长度
        temporal_features: Optional[torch.Tensor] = None
    ):
        """
        Args:
            target_stock_features: 目标股票的原始特征 (OHLCV, 技术指标等)
            relationship_features: Stage1提取的关系特征
            targets: 预测目标 (如未来收益率)
            seq_len: lookback窗口长度
            temporal_features: 时间特征 (可选)
        """
        self.seq_len = seq_len

        # 合并特征
        if temporal_features is not None:
            combined_features = torch.cat([
                target_stock_features,
                relationship_features,
                temporal_features
            ], dim=1)
        else:
            combined_features = torch.cat([
                target_stock_features,
                relationship_features
            ], dim=1)

        self.features = combined_features
        self.targets = targets

        # 计算有效样本数
        self.num_samples = len(self.features) - seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 提取序列 [seq_len, num_features]
        sequence = self.features[idx:idx + self.seq_len]

        # 目标值 (序列最后一个时间步之后的值)
        target = self.targets[idx + self.seq_len]

        return sequence, target


# ============================================================================
# TFT接口 (使用PyTorch Forecasting库)
# ============================================================================

class TFTWrapper:
    """
    TFT (Temporal Fusion Transformer) 包装器

    需要安装: pip install pytorch-forecasting
    """

    def __init__(
        self,
        max_encoder_length: int = 60,
        max_prediction_length: int = 1,
        hidden_size: int = 64,
        attention_head_size: int = 4,
        dropout: float = 0.1,
        hidden_continuous_size: int = 16
    ):
        """
        Args:
            max_encoder_length: 历史窗口长度
            max_prediction_length: 预测horizon
            hidden_size: 隐藏层维度
            attention_head_size: attention head数量
        """
        try:
            from pytorch_forecasting import TemporalFusionTransformer
            from pytorch_forecasting.data import TimeSeriesDataSet
        except ImportError:
            raise ImportError(
                "TFT requires pytorch-forecasting. Install with: pip install pytorch-forecasting"
            )

        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length

        # 配置
        self.config = {
            "hidden_size": hidden_size,
            "attention_head_size": attention_head_size,
            "dropout": dropout,
            "hidden_continuous_size": hidden_continuous_size
        }

        self.model = None

    def create_dataset(self, data, time_varying_known_reals, time_varying_unknown_reals):
        """
        创建TFT数据集

        Args:
            data: pandas DataFrame
            time_varying_known_reals: 已知的时序特征 (如时间编码)
            time_varying_unknown_reals: 未知的时序特征 (如关系特征)
        """
        from pytorch_forecasting.data import TimeSeriesDataSet

        # 创建TimeSeriesDataSet
        # 详细配置见pytorch-forecasting文档
        pass

    def fit(self, train_dataloader, val_dataloader, max_epochs=50):
        """训练TFT模型"""
        pass

    def predict(self, test_dataloader):
        """预测"""
        pass


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == '__main__':
    """
    示例: 如何使用时序预测器
    """

    # 配置
    batch_size = 32
    seq_len = 60
    num_stock_features = 30  # 目标股票的技术指标数量
    relationship_dim = 32    # Stage1提取的关系特征维度
    temporal_dim = 10        # 时间特征维度 (day, week, month等)

    total_input_dim = num_stock_features + relationship_dim + temporal_dim

    print("=" * 60)
    print("方案1: LSTM Temporal Predictor")
    print("=" * 60)

    lstm_model = LSTMTemporalPredictor(
        input_dim=total_input_dim,
        hidden_dim=128,
        num_layers=2,
        output_dim=1,
        use_attention=True
    )

    # 模拟输入
    x = torch.randn(batch_size, seq_len, total_input_dim)
    predictions = lstm_model(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {predictions.shape}")
    print(f"参数量: {sum(p.numel() for p in lstm_model.parameters()):,}")
    print()

    print("=" * 60)
    print("方案2: GRU Temporal Predictor (更轻量)")
    print("=" * 60)

    gru_model = GRUTemporalPredictor(
        input_dim=total_input_dim,
        hidden_dim=128,
        num_layers=2,
        output_dim=1
    )

    predictions = gru_model(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {predictions.shape}")
    print(f"参数量: {sum(p.numel() for p in gru_model.parameters()):,}")
    print()

    print("=" * 60)
    print("方案3: TCN Temporal Predictor (最快)")
    print("=" * 60)

    tcn_model = TCNTemporalPredictor(
        input_dim=total_input_dim,
        num_channels=[64, 128, 128, 64],
        kernel_size=3,
        output_dim=1
    )

    predictions = tcn_model(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {predictions.shape}")
    print(f"参数量: {sum(p.numel() for p in tcn_model.parameters()):,}")
    print()

    print("=" * 60)
    print("数据集创建示例")
    print("=" * 60)

    num_days = 1000

    # 模拟数据
    target_stock_features = torch.randn(num_days, num_stock_features)
    relationship_features = torch.randn(num_days, relationship_dim)
    temporal_features = torch.randn(num_days, temporal_dim)
    targets = torch.randn(num_days, 1)

    # 创建数据集
    dataset = TemporalDataset(
        target_stock_features=target_stock_features,
        relationship_features=relationship_features,
        targets=targets,
        seq_len=60,
        temporal_features=temporal_features
    )

    print(f"数据集大小: {len(dataset)}")
    print(f"每个样本: sequence {dataset[0][0].shape}, target {dataset[0][1].shape}")
    print()

    print("=" * 60)
    print("推荐配置")
    print("=" * 60)
    print("""
    小规模 (< 10只股票):
      - LSTM: hidden=64, layers=2
      - 序列长度: 60天
      - 预测horizon: 1-5天

    中等规模 (10-30只股票):
      - LSTM: hidden=128, layers=2, attention=True
      - 序列长度: 90天
      - 预测horizon: 1-10天

    大规模 (30+只股票或分钟线):
      - TCN: channels=[64,128,128,64]
      - 序列长度: 根据内存
      - GRU替代LSTM (更轻量)
    """)
