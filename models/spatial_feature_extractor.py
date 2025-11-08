"""
Stage1: 空间特征提取器 (Spatial Feature Extractor)

扩展StaticSensorTransformer，添加中间特征提取功能
用于提取股票间的关系特征，输入到Stage3时序模型
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional
from models.static_transformer import StaticSensorTransformer


class SpatialFeatureExtractor(StaticSensorTransformer):
    """
    空间特征提取器 - Stage1增强版

    在原始StaticSensorTransformer基础上:
    1. 保留所有原始功能 (预测目标值)
    2. 添加中间特征提取接口
    3. 支持attention权重提取
    4. 支持encoder输出提取

    用途:
    - Stage1训练: 正常训练，预测股票价格/收益
    - 特征提取: 提取关系特征，输入Stage3
    """

    def __init__(
        self,
        num_boundary_sensors,
        num_target_sensors,
        d_model=128,
        nhead=8,
        num_layers=3,
        dropout=0.1,
        enable_feature_extraction=True
    ):
        super().__init__(
            num_boundary_sensors=num_boundary_sensors,
            num_target_sensors=num_target_sensors,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout
        )

        self.enable_feature_extraction = enable_feature_extraction
        self.nhead = nhead

        # 缓存中间结果 (用于特征提取)
        self._encoder_output = None
        self._attention_weights = None
        self._embeddings = None

        # 注册hook来捕获attention权重
        if enable_feature_extraction:
            self._register_attention_hooks()

    def _register_attention_hooks(self):
        """注册forward hook来捕获attention权重"""

        def get_attention_hook(module, input, output):
            """
            TransformerEncoderLayer的hook
            捕获attention权重
            """
            # PyTorch的MultiheadAttention在forward时可以返回attention权重
            # 但TransformerEncoderLayer默认不返回
            # 需要手动访问attn_weights
            pass

        # 由于TransformerEncoderLayer不直接暴露attention权重
        # 我们需要重写forward方法或使用自定义Transformer
        # 这里提供一个替代方案: 使用自定义的forward_with_attention

    def forward_with_features(
        self,
        boundary_conditions: torch.Tensor,
        return_attention: bool = True,
        return_encoder_output: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        带中间特征的forward传播

        Args:
            boundary_conditions: [batch, num_boundary_sensors] 输入数据
            return_attention: 是否返回attention权重
            return_encoder_output: 是否返回encoder输出

        Returns:
            predictions: [batch, num_target_sensors] 预测结果
            features: 字典，包含中间特征
                - 'embeddings': [batch, num_sensors, d_model]
                - 'encoder_output': [batch, num_sensors, d_model]
                - 'attention_weights': [batch, num_layers, num_heads, num_sensors, num_sensors]
                - 'pooled_features': [batch, d_model]
        """
        batch_size = boundary_conditions.shape[0]
        features = {}

        # 1. Embed boundary conditions
        x = boundary_conditions.unsqueeze(-1)  # (batch, sensors, 1)
        x = self.boundary_embedding(x) + self.boundary_position_encoding.unsqueeze(0)
        # (batch, sensors, d_model)

        if return_encoder_output:
            features['embeddings'] = x.clone()

        # 2. Transform (需要自定义以获取attention)
        if return_attention:
            # 使用自定义transformer forward来获取attention
            encoder_output, attention_weights = self._forward_transformer_with_attention(x)
            features['attention_weights'] = attention_weights
        else:
            encoder_output = self.transformer(x)

        if return_encoder_output:
            features['encoder_output'] = encoder_output

        # 3. Global pooling
        x_pooled = encoder_output.permute(0, 2, 1)  # (batch, d_model, sensors)
        x_pooled = self.global_pool(x_pooled).squeeze(-1)  # (batch, d_model)

        features['pooled_features'] = x_pooled

        # 4. Output projection
        predictions = self.output_projection(x_pooled)

        return predictions, features

    def _forward_transformer_with_attention(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        自定义transformer forward，返回attention权重

        Args:
            x: [batch, num_sensors, d_model]

        Returns:
            output: [batch, num_sensors, d_model]
            attention_weights: [batch, num_layers, num_heads, num_sensors, num_sensors]
        """
        attention_weights_list = []

        # 逐层执行transformer
        for layer in self.transformer.layers:
            # 使用MultiheadAttention的输出attention选项
            # 注意: 需要修改TransformerEncoderLayer或直接调用self_attn

            # 保存原始输入用于residual connection
            residual = x

            # Self-attention (手动调用)
            attn_output, attn_weights = layer.self_attn(
                x, x, x,
                need_weights=True,
                average_attn_weights=False  # 返回每个head的权重
            )
            # attn_output: [batch, num_sensors, d_model]
            # attn_weights: [batch, num_heads, num_sensors, num_sensors]

            attention_weights_list.append(attn_weights)

            # Dropout + Residual
            x = residual + layer.dropout1(attn_output)

            # Layer norm
            x = layer.norm1(x)

            # Feedforward
            residual = x
            ff_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
            x = residual + layer.dropout2(ff_output)
            x = layer.norm2(x)

        # Stack attention weights from all layers
        # [num_layers, batch, num_heads, num_sensors, num_sensors]
        attention_weights = torch.stack(attention_weights_list, dim=1)
        # [batch, num_layers, num_heads, num_sensors, num_sensors]

        return x, attention_weights

    def forward(self, boundary_conditions: torch.Tensor) -> torch.Tensor:
        """
        标准forward (保持向后兼容)
        不返回中间特征，仅返回预测
        """
        return super().forward(boundary_conditions)

    def extract_relationship_features(
        self,
        boundary_conditions: torch.Tensor,
        target_stock_idx: int,
        feature_extractor: nn.Module
    ) -> torch.Tensor:
        """
        提取关系特征的便捷方法

        Args:
            boundary_conditions: [batch, num_boundary_sensors]
            target_stock_idx: 目标股票在输入中的索引
            feature_extractor: 关系特征提取器 (AttentionBasedExtractor等)

        Returns:
            relationship_features: [batch, relationship_dim]
        """
        self.eval()

        with torch.no_grad():
            # 获取中间特征
            _, features = self.forward_with_features(
                boundary_conditions,
                return_attention=True,
                return_encoder_output=True
            )

            # 使用提取器提取关系特征
            encoder_output = features['encoder_output']
            attention_weights = features['attention_weights']

            # 平均所有层的attention
            avg_attention = attention_weights.mean(dim=1)  # [batch, num_heads, num_sensors, num_sensors]

            # 调用特征提取器
            relationship_features = feature_extractor(
                attention_weights=avg_attention,
                encoder_output=encoder_output,
                target_stock_idx=target_stock_idx
            )

        return relationship_features

    def get_encoder_output(self, boundary_conditions: torch.Tensor) -> torch.Tensor:
        """
        快速获取encoder输出

        Returns:
            encoder_output: [batch, num_sensors, d_model]
        """
        _, features = self.forward_with_features(
            boundary_conditions,
            return_attention=False,
            return_encoder_output=True
        )
        return features['encoder_output']

    def get_attention_weights(self, boundary_conditions: torch.Tensor) -> torch.Tensor:
        """
        快速获取attention权重

        Returns:
            attention_weights: [batch, num_layers, num_heads, num_sensors, num_sensors]
        """
        _, features = self.forward_with_features(
            boundary_conditions,
            return_attention=True,
            return_encoder_output=False
        )
        return features['attention_weights']


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == '__main__':
    """
    示例: 如何使用SpatialFeatureExtractor
    """

    # 配置
    num_stocks = 20
    num_indices = 3
    num_boundary_sensors = num_stocks + num_indices  # 总共23个输入信号
    num_target_sensors = 5  # 预测5个目标 (如5天后的价格)

    # 创建模型
    model = SpatialFeatureExtractor(
        num_boundary_sensors=num_boundary_sensors,
        num_target_sensors=num_target_sensors,
        d_model=128,
        nhead=8,
        num_layers=3,
        enable_feature_extraction=True
    )

    # 模拟输入数据
    batch_size = 32
    boundary_conditions = torch.randn(batch_size, num_boundary_sensors)

    print("=" * 60)
    print("示例1: 标准forward (用于训练)")
    print("=" * 60)

    predictions = model(boundary_conditions)
    print(f"输入形状: {boundary_conditions.shape}")
    print(f"预测形状: {predictions.shape}")
    print()

    print("=" * 60)
    print("示例2: 带中间特征的forward (用于分析)")
    print("=" * 60)

    predictions, features = model.forward_with_features(
        boundary_conditions,
        return_attention=True,
        return_encoder_output=True
    )

    print(f"预测形状: {predictions.shape}")
    print("\n中间特征:")
    for key, value in features.items():
        print(f"  {key}: {value.shape}")
    print()

    print("=" * 60)
    print("示例3: 提取encoder输出")
    print("=" * 60)

    encoder_output = model.get_encoder_output(boundary_conditions)
    print(f"Encoder输出形状: {encoder_output.shape}")
    print()

    print("=" * 60)
    print("示例4: 提取attention权重")
    print("=" * 60)

    attention_weights = model.get_attention_weights(boundary_conditions)
    print(f"Attention权重形状: {attention_weights.shape}")
    print(f"  - Batch size: {attention_weights.shape[0]}")
    print(f"  - Num layers: {attention_weights.shape[1]}")
    print(f"  - Num heads: {attention_weights.shape[2]}")
    print(f"  - Num sensors: {attention_weights.shape[3]} x {attention_weights.shape[4]}")
    print()

    print("=" * 60)
    print("示例5: 结合关系特征提取器")
    print("=" * 60)

    from relationship_extractors import EmbeddingBasedExtractor

    # 创建特征提取器
    feature_extractor = EmbeddingBasedExtractor(
        d_model=128,
        output_dim=32,
        pooling_method='concat'
    )

    # 提取关系特征
    target_stock_idx = 0  # 提取第一只股票的关系特征
    relationship_features = model.extract_relationship_features(
        boundary_conditions,
        target_stock_idx,
        feature_extractor
    )

    print(f"关系特征形状: {relationship_features.shape}")
    print(f"这些特征可以作为Stage3的输入!")
