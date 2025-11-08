"""
关系特征提取器 - 从Stage1 Transformer中提取股票间关系特征

三种方案:
1. AttentionBasedExtractor: 使用attention权重
2. EmbeddingBasedExtractor: 使用transformer输出embedding
3. ContrastiveExtractor: 使用对比学习 (高级)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class AttentionBasedExtractor(nn.Module):
    """
    方案1: 基于Attention权重的关系特征提取

    优点:
    - 直观可解释
    - 计算开销小
    - 可视化友好

    输出维度: num_stocks + aggregated_features
    """

    def __init__(
        self,
        num_stocks: int,
        num_indices: int = 3,  # 大盘指数数量
        output_dim: int = 32
    ):
        super().__init__()
        self.num_stocks = num_stocks
        self.num_indices = num_indices

        # 可选: 对attention权重做非线性变换
        input_dim = num_stocks + num_indices
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def extract_from_attention(
        self,
        attention_weights: torch.Tensor,
        target_stock_idx: int,
        stock_indices: List[int],
        index_indices: List[int]
    ) -> torch.Tensor:
        """
        从attention矩阵提取目标股票的关系特征

        Args:
            attention_weights: [batch, num_heads, num_signals, num_signals]
            target_stock_idx: 目标股票在序列中的索引
            stock_indices: 其他股票的索引列表
            index_indices: 大盘指数的索引列表

        Returns:
            relationship_features: [batch, output_dim]
        """
        batch_size, num_heads, _, _ = attention_weights.shape

        # 1. 提取目标股票的attention分布
        # [batch, num_heads, num_signals]
        target_attention = attention_weights[:, :, target_stock_idx, :]

        # 2. 平均所有attention heads
        # [batch, num_signals]
        avg_attention = target_attention.mean(dim=1)

        # 3. 分组聚合关系特征
        features = []

        # 3.1 对每只股票的attention权重
        stock_attention = avg_attention[:, stock_indices]  # [batch, num_stocks]
        features.append(stock_attention)

        # 3.2 对大盘指数的attention权重 (市场相关性)
        index_attention = avg_attention[:, index_indices]  # [batch, num_indices]
        features.append(index_attention)

        # 3.3 聚合统计量
        # - 最大attention (最相关的股票)
        max_stock_attention = stock_attention.max(dim=1, keepdim=True)[0]

        # - 平均attention (整体市场影响)
        mean_stock_attention = stock_attention.mean(dim=1, keepdim=True)

        # - 注意力分散度 (市场关联分散程度)
        attention_entropy = -torch.sum(
            stock_attention * torch.log(stock_attention + 1e-9),
            dim=1,
            keepdim=True
        )

        features.extend([max_stock_attention, mean_stock_attention, attention_entropy])

        # 4. 合并所有特征
        combined = torch.cat(features, dim=1)  # [batch, num_stocks + num_indices + 3]

        # 5. 可选: 投影到目标维度
        output = self.projection(combined)  # [batch, output_dim]

        return output

    def forward(
        self,
        attention_weights: torch.Tensor,
        target_stock_idx: int,
        metadata: Dict[str, List[int]]
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            attention_weights: Transformer的attention权重
            target_stock_idx: 目标股票索引
            metadata: 包含stock_indices和index_indices的字典
        """
        return self.extract_from_attention(
            attention_weights,
            target_stock_idx,
            metadata['stock_indices'],
            metadata['index_indices']
        )


class EmbeddingBasedExtractor(nn.Module):
    """
    方案2: 基于Transformer Embedding的关系特征提取

    优点:
    - 信息丰富 (包含非线性关系)
    - 维度可控
    - 适合做进一步的时序建模

    输出维度: output_dim (可配置)
    """

    def __init__(
        self,
        d_model: int = 128,           # Transformer的模型维度
        output_dim: int = 32,         # 输出关系特征维度
        pooling_method: str = 'concat'  # 'mean', 'max', 'concat', 'attention'
    ):
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim
        self.pooling_method = pooling_method

        # 根据pooling方法确定输入维度
        if pooling_method == 'concat':
            # 目标股票 + 全局pooling
            projection_input_dim = d_model * 2
        else:
            projection_input_dim = d_model

        # 降维投影层
        self.projection = nn.Sequential(
            nn.Linear(projection_input_dim, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim)
        )

        # 如果使用attention pooling
        if pooling_method == 'attention':
            self.attention_pooling = nn.Sequential(
                nn.Linear(d_model, 1),
                nn.Softmax(dim=1)
            )

    def extract_from_embeddings(
        self,
        encoder_output: torch.Tensor,
        target_stock_idx: int
    ) -> torch.Tensor:
        """
        从Transformer encoder输出提取关系特征

        Args:
            encoder_output: [batch, num_signals, d_model]
            target_stock_idx: 目标股票索引

        Returns:
            relationship_features: [batch, output_dim]
        """
        batch_size, num_signals, d_model = encoder_output.shape

        # 1. 提取目标股票的embedding
        target_embedding = encoder_output[:, target_stock_idx, :]  # [batch, d_model]

        # 2. 全局市场信息聚合
        if self.pooling_method == 'mean':
            # 平均pooling
            global_embedding = encoder_output.mean(dim=1)  # [batch, d_model]
            combined = global_embedding

        elif self.pooling_method == 'max':
            # 最大pooling
            global_embedding = encoder_output.max(dim=1)[0]  # [batch, d_model]
            combined = global_embedding

        elif self.pooling_method == 'concat':
            # 拼接目标股票和全局信息
            global_embedding = encoder_output.mean(dim=1)  # [batch, d_model]
            combined = torch.cat([target_embedding, global_embedding], dim=1)
            # [batch, d_model * 2]

        elif self.pooling_method == 'attention':
            # Attention weighted pooling
            attention_scores = self.attention_pooling(encoder_output)  # [batch, num_signals, 1]
            global_embedding = torch.sum(
                encoder_output * attention_scores,
                dim=1
            )  # [batch, d_model]
            combined = global_embedding

        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")

        # 3. 投影到目标维度
        relationship_features = self.projection(combined)  # [batch, output_dim]

        return relationship_features

    def forward(
        self,
        encoder_output: torch.Tensor,
        target_stock_idx: int
    ) -> torch.Tensor:
        return self.extract_from_embeddings(encoder_output, target_stock_idx)


class HybridExtractor(nn.Module):
    """
    混合提取器: 结合Attention和Embedding

    最佳实践: 同时利用attention的可解释性和embedding的表达能力
    """

    def __init__(
        self,
        num_stocks: int,
        num_indices: int,
        d_model: int,
        output_dim: int = 64,
        attention_dim: int = 16,
        embedding_dim: int = 48
    ):
        super().__init__()

        # 两个子提取器
        self.attention_extractor = AttentionBasedExtractor(
            num_stocks=num_stocks,
            num_indices=num_indices,
            output_dim=attention_dim
        )

        self.embedding_extractor = EmbeddingBasedExtractor(
            d_model=d_model,
            output_dim=embedding_dim,
            pooling_method='concat'
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(attention_dim + embedding_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(
        self,
        attention_weights: torch.Tensor,
        encoder_output: torch.Tensor,
        target_stock_idx: int,
        metadata: Dict[str, List[int]]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
            relationship_features: [batch, output_dim]
            intermediates: 中间特征字典 (用于可视化和调试)
        """
        # 提取attention特征
        attention_features = self.attention_extractor(
            attention_weights,
            target_stock_idx,
            metadata
        )

        # 提取embedding特征
        embedding_features = self.embedding_extractor(
            encoder_output,
            target_stock_idx
        )

        # 融合
        combined = torch.cat([attention_features, embedding_features], dim=1)
        relationship_features = self.fusion(combined)

        # 返回中间结果用于可视化
        intermediates = {
            'attention_features': attention_features,
            'embedding_features': embedding_features,
            'attention_weights': attention_weights[:, :, target_stock_idx, :]
        }

        return relationship_features, intermediates


# ============================================================================
# 辅助函数
# ============================================================================

def extract_relationship_features_batch(
    stage1_model: nn.Module,
    data: torch.Tensor,
    target_stock_idx: int,
    extractor: nn.Module,
    metadata: Optional[Dict] = None,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    批量提取关系特征

    Args:
        stage1_model: 训练好的Stage1模型
        data: [batch, num_signals, seq_len] 输入数据
        target_stock_idx: 目标股票索引
        extractor: 关系特征提取器
        metadata: 元数据 (stock/index indices)
        device: 设备

    Returns:
        relationship_features: [batch, output_dim]
    """
    stage1_model.eval()
    extractor.eval()

    with torch.no_grad():
        # 1. 获取Stage1的中间输出
        # 假设Stage1模型有这些方法
        encoder_output = stage1_model.get_encoder_output(data)
        attention_weights = stage1_model.get_attention_weights(data)

        # 2. 提取关系特征
        if isinstance(extractor, HybridExtractor):
            relationship_features, _ = extractor(
                attention_weights,
                encoder_output,
                target_stock_idx,
                metadata
            )
        elif isinstance(extractor, AttentionBasedExtractor):
            relationship_features = extractor(
                attention_weights,
                target_stock_idx,
                metadata
            )
        elif isinstance(extractor, EmbeddingBasedExtractor):
            relationship_features = extractor(
                encoder_output,
                target_stock_idx
            )
        else:
            raise ValueError(f"Unknown extractor type: {type(extractor)}")

    return relationship_features


def visualize_attention_relationships(
    attention_weights: torch.Tensor,
    stock_names: List[str],
    target_stock_idx: int,
    save_path: str = None
):
    """
    可视化attention权重，展示股票间关系

    Args:
        attention_weights: [num_heads, num_stocks, num_stocks]
        stock_names: 股票名称列表
        target_stock_idx: 目标股票索引
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 平均所有heads
    avg_attention = attention_weights.mean(dim=0).cpu().numpy()

    # 提取目标股票的attention
    target_attention = avg_attention[target_stock_idx, :]

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 1. 热力图
    sns.heatmap(
        avg_attention,
        xticklabels=stock_names,
        yticklabels=stock_names,
        cmap='YlOrRd',
        ax=axes[0]
    )
    axes[0].set_title('Attention Heatmap (All Stocks)')

    # 2. 柱状图 - 目标股票的attention分布
    sorted_indices = target_attention.argsort()[::-1]
    sorted_names = [stock_names[i] for i in sorted_indices]
    sorted_values = target_attention[sorted_indices]

    axes[1].barh(range(len(sorted_names)), sorted_values)
    axes[1].set_yticks(range(len(sorted_names)))
    axes[1].set_yticklabels(sorted_names)
    axes[1].set_xlabel('Attention Weight')
    axes[1].set_title(f'Attention Distribution for {stock_names[target_stock_idx]}')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == '__main__':
    """
    示例代码: 如何使用关系特征提取器
    """

    # 配置
    batch_size = 32
    num_stocks = 20
    num_indices = 3
    d_model = 128
    num_heads = 8

    # 模拟数据
    encoder_output = torch.randn(batch_size, num_stocks + num_indices, d_model)
    attention_weights = torch.softmax(
        torch.randn(batch_size, num_heads, num_stocks + num_indices, num_stocks + num_indices),
        dim=-1
    )

    target_stock_idx = 0  # 预测第一只股票

    metadata = {
        'stock_indices': list(range(num_stocks)),
        'index_indices': list(range(num_stocks, num_stocks + num_indices))
    }

    # ===== 方案1: Attention特征 =====
    print("方案1: Attention-based Extractor")
    extractor1 = AttentionBasedExtractor(
        num_stocks=num_stocks,
        num_indices=num_indices,
        output_dim=32
    )

    features1 = extractor1(attention_weights, target_stock_idx, metadata)
    print(f"输出形状: {features1.shape}")  # [32, 32]
    print(f"特征维度: 32\n")

    # ===== 方案2: Embedding特征 =====
    print("方案2: Embedding-based Extractor")
    extractor2 = EmbeddingBasedExtractor(
        d_model=d_model,
        output_dim=32,
        pooling_method='concat'
    )

    features2 = extractor2(encoder_output, target_stock_idx)
    print(f"输出形状: {features2.shape}")  # [32, 32]
    print(f"特征维度: 32\n")

    # ===== 混合方案 =====
    print("混合方案: Hybrid Extractor")
    extractor3 = HybridExtractor(
        num_stocks=num_stocks,
        num_indices=num_indices,
        d_model=d_model,
        output_dim=64,
        attention_dim=16,
        embedding_dim=48
    )

    features3, intermediates = extractor3(
        attention_weights,
        encoder_output,
        target_stock_idx,
        metadata
    )
    print(f"输出形状: {features3.shape}")  # [32, 64]
    print(f"特征维度: 64")
    print(f"中间特征键: {intermediates.keys()}")
