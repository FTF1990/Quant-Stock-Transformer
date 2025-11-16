"""
完整演示：从SST提取内部特征用于时序增强

展示：
1. 常规推理 vs 特征提取推理
2. 提取attention weights和encoder output
3. 计算残差（T日和T+1日）
4. 构建60天历史序列
5. 降维为LSTM输入
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.spatial_feature_extractor import SpatialFeatureExtractor


# ============================================================================
# Part 1: 创建双输出SST模型
# ============================================================================

class DualOutputSST(SpatialFeatureExtractor):
    """
    双输出SST：同时预测T日和T+1日

    输出1（pred_T）：同时刻预测（纯空间响应）
    输出2（pred_T1）：次日预测（空间+时序）
    """

    def __init__(self, num_boundary_sensors, num_target_sensors, **kwargs):
        super().__init__(num_boundary_sensors, num_target_sensors, **kwargs)

        # 双输出头
        self.output_projection_T = nn.Linear(self.d_model, num_target_sensors)
        self.output_projection_T1 = nn.Linear(self.d_model, num_target_sensors)

        # 初始化新的输出层
        nn.init.xavier_uniform_(self.output_projection_T.weight)
        nn.init.xavier_uniform_(self.output_projection_T1.weight)

    def forward(self, boundary_conditions):
        """
        标准forward：返回双输出

        Args:
            boundary_conditions: [batch, num_boundary_sensors]

        Returns:
            pred_T: [batch, num_target_sensors] - T日预测
            pred_T1: [batch, num_target_sensors] - T+1日预测
        """
        batch_size = boundary_conditions.shape[0]

        # 1. Embed
        x = boundary_conditions.unsqueeze(-1)
        x = self.boundary_embedding(x) + self.boundary_position_encoding.unsqueeze(0)

        # 2. Transform
        x = self.transformer(x)

        # 3. Pool
        x_pooled = x.permute(0, 2, 1)
        x_pooled = self.global_pool(x_pooled).squeeze(-1)

        # 4. 双输出
        pred_T = self.output_projection_T(x_pooled)
        pred_T1 = self.output_projection_T1(x_pooled)

        return pred_T, pred_T1

    def forward_with_features(
        self,
        boundary_conditions,
        return_attention=True,
        return_encoder_output=True
    ):
        """
        带特征的双输出forward

        Returns:
            (pred_T, pred_T1): 双输出预测
            features: 字典，包含中间特征
        """
        batch_size = boundary_conditions.shape[0]
        features = {}

        # 1. Embed
        x = boundary_conditions.unsqueeze(-1)
        x = self.boundary_embedding(x) + self.boundary_position_encoding.unsqueeze(0)

        if return_encoder_output:
            features['embeddings'] = x.clone()

        # 2. Transform (with attention)
        if return_attention:
            encoder_output, attention_weights = self._forward_transformer_with_attention(x)
            features['attention_weights'] = attention_weights
        else:
            encoder_output = self.transformer(x)

        if return_encoder_output:
            features['encoder_output'] = encoder_output

        # 3. Pool
        x_pooled = encoder_output.permute(0, 2, 1)
        x_pooled = self.global_pool(x_pooled).squeeze(-1)
        features['pooled_features'] = x_pooled

        # 4. 双输出
        pred_T = self.output_projection_T(x_pooled)
        pred_T1 = self.output_projection_T1(x_pooled)

        return (pred_T, pred_T1), features


# ============================================================================
# Part 2: 简单的降维特征提取器
# ============================================================================

class SimpleAttentionExtractor(nn.Module):
    """从Attention提取图论特征"""

    def __init__(self, output_dim=10):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, attention_weights, target_stock_idx):
        """
        Args:
            attention_weights: [batch, num_layers, num_heads, num_sensors, num_sensors]
            target_stock_idx: int

        Returns:
            features: [batch, output_dim]
        """
        batch_size = attention_weights.shape[0]

        # 平均所有层和head
        avg_attn = attention_weights.mean(dim=[1, 2])  # [batch, num_sensors, num_sensors]

        # 提取目标股票的attention分布
        target_attn = avg_attn[:, target_stock_idx, :]  # [batch, num_sensors]

        # 提取特征
        features = []

        # 1. Top-5权重
        top5_weights, top5_indices = torch.topk(target_attn, k=5, dim=-1)
        features.append(top5_weights)  # [batch, 5]

        # 2. 熵（集中度）
        entropy = -(target_attn * torch.log(target_attn + 1e-10)).sum(dim=-1, keepdim=True)
        features.append(entropy)  # [batch, 1]

        # 3. 最大权重
        max_weight = target_attn.max(dim=-1, keepdim=True)[0]
        features.append(max_weight)  # [batch, 1]

        # 4. 平均权重
        mean_weight = target_attn.mean(dim=-1, keepdim=True)
        features.append(mean_weight)  # [batch, 1]

        # 5. 方差
        var_weight = target_attn.var(dim=-1, keepdim=True)
        features.append(var_weight)  # [batch, 1]

        # 6. 偏度（简化版）
        std = target_attn.std(dim=-1, keepdim=True)
        skew = ((target_attn - target_attn.mean(dim=-1, keepdim=True)) ** 3).mean(dim=-1, keepdim=True) / (std ** 3 + 1e-10)
        features.append(skew)  # [batch, 1]

        # 合并 (5+1+1+1+1+1=10维)
        features = torch.cat(features, dim=-1)

        return features


class SimpleEncoderExtractor(nn.Module):
    """从Encoder Output提取特征"""

    def __init__(self, d_model=128, output_dim=32):
        super().__init__()

        # 简单的全连接降维
        self.fc = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, encoder_output, target_stock_idx):
        """
        Args:
            encoder_output: [batch, num_sensors, d_model]
            target_stock_idx: int

        Returns:
            features: [batch, output_dim]
        """
        # 提取目标股票的embedding
        target_emb = encoder_output[:, target_stock_idx, :]  # [batch, d_model]

        # 全局平均（市场整体状态）
        global_emb = encoder_output.mean(dim=1)  # [batch, d_model]

        # 拼接
        combined = torch.cat([target_emb, global_emb], dim=-1)  # [batch, d_model*2]

        # 降维
        features = self.fc(combined)  # [batch, output_dim]

        return features


class SimpleResidualExtractor(nn.Module):
    """从残差序列提取统计特征"""

    def __init__(self, output_dim=8):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, residual_sequence):
        """
        Args:
            residual_sequence: [seq_len] - 历史残差序列

        Returns:
            features: [output_dim]
        """
        # 确保是1D tensor
        if residual_sequence.dim() > 1:
            residual_sequence = residual_sequence.squeeze()

        features = []

        # 1. 均值
        features.append(residual_sequence.mean())

        # 2. 标准差
        features.append(residual_sequence.std())

        # 3. 最大值
        features.append(residual_sequence.max())

        # 4. 最小值
        features.append(residual_sequence.min())

        # 5. 偏度（简化）
        mean = residual_sequence.mean()
        std = residual_sequence.std()
        skew = ((residual_sequence - mean) ** 3).mean() / (std ** 3 + 1e-10)
        features.append(skew)

        # 6. 峰度（简化）
        kurt = ((residual_sequence - mean) ** 4).mean() / (std ** 4 + 1e-10)
        features.append(kurt)

        # 7. 一阶自相关（如果序列长度>1）
        if len(residual_sequence) > 1:
            acf1 = torch.corrcoef(torch.stack([
                residual_sequence[:-1],
                residual_sequence[1:]
            ]))[0, 1]
            if torch.isnan(acf1):
                acf1 = torch.tensor(0.0)
        else:
            acf1 = torch.tensor(0.0)
        features.append(acf1)

        # 8. 趋势（最后5个值的平均 - 前5个值的平均）
        if len(residual_sequence) >= 10:
            trend = residual_sequence[-5:].mean() - residual_sequence[:5].mean()
        else:
            trend = torch.tensor(0.0)
        features.append(trend)

        # 堆叠
        features = torch.stack(features)

        return features


# ============================================================================
# Part 3: 主演示流程
# ============================================================================

def main():
    print("=" * 80)
    print("SST内部特征提取完整演示")
    print("=" * 80)
    print()

    # ========================================================================
    # 配置
    # ========================================================================

    num_stocks = 20
    num_indices = 3
    num_boundary_sensors = num_stocks + num_indices  # 23
    num_target_sensors = 1  # 预测单个目标（如某只股票的收益）

    num_days = 60  # 历史天数
    target_stock_idx = 5  # 假设我们关注第5只股票

    # ========================================================================
    # Step 1: 创建双输出SST模型
    # ========================================================================

    print("Step 1: 创建双输出SST模型")
    print("-" * 80)

    model = DualOutputSST(
        num_boundary_sensors=num_boundary_sensors,
        num_target_sensors=num_target_sensors,
        d_model=128,
        nhead=8,
        num_layers=3,
        dropout=0.1,
        enable_feature_extraction=True
    )

    model.eval()

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # ========================================================================
    # Step 2: 模拟历史数据
    # ========================================================================

    print("Step 2: 准备历史数据（60天）")
    print("-" * 80)

    # 模拟边界条件（指数 + 股票）
    np.random.seed(42)
    historical_data = torch.randn(num_days, num_boundary_sensors)

    # 模拟真实值（T日和T+1日）
    true_values_T = torch.randn(num_days, num_target_sensors) * 0.02  # 日收益率 ~2%
    true_values_T1 = torch.randn(num_days, num_target_sensors) * 0.02

    print(f"历史数据形状: {historical_data.shape}")
    print(f"真实值T形状: {true_values_T.shape}")
    print(f"真实值T+1形状: {true_values_T1.shape}")
    print()

    # ========================================================================
    # Step 3: 对比常规推理 vs 特征提取推理
    # ========================================================================

    print("Step 3: 对比两种推理模式")
    print("-" * 80)

    # 单天数据
    boundary_conditions = historical_data[0:1]  # [1, 23]

    # 模式1：常规推理
    print("【模式1】常规推理（仅用于预测）")
    with torch.no_grad():
        pred_T, pred_T1 = model(boundary_conditions)

    print(f"  输入: {boundary_conditions.shape}")
    print(f"  输出 pred_T: {pred_T.shape}")
    print(f"  输出 pred_T1: {pred_T1.shape}")
    print(f"  ✗ 无法获取中间特征")
    print()

    # 模式2：特征提取推理
    print("【模式2】特征提取推理（用于分析和增强）")
    with torch.no_grad():
        (pred_T2, pred_T12), features = model.forward_with_features(
            boundary_conditions,
            return_attention=True,
            return_encoder_output=True
        )

    print(f"  输入: {boundary_conditions.shape}")
    print(f"  输出 pred_T: {pred_T2.shape}")
    print(f"  输出 pred_T1: {pred_T12.shape}")
    print(f"  ✓ 中间特征:")
    for key, value in features.items():
        print(f"    - {key}: {value.shape}")

    # 验证两种模式结果一致
    print(f"\n  验证: 两种模式的预测是否一致?")
    print(f"    pred_T差异: {(pred_T - pred_T2).abs().max().item():.2e}")
    print(f"    pred_T1差异: {(pred_T1 - pred_T12).abs().max().item():.2e}")
    print()

    # ========================================================================
    # Step 4: 提取60天历史的所有特征
    # ========================================================================

    print("Step 4: 提取60天历史特征")
    print("-" * 80)

    all_features = {
        'attention_weights': [],
        'encoder_output': [],
        'residual_T': [],
        'residual_T1': []
    }

    with torch.no_grad():
        for day in range(num_days):
            # 当天边界条件
            bc = historical_data[day:day+1]  # [1, 23]

            # 提取特征
            (pred_T, pred_T1), feats = model.forward_with_features(
                bc,
                return_attention=True,
                return_encoder_output=True
            )

            # 计算残差
            residual_T = true_values_T[day:day+1] - pred_T
            residual_T1 = true_values_T1[day:day+1] - pred_T1

            # 保存
            all_features['attention_weights'].append(feats['attention_weights'])
            all_features['encoder_output'].append(feats['encoder_output'])
            all_features['residual_T'].append(residual_T)
            all_features['residual_T1'].append(residual_T1)

    # 合并成序列
    attention_seq = torch.cat(all_features['attention_weights'], dim=0)
    encoder_seq = torch.cat(all_features['encoder_output'], dim=0)
    residual_T_seq = torch.cat(all_features['residual_T'], dim=0)
    residual_T1_seq = torch.cat(all_features['residual_T1'], dim=0)

    print(f"Attention序列: {attention_seq.shape}")
    print(f"  - [60天, {model.transformer.num_layers}层, {model.nhead}头, {num_boundary_sensors}传感器, {num_boundary_sensors}传感器]")
    print(f"\nEncoder序列: {encoder_seq.shape}")
    print(f"  - [60天, {num_boundary_sensors}传感器, 128维]")
    print(f"\n残差T序列: {residual_T_seq.shape}")
    print(f"残差T1序列: {residual_T1_seq.shape}")
    print()

    # ========================================================================
    # Step 5: 降维 - 构建LSTM输入
    # ========================================================================

    print("Step 5: 降维并构建LSTM输入")
    print("-" * 80)

    # 创建降维器
    attention_extractor = SimpleAttentionExtractor(output_dim=10)
    encoder_extractor = SimpleEncoderExtractor(d_model=128, output_dim=32)
    residual_extractor = SimpleResidualExtractor(output_dim=8)

    lstm_input_list = []

    for day in range(num_days):
        # 提取attention特征（10维）
        attn_feat = attention_extractor(
            all_features['attention_weights'][day],
            target_stock_idx=target_stock_idx
        )  # [1, 10]

        # 提取encoder特征（32维）
        enc_feat = encoder_extractor(
            all_features['encoder_output'][day],
            target_stock_idx=target_stock_idx
        )  # [1, 32]

        # 残差特征（直接拼接T和T1）
        res_feat = torch.cat([
            all_features['residual_T'][day],
            all_features['residual_T1'][day]
        ], dim=-1)  # [1, 2]

        # 合并 (10 + 32 + 2 = 44维)
        day_feat = torch.cat([attn_feat, enc_feat, res_feat], dim=-1)
        lstm_input_list.append(day_feat)

    # 构建LSTM输入序列
    lstm_input = torch.cat(lstm_input_list, dim=0)  # [60, 44]
    lstm_input = lstm_input.unsqueeze(0)  # [1, 60, 44] - 添加batch维度

    print(f"LSTM输入形状: {lstm_input.shape}")
    print(f"  - 1个样本")
    print(f"  - 60个时间步")
    print(f"  - 44维特征 (10注意力 + 32编码器 + 2残差)")
    print()

    # ========================================================================
    # Step 6: 额外的残差统计特征
    # ========================================================================

    print("Step 6: 提取残差统计特征")
    print("-" * 80)

    # 提取T日残差的统计特征
    residual_T_stats = residual_extractor(residual_T_seq.squeeze())
    print(f"残差T统计特征: {residual_T_stats.shape}")
    print(f"  包含: 均值、标准差、最大/最小值、偏度、峰度、自相关、趋势")

    # 提取T1日残差的统计特征
    residual_T1_stats = residual_extractor(residual_T1_seq.squeeze())
    print(f"残差T1统计特征: {residual_T1_stats.shape}")
    print()

    # ========================================================================
    # Step 7: 最终总结
    # ========================================================================

    print("=" * 80)
    print("总结：从SST提取到的所有信息")
    print("=" * 80)

    print("\n【原始维度】（未降维）")
    print(f"  - Attention权重: {attention_seq.shape} = {np.prod(attention_seq.shape[1:]):,}维/天")
    print(f"  - Encoder输出: {encoder_seq.shape} = {np.prod(encoder_seq.shape[1:]):,}维/天")
    print(f"  - 残差: {residual_T_seq.shape[0]}天 × 2个残差")

    print("\n【降维后】（用于LSTM）")
    print(f"  - Attention特征: 10维")
    print(f"  - Encoder特征: 32维")
    print(f"  - 残差特征: 2维")
    print(f"  - 总计: 44维/天")

    print("\n【LSTM输入】")
    print(f"  - 形状: {lstm_input.shape}")
    print(f"  - 说明: 60个时间步，每步44维压缩特征")

    print("\n【可选：残差统计特征】")
    print(f"  - 残差T统计: {residual_T_stats.shape}")
    print(f"  - 残差T1统计: {residual_T1_stats.shape}")
    print(f"  - 说明: 可以作为LSTM的额外输入或初始状态")

    print("\n" + "=" * 80)
    print("✓ 演示完成！")
    print("=" * 80)
    print("\n下一步:")
    print("  1. 使用LSTM(input_size=44, hidden_size=64, num_layers=2)")
    print("  2. 输入lstm_input训练时序增强模型")
    print("  3. 输出可以是：")
    print("     - 修正后的预测")
    print("     - 残差预测")
    print("     - 置信区间")


if __name__ == '__main__':
    main()
