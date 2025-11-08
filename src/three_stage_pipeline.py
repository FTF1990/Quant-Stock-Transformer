"""
三阶段股票预测Pipeline

完整流程:
Stage1: 训练空间特征提取器 (跨股票关系学习)
Stage2: 残差提升 (可选)
Stage3: 时序预测器 (单股票 + 关系特征)
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import joblib
from tqdm import tqdm


class ThreeStagePipeline:
    """
    三阶段股票预测完整流程

    使用方法:
    1. 训练Stage1 (空间特征提取)
    2. 提取关系特征
    3. 训练Stage3 (时序预测)
    4. 推理
    """

    def __init__(
        self,
        stock_codes: List[str],           # 股票代码列表
        index_codes: List[str],           # 指数代码列表
        target_stock: str,                # 目标股票
        feature_columns: List[str],       # 技术指标列名
        relationship_dim: int = 32,       # 关系特征维度
        seq_len: int = 60,                # 时序窗口长度
        device: str = 'cpu'
    ):
        """
        Args:
            stock_codes: ['000001', '000002', ...]
            index_codes: ['sh000001', 'sz399001', ...]
            target_stock: '000001'
            feature_columns: ['close', 'volume', 'MA5', 'MA20', ...]
            relationship_dim: 关系特征输出维度
            seq_len: 时序lookback长度
        """
        self.stock_codes = stock_codes
        self.index_codes = index_codes
        self.target_stock = target_stock
        self.feature_columns = feature_columns
        self.relationship_dim = relationship_dim
        self.seq_len = seq_len
        self.device = device

        # 计算维度
        self.num_stocks = len(stock_codes)
        self.num_indices = len(index_codes)
        self.num_features_per_signal = len(feature_columns)
        self.num_boundary_sensors = (self.num_stocks + self.num_indices) * self.num_features_per_signal

        # 模型
        self.stage1_model = None
        self.stage2_model = None
        self.stage3_model = None
        self.relationship_extractor = None

        # 数据
        self.scaler = None
        self.data = None

    # ========================================================================
    # Stage1: 空间特征提取
    # ========================================================================

    def build_stage1(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3
    ):
        """构建Stage1模型"""
        from models.spatial_feature_extractor import SpatialFeatureExtractor

        self.stage1_model = SpatialFeatureExtractor(
            num_boundary_sensors=self.num_boundary_sensors,
            num_target_sensors=self.num_features_per_signal,  # 预测目标股票的特征
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            enable_feature_extraction=True
        ).to(self.device)

        print(f"✓ Stage1 model built: {sum(p.numel() for p in self.stage1_model.parameters()):,} parameters")

    def train_stage1(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        num_epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-4
    ):
        """
        训练Stage1

        Args:
            train_data: DataFrame with columns [stock1_close, stock1_volume, ..., stock2_close, ...]
            val_data: 验证集
        """
        # 准备数据
        X_train, y_train = self._prepare_stage1_data(train_data)
        X_val, y_val = self._prepare_stage1_data(val_data)

        # DataLoader
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

        # 训练
        optimizer = torch.optim.Adam(self.stage1_model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            # 训练
            self.stage1_model.train()
            train_loss = 0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                predictions = self.stage1_model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # 验证
            self.stage1_model.eval()
            val_loss = 0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    predictions = self.stage1_model(batch_X)
                    loss = criterion(predictions, batch_y)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.stage1_model.state_dict(), 'stage1_best.pth')

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        print(f"✓ Stage1 training completed. Best val loss: {best_val_loss:.6f}")

    def _prepare_stage1_data(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        准备Stage1训练数据

        Args:
            df: DataFrame with multi-stock features

        Returns:
            X: [num_samples, num_boundary_sensors]
            y: [num_samples, num_target_sensors]
        """
        # 提取所有股票和指数的特征作为输入
        boundary_cols = []
        for code in self.stock_codes + self.index_codes:
            for feat in self.feature_columns:
                col_name = f"{code}_{feat}"
                if col_name in df.columns:
                    boundary_cols.append(col_name)

        # 提取目标股票的特征作为输出
        target_cols = [f"{self.target_stock}_{feat}" for feat in self.feature_columns]

        X = torch.FloatTensor(df[boundary_cols].values)
        y = torch.FloatTensor(df[target_cols].values)

        return X, y

    # ========================================================================
    # 关系特征提取
    # ========================================================================

    def build_relationship_extractor(self, extractor_type: str = 'hybrid'):
        """
        构建关系特征提取器

        Args:
            extractor_type: 'attention', 'embedding', 'hybrid'
        """
        from models.relationship_extractors import (
            AttentionBasedExtractor,
            EmbeddingBasedExtractor,
            HybridExtractor
        )

        if extractor_type == 'attention':
            self.relationship_extractor = AttentionBasedExtractor(
                num_stocks=self.num_stocks,
                num_indices=self.num_indices,
                output_dim=self.relationship_dim
            ).to(self.device)

        elif extractor_type == 'embedding':
            self.relationship_extractor = EmbeddingBasedExtractor(
                d_model=self.stage1_model.d_model,
                output_dim=self.relationship_dim,
                pooling_method='concat'
            ).to(self.device)

        elif extractor_type == 'hybrid':
            self.relationship_extractor = HybridExtractor(
                num_stocks=self.num_stocks,
                num_indices=self.num_indices,
                d_model=self.stage1_model.d_model,
                output_dim=self.relationship_dim
            ).to(self.device)

        else:
            raise ValueError(f"Unknown extractor type: {extractor_type}")

        print(f"✓ Relationship extractor built: {extractor_type}")

    def extract_relationship_features(
        self,
        df: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        对整个数据集提取关系特征

        Args:
            df: 原始数据
            save_path: 保存路径

        Returns:
            df_with_relationships: 添加了关系特征的DataFrame
        """
        self.stage1_model.eval()
        self.relationship_extractor.eval()

        # 准备输入
        X, _ = self._prepare_stage1_data(df)
        X = X.to(self.device)

        # 批量提取
        batch_size = 256
        num_batches = (len(X) + batch_size - 1) // batch_size
        all_relationship_features = []

        print("Extracting relationship features...")
        for i in tqdm(range(num_batches)):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X))
            batch_X = X[start_idx:end_idx]

            with torch.no_grad():
                # 获取Stage1中间输出
                _, features = self.stage1_model.forward_with_features(
                    batch_X,
                    return_attention=True,
                    return_encoder_output=True
                )

                encoder_output = features['encoder_output']
                attention_weights = features['attention_weights']

                # 平均所有层的attention
                avg_attention = attention_weights.mean(dim=1)

                # 目标股票索引
                target_stock_idx = self.stock_codes.index(self.target_stock) * self.num_features_per_signal

                # 构建metadata
                stock_indices = list(range(self.num_stocks * self.num_features_per_signal))
                index_indices = list(range(
                    self.num_stocks * self.num_features_per_signal,
                    (self.num_stocks + self.num_indices) * self.num_features_per_signal
                ))
                metadata = {
                    'stock_indices': stock_indices,
                    'index_indices': index_indices
                }

                # 提取关系特征
                if hasattr(self.relationship_extractor, 'forward'):
                    if isinstance(self.relationship_extractor, type(self.relationship_extractor).__bases__[0]):
                        # HybridExtractor
                        rel_features, _ = self.relationship_extractor(
                            avg_attention,
                            encoder_output,
                            target_stock_idx,
                            metadata
                        )
                    else:
                        rel_features = self.relationship_extractor(
                            encoder_output,
                            target_stock_idx
                        )

                all_relationship_features.append(rel_features.cpu())

        # 合并
        relationship_features = torch.cat(all_relationship_features, dim=0)
        relationship_features = relationship_features.numpy()

        # 添加到DataFrame
        df_result = df.copy()
        for i in range(self.relationship_dim):
            df_result[f'relationship_{i}'] = relationship_features[:, i]

        if save_path:
            df_result.to_csv(save_path, index=False)
            print(f"✓ Relationship features saved to {save_path}")

        return df_result

    # ========================================================================
    # Stage3: 时序预测
    # ========================================================================

    def build_stage3(
        self,
        model_type: str = 'lstm',
        hidden_dim: int = 128,
        num_layers: int = 2
    ):
        """
        构建Stage3模型

        Args:
            model_type: 'lstm', 'gru', 'tcn'
        """
        from models.temporal_predictor import (
            LSTMTemporalPredictor,
            GRUTemporalPredictor,
            TCNTemporalPredictor
        )

        # 计算输入维度
        input_dim = self.num_features_per_signal + self.relationship_dim

        if model_type == 'lstm':
            self.stage3_model = LSTMTemporalPredictor(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                output_dim=1,
                use_attention=True
            ).to(self.device)

        elif model_type == 'gru':
            self.stage3_model = GRUTemporalPredictor(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                output_dim=1
            ).to(self.device)

        elif model_type == 'tcn':
            self.stage3_model = TCNTemporalPredictor(
                input_dim=input_dim,
                num_channels=[64, 128, 128, 64],
                output_dim=1
            ).to(self.device)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        print(f"✓ Stage3 model built: {model_type}, {sum(p.numel() for p in self.stage3_model.parameters()):,} parameters")

    def train_stage3(
        self,
        df_with_relationships: pd.DataFrame,
        target_column: str = 'target_return_1d',
        train_ratio: float = 0.8,
        num_epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-4
    ):
        """
        训练Stage3

        Args:
            df_with_relationships: 包含关系特征的DataFrame
            target_column: 预测目标列名
        """
        from models.temporal_predictor import TemporalDataset

        # 准备数据
        target_stock_features, relationship_features, targets = self._prepare_stage3_data(
            df_with_relationships,
            target_column
        )

        # 分割训练/验证集
        split_idx = int(len(targets) * train_ratio)

        train_dataset = TemporalDataset(
            target_stock_features=target_stock_features[:split_idx],
            relationship_features=relationship_features[:split_idx],
            targets=targets[:split_idx],
            seq_len=self.seq_len
        )

        val_dataset = TemporalDataset(
            target_stock_features=target_stock_features[split_idx:],
            relationship_features=relationship_features[split_idx:],
            targets=targets[split_idx:],
            seq_len=self.seq_len
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

        # 训练
        optimizer = torch.optim.Adam(self.stage3_model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            # 训练
            self.stage3_model.train()
            train_loss = 0

            for batch_seq, batch_target in train_loader:
                batch_seq = batch_seq.to(self.device)
                batch_target = batch_target.to(self.device)

                optimizer.zero_grad()
                predictions = self.stage3_model(batch_seq)
                loss = criterion(predictions, batch_target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # 验证
            self.stage3_model.eval()
            val_loss = 0

            with torch.no_grad():
                for batch_seq, batch_target in val_loader:
                    batch_seq = batch_seq.to(self.device)
                    batch_target = batch_target.to(self.device)

                    predictions = self.stage3_model(batch_seq)
                    loss = criterion(predictions, batch_target)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.stage3_model.state_dict(), 'stage3_best.pth')

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        print(f"✓ Stage3 training completed. Best val loss: {best_val_loss:.6f}")

    def _prepare_stage3_data(
        self,
        df: pd.DataFrame,
        target_column: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """准备Stage3训练数据"""

        # 目标股票特征
        target_stock_cols = [f"{self.target_stock}_{feat}" for feat in self.feature_columns]
        target_stock_features = torch.FloatTensor(df[target_stock_cols].values)

        # 关系特征
        relationship_cols = [f'relationship_{i}' for i in range(self.relationship_dim)]
        relationship_features = torch.FloatTensor(df[relationship_cols].values)

        # 预测目标
        targets = torch.FloatTensor(df[target_column].values).unsqueeze(1)

        return target_stock_features, relationship_features, targets

    # ========================================================================
    # 推理
    # ========================================================================

    def predict(
        self,
        df: pd.DataFrame,
        return_features: bool = False
    ) -> np.ndarray:
        """
        完整推理流程

        Args:
            df: 输入数据 (最近seq_len天的数据)
            return_features: 是否返回中间特征

        Returns:
            predictions: 预测结果
        """
        # 1. 提取关系特征
        df_with_rel = self.extract_relationship_features(df)

        # 2. 准备Stage3输入
        target_stock_features, relationship_features, _ = self._prepare_stage3_data(
            df_with_rel,
            target_column='dummy'  # 推理时不需要target
        )

        # 取最后seq_len天
        sequence = torch.cat([
            target_stock_features[-self.seq_len:],
            relationship_features[-self.seq_len:]
        ], dim=1).unsqueeze(0).to(self.device)

        # 3. Stage3预测
        self.stage3_model.eval()
        with torch.no_grad():
            predictions = self.stage3_model(sequence)

        predictions = predictions.cpu().numpy()

        if return_features:
            return predictions, df_with_rel
        else:
            return predictions

    # ========================================================================
    # 保存/加载
    # ========================================================================

    def save_pipeline(self, save_dir: str):
        """保存完整pipeline"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.stage1_model.state_dict(), save_dir / 'stage1.pth')
        torch.save(self.stage3_model.state_dict(), save_dir / 'stage3.pth')
        torch.save(self.relationship_extractor.state_dict(), save_dir / 'relationship_extractor.pth')

        # 保存配置
        config = {
            'stock_codes': self.stock_codes,
            'index_codes': self.index_codes,
            'target_stock': self.target_stock,
            'feature_columns': self.feature_columns,
            'relationship_dim': self.relationship_dim,
            'seq_len': self.seq_len
        }
        joblib.dump(config, save_dir / 'config.pkl')

        print(f"✓ Pipeline saved to {save_dir}")

    def load_pipeline(self, save_dir: str):
        """加载完整pipeline"""
        save_dir = Path(save_dir)

        # 加载配置
        config = joblib.load(save_dir / 'config.pkl')
        self.__init__(**config)

        # 重建模型
        self.build_stage1()
        self.build_relationship_extractor()
        self.build_stage3()

        # 加载权重
        self.stage1_model.load_state_dict(torch.load(save_dir / 'stage1.pth'))
        self.stage3_model.load_state_dict(torch.load(save_dir / 'stage3.pth'))
        self.relationship_extractor.load_state_dict(torch.load(save_dir / 'relationship_extractor.pth'))

        print(f"✓ Pipeline loaded from {save_dir}")


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == '__main__':
    """
    完整使用流程示例
    """

    # 配置
    stock_codes = ['000001', '000002', '600000', '600036']
    index_codes = ['sh000001', 'sz399001', 'sz399006']
    target_stock = '000001'
    feature_columns = ['close', 'volume', 'MA5', 'MA20', 'RSI']

    # 创建pipeline
    pipeline = ThreeStagePipeline(
        stock_codes=stock_codes,
        index_codes=index_codes,
        target_stock=target_stock,
        feature_columns=feature_columns,
        relationship_dim=32,
        seq_len=60,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print("=" * 60)
    print("三阶段Pipeline示例")
    print("=" * 60)

    # 模拟数据
    num_days = 1000
    data = {}

    for code in stock_codes + index_codes:
        for feat in feature_columns:
            data[f"{code}_{feat}"] = np.random.randn(num_days)

    # 添加预测目标
    data[f"{target_stock}_target_return_1d"] = np.random.randn(num_days) * 0.02

    df = pd.DataFrame(data)

    # 分割数据
    train_df = df.iloc[:800]
    val_df = df.iloc[800:900]
    test_df = df.iloc[900:]

    print("\n" + "=" * 60)
    print("Step 1: 构建并训练Stage1")
    print("=" * 60)

    pipeline.build_stage1(d_model=128, nhead=8, num_layers=3)
    # pipeline.train_stage1(train_df, val_df, num_epochs=5)  # 示例用，减少epoch

    print("\n" + "=" * 60)
    print("Step 2: 构建关系特征提取器并提取特征")
    print("=" * 60)

    pipeline.build_relationship_extractor(extractor_type='hybrid')
    # df_with_rel = pipeline.extract_relationship_features(
    #     df,
    #     save_path='data_with_relationships.csv'
    # )

    print("\n" + "=" * 60)
    print("Step 3: 构建并训练Stage3")
    print("=" * 60)

    pipeline.build_stage3(model_type='lstm', hidden_dim=128, num_layers=2)
    # pipeline.train_stage3(
    #     df_with_rel,
    #     target_column=f"{target_stock}_target_return_1d",
    #     num_epochs=5
    # )

    print("\n" + "=" * 60)
    print("Step 4: 推理")
    print("=" * 60)

    # predictions = pipeline.predict(test_df)
    # print(f"预测结果形状: {predictions.shape}")

    print("\n" + "=" * 60)
    print("Step 5: 保存Pipeline")
    print("=" * 60)

    # pipeline.save_pipeline('saved_models/three_stage_pipeline')

    print("\n✓ Pipeline示例完成!")
