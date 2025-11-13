import numpy as np
import torch
import torch.nn as nn
from model.layers import TimeCon
from model.layers import GCN
from model.layers.Mamba import Mamba
from model.layers.Mamba import Encoder,EncoderLayer,EncoderLayerV2

class STTemporalSpatialModule(nn.Module):
    def __init__(self, spatial_mamba, temporal_mamba):
        """
        初始化 ST 分支模块。

        Args:
            spatial_mamba (nn.Module): 空间维度的 Mamba 模块。
            temporal_mamba (nn.Module): 时间维度的 Mamba 模块。
        """
        super(STTemporalSpatialModule, self).__init__()
        self.spatial_mamba = spatial_mamba
        self.temporal_mamba = temporal_mamba

    def forward(self, filter_feature):
        """
        前向传播方法。

        Args:
            filter_feature (torch.Tensor): 输入特征，形状为 [b, J3, seq_len]。

        Returns:
            torch.Tensor: 处理后的特征，形状为 [b, seq_len, J3]。
        """
        # [b, J3, seq_len] -> [b, J3, seq_len]
        spatial_feature_att, attns = self.spatial_mamba(filter_feature)
        # [b, seq_len, J3] -> [b, seq_len, J3]->[b, J3, seq_len]
        spatial_temporal_feature_att, attns = self.temporal_mamba(spatial_feature_att.transpose(1, 2))
        spatial_temporal_feature_att = spatial_temporal_feature_att.transpose(1, 2)
        return spatial_temporal_feature_att

class TSTemporalSpatialModule(nn.Module):
    def __init__(self, temporal_mamba, spatial_mamba):
        """
        初始化 TS 分支模块。

        Args:
            temporal_mamba (nn.Module): 时间维度的 Mamba 模块。
            spatial_mamba (nn.Module): 空间维度的 Mamba 模块。
        """
        super(TSTemporalSpatialModule, self).__init__()
        self.temporal_mamba = temporal_mamba
        self.spatial_mamba = spatial_mamba

    def forward(self, filter_feature):
        """
        前向传播方法。

        Args:
            filter_feature (torch.Tensor): 输入特征，形状为 [b, J3, seq_len]。

        Returns:
            torch.Tensor: 处理后的特征，形状为 [b, J3, seq_len]。
        """
        # [b, J3, seq_len]->[b, seq_len, J3]
        temoral_feature_att, attns = self.temporal_mamba(filter_feature.transpose(1, 2))
        # [b, seq_len, J3]->[b, J3, seq_len]
        temporal_spatial_feature_att, attns = self.spatial_mamba(temoral_feature_att.transpose(1, 2))
        return temporal_spatial_feature_att
    
class TTTemporalModule(nn.Module):
    def __init__(self, temporal_mamba1, temporal_mamba2):
        """
        初始化 TT 分支模块（时间-时间）
        
        Args:
            temporal_mamba1 (nn.Module): 第一个时间维度Mamba模块
            temporal_mamba2 (nn.Module): 第二个时间维度Mamba模块
        """
        super(TTTemporalModule, self).__init__()
        self.temporal_mamba1 = temporal_mamba1
        self.temporal_mamba2 = temporal_mamba2

    def forward(self, filter_feature):
        """
        输入特征形状: [b, J3, seq_len]
        输出特征形状: [b, J3, seq_len]
        """
        # 第一次时间处理
        # [b, J3, seq_len] -> [b, seq_len, J3]
        temporal_feat, _ = self.temporal_mamba1(filter_feature.transpose(1, 2).contiguous())
        
        # 第二次时间处理（保持时间维度在中间）
        # [b, seq_len, J3] -> [b, seq_len, J3]
        temporal_feat, _ = self.temporal_mamba2(temporal_feat)
        
        # 恢复原始维度
        # [b, seq_len, J3] -> [b, J3, seq_len]
        return temporal_feat.transpose(1, 2).contiguous()
    
class SSSpatialModule(nn.Module):
    def __init__(self, spatial_mamba1, spatial_mamba2):
        """
        初始化 SS 分支模块（空间-空间）
        
        Args:
            spatial_mamba1 (nn.Module): 第一个空间维度Mamba模块
            spatial_mamba2 (nn.Module): 第二个空间维度Mamba模块
        """
        super(SSSpatialModule, self).__init__()
        self.spatial_mamba1 = spatial_mamba1
        self.spatial_mamba2 = spatial_mamba2

    def forward(self, filter_feature):
        """
        输入特征形状: [b, J3, seq_len]
        输出特征形状: [b, J3, seq_len]
        """
        # 第一次空间处理
        # [b, J3, seq_len] -> [b, J3, seq_len]
        spatial_feat, _ = self.spatial_mamba1(filter_feature)
        
        # 第二次空间处理
        # [b, J3, seq_len] -> [b, J3, seq_len]
        spatial_feat, _ = self.spatial_mamba2(spatial_feat)
        return spatial_feat
