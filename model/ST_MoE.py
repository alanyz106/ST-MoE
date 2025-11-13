import numpy as np
import torch
import torch.nn as nn
from model.layers import TimeCon
from model.layers import GCN
from model.layers.Mamba import Mamba
from model.layers.Mamba import Encoder,EncoderLayer,EncoderLayerV2
from model.Experts import TSTemporalSpatialModule
from model.Experts import STTemporalSpatialModule
from model.Experts import TTTemporalModule
from model.Experts import SSSpatialModule
import torch.nn.functional as F

class SharedMambaBlocks(nn.Module):
    """共享时空Mamba模块的基类"""
    def __init__(self, mid_feature, num_kpt, depth=1):
        super().__init__()
        # 构建共享的时空编码器
        self.shared_temporal = self._build_encoder(d_model=num_kpt*3, depth=depth)
        self.shared_spatial = self._build_encoder(d_model=mid_feature, depth=depth)
        
    def _build_encoder(self, d_model, depth):
        return Encoder(
            [
                EncoderLayer(
                    Mamba(
                        d_model=d_model,
                        d_state=32,
                        d_conv=2,
                        expand=1
                    ),
                    Mamba(
                        d_model=d_model,
                        d_state=32,
                        d_conv=2,
                        expand=1
                    ),
                    d_model=d_model,
                    dropout=0.1,
                    activation='gelu'
                ) for _ in range(depth)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )

class FourExpertsPool_Shared(SharedMambaBlocks):
    def __init__(self, mid_feature, num_kpt, depth=1, num_experts=4):
        super().__init__(mid_feature, num_kpt, depth)
        
        # 初始化专家模块（共享核心参数）
        self.ts_expert = TSTemporalSpatialModule(
            temporal_mamba=self.shared_temporal,
            spatial_mamba=self.shared_spatial
        )
        
        self.st_expert = STTemporalSpatialModule(
            spatial_mamba=self.shared_spatial,
            temporal_mamba=self.shared_temporal
        )
        
        self.tt_expert = TTTemporalModule(
            temporal_mamba1=self.shared_temporal,
            temporal_mamba2=self.shared_temporal
        )
        
        self.ss_expert = SSSpatialModule(
            spatial_mamba1=self.shared_spatial,
            spatial_mamba2=self.shared_spatial
        )

        self.experts = nn.ModuleList([
            self.ts_expert, self.st_expert,
            self.ss_expert, self.tt_expert
        ])

    def forward(self, x):
        expert_outputs = [expert(x) for expert in self.experts]
        return torch.stack(expert_outputs, dim=1)
           
class FourExpertsPool(nn.Module):
    def __init__(self, mid_feature, num_kpt, depth=1, num_experts=4):
        super().__init__()
        
        # 统一Encoder构建函数
        def build_encoder(d_model, depth):
            return Encoder(
                [
                    EncoderLayer(
                        Mamba(
                            d_model=d_model,
                            d_state=32,
                            d_conv=2,
                            expand=1
                        ),
                        Mamba(
                            d_model=d_model,
                            d_state=32,
                            d_conv=2,
                            expand=1
                        ),
                        d_model=d_model,
                        dropout=0.1,
                        activation='gelu'
                    ) for _ in range(depth)
                ],
                norm_layer=nn.LayerNorm(d_model)
            )

        # TS Expert (保持原有结构)
        self.ts_expert = TSTemporalSpatialModule(
            temporal_mamba=build_encoder(num_kpt*3, depth),
            spatial_mamba=build_encoder(mid_feature, depth)
        )

        # ST Expert 
        self.st_expert = STTemporalSpatialModule(
            spatial_mamba=build_encoder(mid_feature, depth),
            temporal_mamba=build_encoder(num_kpt*3, depth)
        )

        # SS Expert
        self.ss_expert = SSSpatialModule(
            spatial_mamba1=build_encoder(mid_feature, depth),
            spatial_mamba2=build_encoder(mid_feature, depth)
        )

        # TT Expert
        self.tt_expert = TTTemporalModule(
            temporal_mamba1=build_encoder(num_kpt*3, depth),
            temporal_mamba2=build_encoder(num_kpt*3, depth)
        )

        self.experts = nn.ModuleList([
            self.ts_expert, self.st_expert,
            self.ss_expert, self.tt_expert
        ])

    def forward(self, x):
        expert_outputs = [expert(x) for expert in self.experts]
        return torch.stack(expert_outputs, dim=1)
    
class ExpertGating(nn.Module):
    def __init__(self, input_dim, num_experts=4, top_k=2):
        super().__init__()
        self.top_k = top_k
        
        # 门控生成器
        self.gate_generator = nn.Linear(input_dim, num_experts)
        self.capture_gating_weights = False  # 新增捕获开关
        self.captured_weights = None         # 存储捕获的权重

        # 初始化参数
        nn.init.xavier_normal_(self.gate_generator.weight)
        nn.init.constant_(self.gate_generator.bias, 0.1)

    def forward(self, x):
        # 输入形状: [B, J3, T]
        B, J3, T = x.shape
        
        # 时空全局平均池化
        pooled = x.mean(dim=[-2])  # [B,seq_len]
        
        # 生成门控权重
        logits = self.gate_generator(pooled)  # [B, 4]

        if self.capture_gating_weights:
            # 捕获原始权重值（非稀疏）
            full_weights = F.softmax(logits, dim=-1)
            self.captured_weights = full_weights.clone().detach()
        
        # Top-K选择
        top_logits, indices = torch.topk(logits, self.top_k, dim=-1)
        sparse_weights = torch.zeros_like(logits).scatter(-1, indices, F.softmax(top_logits, dim=-1))
        
        return sparse_weights, indices
    
class MoELayer(nn.Module):
    def __init__(self, mid_feature, num_kpt, opt):
        super().__init__()
        self.expert_pool = FourExpertsPool_Shared(
            mid_feature=opt.seq_len,
            num_kpt=num_kpt,
            depth=opt.depth,
            num_experts=opt.num_experts
        )
        self.top_k = opt.top_k
        self.gate = ExpertGating(
            input_dim=mid_feature,
            num_experts=opt.num_experts,
            top_k=opt.top_k
        )
        # 添加标记用于捕获专家输出
        self.capture_expert_output = False
        self.expert_outputs = None  # 用于存储专家输出
         # 残差系数（可学习）
        # self.res_coef = nn.Parameter(torch.tensor(0.1))
        # self.norm = nn.LayerNorm(mid_feature)

    def forward(self, x):
        # residual = x  # [B, J3, T]
        B, J3, T = x.shape
         # 标准化
        # x = self.norm(x)
        # 获取所有专家输出
        expert_outputs = self.expert_pool(x)  # [B, 4, J3, T]

        # 如果设置了捕获标记，保存专家输出
        if self.capture_expert_output:
            # 平均池化减少序列长度维度
            self.expert_outputs = expert_outputs.mean(dim=-1)  # [B, 4, J3]
            # 展平空间维度
            self.expert_outputs = self.expert_outputs.reshape(
                self.expert_outputs.shape[0], 
                self.expert_outputs.shape[1], 
                -1
            )  # [B, 4, J3] -> [B, 4, num_kpt*3]
            
        # 获取门控权重
        gate_weights, indices = self.gate(x)  # [B, 4], [B, 2]
        
         # 动态选择专家
        selected_experts = torch.gather(
            expert_outputs,
            dim=1,
            index=indices.view(B, self.top_k, 1, 1).expand(-1, -1, J3, T)
        )  # [B, 2, J3, T]
        
        # 加权融合
        weights = gate_weights.gather(1, indices).view(B, self.top_k, 1, 1)  # [B, 2, 1, 1]
        moe_out = (selected_experts * weights).sum(dim=1)  # [B, J3, T]
        
        # 残差连接
        # return moe_out + self.res_coef * residual
        return moe_out
       
class ST_MoE(nn.Module):
    def __init__(self, seq_len, d_model, opt, num_kpt, dataset):
        super(ST_MoE, self).__init__()
        self.opt = opt
        self.mid_feature = opt.seq_len
        self.dataset = dataset
        self.seq_len = seq_len
        self.num_kpt = num_kpt
        self.dct, self.idct = self.get_dct_matrix(self.num_kpt*3)

        self.w_sp = self.opt.w_sp
        self.w_tp = self.opt.w_tp

        self.GCNQ1 = GCN.GCN(input_feature=self.mid_feature,
                             hidden_feature=d_model,
                             p_dropout=0.3,
                             num_stage=self.opt.num_stage,
                             node_n=num_kpt * 3)#2

        self.GCNQ2 = GCN.GCN(input_feature=self.mid_feature,
                             hidden_feature=d_model,
                             p_dropout=0.3,
                             num_stage=self.opt.num_stage,
                             node_n=num_kpt * 3)
        
        # 初始化MoE层
        self.moe_layers = nn.ModuleList([
            MoELayer(
                mid_feature=self.mid_feature,
                num_kpt=num_kpt,
                opt=opt
            ) for _ in range(opt.nlayer)
        ])

        self.timecon = TimeCon.timecon_plus()

    def forward(self, input_ori, gt):
        # [96, 3, 45, 75],[bs,n,j*3,seqlen]
        input = torch.matmul(self.dct, input_ori)

        if self.dataset == "Mocap" or self.dataset == "CHI3D":
            input = input
        elif self.dataset == "Human3.6M":
            input = input.unsqueeze(dim=1)
            input_ori = input_ori.unsqueeze(dim=1)
            gt = gt.unsqueeze(dim=1)

        num_person = np.shape(input)[1]


        for i in range(num_person):
            people_in = input[:, i, :, :].clone()
            # people_in = input[:, i, :, :]
            if i == 0:
                people_feature_all = self.GCNQ1(people_in).unsqueeze(1).clone()
            else:
                people_feature_all = torch.cat([people_feature_all, self.GCNQ1(people_in).unsqueeze(1).clone()], 1)
        # people_feature_all [96, 3, 45, 75],[bs,n,j*3,seqlen]

        for i in range(num_person):

            people_feature = people_feature_all[:, i, :, :]
            
            filter_feature = people_feature.clone()
            
            # 通过MoE层处理
            for moe_layer in self.moe_layers:
                filter_feature = moe_layer(filter_feature)  # [B, J3, T]
                
            feature = filter_feature + people_feature.clone()

            feature = self.GCNQ2(feature)
            feature = torch.matmul(self.idct, feature)
            feature = feature.transpose(1, 2)

            if i == 0:
                predic = feature.unsqueeze(1).clone()
            else:
                predic = torch.cat([predic, feature.unsqueeze(1).clone()], 1)

        # predic[16, 3, 75, 45] gt[16, 3, 45, 75]
        loss = self.mix_loss(predic, gt)


        if self.dataset == "Mocap" or self.dataset == "CHI3D":
            return predic, loss
        elif self.dataset == "Human3.6M":
            return predic[:, 0, :, :], loss

    def mix_loss(self, predic, gt):

        gt = gt.transpose(2, 3)
        bs, n, sql, _ = gt.shape #torch.Size([96, 3, 75, 45])

        spacial_loss_pred = torch.mean(torch.norm((predic[:, :, self.opt.frame_in:, :] - gt[:, :, self.opt.frame_in:, :]), dim=3))
        spacial_loss_ori = torch.mean(torch.norm((predic[:, :, :self.opt.frame_in, :] - gt[:, :, :self.opt.frame_in, :]), dim=3))
        spacial_loss = spacial_loss_pred + spacial_loss_ori * 0.1

        temporal_loss = 0


        for idx_person in range(n):

            # tempo_pre [96, 192, 1, 1] tempo_ref [96, 192, 1, 1]
            tempo_pre = self.timecon(predic[:, idx_person, :, :].unsqueeze(1))
            tempo_ref = self.timecon(gt[:, idx_person, :, :].unsqueeze(1))
            
            temporal_loss += torch.mean(torch.norm(tempo_pre-tempo_ref, dim=3))

        loss = self.w_sp * spacial_loss + self.w_tp * temporal_loss 
        # loss = self.w_sp * spacial_loss + self.w_tp * temporal_loss

        return loss


    def get_dct_matrix(self, N):
        # Computes the discrete cosine transform (DCT) matrix and its inverse (IDCT)
        dct_m = np.eye(N)
        for k in np.arange(N):
            for i in np.arange(N):
                w = np.sqrt(2 / N)
                if k == 0:
                    w = np.sqrt(1 / N)
                dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
        idct_m = np.linalg.inv(dct_m)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dct_m = torch.tensor(dct_m).float().to(device)
        idct_m = torch.tensor(idct_m).float().to(device)
        return dct_m, idct_m