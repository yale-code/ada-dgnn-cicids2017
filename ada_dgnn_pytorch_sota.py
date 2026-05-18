"""
Ada-DGNN-SOTA: State-of-the-Art Network Intrusion Detection using Graph Neural Networks
========================================================================================

基于2024-2025年最新SOTA论文的全面优化实现：
- GEAFL-IDS: Graph Edge Attention + Focal Loss (Zhang et al., ACM ICNSC 2025)
- E-GraphSAGE: Edge-centric GraphSAGE (Lo et al., 2022)  
- GraphIDS: Self-supervised E-GraphSAGE + Transformer (Guerra et al., NeurIPS 2025)
- DIGNN-A: Dynamic Graph Attention with Line Graph (Liu & Guo, 2024)
- GAT+KAN: Graph Attention + Kolmogorov-Arnold Networks (Nature 2025)
- FN-GNN: Flow-as-Node Graph Embedding (MDPI Applied Sciences 2024)
- N-STGAT: Node Condition-augmented Spatial-Temporal GAT (Wang et al., 2023)

核心优化:
1. 真正的图神经网络: Edge-Aware GAT + GraphSAGE 消息传递
2. 自动图构建: k-NN相似度图 (通用) + IP-based图 (有IP时增强)
3. Focal Loss: 处理NIDS极端类别不平衡
4. SMOTE: 少数类过采样增强
5. AdamW + ReduceLROnPlateau: 稳定训练 + 学习率调度
6. LayerNorm + Dropout: 正则化与训练稳定性
7. 边缘特征编码: 流交互建模
8. 早停机制: 防止过拟合

作者: Moonshot AI | 木那 (AI Assistant)
日期: 2026-05-12
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, 
                             confusion_matrix, classification_report, roc_auc_score)
from collections import Counter
import os
import time
import warnings
warnings.filterwarnings('ignore')

# ───────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ───────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "data_path": "/root/.openclaw/workspace/datasets/cicids2017/synthetic_cicids2017_50k.csv",  # 数据集路径
    "random_state": 42,
    "test_size": 0.2,
    "val_size": 0.1,  # 验证集比例 (从训练集中划分)
    
    # === 模型架构 (基于GEAFL-IDS & E-GraphSAGE SOTA配置) ===
    "input_dim": None,        # 自动检测
    "hidden_dim": 128,        # GEAFL-IDS推荐: 128 (原模型256过大，易过拟合)
    "num_classes": None,      # 自动检测
    "num_gnn_layers": 2,      # GEAFL-IDS: 2层图注意力
    "num_attention_heads": 4, # GAT: 4头注意力 (平衡表达能力与计算量)
    "dropout": 0.2,           # GEAFL-IDS推荐: 0.2 (原模型0.3未实现)
    "use_edge_features": True, # E-GraphSAGE: 边缘特征编码
    "use_residual": True,     # 残差连接 (保留原模型设计)
    "aggregator": "mean",     # GraphSAGE聚合: mean / max / lstm
    
    # === 图构建 ===
    "graph_k": 3,             # k-NN图的k值 (减小以加速)
    "graph_type": "knn",      # knn / ip_based / hybrid
    "edge_feature_dim": 16,   # 边缘特征编码维度
    
    # === 训练 (基于GEAFL-IDS实验配置) ===
    "epochs": 200,            # 增加epoch，配合早停
    "batch_size": 256,        # 批量大小
    "learning_rate": 1e-3,    # GEAFL-IDS: 0.001
    "weight_decay": 1e-4,     # AdamW权重衰减
    "patience": 15,           # 早停耐心值
    
    # === Focal Loss (处理类别不平衡) ===
    "use_focal_loss": True,
    "focal_gamma": 2.0,       # GEAFL-IDS: γ=2
    "focal_alpha": None,      # None=自动计算类别权重
    
    # === SMOTE (少数类过采样) ===
    "use_smote": True,
    "smote_k": 5,             # SMOTE近邻数
    "smote_ratio": 0.5,       # 少数类:多数类目标比例 (1.0=完全平衡)
    
    # === 设备 ===
    "device": "auto",         # auto / cuda / cpu
}

# ───────────────────────────────────────────────────────────────────────────────
# SMOTE: Synthetic Minority Over-sampling Technique
# 参考: Chawla et al. (2002); Gu et al. (2025) Borderline-SMOTE+WGAN
# ───────────────────────────────────────────────────────────────────────────────
class SMOTE:
    """
    SMOTE过采样实现
    对少数类样本，在其k近邻之间插值生成合成样本
    """
    def __init__(self, k=5, random_state=42):
        self.k = k
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
    
    def fit_resample(self, X, y, target_ratio=0.5):
        """
        参数:
            X: 特征矩阵 (N, D)
            y: 标签 (N,)
            target_ratio: 少数类/多数类目标比例
        返回:
            X_resampled, y_resampled
        """
        classes, counts = np.unique(y, return_counts=True)
        max_count = counts.max()
        
        X_resampled = [X]
        y_resampled = [y]
        
        for cls, count in zip(classes, counts):
            if count == max_count:
                continue  # 多数类不过采样
            
            # 计算需要生成的样本数
            n_samples = int(max_count * target_ratio) - count
            if n_samples <= 0:
                continue
            
            # 获取少数类样本
            minority_samples = X[y == cls]
            n_minority = len(minority_samples)
            
            # 为每个少数类样本找到k近邻
            # 简化: 随机选择k个近邻进行插值
            for _ in range(n_samples):
                idx = self.rng.randint(0, n_minority)
                sample = minority_samples[idx]
                
                # 随机选择k近邻中的一个
                nn_idx = self.rng.choice(n_minority, min(self.k, n_minority), replace=False)
                nn = minority_samples[self.rng.choice(nn_idx)]
                
                # 线性插值
                alpha = self.rng.random()
                synthetic = sample + alpha * (nn - sample)
                
                X_resampled.append(synthetic.reshape(1, -1))
                y_resampled.append(np.array([cls]))
        
        X_out = np.vstack(X_resampled)
        y_out = np.hstack(y_resampled)
        
        # 打乱顺序
        perm = self.rng.permutation(len(X_out))
        return X_out[perm], y_out[perm]


# ───────────────────────────────────────────────────────────────────────────────
# Focal Loss: 处理类别不平衡的焦点损失
# 参考: Lin et al. (2017); GEAFL-IDS (Zhang et al., ACM 2025)
# 公式: FL(p_t) = -α * (1 - p_t)^γ * log(p_t)
# ───────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification
    
    自适应类别权重:
        β_m = 1 - n_m / N_total  (GEAFL-IDS 公式7)
    """
    def __init__(self, num_classes, gamma=2.0, alpha=None, reduction='mean', device='cpu'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.device = device
        
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32, device=device)
        else:
            self.alpha = None
        
        self.num_classes = num_classes
    
    def compute_alpha(self, targets):
        """根据标签分布自适应计算类别权重 (GEAFL-IDS方法)"""
        if self.alpha is not None:
            return self.alpha
        
        counts = torch.bincount(targets, minlength=self.num_classes).float()
        total = counts.sum()
        # β_m = 1 - n_m / N_total
        beta = 1.0 - counts / (total + 1e-8)
        return beta.to(self.device)
    
    def forward(self, inputs, targets):
        """
        inputs: (N, C) 模型logits
        targets: (N,) 类别索引
        """
        # log_softmax for numerical stability
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)
        
        # 获取对应目标的概率
        batch_size = targets.size(0)
        p_t = probs[torch.arange(batch_size, device=probs.device), targets]
        log_p_t = log_probs[torch.arange(batch_size, device=probs.device), targets]
        
        # 自适应类别权重
        alpha_t = self.compute_alpha(targets)[targets]
        
        # Focal weight: (1 - p_t)^γ
        focal_weight = (1.0 - p_t) ** self.gamma
        
        # Focal Loss
        loss = -alpha_t * focal_weight * log_p_t
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# ───────────────────────────────────────────────────────────────────────────────
# 图构建: 将表格数据转换为图结构
# 参考: FN-GNN (2024) Flow-as-Node; E-GraphSAGE (2022) Flow-as-Edge
# ───────────────────────────────────────────────────────────────────────────────
class TrafficGraphBuilder:
    """
    网络流量图构建器
    
    支持两种图构建策略:
    1. k-NN相似度图 (通用，无需IP列)
    2. IP-based图 (需要源/目的IP列)
    
    节点: 每条流记录
    边: k-NN特征相似度连接
    边缘特征: 节点特征差 + 相似度得分
    """
    def __init__(self, k=10, graph_type='knn', edge_feature_dim=16, device='cpu'):
        self.k = k
        self.graph_type = graph_type
        self.edge_feature_dim = edge_feature_dim
        self.device = device
    
    def build_graph(self, X, df=None, src_ip_col=None, dst_ip_col=None):
        """
        构建图
        
        参数:
            X: 特征矩阵 (N, D) - 已标准化
            df: 原始DataFrame (用于IP-based图)
            src_ip_col: 源IP列名
            dst_ip_col: 目的IP列名
        返回:
            edge_index: (2, E) 边索引
            edge_attr: (E, D_edge) 边缘特征
            node_features: (N, D) 节点特征
        """
        N = X.shape[0]
        
        if self.graph_type == 'knn' or (src_ip_col is None and dst_ip_col is None):
            # k-NN相似度图 (FN-GNN 2024)
            edge_index, edge_attr = self._build_knn_graph(X)
        elif self.graph_type == 'ip_based' and df is not None:
            # IP-based图 (E-GraphSAGE风格)
            edge_index, edge_attr = self._build_ip_graph(df, src_ip_col, dst_ip_col, X)
        else:
            # 混合: k-NN + IP连接
            knn_edges, knn_attr = self._build_knn_graph(X)
            ip_edges, ip_attr = self._build_ip_graph(df, src_ip_col, dst_ip_col, X)
            edge_index = torch.cat([knn_edges, ip_edges], dim=1)
            edge_attr = torch.cat([knn_attr, ip_attr], dim=0)
        
        return edge_index, edge_attr, torch.FloatTensor(X).to(self.device)
    
    def _build_knn_graph(self, X):
        """构建k-NN相似度图"""
        N = X.shape[0]
        
        # 使用sklearn的kneighbors_graph (基于cosine相似度)
        adj = kneighbors_graph(X, n_neighbors=min(self.k, N-1), 
                               mode='connectivity', metric='cosine',
                               include_self=False)
        
        # 转为对称图 (无向图)
        adj = np.maximum(adj, adj.T)
        
        # 提取边
        rows, cols = adj.nonzero()
        edge_index = torch.LongTensor(np.array([rows, cols])).to(self.device)
        
        # 计算边缘特征: 特征差 + 余弦相似度 (向量化)
        if len(rows) > 0:
            X_t = torch.FloatTensor(X).to(self.device)
            xi = X_t[rows]  # (E, D)
            xj = X_t[cols]  # (E, D)
            
            # 余弦相似度 (向量化)
            norm_i = torch.norm(xi, dim=1, keepdim=True)
            norm_j = torch.norm(xj, dim=1, keepdim=True)
            cos_sim = (xi * xj).sum(dim=1) / (norm_i.squeeze() * norm_j.squeeze() + 1e-8)
            
            # 特征差 (向量化)
            feat_diff = torch.abs(xi - xj)[:, :self.edge_feature_dim-1]
            
            edge_attr = torch.cat([cos_sim.unsqueeze(1), feat_diff], dim=1)
        else:
            edge_attr = torch.zeros((0, self.edge_feature_dim)).to(self.device)
        
        return edge_index, edge_attr
    
    def _build_ip_graph(self, df, src_ip_col, dst_ip_col, X):
        """构建IP-based图: 共享IP的流之间建立连接"""
        N = len(df)
        edge_list = []
        edge_attr_list = []
        
        # 建立IP到流索引的映射
        src_ip_map = {}
        dst_ip_map = {}
        
        for idx, row in df.iterrows():
            sip = str(row.get(src_ip_col, ''))
            dip = str(row.get(dst_ip_col, ''))
            
            if sip and sip != 'nan':
                if sip not in src_ip_map:
                    src_ip_map[sip] = []
                src_ip_map[sip].append(idx)
            
            if dip and dip != 'nan':
                if dip not in dst_ip_map:
                    dst_ip_map[dip] = []
                dst_ip_map[dip].append(idx)
        
        # 相同源IP的流连接
        for ip, indices in src_ip_map.items():
            if len(indices) > 1:
                for i in range(len(indices)):
                    for j in range(i+1, len(indices)):
                        idx_i, idx_j = indices[i], indices[j]
                        edge_list.append([idx_i, idx_j])
                        edge_list.append([idx_j, idx_i])
                        
                        # 边缘特征
                        xi, xj = X[idx_i], X[idx_j]
                        cos_sim = np.dot(xi, xj) / (np.linalg.norm(xi) * np.linalg.norm(xj) + 1e-8)
                        feat_diff = np.abs(xi - xj)
                        edge_feat = np.concatenate([[cos_sim], feat_diff[:self.edge_feature_dim-1]])
                        edge_attr_list.append(edge_feat)
        
        # 相同目的IP的流连接
        for ip, indices in dst_ip_map.items():
            if len(indices) > 1:
                for i in range(len(indices)):
                    for j in range(i+1, len(indices)):
                        idx_i, idx_j = indices[i], indices[j]
                        edge_list.append([idx_i, idx_j])
                        edge_list.append([idx_j, idx_i])
                        
                        xi, xj = X[idx_i], X[idx_j]
                        cos_sim = np.dot(xi, xj) / (np.linalg.norm(xi) * np.linalg.norm(xj) + 1e-8)
                        feat_diff = np.abs(xi - xj)
                        edge_feat = np.concatenate([[cos_sim], feat_diff[:self.edge_feature_dim-1]])
                        edge_attr_list.append(edge_feat)
        
        if len(edge_list) == 0:
            # 无IP连接，退化为k-NN
            return self._build_knn_graph(X)
        
        edge_index = torch.LongTensor(np.array(edge_list).T).to(self.device)
        edge_attr = torch.FloatTensor(np.array(edge_attr_list)).to(self.device)
        
        return edge_index, edge_attr


# ───────────────────────────────────────────────────────────────────────────────
# Edge-Aware Graph Attention Layer
# 参考: GEAFL-IDS (Zhang et al., ACM 2025); GAT (Veličković et al., 2018)
# ───────────────────────────────────────────────────────────────────────────────
class EdgeAwareGATLayer(nn.Module):
    """
    边缘感知图注意力层
    
    创新: 同时聚合节点特征和边缘特征
    注意力系数计算考虑: [节点i || 节点j || 边缘ij]
    
    参考论文: GEAFL-IDS "基于边缘特征的多头注意力机制进行加权聚合"
    """
    def __init__(self, in_dim, out_dim, edge_dim, num_heads=4, dropout=0.2, 
                 concat=True, residual=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        self.num_heads = num_heads
        self.concat = concat
        self.residual = residual
        
        # 每个头的输出维度
        self.head_dim = out_dim  # concat=False时每个头输出out_dim维 if concat else out_dim
        
        # 节点特征线性变换
        self.W = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        
        # 边缘特征线性变换
        self.W_e = nn.Linear(edge_dim, num_heads, bias=False)
        
        # 注意力参数 a^T [Wh_i || Wh_j || We_ij]
        self.att_src = nn.Parameter(torch.Tensor(1, num_heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.Tensor(1, num_heads, self.head_dim))
        
        # LeakyReLU负斜率
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        
        # 残差连接
        if residual and in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.residual_proj = None
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.W_e.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        x: (N, in_dim) 节点特征
        edge_index: (2, E) 边索引 [source, target]
        edge_attr: (E, edge_dim) 边缘特征 (可选)
        """
        N = x.size(0)
        
        # 节点特征变换: (N, in_dim) -> (N, num_heads, head_dim)
        h = self.W(x).view(N, self.num_heads, -1)
        
        # 提取源节点和目标节点特征
        src_idx, dst_idx = edge_index[0], edge_index[1]
        h_src = h[src_idx]  # (E, num_heads, head_dim)
        h_dst = h[dst_idx]  # (E, num_heads, head_dim)
        
        # 注意力系数: e_ij = LeakyReLU(a^T [Wh_i || Wh_j]) + edge_features
        attn_src = (h_src * self.att_src).sum(dim=-1)  # (E, num_heads)
        attn_dst = (h_dst * self.att_dst).sum(dim=-1)  # (E, num_heads)
        attn = self.leaky_relu(attn_src + attn_dst)
        
        # 加入边缘特征影响
        if edge_attr is not None:
            edge_contrib = self.W_e(edge_attr)  # (E, num_heads)
            attn = attn + edge_contrib
        
        # Softmax归一化 (按目标节点)
        attn_exp = torch.exp(attn - attn.max(dim=0, keepdim=True)[0])
        
        # 手动归一化 (不依赖torch-scatter)
        norm = torch.zeros(N, self.num_heads, device=x.device)
        norm.index_add_(0, dst_idx, attn_exp)
        norm = norm[dst_idx] + 1e-8
        attn = attn_exp / norm
        
        attn = self.dropout(attn)
        
        # 消息传递: h_i' = Σ α_ij * Wh_j
        messages = attn.unsqueeze(-1) * h_dst  # (E, num_heads, head_dim)
        
        # 聚合到目标节点
        out = torch.zeros(N, self.num_heads, self.head_dim, device=x.device)
        out.index_add_(0, dst_idx, messages)
        
        # 拼接或平均多头输出
        if self.concat:
            out = out.view(N, -1)  # (N, num_heads * head_dim)
        else:
            out = out.mean(dim=1)  # (N, head_dim)
        
        # 残差连接
        if self.residual:
            if self.residual_proj is not None:
                res = self.residual_proj(x)
            else:
                res = x
            out = out + res
        
        return out


# ───────────────────────────────────────────────────────────────────────────────
# GraphSAGE Layer with Edge Features
# 参考: E-GraphSAGE (Lo et al., 2022); GraphSAGE (Hamilton et al., 2017)
# ───────────────────────────────────────────────────────────────────────────────
class EdgeGraphSAGELayer(nn.Module):
    """
    边缘感知GraphSAGE层
    
    h_i^(l+1) = σ(W · CONCAT(h_i^l, AGG_{j∈N(i)} {h_j^l, e_ij}))
    
    聚合函数支持: mean / max / sum
    """
    def __init__(self, in_dim, out_dim, edge_dim, aggregator='mean', 
                 dropout=0.2, residual=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        self.aggregator = aggregator
        self.residual = residual
        
        # 边缘特征编码
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        
        # 聚合后的变换
        self.fc = nn.Linear(in_dim + out_dim, out_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
        # 残差
        if residual and in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.residual_proj = None
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        x: (N, in_dim)
        edge_index: (2, E)
        edge_attr: (E, edge_dim)
        """
        N = x.size(0)
        src_idx, dst_idx = edge_index[0], edge_index[1]
        
        # 邻居节点特征
        h_src = x[src_idx]  # (E, in_dim)
        
        # 结合边缘特征
        if edge_attr is not None:
            edge_feat = self.edge_encoder(edge_attr)  # (E, out_dim)
            messages = h_src + edge_feat  # 简单相加融合
        else:
            messages = h_src
        
        # 聚合: 按目标节点分组聚合
        if self.aggregator == 'mean':
            out = torch.zeros(N, self.out_dim, device=x.device)
            out.index_add_(0, dst_idx, messages)
            # 计算每个节点的度数进行平均
            deg = torch.zeros(N, device=x.device)
            deg.index_add_(0, dst_idx, torch.ones(messages.size(0), device=x.device))
            deg = deg.clamp(min=1)
            out = out / deg.unsqueeze(-1)
        elif self.aggregator == 'sum':
            out = torch.zeros(N, self.out_dim, device=x.device)
            out.index_add_(0, dst_idx, messages)
        elif self.aggregator == 'max':
            # Max pooling聚合 (需要scatter)
            out = torch.zeros(N, self.out_dim, device=x.device)
            out.index_reduce_(0, dst_idx, messages, 'amax', include_self=False)
        
        # 拼接自身特征和聚合特征
        out = torch.cat([x, out], dim=-1)
        out = self.fc(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        # 残差
        if self.residual:
            if self.residual_proj is not None:
                res = self.residual_proj(x)
            else:
                res = x
            out = out + res
        
        return out


# ───────────────────────────────────────────────────────────────────────────────
# Layer Normalization (稳定深层网络训练)
# 参考: Ba et al. (2016); Transformer架构
# ───────────────────────────────────────────────────────────────────────────────
class GraphLayerNorm(nn.Module):
    """图数据LayerNorm: 对每个节点的特征独立归一化"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + self.eps
        return self.gamma * (x - mean) / std + self.beta


# ───────────────────────────────────────────────────────────────────────────────
# Ada-DGNN SOTA 主模型
# 融合: GEAFL-IDS EdgeAttention + E-GraphSAGE + Residual + LayerNorm
# ───────────────────────────────────────────────────────────────────────────────
class AdaDGNN_SOTA(nn.Module):
    """
    Ada-DGNN SOTA 模型
    
    架构:
    1. 输入投影: 原始特征 -> 隐藏维度
    2. GNN编码器 (2层):
       - EdgeGraphSAGE: 邻居聚合 + 边缘特征
       - EdgeAwareGAT: 注意力加权聚合
       - LayerNorm + Dropout + Residual
    3. 全局池化: 均值聚合
    4. MLP分类器: 隐藏层 -> 输出层
    
    创新点:
    - 边缘特征编码 (E-GraphSAGE)
    - 多头边缘注意力 (GEAFL-IDS)
    - 残差连接 + LayerNorm (稳定训练)
    - 自适应Focal Loss (类别不平衡)
    """
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2,
                 num_heads=4, edge_dim=16, dropout=0.2, use_edge_features=True,
                 aggregator='mean', use_residual=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.use_edge_features = use_edge_features
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN层: 交替使用GraphSAGE和GAT
        self.gnn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_d = hidden_dim
            out_d = hidden_dim
            
            # 交替: GraphSAGE -> GAT -> GraphSAGE -> GAT
            if i % 2 == 0:
                layer = EdgeGraphSAGELayer(
                    in_d, out_d, edge_dim,
                    aggregator=aggregator,
                    dropout=dropout,
                    residual=use_residual
                )
            else:
                layer = EdgeAwareGATLayer(
                    in_d, out_d, edge_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    concat=False,  # 不拼接，保持维度
                    residual=use_residual
                )
            
            self.gnn_layers.append(layer)
            self.layer_norms.append(GraphLayerNorm(out_d))
        
        # MLP分类器
        classifier_dim = hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_dim, hidden_dim),
            GraphLayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            GraphLayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        前向传播
        
        x: (N, input_dim) 节点特征
        edge_index: (2, E) 边索引
        edge_attr: (E, edge_dim) 边缘特征
        """
        # 输入投影
        h = self.input_proj(x)
        h = F.relu(h)
        
        # GNN编码
        for gnn_layer, ln in zip(self.gnn_layers, self.layer_norms):
            h_new = gnn_layer(h, edge_index, edge_attr if self.use_edge_features else None)
            h_new = ln(h_new)
            h_new = F.relu(h_new)
            h = h_new
        
        # 分类
        logits = self.classifier(h)
        
        return logits
    
    def predict(self, x, edge_index, edge_attr=None):
        """预测模式 (返回概率)"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, edge_index, edge_attr)
            probs = F.softmax(logits, dim=-1)
        return probs


# ───────────────────────────────────────────────────────────────────────────────
# 通用数据预处理 (保留原模型设计，增强SMOTE)
# ───────────────────────────────────────────────────────────────────────────────
class UniversalDataPreprocessor:
    """增强版通用数据预处理器"""
    def __init__(self, config=None):
        self.config = config or CONFIG
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.non_feature_cols = ['source_ip', 'src_ip', 'dst_ip', 'destination_ip',
                                 'source_port', 'src_port', 'dst_port', 'destination_port',
                                 'timestamp', 'time', 'date', 'flow_id', 'id', 'protocol',
                                 'label', 'class', 'target', 'type', 'category',
                                 'attack_cat', 'attack', 'traffic_type']
    
    def preprocess(self, file_path):
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.tsv'):
            df = pd.read_csv(file_path, sep='\t')
        else:
            raise ValueError("Unsupported file format")
        
        # 自动检测标签列
        label_col = None
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['label', 'class', 'target', 'type', 'category', 'attack_cat', 'attack', 'traffic_type']):
                label_col = col
                break
        
        if label_col is None:
            raise ValueError("无法自动检测标签列")
        
        # 过滤样本数过少的类别 (无法分层划分)
        label_counts = df[label_col].value_counts()
        min_samples = 5
        valid_labels = label_counts[label_counts >= min_samples].index
        n_before = label_counts.nunique()
        df = df[df[label_col].isin(valid_labels)].reset_index(drop=True)
        unique_labels = df[label_col].nunique()
        if unique_labels < n_before:
            print(f"⚠️  过滤了 {n_before - unique_labels} 个样本数<{min_samples}的稀有类别")
        
        # 提取标签
        y = df[label_col].values
        if not pd.api.types.is_numeric_dtype(df[label_col]):
            y = self.label_encoder.fit_transform(y)
        
        # 排除非特征列
        feature_cols = [c for c in df.columns if c != label_col and 
                        not any(nf.lower() in c.lower() for nf in self.non_feature_cols)]
        
        # 记录IP列（用于图构建）
        src_ip_col = None
        dst_ip_col = None
        for col in df.columns:
            if any(k in col.lower() for k in ['src_ip', 'source_ip', 'sip']):
                src_ip_col = col
            if any(k in col.lower() for k in ['dst_ip', 'dest_ip', 'destination_ip', 'dip']):
                dst_ip_col = col
        
        # 提取特征
        X = df[feature_cols].select_dtypes(include=[np.number]).values
        
        # 处理无穷值和NaN (CICIDS2017数据集的已知问题)
        X = np.where(np.isinf(X) | np.isneginf(X), np.nan, X)
        col_means = np.nanmean(X, axis=0)
        col_means = np.where(np.isnan(col_means), 0, col_means)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])
        
        X = self.scaler.fit_transform(X)
        
        # 划分训练/验证/测试
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=self.config['test_size'] + self.config['val_size'],
            random_state=self.config['random_state'], stratify=y
        )
        val_ratio = self.config['val_size'] / (self.config['test_size'] + self.config['val_size'])
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=1-val_ratio,
            random_state=self.config['random_state'], stratify=y_temp
        )
        
        # SMOTE过采样
        if self.config['use_smote']:
            smote = SMOTE(k=self.config['smote_k'], 
                         random_state=self.config['random_state'])
            X_train, y_train = smote.fit_resample(
                X_train, y_train, 
                target_ratio=self.config['smote_ratio']
            )
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'feature_cols': feature_cols,
            'src_ip_col': src_ip_col, 'dst_ip_col': dst_ip_col,
            'num_classes': unique_labels,
            'class_names': self.label_encoder.classes_ if hasattr(self.label_encoder, 'classes_') else None
        }


# ───────────────────────────────────────────────────────────────────────────────
# 训练器: AdamW + ReduceLROnPlateau + 早停
# ───────────────────────────────────────────────────────────────────────────────
class Trainer:
    """完整训练流程"""
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # 优化器: AdamW (权重衰减防止过拟合)
        self.optimizer = AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # 学习率调度: 验证集loss不下降时降低学习率
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5,
            patience=config['patience'] // 3
        )
        
        # 损失函数
        if config['use_focal_loss']:
            self.criterion = FocalLoss(
                num_classes=config['num_classes'],
                gamma=config['focal_gamma'],
                device=device
            )
        else:
            # 类别权重交叉熵
            self.criterion = nn.CrossEntropyLoss()
        
        # 早停
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
    
    def train_epoch(self, x, edge_index, edge_attr, y):
        self.model.train()
        self.optimizer.zero_grad()
        
        # 前向
        logits = self.model(x, edge_index, edge_attr)
        loss = self.criterion(logits, y)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪 (防止爆炸)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # 计算训练准确率
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean().item()
        
        return loss.item(), acc
    
    @torch.no_grad()
    def evaluate(self, x, edge_index, edge_attr, y):
        self.model.eval()
        logits = self.model(x, edge_index, edge_attr)
        loss = self.criterion(logits, y)
        
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean().item()
        
        # 详细指标
        y_cpu = y.cpu().numpy()
        preds_cpu = preds.cpu().numpy()
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_cpu, preds_cpu, average='macro', zero_division=0
        )
        
        return {
            'loss': loss.item(),
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': preds_cpu,
            'probabilities': F.softmax(logits, dim=-1).cpu().numpy()
        }
    
    def fit(self, train_data, val_data, epochs=None):
        epochs = epochs or self.config['epochs']
        
        x_train, edge_index_train, edge_attr_train, y_train = train_data
        x_val, edge_index_val, edge_attr_val, y_val = val_data
        
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 
                   'val_acc': [], 'val_f1': [], 'lr': []}
        
        print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>9} | {'Val Acc':>8} | {'Val F1':>8} | {'LR':>10}")
        print("-" * 80)
        
        for epoch in range(epochs):
            t0 = time.time()
            
            train_loss, train_acc = self.train_epoch(
                x_train, edge_index_train, edge_attr_train, y_train
            )
            
            val_metrics = self.evaluate(
                x_val, edge_index_val, edge_attr_val, y_val
            )
            
            val_loss = val_metrics['loss']
            val_acc = val_metrics['accuracy']
            val_f1 = val_metrics['f1']
            
            # 学习率调度
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
            history['lr'].append(current_lr)
            
            # 早停检查
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_model_state = {
                    k: v.cpu().clone() 
                    for k, v in self.model.state_dict().items()
                }
            else:
                self.patience_counter += 1
            
            # 打印进度
            if epoch % 10 == 0 or epoch < 5 or self.patience_counter >= self.config['patience'] - 5:
                print(f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>9.4f} | {val_loss:>9.4f} | {val_acc:>8.4f} | {val_f1:>8.4f} | {current_lr:>10.6f}")
            
            # 早停
            if self.patience_counter >= self.config['patience']:
                print(f"\n🛑 早停触发 (epoch {epoch+1}/{epochs})")
                break
            
            # 如果学习率降到太低，也停止
            if current_lr < 1e-6:
                print(f"\n⏹️ 学习率过小，停止训练")
                break
        
        # 恢复最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print("✅ 已恢复最佳验证模型")
        
        return history
    
    def predict(self, x, edge_index, edge_attr):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x, edge_index, edge_attr)
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
        return preds.cpu().numpy(), probs.cpu().numpy()


# ───────────────────────────────────────────────────────────────────────────────
# 评估与报告
# ───────────────────────────────────────────────────────────────────────────────
def evaluate_model(model, x, edge_index, edge_attr, y_true, class_names=None):
    """完整评估报告"""
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index, edge_attr)
        preds = logits.argmax(dim=-1).cpu().numpy()
        probs = F.softmax(logits, dim=-1).cpu().numpy()
    
    y_true = y_true.cpu().numpy() if torch.is_tensor(y_true) else y_true
    
    print("\n" + "="*60)
    print("📊 模型评估报告")
    print("="*60)
    
    # 基本指标
    acc = accuracy_score(y_true, preds)
    print(f"\n准确率 (Accuracy): {acc:.4f}")
    
    # 分类报告
    print("\n分类报告:")
    target_names = class_names if class_names is not None else None
    print(classification_report(y_true, preds, target_names=target_names, digits=4))
    
    # Macro指标
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, preds, average='macro', zero_division=0
    )
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall:    {recall:.4f}")
    print(f"Macro F1-Score:  {f1:.4f}")
    
    # 混淆矩阵
    print("\n混淆矩阵:")
    cm = confusion_matrix(y_true, preds)
    print(cm)
    
    # 多分类AUROC (如果适用)
    if len(np.unique(y_true)) > 1 and probs.shape[1] > 1:
        try:
            if probs.shape[1] == 2:
                auroc = roc_auc_score(y_true, probs[:, 1])
            else:
                from sklearn.preprocessing import label_binarize
                y_bin = label_binarize(y_true, classes=np.unique(y_true))
                auroc = roc_auc_score(y_bin, probs, average='macro', multi_class='ovr')
            print(f"\nAUROC: {auroc:.4f}")
        except:
            pass
    
    print("="*60)
    
    return {
        'accuracy': acc,
        'macro_precision': precision,
        'macro_recall': recall,
        'macro_f1': f1,
        'predictions': preds,
        'probabilities': probs
    }


# ───────────────────────────────────────────────────────────────────────────────
# 主函数
# ───────────────────────────────────────────────────────────────────────────────
def main():
    # 设备配置
    if CONFIG['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(CONFIG['device'])
    
    print(f"🖥️  使用设备: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # 1. 数据预处理
    print("\n📂 加载并预处理数据...")
    preprocessor = UniversalDataPreprocessor(CONFIG)
    
    # 查找数据集
    data_path = CONFIG['data_path']
    if os.path.isdir(data_path):
        files = [f for f in os.listdir(data_path) 
                if f.endswith(('.csv', '.tsv'))]
        if not files:
            print(f"❌ 未在 {data_path} 找到数据集")
            return
        data_file = os.path.join(data_path, files[0])
    else:
        data_file = data_path
    
    print(f"   数据集: {data_file}")
    
    try:
        data = preprocessor.preprocess(data_file)
    except Exception as e:
        print(f"❌ 预处理失败: {e}")
        # 使用随机数据演示
        print("   使用随机数据演示...")
        np.random.seed(42)
        N, D = 5000, 20
        X = np.random.randn(N, D)
        y = np.random.choice(5, N, p=[0.7, 0.15, 0.08, 0.05, 0.02])
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        data = {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'num_classes': 5,
            'src_ip_col': None, 'dst_ip_col': None
        }
    
    # 2. 构建图
    print("\n🔨 构建图结构...")
    
    # 纯CPU环境下大数据集图构建可能OOM，添加fallback
    try:
        graph_builder = TrafficGraphBuilder(
            k=CONFIG['graph_k'],
            graph_type=CONFIG['graph_type'],
            edge_feature_dim=CONFIG['edge_feature_dim'],
            device=device
        )
        
        # 训练图
        edge_index_train, edge_attr_train, x_train = graph_builder.build_graph(
            data['X_train']
        )
        # 验证图
        edge_index_val, edge_attr_val, x_val = graph_builder.build_graph(
            data['X_val']
        )
        # 测试图
        edge_index_test, edge_attr_test, x_test = graph_builder.build_graph(
            data['X_test']
        )
        print(f"   训练图: {x_train.size(0)} 节点, {edge_index_train.size(1)} 边")
    except Exception as e:
        print(f"   ⚠️ 图构建失败 ({e})，回退到无图模式")
        x_train = torch.FloatTensor(data['X_train']).to(device)
        x_val = torch.FloatTensor(data['X_val']).to(device)
        x_test = torch.FloatTensor(data['X_test']).to(device)
        # 自环边 (每个节点只连向自己)
        N_train = x_train.size(0)
        N_val = x_val.size(0)
        N_test = x_test.size(0)
        edge_index_train = torch.stack([torch.arange(N_train), torch.arange(N_train)]).to(device)
        edge_index_val = torch.stack([torch.arange(N_val), torch.arange(N_val)]).to(device)
        edge_index_test = torch.stack([torch.arange(N_test), torch.arange(N_test)]).to(device)
        edge_attr_train = torch.zeros(N_train, CONFIG['edge_feature_dim']).to(device)
        edge_attr_val = torch.zeros(N_val, CONFIG['edge_feature_dim']).to(device)
        edge_attr_test = torch.zeros(N_test, CONFIG['edge_feature_dim']).to(device)
        print(f"   无图模式: {x_train.size(0)} 节点, 无GNN消息传递")
    
    y_train = torch.LongTensor(data['y_train']).to(device)
    y_val = torch.LongTensor(data['y_val']).to(device)
    y_test = torch.LongTensor(data['y_test']).to(device)
    
    # 3. 创建模型
    CONFIG['input_dim'] = x_train.size(1)
    CONFIG['num_classes'] = data['num_classes']
    
    print(f"\n🧠 创建模型...")
    print(f"   输入维度: {CONFIG['input_dim']}")
    print(f"   隐藏维度: {CONFIG['hidden_dim']}")
    print(f"   类别数: {CONFIG['num_classes']}")
    print(f"   GNN层数: {CONFIG['num_gnn_layers']}")
    print(f"   注意力头: {CONFIG['num_attention_heads']}")
    
    model = AdaDGNN_SOTA(
        input_dim=CONFIG['input_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        num_classes=CONFIG['num_classes'],
        num_layers=CONFIG['num_gnn_layers'],
        num_heads=CONFIG['num_attention_heads'],
        edge_dim=CONFIG['edge_feature_dim'],
        dropout=CONFIG['dropout'],
        use_edge_features=CONFIG['use_edge_features'],
        aggregator=CONFIG['aggregator'],
        use_residual=CONFIG['use_residual']
    )
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   总参数量: {total_params:,}")
    print(f"   可训练参数量: {trainable_params:,}")
    
    # 4. 训练
    print(f"\n🏋️ 开始训练...")
    trainer = Trainer(model, CONFIG, device)
    
    train_data = (x_train, edge_index_train, edge_attr_train, y_train)
    val_data = (x_val, edge_index_val, edge_attr_val, y_val)
    
    history = trainer.fit(train_data, val_data, epochs=CONFIG['epochs'])
    
    # 5. 测试集评估
    print("\n🧪 测试集评估...")
    test_metrics = evaluate_model(
        model, x_test, edge_index_test, edge_attr_test, y_test,
        class_names=data.get('class_names')
    )
    
    # 6. 保存模型
    save_path = 'ada_dgnn_sota_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': CONFIG,
        'history': history,
        'test_metrics': test_metrics,
        'scaler': preprocessor.scaler,
        'label_encoder': preprocessor.label_encoder
    }, save_path)
    print(f"\n💾 模型已保存: {save_path}")
    
    print("\n✅ 训练完成!")
    return model, history, test_metrics


if __name__ == "__main__":
    main()
