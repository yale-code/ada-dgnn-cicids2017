# Ada-DGNN SOTA 优化分析报告

## 一、当前模型诊断

### 严重问题

| 问题 | 影响 | 修复优先级 |
|------|------|-----------|
| **无反向传播** | 模型根本不学习，`train_epoch`只计算loss不更新权重 | 🔴 P0 |
| **不是真正的GNN** | 只是带残差连接的MLP，无图结构、无消息传递、无邻居聚合 | 🔴 P0 |
| **Dropout未实现** | 配置有`dropout=0.3`但代码中完全没有使用 | 🟡 P1 |
| **无类别不平衡处理** | NIDS数据集极度不平衡（正常>99%），模型会偏向多数类 | 🔴 P0 |
| **无图构建** | 网络流数据是表格形式，未转化为图结构 | 🔴 P0 |
| **优化器缺失** | 无Adam，无学习率调度，SGD都没实现 | 🟡 P1 |
| **无归一化层** | 无LayerNorm/BatchNorm，深层网络训练不稳定 | 🟡 P1 |
| **无边缘特征** | E-GraphSAGE等SOTA方法证明边缘特征至关重要 | 🟡 P1 |

### 当前模型本质
```
Input → MLP → ReLU → Residual → MLP → ReLU → Residual → ... → Softmax
```
这不是DGNN（Dynamic Graph Neural Network），这是一个带残差连接的深度前馈网络。

---

## 二、2024-2025 SOTA 方法综述

基于检索到的最新论文，NIDS领域GNN方向的关键进展：

### 1. GEAFL-IDS (Zhang et al., 2025c) — ACM会议论文
- **核心**: 图边缘注意力 + Focal Loss
- **架构**: 2层边缘注意力（第1层3头，第2层1头），隐藏层128
- **图构建**: 流作为边，通过k-NN/相似度构建图
- **类别不平衡**: Focal Loss with β_m = 1 - n_m/N_total, γ=2
- **结果**: NF-BoT-IoT 83.08%, NF-UNSW-NB15 97.87%
- **配置**: Dropout=0.2, Adam, lr=0.001

### 2. GraphIDS (Guerra et al., NeurIPS 2025)
- **核心**: 自监督E-GraphSAGE + Transformer Autoencoder
- **创新**: 无需标签预训练，重构误差检测异常
- **结果**: 99.98% PR-AUC, 99.61% macro F1
- **实现**: https://github.com/lorenzo9uerra/GraphIDS

### 3. DIGNN-A (Liu & Guo, 2024)
- **核心**: 动态图 + 注意力 + 线图快照
- **创新**: 图快照分割 → 线图转换（边分类→节点分类）
- **数据集**: UNSW-NB15, NF-ToN-IoT-v2

### 4. GConvTrans (2025 PMC)
- **核心**: GCN层 + Transformer编码器 混合架构
- **优势**: 同时捕捉局部图结构和全局序列上下文
- **结果**: CSE-CIC-IDS2018 测试96.94%

### 5. GAT + KAN (2025 Nature)
- **核心**: 图注意力网络 + Kolmogorov-Arnold网络
- **创新**: KAN替代MLP进行下游分类，B-spline激活函数
- **应用**: 智能电网入侵检测

### 6. GNN-IDS (Sun et al., ARES 2024)
- **核心**: 攻击图 + 实时测量 联合建模
- **创新**: 静态攻击图与动态测量结合
- **优势**: 不仅检测异常，还能识别攻击路径

### 7. FIR-GNN (2025)
- **核心**: Flow Interaction Representation
- **创新**: 流交互表示学习

### 8. 自监督GNN (Xu et al., 2024; Caville et al., 2022)
- **Anomal-E**: E-GraphSAGE + 互信息最大化 + PCA/IF
- **NEGSC**: 生成子图对比学习，注意力聚合边缘特征
- **TS-IDS**: 流量感知自监督学习
- **CoGN**: 协作对比机制

### 9. 类别不平衡处理 (2024-2025)
- **VAE-GAN-Guided**: VAE-GAN跨类生成数据增强
- **KGSMOTE**: 核密度估计几何SMOTE
- **SMOTEENN**: SMOTE + Edited Nearest Neighbor
- **Borderline-SMOTE + WGAN**: 边界样本过采样
- **Class-wise Focal Loss + XGBoost**: XIDINTFL-VAE

### 10. 图构建策略
| 方法 | 节点 | 边 | 代表论文 |
|------|------|-----|---------|
| **IP-based** | 主机/IP地址 | 流连接通信端点 | Lo et al. 2022 |
| **Flow-as-node** | 每条流记录 | 相同源IP的流相连 | FN-GNN 2024 |
| **Flow-as-edge** | 主机/端口 | 流作为边特征 | E-GraphSAGE 2022 |
| **k-NN相似度** | 流记录 | 特征空间Top-k相似 | GAT-IoT 2025 |
| **Line graph** | 原始图的边 | 共享端点的边相连 | DIGNN-A 2024 |
| **攻击图** | 网络事件/漏洞 | 攻击路径关系 | GNN-IDS 2024 |

---

## 三、优化方案设计

### 方案A: PyTorch Geometric 完整SOTA实现（推荐）

基于 **GEAFL-IDS + E-GraphSAGE + Focal Loss** 的融合架构：

```
┌─────────────────────────────────────────────────────────────┐
│  输入: 表格流数据 (CSV/TSV)                                   │
│                                                             │
│  预处理层                                                   │
│  ├── 数值特征标准化 (Z-score)                                │
│  ├── 类别特征编码 (可选)                                     │
│  ├── SMOTE过采样 (处理类别不平衡)                            │
│  └── 图构建: k-NN相似度图 / IP-based图 / 混合                │
│                                                             │
│  图神经网络层                                                │
│  ├── EdgeAttention Layer 1 (3 heads, hidden=128)          │
│  │   └── 边缘特征加权聚合 → ReLU → Dropout(0.2)             │
│  ├── EdgeAttention Layer 2 (1 head, hidden=128)             │
│  │   └── 边缘特征加权聚合 → ReLU → Dropout(0.2)             │
│  └── 边缘嵌入 → 节点嵌入 (均值聚合)                          │
│                                                             │
│  分类层                                                     │
│  ├── MLP (128 → 64 → num_classes)                          │
│  ├── LayerNorm (稳定训练)                                   │
│  └── Softmax                                                │
│                                                             │
│  训练                                                       │
│  ├── Focal Loss (γ=2, β_m自适应)                            │
│  ├── Adam优化器 (lr=0.001)                                  │
│  ├── ReduceLROnPlateau调度                                  │
│  └── 早停 (patience=15)                                     │
└─────────────────────────────────────────────────────────────┘
```

### 方案B: NumPy 增强版（轻量、无依赖）

在纯NumPy框架下尽可能修复和增强：
- 修复反向传播（数值梯度或简化SGD）
- k-NN图构建（特征相似度）
- 简化单头注意力
- Focal Loss
- SMOTE（简化版）
- 真正的Dropout
- LayerNorm（简化版）

---

## 四、关键技术实现要点

### 4.1 Focal Loss（类别不平衡）
```python
FL(p_t) = -β_m * (1 - p_t)^γ * log(p_t)
β_m = 1 - n_m / N_total   # 类别权重
γ = 2                      # 聚焦参数
```
- 降低易分类样本权重
- 增加难分类/少数类样本关注

### 4.2 边缘注意力（Edge Attention）
```python
# 边缘(i,j)的注意力系数
α_ij = softmax(LeakyReLU(a^T [Wh_i || Wh_j || We_ij]))
# 聚合
h_i' = Σ α_ij * (Wh_j + We_ij)
```
- 同时考虑节点特征和边缘特征
- 多头注意力提供多角度表示

### 4.3 图构建（k-NN）
```python
# 特征相似度图
A[i,j] = 1 if j in top-k_similar(i) else 0
# 对称化
A = A | A.T
```

### 4.4 E-GraphSAGE 消息传递
```python
h_i^(l+1) = σ(W^l · CONCAT(h_i^l, AGG_{j∈N(i)} {h_j^l, e_ij}))
```
- 聚合邻居节点特征 + 边缘特征
- 支持归纳式学习（新节点无需重训练）

### 4.5 SMOTE 过采样
```python
# 对少数类样本，在其k近邻之间插值生成新样本
x_new = x_i + λ * (x_j - x_i), λ∈[0,1]
```

---

## 五、预期性能提升

| 指标 | 当前模型 | SOTA优化后 |
|------|---------|-----------|
| 是否真正学习 | ❌ 否 | ✅ 是 |
| 准确率 | ~随机 | 90-99%（依数据集） |
| Macro F1 | ~0 | 0.70-0.95 |
| 少数类召回 | ~0 | 显著提升 |
| 训练稳定性 | 差 | Adam + LayerNorm稳定 |

---

## 六、文件输出

| 文件 | 说明 |
|------|------|
| `ada_dgnn_pytorch_sota.py` | PyTorch Geometric完整SOTA实现 |
| `ada_dgnn_numpy_sota.py` | NumPy增强版（修复bug+关键优化） |
| `requirements_sota.txt` | 依赖清单 |

---

*分析基于2024-2025年ACM、IEEE、Nature、NeurIPS等顶级会议/期刊论文*
*核心参考: GEAFL-IDS(2025), GraphIDS(NeurIPS2025), DIGNN-A(2024), GConvTrans(2025), GAT+KAN(2025)*
