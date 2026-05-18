"""
Multi-Scale Graph Neural Networks
=================================
Implements multi-hop graph convolutions to capture patterns at different scales.
Reference: Multi-scale GNN architectures for NIDS (2024-2025)

Key innovations:
1. Parallel graph convolutions at 1-hop, 2-hop, and 3-hop distances
2. Adaptive fusion of multi-scale features
3. Edge-aware attention with edge features
4. Residual connections and layer normalization for stability
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EdgeAwareGATLayer(nn.Module):
    """
    Enhanced Edge-Aware Graph Attention Layer
    - Multi-head attention
    - Edge feature integration
    - Residual connections
    - Layer normalization
    """
    def __init__(self, in_dim, out_dim, edge_dim, num_heads=4, dropout=0.2, concat=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        self.num_heads = num_heads
        self.concat = concat

        # Output dimension per head - out_dim must be divisible by num_heads
        assert out_dim % num_heads == 0, f"out_dim ({out_dim}) must be divisible by num_heads ({num_heads})"
        self.head_dim = out_dim // num_heads

        # Feature transformations
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.W_e = nn.Linear(edge_dim, num_heads, bias=False)

        # Attention parameters (source and target)
        self.att_src = nn.Parameter(torch.Tensor(1, num_heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.Tensor(1, num_heads, self.head_dim))

        # Edge attention parameter
        self.att_edge = nn.Parameter(torch.Tensor(1, num_heads, 1))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)

        # Residual projection
        if in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.residual_proj = None

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.W_e.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.xavier_uniform_(self.att_edge)

    def forward(self, x, edge_index, edge_attr=None):
        """
        x: (N, in_dim) - node features
        edge_index: (2, E) - edge indices [source, target]
        edge_attr: (E, edge_dim) - edge features
        """
        N = x.size(0)

        # Transform node features
        h = self.W(x)  # (N, out_dim)

        # Split into heads: (N, num_heads * head_dim) -> (N, num_heads, head_dim)
        h = h.view(N, self.num_heads, self.head_dim)

        # Extract source and target features
        src_idx, dst_idx = edge_index[0], edge_index[1]
        h_src = h[src_idx]  # (E, num_heads, head_dim)
        h_dst = h[dst_idx]  # (E, num_heads, head_dim)

        # Compute attention coefficients
        attn_src = (h_src * self.att_src).sum(dim=-1)  # (E, num_heads)
        attn_dst = (h_dst * self.att_dst).sum(dim=-1)  # (E, num_heads)
        attn = self.leaky_relu(attn_src + attn_dst)  # (E, num_heads)

        # Incorporate edge features
        if edge_attr is not None:
            edge_contrib = self.W_e(edge_attr)  # (E, num_heads)
            edge_attn = self.att_edge.squeeze(-1)  # (num_heads,)
            attn = attn + edge_contrib * edge_attn

        # Softmax normalization
        attn_exp = torch.exp(attn - attn.max(dim=0, keepdim=True)[0])

        # Manual scatter-based softmax
        norm = torch.zeros(N, self.num_heads, device=x.device)
        norm.index_add_(0, dst_idx, attn_exp)
        norm = norm[dst_idx] + 1e-8
        attn = attn_exp / norm
        attn = self.dropout(attn)

        # Message passing
        messages = attn.unsqueeze(-1) * h_src  # (E, num_heads, head_dim)

        # Aggregate to destination nodes
        out = torch.zeros(N, self.num_heads, self.head_dim, device=x.device)
        out.index_add_(0, dst_idx, messages)

        # Combine heads
        if self.concat:
            out = out.view(N, -1)  # (N, num_heads * head_dim) = (N, out_dim)
        else:
            out = out.mean(dim=1)  # (N, head_dim)
            # When not concatenating, expand to out_dim
            if self.head_dim != self.out_dim:
                # Project to out_dim
                out = out.repeat(1, self.num_heads)[:, :self.out_dim]

        # Layer normalization
        out = self.layer_norm(out)

        # Residual connection
        if self.residual_proj is not None:
            residual = self.residual_proj(x)
        else:
            residual = x
        out = out + residual

        return out


class MultiHopGraphConv(nn.Module):
    """
    Multi-hop graph convolution that aggregates information from k-hop neighbors.
    """
    def __init__(self, in_dim, out_dim, hop=1, dropout=0.2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hop = hop

        self.transform = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()
        self.layer_norm = nn.LayerNorm(out_dim)

        if in_dim != out_dim:
            self.residual = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.residual = None

    def forward(self, x, adj_matrix):
        """
        x: (N, in_dim) - node features
        adj_matrix: (N, N) - sparse or dense adjacency matrix
        """
        N = x.size(0)
        h = x

        # Multi-hop aggregation
        for _ in range(self.hop):
            if isinstance(adj_matrix, torch.Tensor):
                # Dense matrix multiplication
                h = torch.matmul(adj_matrix, h)
            else:
                # Sparse matrix multiplication (TODO: implement if needed)
                h = torch.matmul(adj_matrix.to_dense(), h)

        # Transform
        out = self.transform(h)
        out = self.layer_norm(out)
        out = self.activation(out)
        out = self.dropout(out)

        # Residual
        if self.residual is not None:
            res = self.residual(x)
        else:
            res = x

        return out + res


class MultiScaleGNN(nn.Module):
    """
    Multi-scale Graph Neural Network combining different hop distances.
    Captures patterns at 1-hop (local), 2-hop (medium), and 3-hop (global) scales.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_scales=3, num_heads=4, edge_dim=16, dropout=0.2):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_scales = num_scales

        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # Multi-scale GAT layers
        self.gat_1hop = EdgeAwareGATLayer(hidden_dim, hidden_dim, edge_dim, num_heads, dropout, concat=False)
        self.gat_2hop = EdgeAwareGATLayer(hidden_dim, hidden_dim, edge_dim, num_heads, dropout, concat=False)
        self.gat_3hop = EdgeAwareGATLayer(hidden_dim, hidden_dim, edge_dim, num_heads, dropout, concat=False)

        # Multi-hop convolution layers (alternative to GAT for some scales)
        self.conv_1hop = MultiHopGraphConv(hidden_dim, hidden_dim, hop=1, dropout=dropout)
        self.conv_2hop = MultiHopGraphConv(hidden_dim, hidden_dim, hop=2, dropout=dropout)
        self.conv_3hop = MultiHopGraphConv(hidden_dim, hidden_dim, hop=3, dropout=dropout)

        # Adaptive fusion weights
        self.scale_weights = nn.Parameter(torch.ones(num_scales))

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_scales, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, out_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def build_adj_from_edges(self, edge_index, num_nodes, edge_weight=None):
        """
        Build adjacency matrix from edge indices.
        """
        adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
        adj[edge_index[0], edge_index[1]] = edge_weight
        return adj

    def power_adj(self, adj, power):
        """
        Compute adjacency matrix raised to power (for multi-hop).
        """
        result = adj.clone()
        for _ in range(power - 1):
            result = torch.matmul(result, adj)
        # Normalize to avoid exploding values
        row_sum = result.sum(dim=1, keepdim=True)
        result = result / (row_sum + 1e-8)
        return result

    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass with multi-scale aggregation.
        """
        N = x.size(0)

        # Input projection
        h = self.input_proj(x)
        h = F.silu(h)

        # Build adjacency matrix for multi-hop conv
        adj = self.build_adj_from_edges(edge_index, N)

        # Multi-scale features
        scale_features = []

        # 1-hop (local neighborhood)
        h_1hop_gat = self.gat_1hop(h, edge_index, edge_attr)
        scale_features.append(h_1hop_gat)

        # 2-hop (medium neighborhood)
        adj_2hop = self.power_adj(adj, 2)
        h_2hop = self.conv_2hop(h, adj_2hop)
        scale_features.append(h_2hop)

        # 3-hop (global neighborhood)
        adj_3hop = self.power_adj(adj, 3)
        h_3hop = self.conv_3hop(h, adj_3hop)
        scale_features.append(h_3hop)

        # Adaptive fusion with learned weights
        weights = F.softmax(self.scale_weights, dim=0)
        weighted_features = [w * f for w, f in zip(weights, scale_features)]

        # Concatenate and fuse
        multi_scale = torch.cat(weighted_features, dim=-1)
        fused = self.fusion(multi_scale)

        # Output projection
        out = self.output_proj(fused)

        return out


class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer for capturing global context.
    Adapted for graph data (node-level sequences).
    """
    def __init__(self, d_model, num_heads=8, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.SiLU()

    def forward(self, src, src_mask=None):
        """
        src: (batch_size, seq_len, d_model)
        """
        # Self-attention
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class GraphTransformerEncoder(nn.Module):
    """
    Graph-aware Transformer encoder that processes node features with global attention.
    """
    def __init__(self, in_dim, hidden_dim, num_layers=2, num_heads=8, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 2, dropout)
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        x: (N, in_dim) - node features
        """
        # Project to hidden dimension
        h = self.input_proj(x)
        h = F.silu(h)

        # Add batch dimension for transformer (treat as sequence of length N)
        h = h.unsqueeze(0)  # (1, N, hidden_dim)

        # Transformer layers
        for layer in self.transformer_layers:
            h = layer(h)

        # Remove batch dimension
        h = h.squeeze(0)  # (N, hidden_dim)

        # Output projection
        out = self.output_proj(h)
        out = self.layer_norm(out)

        return out


if __name__ == "__main__":
    # Test EdgeAwareGATLayer
    print("Testing EdgeAwareGATLayer...")
    gat = EdgeAwareGATLayer(64, 128, edge_dim=16, num_heads=4, dropout=0.1)
    x = torch.randn(100, 64)
    edge_index = torch.randint(0, 100, (2, 500))
    edge_attr = torch.randn(500, 16)
    y = gat(x, edge_index, edge_attr)
    print(f"Input: {x.shape}, Output: {y.shape}")

    # Test MultiScaleGNN
    print("\nTesting MultiScaleGNN...")
    msgnn = MultiScaleGNN(64, 128, 64, num_scales=3, num_heads=4, edge_dim=16, dropout=0.1)
    y = msgnn(x, edge_index, edge_attr)
    print(f"Input: {x.shape}, Output: {y.shape}")

    # Test GraphTransformerEncoder
    print("\nTesting GraphTransformerEncoder...")
    transformer = GraphTransformerEncoder(64, 128, num_layers=2, num_heads=8, dropout=0.1)
    y = transformer(x)
    print(f"Input: {x.shape}, Output: {y.shape}")

    print("\nAll tests passed!")
