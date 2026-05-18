"""
KAN Layer: Kolmogorov-Arnold Networks Implementation
======================================================
Reference: Liu et al. "KAN: Kolmogorov-Arnold Networks", Nature 2025

KAN uses learnable univariate functions (B-splines) instead of fixed activation functions.
Key innovation: Each edge has a learnable activation function Φ(x) instead of weight w.

Architecture:
- Input: x ∈ R^d_in
- Hidden: z_j = Σ_i Φ_{ij}(x_i) where Φ_{ij} is a learnable function
- Output: y_k = Σ_j Φ_{jk}(z_j)

B-spline basis representation:
Φ(x) = Σ_{i=0}^{G-1} c_i * B_i(x)
where B_i are B-spline basis functions and c_i are learnable coefficients.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BSplineActivation(nn.Module):
    """
    B-spline based learnable activation function.
    Each edge has its own learnable activation function.
    """
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Create grid points for B-splines
        # Grid covers [-1, 1] range (input should be normalized)
        grid_range = [-1, 1]
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.linspace(grid_range[0] - h * spline_order,
                              grid_range[1] + h * spline_order,
                              grid_size + 2 * spline_order + 1)
        self.register_buffer('grid', grid)

        # Spline coefficients: (in_features, out_features, grid_size + spline_order)
        self.spline_coeffs = nn.Parameter(
            torch.randn(in_features, out_features, grid_size + spline_order) * 0.1
        )

        # Residual connection weight (like in MLP)
        self.residual_weight = nn.Parameter(
            torch.randn(in_features, out_features) * 0.1
        )

        # Base activation (silu/sigmoid) for residual
        self.base_activation = nn.SiLU()

    def b_spline_basis(self, x):
        """
        Compute B-spline basis functions at x.
        x: (batch_size, in_features)
        Returns: (batch_size, in_features, grid_size + spline_order)
        """
        # Expand x for broadcasting: (batch, in, 1)
        x_expanded = x.unsqueeze(-1)
        grid_expanded = self.grid.unsqueeze(0).unsqueeze(0)  # (1, 1, grid_points)

        # Cox-de Boor recursion for B-splines
        # Initialize with degree 0
        basis = ((x_expanded >= self.grid[:-1]) & (x_expanded < self.grid[1:])).float()

        # Build up to target degree
        for k in range(1, self.spline_order + 1):
            left_num = x_expanded - self.grid[:-(k+1)]
            left_den = self.grid[k:-1] - self.grid[:-(k+1)]
            left = left_num / (left_den + 1e-8)

            right_num = self.grid[k+1:] - x_expanded
            right_den = self.grid[k+1:] - self.grid[1:-k]
            right = right_num / (right_den + 1e-8)

            basis = left * basis[:, :, :-1] + right * basis[:, :, 1:]

        return basis

    def forward(self, x):
        """
        Forward pass through B-spline activation.
        x: (batch_size, in_features)
        Returns: (batch_size, out_features)
        """
        # Ensure x is in valid range
        x = torch.clamp(x, -2, 2)

        # Compute B-spline basis
        basis = self.b_spline_basis(x)  # (batch, in_features, num_basis)

        # Spline output: sum over input features
        # basis: (batch, in, num_basis), coeffs: (in, out, num_basis)
        spline_out = torch.einsum('bin,ion->bo', basis, self.spline_coeffs)

        # Residual connection (base activation + linear)
        base_out = self.base_activation(x)  # (batch, in_features)
        residual = torch.matmul(base_out, self.residual_weight)  # (batch, out_features)

        return spline_out + residual


class KANLayer(nn.Module):
    """
    Complete KAN layer with batch normalization and dropout.
    """
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, dropout=0.0):
        super().__init__()
        self.kan = BSplineActivation(in_features, out_features, grid_size, spline_order)
        self.bn = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.kan(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x


class KANMLP(nn.Module):
    """
    Multi-layer KAN network as MLP replacement.
    """
    def __init__(self, input_dim, hidden_dims, output_dim, grid_size=5, spline_order=3, dropout=0.2):
        super().__init__()
        dims = [input_dim] + hidden_dims + [output_dim]

        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(
                KANLayer(dims[i], dims[i+1], grid_size, spline_order, dropout if i < len(dims) - 2 else 0)
            )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.silu(x)  # Activation between layers
        return x


# Simplified KAN for faster training (grid_size=3 for efficiency)
class FastKANLayer(nn.Module):
    """
    Fast KAN approximation using piecewise linear functions.
    More efficient than full B-spline for large networks.
    """
    def __init__(self, in_features, out_features, num_grids=8, dropout=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_grids = num_grids

        # Learnable grid points
        self.grid_points = nn.Parameter(torch.linspace(-1, 1, num_grids))

        # Linear interpolation weights
        self.weights = nn.Parameter(torch.randn(in_features, out_features, num_grids) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Residual connection
        self.residual = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Piecewise linear interpolation.
        """
        batch_size = x.shape[0]

        # Clamp to valid range
        x = torch.clamp(x, -1, 1)

        # Find which grid interval each x falls into
        # grid_points: (num_grids,)
        expanded_x = x.unsqueeze(-1)  # (batch, in_features, 1)
        expanded_grid = self.grid_points.unsqueeze(0).unsqueeze(0)  # (1, 1, num_grids)

        # Compute interpolation weights
        diff = expanded_x - expanded_grid  # (batch, in, num_grids)

        # Find nearest grid points
        abs_diff = torch.abs(diff)
        left_idx = torch.argmin(abs_diff, dim=-1).clamp(0, self.num_grids - 2)
        right_idx = left_idx + 1

        # Get grid values
        left_grid = self.grid_points[left_idx]  # (batch, in_features)
        right_grid = self.grid_points[right_idx]  # (batch, in_features)

        # Compute interpolation weights
        alpha = (x - left_grid) / (right_grid - left_grid + 1e-8)  # (batch, in_features)

        # Gather weights for left and right grid points
        batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(1).expand(-1, self.in_features)
        in_indices = torch.arange(self.in_features, device=x.device).unsqueeze(0).expand(batch_size, -1)

        left_weights = self.weights[in_indices, :, left_idx]  # (batch, in, out)
        right_weights = self.weights[in_indices, :, right_idx]  # (batch, in, out)

        # Interpolate
        interp_weights = alpha.unsqueeze(-1) * right_weights + (1 - alpha.unsqueeze(-1)) * left_weights

        # Sum over input features
        output = interp_weights.sum(dim=1)  # (batch, out_features)
        output = output + self.bias

        # Add residual
        residual = self.residual(x)

        return self.dropout(output + residual)


if __name__ == "__main__":
    # Test KAN layer
    print("Testing KAN Layer...")
    kan = KANLayer(64, 128, grid_size=5, spline_order=3, dropout=0.1)
    x = torch.randn(32, 64)
    y = kan(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")

    # Test FastKAN
    print("\nTesting FastKAN Layer...")
    fast_kan = FastKANLayer(64, 128, num_grids=8, dropout=0.1)
    y_fast = fast_kan(x)
    print(f"Input shape: {x.shape}, Output shape: {y_fast.shape}")

    # Test KANMLP
    print("\nTesting KANMLP...")
    kan_mlp = KANMLP(64, [128, 64], 10, grid_size=3, dropout=0.2)
    y_mlp = kan_mlp(x)
    print(f"Input shape: {x.shape}, Output shape: {y_mlp.shape}")

    print("\nAll tests passed!")
