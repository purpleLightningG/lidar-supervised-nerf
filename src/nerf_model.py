"""
nerf_model.py — Vanilla NeRF architecture (Mildenhall et al., 2020).

A NeRF is just an MLP that maps:
    (x, y, z, view_dir) → (rgb, density σ)

The trick is:
  1. Encode position and direction with sinusoidal frequencies (positional encoding)
     so the MLP can fit high-frequency content despite being a smooth function
  2. Predict density σ from position only (view-independent geometry)
  3. Predict color from position + view direction (view-dependent appearance)

The same MLP architecture is used for both baseline and depth-supervised models.
The only difference is the loss function during training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding.

    Maps each scalar coordinate x → [x, sin(2^0 πx), cos(2^0 πx), sin(2^1 πx), cos(2^1 πx), ...]

    With L frequencies, this turns a D-dim input into a (D * (2L + 1))-dim feature.
    """

    def __init__(self, num_frequencies: int, include_input: bool = True):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        # Frequencies are 2^0, 2^1, ..., 2^(L-1) times pi
        self.register_buffer(
            'freq_bands',
            2.0 ** torch.arange(num_frequencies, dtype=torch.float32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., D) → (..., D * (2L + 1)) or (..., D * 2L) if not include_input."""
        encoded = [x] if self.include_input else []
        for freq in self.freq_bands:
            encoded.append(torch.sin(freq * x))
            encoded.append(torch.cos(freq * x))
        return torch.cat(encoded, dim=-1)

    @property
    def output_dim_multiplier(self) -> int:
        """Multiplier on input dim: 2L if not including input, 2L+1 otherwise."""
        return 2 * self.num_frequencies + (1 if self.include_input else 0)


class NeRFMLP(nn.Module):
    """Vanilla NeRF MLP.

    Architecture (from the original paper):
      - 8 hidden layers, 256 units each
      - Skip connection at layer 4 (concat input position encoding)
      - σ predicted from layer 8 output (positions only)
      - Color predicted from layer 8 output + view direction encoding

    Input position is encoded with 10 frequencies → 63 dims (3 * 21)
    Input direction is encoded with 4 frequencies → 27 dims (3 * 9)
    """

    def __init__(
        self,
        num_pos_freqs: int = 10,
        num_dir_freqs: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 8,
        skip_layer: int = 4,
    ):
        super().__init__()
        self.skip_layer = skip_layer

        # Encoders for position and direction
        self.pos_encoder = PositionalEncoding(num_pos_freqs)
        self.dir_encoder = PositionalEncoding(num_dir_freqs)

        pos_input_dim = 3 * self.pos_encoder.output_dim_multiplier  # 3 * 21 = 63
        dir_input_dim = 3 * self.dir_encoder.output_dim_multiplier  # 3 * 9  = 27

        # Backbone: position → density features
        layers = []
        for i in range(num_layers):
            if i == 0:
                in_dim = pos_input_dim
            elif i == skip_layer:
                # Skip connection: concat original position encoding
                in_dim = hidden_dim + pos_input_dim
            else:
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim))
        self.backbone = nn.ModuleList(layers)

        # Density head: 1 scalar from final backbone activation
        self.density_head = nn.Linear(hidden_dim, 1)

        # Color head: combine backbone features + view direction
        self.feature_head = nn.Linear(hidden_dim, hidden_dim)
        self.color_layer_1 = nn.Linear(hidden_dim + dir_input_dim, hidden_dim // 2)
        self.color_layer_2 = nn.Linear(hidden_dim // 2, 3)

    def forward(self, positions: torch.Tensor, directions: torch.Tensor):
        """
        Args:
            positions:  (N, 3) sample positions in world coords
            directions: (N, 3) view directions (unit vectors)

        Returns:
            density: (N,) raw σ values (post-ReLU done in renderer)
            color:   (N, 3) sigmoid-output RGB in [0, 1]
        """
        pos_enc = self.pos_encoder(positions)
        dir_enc = self.dir_encoder(directions)

        # Backbone with skip connection
        x = pos_enc
        for i, layer in enumerate(self.backbone):
            if i == self.skip_layer:
                x = torch.cat([x, pos_enc], dim=-1)
            x = F.relu(layer(x))

        # Density head (no activation here — applied in volume renderer)
        density = self.density_head(x).squeeze(-1)

        # Color head: project features, concat with view direction, two more layers
        feature = self.feature_head(x)
        color_input = torch.cat([feature, dir_enc], dim=-1)
        color = F.relu(self.color_layer_1(color_input))
        color = torch.sigmoid(self.color_layer_2(color))

        return density, color


class CoarseFineNeRF(nn.Module):
    """The standard NeRF setup uses two MLPs:
      - Coarse network for initial uniform sampling
      - Fine network for importance-sampled refinement
    """

    def __init__(self, **mlp_kwargs):
        super().__init__()
        self.coarse = NeRFMLP(**mlp_kwargs)
        self.fine = NeRFMLP(**mlp_kwargs)


if __name__ == '__main__':
    # Smoke test
    model = NeRFMLP()
    n_pts = 1024
    pos = torch.randn(n_pts, 3)
    dir = torch.randn(n_pts, 3)
    dir = dir / dir.norm(dim=-1, keepdim=True)

    density, color = model(pos, dir)
    print(f'Density shape: {density.shape}, range: [{density.min():.3f}, {density.max():.3f}]')
    print(f'Color shape:   {color.shape}, range: [{color.min():.3f}, {color.max():.3f}]')

    n_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {n_params:,}')
