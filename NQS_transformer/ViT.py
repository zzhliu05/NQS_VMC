from __future__ import annotations

import math

import torch
from torch import Tensor, nn




def count_model_parameters(model: nn.Module, verbose: bool = True) -> int:
    total_params = sum(param.numel() for param in model.parameters())
    if verbose:
        print(f"[Model] Total parameters: {total_params}")
    return total_params


class LinearTokenEmbedding(nn.Module):
    """Tokenize a site configuration and project each token to d_model.

    Input shapes:
    - (num_sites,)
    - (batch_size, num_sites)

    Output shape:
    - (batch_size, num_tokens, d_model)
    """

    def __init__(self, num_sites: int, token_size: int, d_model: int, bias: bool = True) -> None:
        super().__init__()
        if num_sites <= 0:
            raise ValueError("num_sites must be positive.")
        if token_size <= 0:
            raise ValueError("token_size must be positive.")
        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        if num_sites % token_size != 0:
            raise ValueError(
                f"num_sites ({num_sites}) must be divisible by token_size ({token_size})."
            )

        self.num_sites = num_sites
        self.token_size = token_size
        self.num_tokens = num_sites // token_size
        self.d_model = d_model
        self.proj = nn.Linear(token_size + 1, d_model, bias=bias)

    def tokenize(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() != 2:
            raise ValueError(
                "Input configuration must have shape (num_sites,) or (batch_size, num_sites)."
            )
        if x.size(-1) != self.num_sites:
            raise ValueError(
                f"Expected the last dimension to be {self.num_sites}, got {x.size(-1)}."
            )

        x = x.to(dtype=self.proj.weight.dtype)
        return x.reshape(x.size(0), self.num_tokens, self.token_size)

    def append_gamma(self, tokens: Tensor, gamma: Tensor | float) -> Tensor:
        if not torch.is_tensor(gamma):
            gamma = torch.tensor(gamma, dtype=tokens.dtype, device=tokens.device)
        else:
            gamma = gamma.to(dtype=tokens.dtype, device=tokens.device)

        if gamma.dim() == 0:
            gamma = gamma.view(1, 1, 1).expand(tokens.size(0), tokens.size(1), 1)
        elif gamma.dim() == 1:
            if gamma.size(0) != tokens.size(0):
                raise ValueError(
                    f"Batch gamma must have shape ({tokens.size(0)},), got {tuple(gamma.shape)}."
                )
            gamma = gamma.view(tokens.size(0), 1, 1).expand(tokens.size(0), tokens.size(1), 1)
        elif gamma.dim() == 2:
            if gamma.shape != (tokens.size(0), tokens.size(1)):
                raise ValueError(
                    "Token-wise gamma must have shape "
                    f"({tokens.size(0)}, {tokens.size(1)}), got {tuple(gamma.shape)}."
                )
            gamma = gamma.unsqueeze(-1)
        elif gamma.dim() == 3 and gamma.shape == (tokens.size(0), tokens.size(1), 1):
            pass
        else:
            raise ValueError(
                "gamma must be a scalar, a batch tensor of shape (batch_size,), "
                "or a token-wise tensor of shape (batch_size, num_tokens)."
            )

        return torch.cat([tokens, gamma], dim=-1)

    def forward(self, x: Tensor, gamma: Tensor | float) -> Tensor:
        tokens = self.tokenize(x)
        tokens_with_gamma = self.append_gamma(tokens, gamma)
        return self.proj(tokens_with_gamma)


class ViTEmbedding(nn.Module):
    """Embedding block for a 1D site configuration.

    This layer converts an N-site configuration into num_tokens tokens and embeds
    each token into a d_model-dimensional space.
    """

    def __init__(self, num_sites: int, token_size: int, d_model: int, bias: bool = True) -> None:
        super().__init__()
        self.token_embedding = LinearTokenEmbedding(
            num_sites=num_sites,
            token_size=token_size,
            d_model=d_model,
            bias=bias,
        )

    def forward(self, x: Tensor, gamma: Tensor | float) -> Tensor:
        return self.token_embedding(x, gamma)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, bias: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})."
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, num_tokens, _ = x.shape

        q = self.q_proj(x).reshape(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).reshape(batch_size, num_tokens, self.d_model)
        return self.out_proj(context)


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_dim: int,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if mlp_dim <= 0:
            raise ValueError("mlp_dim must be positive.")

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            bias=bias,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim, bias=bias),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_sites: int,
        token_size: int,
        d_model: int,
        num_heads: int,
        mlp_dim: int,
        num_layers: int,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        self.embedding = ViTEmbedding(
            num_sites=num_sites,
            token_size=token_size,
            d_model=d_model,
            bias=bias,
        )
        num_tokens = num_sites // token_size
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_tokens, d_model))
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    bias=bias,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, gamma: Tensor | float) -> Tensor:
        x = self.embedding(x, gamma)
        x = self.dropout(x + self.pos_embedding)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class ViTWaveFunction(nn.Module):
    """ViT-style wavefunction model with complex-valued outputs.

    Pipeline:
    1. tokenize + embedding
    2. transformer encoder
    3. sum over token dimension
    4. linear heads for real and imaginary parts

    Output shape:
    - (batch_size, num_outputs)
    """

    def __init__(
        self,
        num_sites: int,
        token_size: int,
        d_model: int,
        num_heads: int,
        mlp_dim: int,
        num_layers: int,
        num_outputs: int,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_outputs <= 0:
            raise ValueError("num_outputs must be positive.")

        self.num_outputs = num_outputs
        self.encoder = TransformerEncoder(
            num_sites=num_sites,
            token_size=token_size,
            d_model=d_model,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            bias=bias,
            dropout=dropout,
        )
        self.real_head = nn.Linear(d_model, num_outputs, bias=bias)
        self.imag_head = nn.Linear(d_model, num_outputs, bias=bias)

    def encode(self, x: Tensor, gamma: Tensor | float) -> Tensor:
        return self.encoder(x, gamma)

    def pooled_features(self, x: Tensor, gamma: Tensor | float) -> Tensor:
        encoded = self.encode(x, gamma)
        return encoded.sum(dim=1)

    def log_wavefunction_matrix(self, x: Tensor, gamma: Tensor | float) -> Tensor:
        pooled = self.pooled_features(x, gamma)
        real = self.real_head(pooled)
        imag = self.imag_head(pooled)
        return real + 1j * imag

    def forward(self, x: Tensor, gamma: Tensor | float) -> Tensor:
        return self.log_wavefunction_matrix(x, gamma)

    def wavefunction_matrix(self, x: Tensor, gamma: Tensor | float) -> Tensor:
        return torch.exp(self.log_wavefunction_matrix(x, gamma))

    def complex_components(self, x: Tensor, gamma: Tensor | float) -> Tensor:
        """Alias kept for code that treats rows as complex output components."""
        return self.log_wavefunction_matrix(x, gamma)

    def phi(self, x: Tensor, gamma: Tensor | float) -> Tensor:
        return self.wavefunction_matrix(x, gamma)

    def log_phi(self, x: Tensor, gamma: Tensor | float) -> Tensor:
        return self.log_wavefunction_matrix(x, gamma)
