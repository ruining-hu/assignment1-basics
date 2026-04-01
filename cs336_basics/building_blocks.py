from typing import Any

import torch
import torch.nn as nn
import math
from einops import einsum, rearrange

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = math.sqrt(2/(in_features + out_features))
        nn.init.trunc_normal_(self.weights, mean=0, std=std, a=-3 * std, b=3 * std)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weights, x, "feature_out feature_in, ... feature_in -> ... feature_out")


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.embeddings = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.embeddings, mean=0, std=1, a=-3, b=3)
    

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.gains = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        norms = torch.sqrt(einsum(x.pow(2), "... d_model -> ...")/x.size(-1) + self.eps)
        return (x * rearrange(self.gains, "d_model -> 1 1 d_model") / rearrange(norms, "... -> ... 1")).to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_hidden: int | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        if not d_hidden:
            d_hidden = int(d_model * 3 / (8 * 64)) * 64
        self.w1 = Linear(d_model, d_hidden, device=device, dtype=dtype)
        self.w2 = Linear(d_hidden, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_hidden, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = self.w1(x)
        return self.w2(w1x * torch.sigmoid(w1x) * self.w3(x))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None) -> None:
        super().__init__()
        self.d_k = d_k
        i = torch.arange(max_seq_len, device=device, requires_grad=False)
        i = rearrange(i, "i -> i 1")
        k = torch.arange(d_k//2, device=device, requires_grad=False)
        k = rearrange(k, "k -> 1 k")
        base_angle = math.pow(theta, 2/d_k)
        self.angles = i / torch.pow(base_angle, k)
        self.cosines = torch.cos(self.angles)
        self.sines = torch.sin(self.angles)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        roped_x = torch.empty_like(x)
        for k in range(self.d_k//2):
            roped_x[..., 2*k] = self.cosines[token_positions, k] * x[..., 2*k] - self.sines[token_positions, k] * x[..., 2*k + 1]
            roped_x[..., 2*k + 1] = self.sines[token_positions, k] * x[..., 2*k] + self.cosines[token_positions, k] * x[..., 2*k + 1]
        
        return roped_x


