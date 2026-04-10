from typing import Any
from collections.abc import Callable, Iterable

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
            d_hidden = int(d_model * 8 / (3 * 64)) * 64
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

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        roped_x = torch.empty_like(x)
        if token_positions is None:
            token_positions = torch.arange(x.shape[-2]).expand(x.shape[:-1])
        for k in range(self.d_k//2):
            roped_x[..., 2*k] = self.cosines[token_positions, k] * x[..., 2*k] - self.sines[token_positions, k] * x[..., 2*k + 1]
            roped_x[..., 2*k + 1] = self.sines[token_positions, k] * x[..., 2*k] + self.cosines[token_positions, k] * x[..., 2*k + 1]
        
        return roped_x


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert dim < len(x.shape)
    maxes = x.amax(dim=dim, keepdim=True)
    exp_x = torch.exp(x - maxes)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    d_k = Q.size(-1)
    d_v = V.size(-1)
    query_seq_length = Q.size(-2)
    key_seq_length = K.size(-2)
    assert mask.shape[-2:] == (query_seq_length, key_seq_length)
    assert not mask.shape[:-2] or mask.shape[:-2] == Q.shape[:-2]
    assert K.size(-1) == d_k
    assert Q.shape[:-2] == K.shape[:-2]

    Q_KT_scaled = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)
    Q_KT_scaled_masked = Q_KT_scaled.masked_fill(~mask, float("-inf"))
    return einsum(softmax(Q_KT_scaled_masked, dim=-1), V, "... queries keys, ... keys d_v -> ... queries d_v")


class MultiheadSelfAttention(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            use_rope: bool = True,
            max_seq_length: int | None = None,
            theta: float | None = None,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.device = device
        self.dtype = dtype
        self.max_seq_length = max_seq_length
        self.theta = theta
        self.d_k = d_model // num_heads
        self.W_Q = Linear(d_model, num_heads * self.d_k, device=device, dtype=dtype)
        self.W_K = Linear(d_model, num_heads * self.d_k, device=device, dtype=dtype)
        self.W_V = Linear(d_model, num_heads * self.d_k, device=device, dtype=dtype)
        self.W_O = Linear(num_heads * self.d_k, d_model, device=device, dtype=dtype)
        if use_rope:
            assert theta
            assert max_seq_length
            self.rope = RotaryPositionalEmbedding(theta=theta, d_k=self.d_k, max_seq_len=max_seq_length, device=device)
        else:
            self.rope = None

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        assert x.size(-1) == self.d_model
        if self.device:
            assert x.device == self.device
        if self.dtype:
            assert x.dtype == self.dtype
        seq_length = x.size(-2)
        Q = rearrange(self.W_Q(x), "... seq_length (num_heads d_k) -> ... num_heads seq_length d_k", d_k=self.d_k)
        K = rearrange(self.W_K(x), "... seq_length (num_heads d_k) -> ... num_heads seq_length d_k", d_k=self.d_k)
        V = rearrange(self.W_V(x), "... seq_length (num_heads d_k) -> ... num_heads seq_length d_k", d_k=self.d_k)
        if self.rope is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        rows = torch.arange(seq_length)[:, None]
        cols = torch.arange(seq_length)[None, :]
        mask = rows >= cols
        attended_values = scaled_dot_product_attention(Q, K, V, mask=mask)
        return self.W_O(rearrange(attended_values, "... num_heads seq_length d_k -> ... seq_length (num_heads d_k)"))


class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int | None = None,
            use_rope: bool = True,
            max_seq_length: int | None = None,
            theta: float | None = None,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.device = device
        self.dtype = dtype
        self.rms_norm_1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.rms_norm_2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.multihead_self_attention = MultiheadSelfAttention(d_model=d_model, num_heads=num_heads, use_rope=use_rope, max_seq_length=max_seq_length, theta=theta, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model=d_model, d_hidden=d_ff, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        layer_1 = x + self.multihead_self_attention(self.rms_norm_1(x), token_positions)
        return layer_1 + self.ffn(self.rms_norm_2(layer_1))


class TransformerLM(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            vocab_size: int,
            context_length: int,
            num_layers: int,
            d_ff: int | None = None,
            use_rope: bool = True,
            theta: float | None = None,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        self.transformer_blocks = [TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, use_rope=use_rope, max_seq_length=context_length, theta=theta, device=device, dtype=dtype) for _ in range(num_layers)]
        self.num_layers = num_layers
        self.rms_norm_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
    
    def forward(self, token_seq: torch.Tensor) -> torch.Tensor:
        embedded_seq = self.embedding(token_seq)
        for i in range(self.num_layers):
            embedded_seq = self.transformer_blocks[i](embedded_seq)
        embedded_seq = self.rms_norm_final(embedded_seq)
        return self.lm_head(embedded_seq)
        

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    assert logits.shape[:-1] == targets.shape
    maxes = logits.amax(dim=-1, keepdim=True)
    logits = logits - maxes
    target_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    losses = torch.log(torch.exp(logits).sum(dim=-1)) - target_logits
    return losses.mean()


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.

        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float, betas: tuple[float, float], weight_decay: float = 0, eps: float = 1e-8) -> None:
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)
    
    def step(self, closure: Callable[[], float] | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            (beta1, beta2) = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                grad = p.grad.data
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2
                alpha_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                p.data *= (1 - lr * weight_decay)
                p.data -= alpha_t * m / (torch.sqrt(v) + eps)
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss



