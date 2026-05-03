import typing
from collections.abc import Callable, Iterable
import os

import torch
import torch.nn as nn
import numpy as np
import math
from einops import einsum, rearrange
from cs336_basics.tokenizer import Tokenizer

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
        return ((x * self.gains) / rearrange(norms, "... -> ... 1")).to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_hidden: int | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        if d_hidden is None:
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
        i = torch.arange(max_seq_len, device=device)
        i = rearrange(i, "i -> i 1")
        k = torch.arange(d_k//2, device=device)
        k = rearrange(k, "k -> 1 k")
        base_angle = math.pow(theta, 2/d_k)
        angles = i / torch.pow(base_angle, k)
        self.register_buffer("cosines", torch.cos(angles))
        self.register_buffer("sines", torch.sin(angles))

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        if token_positions is None:
            token_positions = torch.arange(x.shape[-2], device=x.device).expand(x.shape[:-1])
        else:
            token_positions = token_positions.expand(x.shape[:-1])
        # assert token_positions.shape == x.shape[:-1], f"token_positions.shape={token_positions.shape} must agree with x.shape[:-1]={x.shape[:-1]}"
        x = rearrange(x, "... (d_k_div_2 two) -> ... d_k_div_2 two", two=2)

        cos = self.cosines[token_positions]  # (..., d_k//2)
        sin = self.sines[token_positions]

        roped_x = torch.stack([x[..., 0] * cos - x[..., 1] * sin, x[..., 0] * sin + x[..., 1] * cos], dim=-1)

        return rearrange(roped_x, "... d_k_div_2 two -> ... (d_k_div_2 two)")


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
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
            rope: RotaryPositionalEmbedding | None = None,
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
            assert theta is not None
            assert max_seq_length is not None
            self.rope = rope or RotaryPositionalEmbedding(theta=theta, d_k=self.d_k, max_seq_len=max_seq_length, device=device)
        else:
            self.rope = None

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        assert x.size(-1) == self.d_model
        # if self.device:
        #     assert x.device == self.device, f"x is on {x.device}, but model is on {self.device}"
        if self.dtype:
            assert x.dtype == self.dtype
        seq_length = x.size(-2)
        Q = rearrange(self.W_Q(x), "... seq_length (num_heads d_k) -> ... num_heads seq_length d_k", d_k=self.d_k)
        K = rearrange(self.W_K(x), "... seq_length (num_heads d_k) -> ... num_heads seq_length d_k", d_k=self.d_k)
        V = rearrange(self.W_V(x), "... seq_length (num_heads d_k) -> ... num_heads seq_length d_k", d_k=self.d_k)
        if self.rope is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        rows = torch.arange(seq_length, device=self.device)[:, None]
        cols = torch.arange(seq_length, device=self.device)[None, :]
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
            rope: RotaryPositionalEmbedding | None = None,
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
        self.rope = rope
        self.rms_norm_1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.rms_norm_2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.multihead_self_attention = MultiheadSelfAttention(d_model=d_model, num_heads=num_heads, use_rope=use_rope, rope=self.rope, max_seq_length=max_seq_length, theta=theta, device=device, dtype=dtype)
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
        if use_rope:
            assert theta is not None
            self.rope = RotaryPositionalEmbedding(theta=theta, d_k=d_model // num_heads, max_seq_len=context_length, device=device)
        else:
            self.rope = None
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, use_rope=use_rope, rope=self.rope, max_seq_length=context_length, theta=theta, device=device, dtype=dtype) for _ in range(num_layers)])
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
    
    def step(self, closure: Callable[[], float] | None = None, lr: float | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
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
                alpha_t = (lr or group["lr"]) * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                p.data *= (1 - (lr or group["lr"]) * weight_decay)
                p.data -= alpha_t * m / (torch.sqrt(v) + eps)
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss


def cosine_lr_schedule(t: int, lr_max: float, lr_min: float, t_warm_up: int, t_cosine: int) -> float:
    # assert t > 0
    if t < t_warm_up:
        return t*lr_max/t_warm_up
    elif t > t_cosine:
        return lr_min
    else:
        return lr_min + (1 + math.cos((t - t_warm_up)*math.pi / (t_cosine - t_warm_up))) * (lr_max - lr_min)/2


def gradient_clipping(params: Iterable[nn.Parameter], clip_norm: float, device: torch.device | None = None) -> None:
    grad_norm = torch.zeros(1, device=device)
    for param in params:
        if param.grad is not None:
            grad_norm += param.grad.data.norm(2) ** 2
    
    grad_norm = torch.sqrt(grad_norm)
    if grad_norm > clip_norm:
        scale_factor = clip_norm / (grad_norm + 1e-6)
        for param in params:
            if param.grad is not None:
                param.grad.data = param.grad.data * scale_factor


def data_loading(x: np.ndarray, batch_size: int, context_length: int, device: torch.device, rng: np.random.Generator = np.random.default_rng()) -> tuple[torch.Tensor, torch.Tensor]:
    assert len(x.shape) == 1
    n = x.size
    assert n >= context_length + 1
    start_indices_chosen = rng.integers(0, n - context_length, size=batch_size)
    result = x[start_indices_chosen[:, None] + np.arange(context_length+1)]
    return (torch.from_numpy(result[:, :-1].astype(np.int64)).to(device), torch.from_numpy(result[:, 1:].astype(np.int64)).to(device))


def val_loading(x: np.ndarray, context_length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    assert len(x.shape) == 1
    n = x.size
    start_indices = np.arange(start=0, stop=n-context_length, step=context_length) # type: ignore
    result = x[start_indices[:, None] + np.arange(context_length+1)]
    return (torch.from_numpy(result[:, :-1].astype(np.int64)).to(device), torch.from_numpy(result[:, 1:].astype(np.int64)).to(device))


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]) -> None:
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(state, out)


def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    state = torch.load(src)
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    return state["iteration"]


class Decoder:
    def __init__(self, transformer: TransformerLM, tokenizer: Tokenizer, split_token: str = "<|endoftext|>") -> None:
        self.transformer_model = transformer
        self.tokenizer = tokenizer
        assert split_token in tokenizer.special_tokens
        self.split_token_id = tokenizer.encode(split_token)[0]
    

    def complete(self, text: str, temp: float = 1.0, top_p: float = 1.0, max_length: int | float = float("inf")) -> str:
        token_ids: list[int] = self.tokenizer.encode(text=text)
        next_token = None
        while next_token != self.split_token_id and len(token_ids) <= max_length:
            with torch.no_grad():
                logits = self.transformer_model(torch.tensor(token_ids, dtype=torch.int64, device=self.transformer_model.embedding.embeddings.device))[-1, :]
            next_token = _sample_from_logits(logits=logits, top_p=top_p, temp=temp)
            token_ids.append(next_token)
        return self.tokenizer.decode(token_ids)


def _sample_from_logits(logits: torch.Tensor, top_p: float, temp: float) -> int:
    assert len(logits.shape) == 1
    assert temp > 0
    probs = softmax(logits / temp, dim=-1)
    probs_sorted, token_ids = torch.sort(probs, dim=-1, descending=True)
    probs_cumsum = torch.cumsum(probs_sorted, dim=-1)

    # shift right by 1 so we include the token that just pushes the cumsum above top_p
    mask = (probs_cumsum - probs_sorted) >= top_p
    probs_sorted[mask] = 0.0

    return token_ids[torch.multinomial(probs_sorted, num_samples=1)].item()




