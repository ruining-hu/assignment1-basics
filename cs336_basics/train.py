from cs336_basics.building_blocks import *
import torch
import numpy as np
import wandb
from dataclasses import dataclass, asdict

@dataclass
class TrainConfig:
        seed: int
        d_model: int
        d_ff: int
        n_layers: int
        n_heads: int
        context_length: int
        theta: float
        lr: float | tuple[float, float]
        weight_decay: float
        beta: tuple[float, float]
        grad_clip: float
        warmup_steps : int
        cosine_steps : int
        batch_size: int
        use_rope: bool
        vocab_size: int = 32000



def train(train_path: str, val_path: str, checkpt_path: str, config: TrainConfig, device: torch.device | None = None, dtype : torch.dtype | None = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer = TransformerLM(
        d_model=config.d_model,
        num_heads=config.n_heads,
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        num_layers=config.n_layers,
        d_ff=config.d_ff,
        use_rope=config.use_rope,
        theta=config.theta,
        device=device,
        dtype=dtype
    )
    rng = np.random.default_rng(seed=config.seed)
    wandb.init(project="cs336-hw1", config=asdict(config))
    data = np.memmap(train_path, dtype=np.uint16, mode='r')
    val_data = np.fromfile(val_path, dtype=np.uint16)
    val_in, val_out = val_loading(val_data, config.context_length, device=device)
    
    n_batches = data.shape[0] // (config.batch_size * config.context_length)
    if isinstance(config.lr, float):
        lr = config.lr
        use_scheduler = False
    else:
        lr = config.lr[0]
        use_scheduler = True
    optimizer = AdamW(params=transformer.parameters(), lr=lr, betas=config.beta, weight_decay=config.weight_decay)

    for i in range(1, n_batches+1):
        optimizer.zero_grad(set_to_none=True)
        train_batch, target_batch = data_loading(x=data, batch_size=config.batch_size, context_length=config.context_length, device=device, rng=rng)
        loss = cross_entropy(transformer(train_batch), target_batch)
        loss.backward()
        gradient_clipping(params=transformer.parameters(), clip_norm=config.grad_clip, device=device)
        if use_scheduler:
            optimizer.step(lr=cosine_lr_schedule(t=i, lr_max=config.lr[0], lr_min=config.lr[1], t_warm_up=config.warmup_steps, t_cosine=config.cosine_steps))
        else:
            optimizer.step()
        wandb.log({"train_loss": loss}, step=i)
        if i % 10 == 0:
            with torch.no_grad():
                val_loss = cross_entropy(transformer(val_in), val_out)
                wandb.log({"val_loss": val_loss}, step=i)
    
    wandb.finish()
    save_checkpoint(transformer, optimizer, iteration=n_batches, out=checkpt_path)
    

    

