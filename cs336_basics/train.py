from cs336_basics.building_blocks import *
import torch
import numpy as np
import wandb
from dataclasses import dataclass, asdict
import logging


def evaluate(transformer: TransformerLM, val_in: torch.Tensor, val_out: torch.Tensor, batch_size) -> float:
    total_loss = 0
    n = val_in.shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            vx = val_in[start:start+batch_size]
            vy = val_out[start:start+batch_size]
            total_loss += cross_entropy(transformer(vx), vy).item() * vx.shape[0]
    return total_loss / n



def train(train_path: str, val_path: str, checkpt_path: str, config: TrainConfig, device: torch.device | None = None, dtype : torch.dtype | None = None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger(__name__)
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
    log.info("model initialized")
    torch.set_float32_matmul_precision('high')
    transformer = torch.compile(transformer)
    log.info("model compiled")
    rng = np.random.default_rng(seed=config.seed)
    wandb.init(project="cs336-hw1", config=asdict(config))
    data = np.load(train_path, mmap_mode='r')
    val_data = np.load(val_path)
    val_in, val_out = val_loading(val_data, config.context_length, device=device)
    log.info("data loaded")
    
    # n_batches = data.shape[0] // (config.batch_size * config.context_length)
    if isinstance(config.lr, float):
        lr = config.lr
        use_scheduler = False
    else:
        lr = config.lr[0]
        use_scheduler = True
    optimizer = AdamW(params=transformer.parameters(), lr=lr, betas=config.beta, weight_decay=config.weight_decay)
    log.info("optimizer initialized")

    # train_batch, target_batch = data_loading(x=data, batch_size=config.batch_size, context_length=config.context_length, device=device, rng=rng)

    for i in range(1, config.n_steps+1):
        optimizer.zero_grad(set_to_none=True)
        train_batch, target_batch = data_loading(x=data, batch_size=config.batch_size, context_length=config.context_length, device=device, rng=rng)
        loss = cross_entropy(transformer(train_batch), target_batch)
        loss.backward()
        grad_norm = gradient_clipping(params=transformer.parameters(), clip_norm=config.grad_clip, device=device)
        if use_scheduler:
            optimizer.step(lr=cosine_lr_schedule(t=i, lr_max=config.lr[0], lr_min=config.lr[1], t_warm_up=config.warmup_steps, t_cosine=config.cosine_steps))
        else:
            optimizer.step()
        log.info(f"step: {i}, train_loss: {loss:.4f}")
        if i % 500 == 0:
            val_loss = evaluate(transformer=transformer, val_in=val_in, val_out=val_out, batch_size=config.batch_size)
            wandb.log({"train_loss": loss, "val_loss": val_loss, "grad_norm": grad_norm}, step=i)
            log.info(f"step: {i}, val_loss: {loss:.4f}")
        else:
            wandb.log({"train_loss": loss, "grad_norm": grad_norm}, step=i)
    
    wandb.finish()
    save_checkpoint(transformer, optimizer, iteration=config.n_steps, config=config, out=checkpt_path)


if __name__ == "__main__":
    config = TrainConfig(
        seed=42,
        d_model=512,
        context_length=256,
        d_ff=1344,
        vocab_size=10000,
        theta=10000.0,
        use_rope=True,
        n_layers=4,
        n_heads=16,
        batch_size=64,
        beta=[0.9, 0.95],
        lr=[5e-3, 5e-5],
        weight_decay=0.1,
        grad_clip=1.0,
        n_steps=20000,
        warmup_steps=2000,
        cosine_steps=20000,
    )
    device = torch.device("cuda")
    train("data/TinyStoriesV2-GPT4-train.npy", "data/TinyStoriesV2-GPT4-valid.npy", "checkpoints/schedule4", config=config, device=device)
    

    

