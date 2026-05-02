from cs336_basics.building_blocks import *
import torch
import numpy as np
import wandb

class TrainConfig():
    def __init__(
            self,
            seed: int,
            d_model: int,
            d_ff: int,
            n_layers: int,
            n_heads: int,
            context_length: int,
            theta: float,
            lr: float | tuple[float, float],
            weight_decay: float,
            beta: tuple[float, float],
            grad_clip: float,
            warmup_steps : int,
            cosine_steps : int,
            batch_size: int,
            use_rope: bool,
            vocab_size: int = 32000,
    ) -> None:
        self.seed = seed
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.context_length = context_length
        self.theta = theta
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta = beta
        self.grad_clip = grad_clip
        self.warmup_steps = warmup_steps
        self.cosine_steps = cosine_steps
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.use_rope = use_rope



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
    wandb.init(project="cs336-hw1", config=config.__dict__)
    data = np.memmap(train_path, dtype=np.uint16, mode='r')
    val_data = np.fromfile(val_path, dtype=np.uint16)
    val_in, val_out = data_loading(val_data, config.batch_size, config.context_length, device=device, rng=rng)
    
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
    

    

