from cs336_basics.train import train, TrainConfig


config = TrainConfig(
    seed=42,
    d_model=512,
    d_ff=1344,
    theta=10_000,
    context_length=256,
    n_layers=4,
    n_heads=16,
    batch_size=64,
    lr=(1e-4, 1e-5),
    beta=(0.9, 0.95),
    weight_decay=0.1,
    grad_clip=1.0,
    warmup_steps=1_000,
    cosine_steps=10_000,
    use_rope=True,
    vocab_size=10_000
)
