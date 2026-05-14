import itertools
import json
from pathlib import Path
from dataclasses import replace
from cs336_basics.train import train
from cs336_basics.building_blocks import TrainConfig

import torch


BASE_CONFIG = TrainConfig(
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


def fmt_lr(x: float) -> str:
    """Format 0.005 as 5e-3-ish for checkpoint names."""
    return f"{x:.0e}".replace("+0", "").replace("-0", "-")


if __name__ == "__main__":
    train_path = "data/TinyStoriesV2-GPT4-train.npy"
    valid_path = "data/TinyStoriesV2-GPT4-valid.npy"

    root_ckpt_dir = Path("checkpoints/lr_sweep")
    root_ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_lrs = [
        3e-3,
        5e-3,
        1e-2
    ]

    min_lr_fracs = [0.0, 1e-3, 1e-1]

    sweep = list(itertools.product(max_lrs, min_lr_fracs))

    for run_id, (max_lr, min_lr_frac) in enumerate(sweep):
        min_lr = max_lr * min_lr_frac

        run_name = (
            f"run{run_id:03d}"
            f"_maxlr_{fmt_lr(max_lr)}"
            f"_minfrac_{fmt_lr(min_lr_frac)}"
        )

        ckpt_dir = root_ckpt_dir / run_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        config = replace(
            BASE_CONFIG,
            lr=[max_lr, min_lr],
        )

        # Save config for later inspection
        with open(ckpt_dir / "config.json", "w") as f:
            json.dump(
                {
                    **config.__dict__,
                    "lr": [max_lr, min_lr],
                    "min_lr_frac": min_lr_frac,
                    "run_name": run_name,
                },
                f,
                indent=2,
            )

        print("=" * 80)
        print(f"Starting {run_name}")
        print(f"max_lr={max_lr:.3g}, min_lr={min_lr:.3g}, min_lr_frac={min_lr_frac:.3g}")
        print(f"checkpoint dir: {ckpt_dir}")
        print("=" * 80)

        try:
            train(
                train_path,
                valid_path,
                str(ckpt_dir),
                config=config,
                device=device,
            )
        except RuntimeError as e:
            print(f"[FAILED] {run_name}: {e}")

        torch.cuda.empty_cache()