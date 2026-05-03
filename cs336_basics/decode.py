from cs336_basics.building_blocks import Decoder, load_checkpoint, TrainConfig, TransformerLM
from tokenizer import Tokenizer
import torch


if __name__ == "__main__":
    device = torch.device("cuda")
    checkpt_name = "tinystories"
    # decoder = Decoder.fromfile(
    #     f"checkpoints/{checkpt_name}.pth",
    #     f"checkpoints/{checkpt_name}.json",
    #     "cs336_basics/trained_vocab_tinystories.pkl",
    #     "cs336_basics/trained_merges_tinystories.pkl",
    #     special_tokens=["<|endoftext|>"],
    #     device=device
    # )
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
        lr=[1e-3, 4e-4],
        weight_decay=0.1,
        grad_clip=1.0,
        n_steps=20000,
        warmup_steps=2000,
        cosine_steps=18000,
    )
    transformer_model = TransformerLM(
        d_model=config.d_model,
        num_heads=config.n_heads,
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        num_layers=config.n_layers,
        d_ff=config.d_ff,
        use_rope=config.use_rope,
        theta=config.theta,
        device=device,
    )
    load_checkpoint(src=f"checkpoints/{checkpt_name}.pth", model=transformer_model, device=device)
    decoder = Decoder(
        transformer=transformer_model,
        tokenizer=Tokenizer.from_files(
            vocab_filepath="cs336_basics/trained_vocab_tinystories.pkl",
            merges_filepath="cs336_basics/trained_merges_tinystories.pkl",
            special_tokens=["<|endoftext|>"]
        )
    )

    while True:
        try:
            text = input("> ")
        except KeyboardInterrupt:
            print()
            break
        print()
        print(decoder.complete(text=text))
