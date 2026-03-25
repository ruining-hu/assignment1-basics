from cs336_basics.BPE import train_bpe
import pickle
import time
import tracemalloc
from contextlib import contextmanager

@contextmanager
def profile(label="block"):
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"[{label}] time={elapsed:.3f}s | mem_current={current/1e6:.2f}MB | mem_peak={peak/1e6:.2f}MB")

if __name__ == "__main__":
    file_path = "data/TinyStoriesV2-GPT4-train.txt"
    with profile("bpe training"):
        vocab, merges = train_bpe(input_path=file_path, vocab_size=10000, special_tokens=["<|endoftext|>"], split_special_token="<|endoftext|>")
    trained_data = {"vocab": vocab, "merges": merges}
    with open("cs336_basics/trained_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("cs336_basics/trained_merges.pkl", "wb") as f:
        pickle.dump(merges, f)