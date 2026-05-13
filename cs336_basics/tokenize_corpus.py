from cs336_basics.tokenizer import Tokenizer
from cs336_basics.pretokenization_example import find_chunk_boundaries
from multiprocessing import Pool
from typing import BinaryIO
from itertools import chain
import numpy as np


def _worker(input_path: str, tokenizer: Tokenizer, start: int, end: int):
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    return tokenizer.encode(chunk)


def tokenize_corpus(vocab_filepath: str, merges_filepath: str, special_tokens: list[str] = [], num_processes: int = 1):
    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)
    corpuses = [
        # "data/TinyStoriesV2-GPT4-train",
        # "data/TinyStoriesV2-GPT4-valid",
        "data/owt_train",
        "data/owt_valid"
    ]
    for path in corpuses:
        f = open(path + ".txt", "br")
        boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode())
        text = f.read()
        if num_processes == 1:
            encoded_text = np.array(tokenizer.encode(text.decode("utf-8", errors="ignore")), dtype=np.uint16)
        else:
            with Pool() as pool:
                encoded_texts = pool.starmap(_worker, [(path + ".txt", tokenizer, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])])
            encoded_text = np.fromiter(chain.from_iterable(encoded_texts), dtype=np.uint16)
        np.save(f"{path}.npy", encoded_text)
        f.close()
        del text, encoded_text


if __name__ == "__main__":
    # tokenize_corpus("cs336_basics/trained_vocab_tinystories.pkl", "cs336_basics/trained_merges_tinystories.pkl", ["<|endoftext|>"])
    tokenize_corpus("cs336_basics/owt_vocab.pkl", "cs336_basics/owt_merges.pkl", ["<|endoftext|>"], num_processes=16)
