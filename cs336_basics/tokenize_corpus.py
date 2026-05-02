from tokenizer import Tokenizer
import numpy as np


def tokenize_corpus(vocab_filepath: str, merges_filepath: str, special_tokens: list[str] = []):
    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)
    corpuses = [
        "data/TinyStoriesV2-GPT4-train",
        "data/TinyStoriesV2-GPT4-valid",
        "data/owt_train",
        "data/owt_valid"
    ]
    for path in corpuses:
        f = open(path + ".txt", "r")
        text = f.read()
        encoded_text = np.array(tokenizer.encode(text), dtype=np.uint16)
        np.save(f"{path}.npy", encoded_text)
        f.close()
        del text, encoded_text


if __name__ == "__main__":
    tokenize_corpus("cs336_basics/trained_vocab_tinystories.pkl", "cs336_basics/trained_merges_tinystories.pkl", ["<|endoftext|>"])
