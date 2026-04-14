from cs336_basics.tokenizer import Tokenizer
import random


def compression_ratio(file_path: str, special_tokens: list[str], vocab_filepath: str, merges_filepath: str) -> float:
    """Sample 10 documents from the file, encode them using the tokenizer, and compute the compression ratio."""
    with open(file_path, "r", encoding="utf-8") as f:
        f.seek(0, 2)
        length = f.tell()
        f.seek(random.randint(0, length//2))  # Seek to a random position in the file
        split_special_token = special_tokens[0]
        mini_chunk_size = 1024  # Read in small chunks to find the special token
        counter = 0
        start = 0
        end = 0

        while True:
            mini_chunk = f.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                start = f.tell()
                break
        
        while True:
            mini_chunk = f.read(mini_chunk_size)  # Read a mini chunk

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                counter += 1
                if counter == 10:  # We want to find the 10th occurrence
                    end = f.tell()
                    break
        
        f.seek(start)
        text = f.read(end - start)

    original_size = len(text.encode("utf-8"))
    print(f"Original size (bytes): {original_size}")

    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)
    encoded_size = len(tokenizer.encode(text))

    return original_size / encoded_size if encoded_size > 0 else float("inf")


if __name__ == "__main__":
    print(compression_ratio(
        file_path="data/owt_valid.txt",
        special_tokens=["<|endoftext|>"],
        vocab_filepath="cs336_basics/owt_vocab.pkl",
        merges_filepath="cs336_basics/owt_merges.pkl"
    ))

    # print(compression_ratio(
    #     file_path="data/TinyStoriesv2-GPT4-valid.txt",
    #     special_tokens=["<|endoftext|>"],
    #     vocab_filepath="cs336_basics/trained_vocab_tinystories.pkl",
    #     merges_filepath="cs336_basics/trained_merges_tinystories.pkl"
    # ))

    # print(compression_ratio(
    #     file_path="data/owt_valid.txt",
    #     special_tokens=["<|endoftext|>"],
    #     vocab_filepath="cs336_basics/trained_vocab_tinystories.pkl",
    #     merges_filepath="cs336_basics/trained_merges_tinystories.pkl"
    # ))
