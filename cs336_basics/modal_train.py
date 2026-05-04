import modal

app = modal.App("bpe-tokenizer")

image = (
    modal.Image.debian_slim()
    .uv_sync()
    .add_local_python_source("cs336_basics")  # mounts your local package into the container
)

volume = modal.Volume.from_name("bpe-output", create_if_missing=True)

@app.function(
    image=image,
    cpu=8,
    memory=102400,
    timeout=36000,
    volumes={"/out": volume},
)
def train_tokenizer(vocab_size: int = 32000):
    import urllib.request, gzip, shutil, pickle

    url = "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz"
    urllib.request.urlretrieve(url, "/tmp/owt_train.txt.gz")

    with gzip.open("/tmp/owt_train.txt.gz", "rb") as f_in, open("/tmp/owt_train.txt", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    from cs336_basics.BPE import train_bpe
    vocab, merges = train_bpe(
        input_path="/tmp/owt_train.txt",
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
        split_special_token="<|endoftext|>",
        num_processes=8
    )

    with open("/out/owt_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("/out/owt_merges.pkl", "wb") as f:
        pickle.dump(merges, f)
    volume.commit()


def tokenize_corpus(vocab_filepath: str, merges_filepath: str, special_tokens: list[str] = []):
    from cs336_basics.tokenizer import Tokenizer
    import urllib.request
    from urllib.parse import urlparse
    import pickle
    import os
    urls = [
        "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz",
        "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz"
    ]
    filenames = []
    for url in urls:
        filename = os.path.basename(urlparse(url).path)
        filenames.append(filename)

        urllib.request.urlretrieve(url, f"/tmp/{filename}")
    
    tokenizer = Tokenizer.from_files(vocab_filepath=vocab_filepath, merges_filepath=merges_filepath, special_tokens=special_tokens)
    for filename in filenames:
        f = open(f"/tmp/{filename}", "r")
        text = f.read()
        f.close()
        encoded_text = tokenizer.encode(text=text)
            

    


@app.local_entrypoint()
def main():
    train_tokenizer.remote(vocab_size=32000)