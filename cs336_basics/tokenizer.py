import pickle
import regex as re
from typing import Iterable, Iterator, Self

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.inverse_vocab: dict[bytes, int] = {}
        for id, token in vocab.items():
            self.inverse_vocab[token] = id


    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> Self:
        with open(vocab_filepath, "rb") as f:
            vocab: dict[int, bytes] = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges: list[tuple[bytes, bytes]] = pickle.load(f)
        
        return cls(vocab, merges, special_tokens)
    

    def encode(self, text: str) -> list[int]:
        if self.special_tokens:
            split: list[str] = re.split("(" + "|".join(re.escape(t) for t in self.special_tokens) + ")", text)
        else:
            split: list[str] = [text]
        
        encoder_cache: dict[bytes, list[int]] = {token: [self.inverse_vocab[token]] for token in self.inverse_vocab}
        encoded: list[int] = []
        last_special_token: str = ""
        for s in split:
            if not s:
                continue # skip empty splits
            if self.special_tokens:
                if last_special_token and last_special_token + s not in self.special_tokens:
                    encoded += encoder_cache[last_special_token.encode("utf-8")]
                    last_special_token = ""
                if last_special_token + s in self.special_tokens:
                    last_special_token += s
                    continue
            for match in re.finditer(PAT, s):
                pretoken = match.group().encode("utf-8")
                if pretoken not in encoder_cache:
                    encoder_cache[pretoken] = encode_pretoken(pretoken, self.merges, self.inverse_vocab) # cache the computation
                encoded += encoder_cache[pretoken] # need to check the time complexity of list add, might need to change to iterated appending
        
        return encoded + encoder_cache[last_special_token.encode("utf-8")] if last_special_token else encoded # need to append any pending special tokens
    

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for s in iterable:
            for token_id in self.encode(s):
                yield token_id
    

    def decode(self, ids: list[int]) -> str:
        decoded: bytes = b""
        for id in ids:
            decoded += self.vocab[id]
        
        return decoded.decode("utf-8", errors="replace")


def encode_pretoken(pretoken: bytes, merges: list[tuple[bytes, bytes]], inverse_vocab: dict[bytes, int]) -> list[int]:
    pretoken_tuple = [bytes([b]) for b in pretoken]
    for merge_pair in merges:
        i = 0
        while i < len(pretoken_tuple)-1:
            if merge_pair[0] == pretoken_tuple[i] and merge_pair[1] == pretoken_tuple[i+1]:
                pretoken_tuple[i] = merge_pair[0] + merge_pair[1]
                pretoken_tuple.pop(i+1)
            
            i += 1
    
    return [inverse_vocab[token] for token in pretoken_tuple]


