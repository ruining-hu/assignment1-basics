import regex as re
from collections import defaultdict, Counter
from cs336_basics.pretokenization_example import find_chunk_boundaries
from typing import BinaryIO
import os
from multiprocessing import Pool


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    split_special_token: str | None = None,
    num_processes: int | None = 4,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    assert special_tokens
    assert vocab_size >= 256 + len(special_tokens)
    
    # initialize the vocabulary
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")
    
    if vocab_size == len(vocab):
        return {}, vocab
    
    # pretokenization
    freq_table: dict[tuple[bytes, ...], int] = {}

    if not split_special_token:
        split_special_token = special_tokens[0]
    if num_processes:
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_processes, split_special_token=split_special_token.encode("utf-8"))
        assert boundaries
        with Pool(processes=num_processes) as pool:
            results = pool.map(worker, [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])])
        assert results
        freq_table = sum([Counter(result) for result in results], start=Counter())
    else:
        freq_table = pretokenization(input_path, 0, None, special_tokens)
    
    # print(freq_table)
    
    # tokenize
    merges: list[tuple[bytes, bytes]] = []
    pair_count: dict[tuple[bytes, bytes], int] = defaultdict(int)
    while len(vocab) < vocab_size:
        pair_count: dict[tuple[bytes, bytes], int] = defaultdict(int)
        for pretoken_tuple in freq_table:
            if len(pretoken_tuple) <= 1:
                continue
            for i in range(len(pretoken_tuple)-1):
                pair_count[pretoken_tuple[i:i+2]] += freq_table[pretoken_tuple]
        
        # print(pair_count)
        
        if not pair_count:
            raise ValueError("desired vocab_size is infeasible")
        
        # get the pair to merge at this step
        max_pair = max(pair_count, key=lambda k: (pair_count[k], k))
        # iterate and merge
        new_freq_table: dict[tuple[bytes, ...], int] = defaultdict(int)
        for pretoken_tuple in freq_table:
            new_freq_table[merge(pretoken_tuple, max_pair)] = freq_table[pretoken_tuple]
        
        freq_table = new_freq_table
        merges.append(max_pair)
        vocab[len(vocab)] = max_pair[0] + max_pair[1]
    
    return vocab, merges


def worker(args):
    return pretokenization(*args)
    

def pretokenization(
    input_path: str,
    start: int,
    end: int | None,
    special_tokens: list[str]
) -> dict[tuple[bytes, ...], int]:
    """
    Compute the frequency table on a specified chunk of the input file. Can be used in parallel.
    """
    freq_table: dict[tuple[bytes, ...], int] = defaultdict(int)
    with open(input_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        length = f.tell()
        f.seek(0)
        if not end:
            end = length
        assert end <= length
        assert start < end
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

        # split on special tokens
        chunk_split = re.split("|".join(re.escape(t) for t in special_tokens), chunk)

        #constuct the frequency table
        for s in chunk_split:
            for match in re.finditer(PAT, s):
                pretoken = match.group().encode("utf-8")
                freq_table[tuple(pretoken[i: i+1] for i in range(len(pretoken)))] += 1
    
    return freq_table
    




def merge(bytes_tuple: tuple[bytes, ...], merge_pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    """
    Returns the bytes_tuple after merging all neighbour merge_pairs.
    """
    if len(bytes_tuple) <= 1:
        return bytes_tuple # no merge can be performed
    merged_bytes: list[bytes] = []
    i = 0
    while i < len(bytes_tuple):
        if bytes_tuple[i:i+2] == merge_pair:
            merged_bytes.append(merge_pair[0] + merge_pair[1])
            i += 2
        else:
            merged_bytes.append(bytes_tuple[i])
            i += 1
    
    return tuple(merged_bytes)




def bpe_example(
    input_str: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    assert vocab_size >= 256 + len(special_tokens)
    
    # initialize the vocabulary
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")
    
    if vocab_size == len(vocab):
        return {}, vocab
    
    # split on special tokens
    str_split: list[str] = re.split("|".join(re.escape(t) for t in special_tokens), input_str)

    # construct frequency table
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    freq_table: dict[tuple[bytes, ...], int] = defaultdict(int)
    for s in str_split:
        for match in re.finditer(PAT, s):
            pretoken: bytes = match.group().encode("utf-8")
            freq_table[tuple(pretoken[i: i+1] for i in range(len(pretoken)))] += 1
    
    # tokenize
    merges: list[tuple[bytes, bytes]] = []
    while len(vocab) < vocab_size:
        pair_count: dict[tuple[bytes, bytes], int] = defaultdict(int)
        for pretoken_tuple in freq_table:
            if len(pretoken_tuple) <= 1:
                continue
            for i in range(len(pretoken_tuple)-1):
                pair_count[pretoken_tuple[i:i+2]] += freq_table[pretoken_tuple]
        
        if not pair_count:
            raise ValueError("desired vocab_size is infeasible")
        
        # get the pair to merge at this step
        max_pair = max(pair_count, key=lambda k: (pair_count[k], k))
        # iterate and merge
        new_freq_table: dict[tuple[bytes, ...], int] = defaultdict(int)
        for pretoken_tuple in freq_table:
            new_freq_table[merge(pretoken_tuple, max_pair)] = freq_table[pretoken_tuple]
        
        freq_table = new_freq_table
        merges.append(max_pair)
        vocab[len(vocab)] = max_pair[0] + max_pair[1]
    
    return vocab, merges






    



    


