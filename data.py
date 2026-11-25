"""Download Tiny Shakespeare and tokenize it with GPT-2 BPE into train.bin / val.bin."""
import os
import urllib.request
import numpy as np
import tiktoken
import torch
from torch.utils.data import Dataset

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = "data"


def prepare(data_dir: str = DATA_DIR, val_frac: float = 0.1):
    os.makedirs(data_dir, exist_ok=True)
    txt_path = os.path.join(data_dir, "input.txt")
    if not os.path.exists(txt_path):
        print(f"Downloading {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, txt_path)

    text = open(txt_path, encoding="utf-8").read()
    enc = tiktoken.get_encoding("gpt2")
    ids = np.array(enc.encode_ordinary(text), dtype=np.uint16)
    print(f"Corpus: {len(text):,} chars -> {len(ids):,} tokens")

    n = int(len(ids) * (1 - val_frac))
    ids[:n].tofile(os.path.join(data_dir, "train.bin"))
    ids[n:].tofile(os.path.join(data_dir, "val.bin"))
    print(f"Wrote train.bin ({n:,}) and val.bin ({len(ids) - n:,})")


class TokenDataset(Dataset):
    def __init__(self, bin_path: str, block_size: int):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, i):
        chunk = self.data[i : i + self.block_size + 1].astype(np.int64)
        return torch.from_numpy(chunk[:-1]), torch.from_numpy(chunk[1:])


if __name__ == "__main__":
    prepare()
