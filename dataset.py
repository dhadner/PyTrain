import os
import urllib.request
import torch
from torch.utils.data import Dataset
import tiktoken

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "data/tinyshakespeare.txt"


def download_data():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(DATA_PATH):
        print("Downloading Tiny Shakespeare dataset...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    with open(DATA_PATH, "r") as f:
        return f.read()


class TextDataset(Dataset):

    def __init__(self, text, block_size, encoder):
        self.tokens = torch.tensor(encoder.encode(text), dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def get_datasets(block_size):
    text = download_data()
    encoder = tiktoken.get_encoding("gpt2")
    split = int(0.9 * len(text))
    train_ds = TextDataset(text[:split], block_size, encoder)
    val_ds = TextDataset(text[split:], block_size, encoder)
    print(f"Train tokens: {len(train_ds.tokens):,} | Val tokens: {len(val_ds.tokens):,}")
    return train_ds, val_ds, encoder
