import os
import urllib.request
import torch
from torch.utils.data import Dataset
import tiktoken

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
TRAIN_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
VALID_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
TRAIN_TXT = os.path.join(DATA_DIR, "TinyStories-train.txt")
VALID_TXT = os.path.join(DATA_DIR, "TinyStories-valid.txt")
TRAIN_PT = os.path.join(DATA_DIR, "tinystories-train-eot.pt")
VALID_PT = os.path.join(DATA_DIR, "tinystories-valid-eot.pt")

EOT_ID = 50256


def _download(url, path):
    if os.path.exists(path):
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Downloading {os.path.basename(path)}...", flush=True)
    last_mb = [0]

    def hook(blocks, block_size, total):
        mb = blocks * block_size // (1024 * 1024)
        if mb >= last_mb[0] + 100:
            last_mb[0] = mb
            print(f"  {mb} / {total // (1024 * 1024)} MB", flush=True)

    urllib.request.urlretrieve(url, path, reporthook=hook)


def _tokenize(text_path, cache_path, encoder):
    if os.path.exists(cache_path):
        return torch.load(cache_path)
    print(f"Tokenizing {os.path.basename(text_path)}...", flush=True)
    with open(text_path, "r") as f:
        text = f.read()
    print(f"  read {len(text) / 1e9:.2f} GB, encoding...", flush=True)
    tokens = torch.tensor(encoder.encode(text, allowed_special={"<|endoftext|>"}), dtype=torch.long)
    print(f"  -> {len(tokens):,} tokens, caching to {cache_path}", flush=True)
    torch.save(tokens, cache_path)
    return tokens


def _doc_ids(tokens):
    is_eot = (tokens == EOT_ID).long()
    prev_eot = torch.zeros_like(is_eot)
    prev_eot[1:] = is_eot[:-1]
    return torch.cumsum(prev_eot, dim=0)


class TokenDataset(Dataset):

    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.doc_ids = _doc_ids(tokens)
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.block_size + 1]
        chunk_doc_ids = self.doc_ids[idx : idx + self.block_size + 1]
        return chunk[:-1], chunk[1:], chunk_doc_ids[:-1]


def get_datasets(block_size):
    encoder = tiktoken.get_encoding("gpt2")
    _download(TRAIN_URL, TRAIN_TXT)
    _download(VALID_URL, VALID_TXT)
    train_tokens = _tokenize(TRAIN_TXT, TRAIN_PT, encoder)
    val_tokens = _tokenize(VALID_TXT, VALID_PT, encoder)
    n_train_docs = (train_tokens == EOT_ID).sum().item()
    n_val_docs = (val_tokens == EOT_ID).sum().item()
    print(f"Train: {len(train_tokens):,} tokens, {n_train_docs:,} docs | Val: {len(val_tokens):,} tokens, {n_val_docs:,} docs", flush=True)
    return TokenDataset(train_tokens, block_size), TokenDataset(val_tokens, block_size), encoder
