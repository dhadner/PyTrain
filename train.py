import math
import os
import time
import torch
from torch.utils.data import DataLoader, RandomSampler

from model import GPT
from dataset import get_datasets

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(PROJECT_DIR, "checkpoint.pt")

CONFIG = {
    "vocab_size": 50257,
    "block_size": 256,
    "n_layer": 8,
    "n_head": 8,
    "n_embd": 512,
}

TRAIN = {
    "batch_size": 64,
    "lr": 1e-3,
    "max_steps": 4000,
    "warmup_steps": 200,
    "min_lr_ratio": 0.1,
    "eval_interval": 200,
    "eval_steps": 20,
}


def lr_schedule(step):
    if step < TRAIN["warmup_steps"]:
        return step / TRAIN["warmup_steps"]
    progress = (step - TRAIN["warmup_steps"]) / (TRAIN["max_steps"] - TRAIN["warmup_steps"])
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return TRAIN["min_lr_ratio"] + (1 - TRAIN["min_lr_ratio"]) * cosine


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, val_loader, device, steps):
    model.eval()
    losses = []
    val_iter = iter(val_loader)
    for _ in range(steps):
        try:
            x, y, doc_ids = next(val_iter)
        except StopIteration:
            break
        x, y, doc_ids = x.to(device), y.to(device), doc_ids.to(device)
        _, loss = model(x, doc_ids=doc_ids, targets=y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float("inf")


def train():
    device = get_device()
    print(f"Using device: {device}", flush=True)

    train_ds, val_ds, encoder = get_datasets(CONFIG["block_size"])
    train_sampler = RandomSampler(train_ds, replacement=True, num_samples=TRAIN["max_steps"] * TRAIN["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=TRAIN["batch_size"], sampler=train_sampler)
    val_loader = DataLoader(val_ds, batch_size=TRAIN["batch_size"], shuffle=False)

    model = GPT(CONFIG).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN["lr"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    train_iter = iter(train_loader)
    t0 = time.time()
    for step in range(1, TRAIN["max_steps"] + 1):
        try:
            x, y, doc_ids = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y, doc_ids = next(train_iter)
        x, y, doc_ids = x.to(device), y.to(device), doc_ids.to(device)
        _, loss = model(x, doc_ids=doc_ids, targets=y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % TRAIN["eval_interval"] == 0:
            val_loss = evaluate(model, val_loader, device, TRAIN["eval_steps"])
            elapsed = time.time() - t0
            lr = scheduler.get_last_lr()[0]
            print(f"Step {step:5d} | lr {lr:.2e} | train loss {loss.item():.4f} | val loss {val_loss:.4f} | {elapsed:.1f}s", flush=True)
            t0 = time.time()

    torch.save({"model": model.state_dict(), "config": CONFIG}, CHECKPOINT_PATH)
    print(f"Saved {CHECKPOINT_PATH}", flush=True)

    model.eval()
    prompt = encoder.encode("Once upon a time")
    idx = torch.tensor([prompt], device=device)
    tokens = model.generate(idx, max_new_tokens=300)
    print("\n--- Sample generation ---")
    print(encoder.decode(tokens[0].tolist()))


if __name__ == "__main__":
    train()
