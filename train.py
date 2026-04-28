import time
import torch
from torch.utils.data import DataLoader

from model import GPT
from dataset import get_datasets

CONFIG = {
    "vocab_size": 50257,   # GPT-2 tokenizer vocab size
    "block_size": 128,
    "n_layer": 4,
    "n_head": 4,
    "n_embd": 128,
}

TRAIN = {
    "batch_size": 32,
    "lr": 3e-4,
    "epochs": 3,
    "eval_interval": 200,
    "eval_steps": 20,
}


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
    for i, (x, y) in enumerate(val_loader):
        if i >= steps:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float("inf")


def train():
    device = get_device()
    print(f"Using device: {device}")

    train_ds, val_ds, encoder = get_datasets(CONFIG["block_size"])
    train_loader = DataLoader(train_ds, batch_size=TRAIN["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=TRAIN["batch_size"], shuffle=False)

    model = GPT(CONFIG).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN["lr"])

    step = 0
    for epoch in range(TRAIN["epochs"]):
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1

            if step % TRAIN["eval_interval"] == 0:
                val_loss = evaluate(model, val_loader, device, TRAIN["eval_steps"])
                elapsed = time.time() - t0
                print(f"Step {step:5d} | train loss {loss.item():.4f} | val loss {val_loss:.4f} | {elapsed:.1f}s")
                t0 = time.time()

        val_loss = evaluate(model, val_loader, device, TRAIN["eval_steps"])
        print(f"--- Epoch {epoch + 1} done | val loss {val_loss:.4f} ---")

    torch.save({"model": model.state_dict(), "config": CONFIG}, "checkpoint.pt")
    print("Saved checkpoint.pt")

    # Generate a sample
    model.eval()
    prompt = encoder.encode("ROMEO:")
    idx = torch.tensor([prompt], device=device)
    tokens = model.generate(idx, max_new_tokens=200)
    print("\n--- Sample generation ---")
    print(encoder.decode(tokens[0].tolist()))


if __name__ == "__main__":
    train()
