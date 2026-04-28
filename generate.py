import argparse
import torch
import tiktoken

from model import GPT


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained model")
    parser.add_argument("--checkpoint", default="checkpoint.pt", help="Path to model checkpoint")
    parser.add_argument("--prompt", default="ROMEO:", help="Text prompt to start generation")
    parser.add_argument("--tokens", type=int, default=300, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = GPT(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    encoder = tiktoken.get_encoding("gpt2")
    idx = torch.tensor([encoder.encode(args.prompt)], device=device)
    tokens = model.generate(idx, max_new_tokens=args.tokens, temperature=args.temperature, top_k=args.top_k)
    print(encoder.decode(tokens[0].tolist()))


if __name__ == "__main__":
    main()
