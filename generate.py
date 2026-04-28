import argparse
import torch
import tiktoken

from model import GPT


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained model")
    parser.add_argument("--checkpoint", default="checkpoint.pt", help="Path to model checkpoint")
    parser.add_argument("--prompt", default="Once upon a time", help="Text prompt to start generation")
    parser.add_argument("--tokens", type=int, default=300, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=40)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = GPT(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    encoder = tiktoken.get_encoding("gpt2")
    eot_id = encoder.eot_token  # 50256
    idx = torch.tensor([encoder.encode(args.prompt)], device=device)
    tokens = model.generate(idx, max_new_tokens=args.tokens, temperature=args.temperature, top_k=args.top_k, stop_at=eot_id)
    token_list = tokens[0].tolist()

    # Trim at the first end-of-text token (if any)
    if eot_id in token_list:
        token_list = token_list[:token_list.index(eot_id)]

    print(encoder.decode(token_list))


if __name__ == "__main__":
    main()
