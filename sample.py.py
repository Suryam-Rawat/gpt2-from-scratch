"""python sample.py --ckpt out/best.pt --prompt "ROMEO:" --tokens 200"""
import argparse
import torch
import tiktoken
from model import GPT, GPTConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="out/best.pt")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--num_samples", type=int, default=3)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    model = GPT(GPTConfig(**ckpt["config"])).to(device).eval()
    model.load_state_dict(ckpt["model"])
    print(f"Loaded {args.ckpt} ({model.num_params()/1e6:.2f}M params)")

    enc = tiktoken.get_encoding("gpt2")
    x = torch.tensor(enc.encode_ordinary(args.prompt), dtype=torch.long, device=device)[None, ...]
    x = x.repeat(args.num_samples, 1)

    with torch.no_grad():
        out = model.generate(x, args.tokens, args.temperature, args.top_k)

    for i in range(args.num_samples):
        print(f"\n{'=' * 60}\nSample {i + 1}\n{'=' * 60}")
        print(enc.decode(out[i].tolist()))


if __name__ == "__main__":
    main()