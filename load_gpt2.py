"""
Load OpenAI's real GPT-2 (124M) weights into this from-scratch class.
If the architecture is correct: params ~124M, loss ~3.0-3.3, coherent generation.
If loss is ~10, the architecture is wrong.

  python load_gpt2.py
"""
import torch
import tiktoken
from model import GPT, GPTConfig
from configs import GPT2_CONFIGS, BLOCK_SIZE, VOCAB_SIZE


def load_pretrained(model, model_type="gpt2"):
    from transformers import GPT2LMHeadModel
    sd = GPT2LMHeadModel.from_pretrained(model_type).state_dict()

    to_transpose = ["attn.c_attn.weight", "attn.c_proj.weight",
                    "mlp.c_fc.weight", "mlp.c_proj.weight"]

    new_sd = {}
    for k, v in sd.items():
        if k.endswith(".attn.bias") or k.endswith(".attn.masked_bias"):
            continue
        if k == "lm_head.weight":
            continue  # tied to wte
        k = k.removeprefix("transformer.")
        if any(k.endswith(t) for t in to_transpose):
            v = v.t()
        new_sd[k] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"missing keys: {list(missing)}")
    print(f"unexpected keys: {list(unexpected)}")


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = GPTConfig(block_size=BLOCK_SIZE, vocab_size=VOCAB_SIZE,
                    dropout=0.0, **GPT2_CONFIGS["gpt2"])
    model = GPT(cfg)
    load_pretrained(model, "gpt2")
    model = model.to(device).eval()
    print(f"Params: {model.num_params()/1e6:.1f}M (expect ~124M)")

    enc = tiktoken.get_encoding("gpt2")

    sample = (
        "The quick brown fox jumps over the lazy dog. "
        "Language models are trained to predict the next token in a sequence. "
    ) * 8
    ids = torch.tensor(enc.encode_ordinary(sample), dtype=torch.long, device=device)[None, ...]
    ids = ids[:, :BLOCK_SIZE]
    with torch.no_grad():
        _, loss = model(ids[:, :-1], ids[:, 1:])
    print(f"\nLoss on natural text: {loss.item():.4f}  (expect ~3.0-3.3)")

    prompt = "Hello, I'm a language model,"
    x = torch.tensor(enc.encode_ordinary(prompt), dtype=torch.long, device=device)[None, ...]
    with torch.no_grad():
        out = model.generate(x, 80, temperature=0.8, top_k=50)
    print(f"\nPrompt: {prompt!r}")
    print("Generated:", enc.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
