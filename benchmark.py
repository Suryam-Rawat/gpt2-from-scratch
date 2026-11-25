"""
Measure training throughput and peak memory: manual attention vs FlashAttention.

  python benchmark.py --config gpt2 --iters 50
"""
import argparse
import contextlib
import time
import torch
from model import GPT, GPTConfig
from configs import GPT2_CONFIGS, BLOCK_SIZE, VOCAB_SIZE


def bench(config_name, flash, iters, warmup, batch_size, device):
    cfg = GPTConfig(block_size=BLOCK_SIZE, vocab_size=VOCAB_SIZE,
                    flash=flash, **GPT2_CONFIGS[config_name])
    model = GPT(cfg).to(device)
    opt = model.configure_optimizers(0.1, 3e-4, (0.9, 0.95),
                                     "cuda" if device == "cuda" else "cpu")

    x = torch.randint(0, VOCAB_SIZE, (batch_size, BLOCK_SIZE), device=device)
    y = torch.randint(0, VOCAB_SIZE, (batch_size, BLOCK_SIZE), device=device)
    amp = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device == "cuda" else contextlib.nullcontext())

    def step():
        opt.zero_grad(set_to_none=True)
        with amp:
            _, loss = model(x, y)
        loss.backward()
        opt.step()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    for _ in range(warmup):
        step()

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        step()
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0

    tokens = iters * x.numel()
    peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
    return tokens / dt, peak


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="gpt2", choices=list(GPT2_CONFIGS))
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=4)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} config={args.config} batch={args.batch_size} "
          f"block={BLOCK_SIZE}\n")

    results = {}
    for label, flash in [("manual attention", False), ("FlashAttention (SDPA)", True)]:
        try:
            tps, peak = bench(args.config, flash, args.iters, args.warmup,
                              args.batch_size, device)
            results[label] = (tps, peak)
            print(f"{label:24s} {tps:12,.0f} tok/s   peak {peak:.2f} GB")
        except torch.cuda.OutOfMemoryError:
            print(f"{label:24s} OOM — reduce --batch_size")
            torch.cuda.empty_cache()

    if len(results) == 2:
        base = results["manual attention"][0]
        fast = results["FlashAttention (SDPA)"][0]
        print(f"\nSpeedup: {fast / base:.2f}x")

    print("\n| Configuration | Tokens/sec | Peak Mem (GB) |")
    print("|---|---|---|")
    for label, (tps, peak) in results.items():
        print(f"| {label} | {tps:,.0f} | {peak:.2f} |")


if __name__ == "__main__":
    main()
