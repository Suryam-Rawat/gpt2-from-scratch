"""
Single-device training.
  python data.py
  python train.py --config tiny --max_iters 5000
"""
import argparse
import math
import os
import time
import torch
from model import GPT, GPTConfig
from data import TokenDataset
from configs import GPT2_CONFIGS, BLOCK_SIZE, VOCAB_SIZE


def get_lr(it, warmup, max_iters, lr, min_lr):
    if it < warmup:
        return lr * (it + 1) / (warmup + 1)
    if it > max_iters:
        return min_lr
    ratio = (it - warmup) / (max_iters - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * ratio))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="tiny", choices=list(GPT2_CONFIGS))
    p.add_argument("--max_iters", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--min_lr", type=float, default=6e-5)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=100)
    p.add_argument("--no_flash", action="store_true")
    p.add_argument("--out_dir", default="out")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_ds = TokenDataset("data/train.bin", BLOCK_SIZE)
    val_ds = TokenDataset("data/val.bin", BLOCK_SIZE)

    def get_batch(ds):
        idx = torch.randint(0, len(ds), (args.batch_size,))
        x = torch.stack([ds[i][0] for i in idx]).to(device)
        y = torch.stack([ds[i][1] for i in idx]).to(device)
        return x, y

    cfg = GPTConfig(
        block_size=BLOCK_SIZE, vocab_size=VOCAB_SIZE,
        flash=not args.no_flash, **GPT2_CONFIGS[args.config],
    )
    model = GPT(cfg).to(device)
    print(f"Model: {args.config} | params: {model.num_params()/1e6:.2f}M | "
          f"device: {device} | flash: {cfg.flash}")

    optimizer = model.configure_optimizers(
        args.weight_decay, args.lr, (0.9, 0.95), "cuda" if device == "cuda" else "cpu"
    )
    autocast_ctx = torch.amp.autocast(device_type=device.split(":")[0], dtype=dtype)

    def estimate_loss():
        model.eval()
        out = {}
        with torch.no_grad():
            for name, ds in [("train", train_ds), ("val", val_ds)]:
                losses = torch.zeros(args.eval_iters)
                for k in range(args.eval_iters):
                    x, y = get_batch(ds)
                    with autocast_ctx:
                        _, loss = model(x, y)
                    losses[k] = loss.item()
                out[name] = losses.mean().item()
        model.train()
        return out

    best_val = float("inf")
    t0 = time.time()
    for it in range(args.max_iters):
        lr = get_lr(it, args.warmup, args.max_iters, args.lr, args.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if it % args.eval_interval == 0:
            losses = estimate_loss()
            print(f"iter {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | "
                  f"lr {lr:.2e} | {time.time() - t0:.1f}s")
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({"model": model.state_dict(), "config": cfg.__dict__, "iter": it},
                           os.path.join(args.out_dir, "best.pt"))

        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            x, y = get_batch(train_ds)
            with autocast_ctx:
                _, loss = model(x, y)
                loss = loss / args.grad_accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    torch.save({"model": model.state_dict(), "config": cfg.__dict__},
               os.path.join(args.out_dir, "final.pt"))
    print(f"Done. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()