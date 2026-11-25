"""
DDP training. Launch:
  torchrun --standalone --nproc_per_node=2 train_ddp.py --config gpt2 --max_iters 10000
"""
import argparse
import math
import os
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from model import GPT, GPTConfig
from data import TokenDataset
from configs import GPT2_CONFIGS, BLOCK_SIZE, VOCAB_SIZE


def is_master():
    return int(os.environ.get("RANK", "0")) == 0


def ddp_setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def get_lr(it, warmup, max_iters, lr, min_lr):
    if it < warmup:
        return lr * (it + 1) / (warmup + 1)
    if it > max_iters:
        return min_lr
    ratio = (it - warmup) / (max_iters - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * ratio))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="gpt2", choices=list(GPT2_CONFIGS))
    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=8, help="per-GPU")
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--no_flash", action="store_true")
    parser.add_argument("--out_dir", default="out_ddp")
    args = parser.parse_args()

    local_rank = ddp_setup()
    device = f"cuda:{local_rank}"
    world_size = dist.get_world_size()
    if is_master():
        os.makedirs(args.out_dir, exist_ok=True)

    cfg = GPTConfig(
        block_size=BLOCK_SIZE, vocab_size=VOCAB_SIZE,
        flash=not args.no_flash, **GPT2_CONFIGS[args.config],
    )
    raw_model = GPT(cfg).to(device)
    model = DDP(raw_model, device_ids=[local_rank])

    if is_master():
        print(f"Model: {args.config} | params: {raw_model.num_params()/1e6:.2f}M | "
              f"world_size: {world_size} | flash: {cfg.flash}")

    optimizer = raw_model.configure_optimizers(
        args.weight_decay, args.lr, (0.9, 0.95), "cuda"
    )
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    train_ds = TokenDataset("data/train.bin", BLOCK_SIZE)
    val_ds = TokenDataset("data/val.bin", BLOCK_SIZE)
    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler)

    def infinite(loader):
        while True:
            for x, y in loader:
                yield x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    train_iter = infinite(train_loader)

    @torch.no_grad()
    def estimate_val():
        model.eval()
        losses = []
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            with autocast_ctx:
                _, loss = model(x, y)
            losses.append(loss.item())
        model.train()
        t = torch.tensor(losses, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        return (t.mean()).item()

    best_val = float("inf")
    t0 = time.time()
    tokens_per_step = world_size * args.batch_size * args.grad_accum * BLOCK_SIZE

    for it in range(args.max_iters):
        lr = get_lr(it, args.warmup, args.max_iters, args.lr, args.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if it % args.eval_interval == 0:
            val_loss = estimate_val()
            elapsed = max(time.time() - t0, 1e-6)
            tps = (it + 1) * tokens_per_step / elapsed
            if is_master():
                print(f"iter {it:5d} | val {val_loss:.4f} | lr {lr:.2e} | "
                      f"{tps:,.0f} tok/s | {elapsed:.1f}s")
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save({"model": raw_model.state_dict(), "config": cfg.__dict__, "iter": it},
                               os.path.join(args.out_dir, "best.pt"))

        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            x, y = next(train_iter)
            with autocast_ctx:
                _, loss = model(x, y)
                loss = loss / args.grad_accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    if is_master():
        torch.save({"model": raw_model.state_dict(), "config": cfg.__dict__},
                   os.path.join(args.out_dir, "final.pt"))
        print(f"Done. Best val: {best_val:.4f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
