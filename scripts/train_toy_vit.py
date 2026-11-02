"""Train the miniature ViT on generated circles, squares and triangles.

A toy task that runs the whole training loop in a few seconds on a CPU.

    python scripts/train_toy_vit.py
    python scripts/train_toy_vit.py --epochs 3 --train-size 300 --no-checkpoint

Outputs (under --output-dir, default artifacts/training_runs/toy_vit):
    metrics.csv, training_curve.png and, unless --no-checkpoint, model.pt
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from tokenverse.vision.image_utils import select_device
from tokenverse.vision.mini_vit import MiniViT, ViTConfig
from tokenverse.vision.synthetic_dataset import CLASS_NAMES, SyntheticShapes


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Toy ViT on synthetic shapes")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--train-size", type=int, default=1200)
    p.add_argument("--val-size", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--patch-size", type=int, default=8)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="cpu, cuda or mps (auto)")
    p.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/training_runs/toy_vit")
    )
    p.add_argument("--no-checkpoint", action="store_true", help="Skip saving model.pt")
    return p.parse_args(argv)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(dim=1) == y).sum().item()
        total += y.numel()
    return correct / total


def save_curve(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tokenverse.visualization import PALETTE, apply_dark_theme

    apply_dark_theme()
    epochs = [r["epoch"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(7, 3.6))
    ax1.plot(
        epochs,
        [r["train_loss"] for r in rows],
        "o-",
        color=PALETTE["cyan"],
        label="train loss",
    )
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss", color=PALETTE["cyan"])
    ax2 = ax1.twinx()
    ax2.plot(
        epochs,
        [r["val_accuracy"] for r in rows],
        "s--",
        color=PALETTE["amber"],
        label="val accuracy",
    )
    ax2.set_ylabel("validation accuracy", color=PALETTE["amber"])
    ax2.set_ylim(0, 1)
    ax1.grid(alpha=0.3)
    ax1.set_title("Mini ViT on synthetic shapes", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    device = select_device(args.device)

    config = ViTConfig(
        image_size=args.image_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.heads,
        mlp_dim=args.embed_dim * 2,
        num_classes=len(CLASS_NAMES),
    )
    train_ds = SyntheticShapes(args.train_size, args.image_size, base_seed=args.seed)
    val_ds = SyntheticShapes(args.val_size, args.image_size, base_seed=args.seed + 1)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, generator=generator
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = MiniViT(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()

    print(f"Device: {device}   train {len(train_ds)}   val {len(val_ds)}")
    print(f"Untrained val accuracy: {evaluate(model, val_loader, device):.3f}")

    rows: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = n = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * y.numel()
            n += y.numel()
        val_acc = evaluate(model, val_loader, device)
        rows.append(
            {"epoch": epoch, "train_loss": running / n, "val_accuracy": val_acc}
        )
        print(f"epoch {epoch:>2}  train loss {running / n:.4f}  val acc {val_acc:.3f}")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["epoch", "train_loss", "val_accuracy"])
        writer.writeheader()
        writer.writerows(rows)
    save_curve(rows, out / "training_curve.png")
    if not args.no_checkpoint:
        torch.save(
            {"config": config.__dict__, "state_dict": model.state_dict()},
            out / "model.pt",
        )
    print(f"Saved outputs to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
