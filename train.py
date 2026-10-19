"""Entrenamiento de la TDNN (MSE como métrica principal) con EMA, SpecAugment y selección por val MSE.

Uso:
  python train.py --cache ~/data/emociones_cache --out runs/tdnn_s0 --seed 0
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ser_lib import (ATTRS, EMA, LABEL_MAX, LABEL_MIN, TDNN, BucketBatchSampler, FeatDataset,
                     bootstrap_ci, ccc_loss, metrics, pad_collate)


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--warmup_epochs", type=int, default=3)
    ap.add_argument("--min_lr_ratio", type=float, default=0.01)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--emb", type=int, default=512)
    ap.add_argument("--head", type=int, default=256)
    ap.add_argument("--drop_frame", type=float, default=0.1)
    ap.add_argument("--drop_head", type=float, default=0.3)
    ap.add_argument("--pool", choices=["asp", "stats"], default="asp")
    ap.add_argument("--no_residual", action="store_true")
    ap.add_argument("--no_specaug", action="store_true")
    ap.add_argument("--ema", type=float, default=0.998)
    ap.add_argument("--ccc_weight", type=float, default=0.0,
                    help="peso de (1-CCC) sumado al MSE; 0 = solo MSE (la métrica principal)")
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    return ap.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def band_std(feats, n_mels):
    s = torch.zeros(n_mels, dtype=torch.float64)
    ss = torch.zeros(n_mels, dtype=torch.float64)
    n = 0
    for f in feats:
        f = f.double()
        s += f.sum(dim=1)
        ss += (f ** 2).sum(dim=1)
        n += f.shape[1]
    mean = s / n
    return (ss / n - mean ** 2).clamp(min=1e-8).sqrt().float()


@torch.no_grad()
def predict(model, loader, device, mu, sd):
    model.eval()
    out = []
    for x, lengths, _ in loader:
        out.append(model(x.to(device, non_blocking=True), lengths.to(device)).float().cpu().numpy())
    z = np.concatenate(out)
    return np.clip(z * sd + mu, LABEL_MIN, LABEL_MAX)  # clip al rango de las etiquetas solo puede bajar el MSE


def make_plots(out, hist, preds, targets, m):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ep = [h["epoch"] for h in hist]
    axes[0].plot(ep, [h["train_mse_z"] for h in hist], color="blue")
    axes[0].set_title("Train MSE (targets estandarizados)")
    axes[0].set_xlabel("Época")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[1].plot(ep, [h["val_mse_raw"] for h in hist], label="val MSE (pesos)", color="orange")
    axes[1].plot(ep, [h["val_mse_ema"] for h in hist], label="val MSE (EMA)", color="green")
    axes[1].set_title("Validación (MSE, escala original)")
    axes[1].set_xlabel("Época")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=120)
    plt.close(fig)

    names = ["Activación", "Valencia", "Dominancia"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    for i, ax in enumerate(axes):
        ax.scatter(targets[:, i], preds[:, i], alpha=0.4, color="teal", edgecolors="black")
        ax.plot([LABEL_MIN, LABEL_MAX], [LABEL_MIN, LABEL_MAX], "r--", lw=2, label="Ideal")
        ax.set_title(f"{names[i]}\nMSE: {m['mse'][i]:.4f} | CCC: {m['ccc'][i]:.4f}")
        ax.set_xlabel("Valor real")
        ax.set_ylabel("Predicción")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out / "scatter_test.png", dpi=120)
    plt.close(fig)


def main():
    args = parse()
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache).expanduser()

    tr, va, te = (torch.load(cache / f"{p}.pt", map_location="cpu", weights_only=False)
                  for p in ("train", "validation", "test"))
    n_mels = tr["feats"][0].shape[0]
    tr_t = tr["labels"].numpy().astype(np.float64)
    va_t = va["labels"].numpy().astype(np.float64)
    te_t = te["labels"].numpy().astype(np.float64)
    mu, sd = tr_t.mean(axis=0), tr_t.std(axis=0)
    scale = band_std(tr["feats"], n_mels)
    z = lambda t: torch.tensor((t - mu) / sd, dtype=torch.float32)

    train_ds = FeatDataset(tr["feats"], z(tr_t), scale, augment=not args.no_specaug)
    val_ds = FeatDataset(va["feats"], z(va_t), scale)
    test_ds = FeatDataset(te["feats"], z(te_t), scale)
    sampler = BucketBatchSampler([f.shape[1] for f in tr["feats"]], args.bs, seed=args.seed)
    pin = device == "cuda"
    persistent = args.workers > 0
    train_loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=pad_collate,
                              num_workers=args.workers, pin_memory=pin, persistent_workers=persistent)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=pad_collate,
                            num_workers=2, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=pad_collate,
                             num_workers=2, pin_memory=pin)

    def build():
        return TDNN(n_mels=n_mels, width=args.width, emb=args.emb, head=args.head,
                    drop_frame=args.drop_frame, drop_head=args.drop_head, pool=args.pool,
                    residual=not args.no_residual)

    model = build().to(device)
    ema = EMA(model, args.ema)
    n_params = sum(p.numel() for p in model.parameters())

    decay, no_decay = [], []
    for p in model.parameters():
        (decay if p.ndim > 1 else no_decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    steps_per_epoch = len(sampler)
    total, warm = args.epochs * steps_per_epoch, args.warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / max(1, warm)
        p = min(1.0, (step - warm) / max(1, total - warm))
        return args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    base = ((te_t - mu) ** 2).mean(axis=0)
    print(f"Params: {n_params / 1e6:.2f} M | train {len(train_ds)} | val {len(val_ds)} | test {len(test_ds)}")
    print(f"Baseline (predecir la media del train) test MSE: "
          + " | ".join(f"{a} {b:.4f}" for a, b in zip(ATTRS, base)))

    hist, best = [], {"score": float("inf"), "epoch": 0, "which": None, "state": None, "val_pred": None}
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        sampler.set_epoch(epoch)
        model.train()
        tot, seen = 0.0, 0
        for x, lengths, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            lengths = lengths.to(device)
            pred = model(x, lengths)
            mse = F.mse_loss(pred, y)
            loss = mse + args.ccc_weight * ccc_loss(pred, y) if args.ccc_weight > 0 else mse
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            ema.update(model)
            tot += mse.item() * x.size(0)
            seen += x.size(0)

        val = {}
        for name, m in (("raw", model), ("ema", ema.shadow)):
            p = predict(m, val_loader, device, mu, sd)
            val[name] = (float(((p - va_t) ** 2).mean()), p, m)
        which = min(val, key=lambda k: val[k][0])
        score = val[which][0]
        improved = score < best["score"] - 1e-4
        if improved:
            best.update(score=score, epoch=epoch, which=which,
                        state={k: v.detach().cpu().clone() for k, v in val[which][2].state_dict().items()},
                        val_pred=val[which][1])
            bad_epochs = 0
        else:
            bad_epochs += 1
        hist.append({"epoch": epoch, "lr": opt.param_groups[0]["lr"], "train_mse_z": tot / seen,
                     "val_mse_raw": val["raw"][0], "val_mse_ema": val["ema"][0]})
        print(f"Epoch {epoch:3d}/{args.epochs} | {time.time() - t0:5.1f}s | lr {hist[-1]['lr']:.2e} | "
              f"train MSE(z) {tot / seen:.4f} | val MSE raw {val['raw'][0]:.4f} ema {val['ema'][0]:.4f}"
              f"{'  *' if improved else ''}")
        if bad_epochs >= args.patience:
            print(f"Early stopping (sin mejora en {args.patience} épocas).")
            break

    print(f"\nMejor val MSE medio: {best['score']:.4f} (época {best['epoch']}, pesos {best['which']})")
    final = build().to(device)
    final.load_state_dict(best["state"])
    torch.save({"state_dict": best["state"], "args": vars(args), "mu": mu, "sd": sd,
                "band_scale": scale}, out / "best.pt")

    val_pred = predict(final, val_loader, device, mu, sd)
    test_pred = predict(final, test_loader, device, mu, sd)
    mv, mt = metrics(val_pred, va_t), metrics(test_pred, te_t)
    ci = bootstrap_ci(test_pred, te_t)
    print("\nTEST (escala original)")
    for i, a in enumerate(ATTRS):
        print(f"  {a:10s} MSE {mt['mse'][i]:.4f} [IC95 {ci[0, i]:.4f}-{ci[1, i]:.4f}] | "
              f"MAE {mt['mae'][i]:.4f} | CCC {mt['ccc'][i]:.4f} | baseline MSE {base[i]:.4f}")
    print(f"  MSE medio test: {mt['mse'].mean():.4f} (baseline {base.mean():.4f})")

    np.save(out / "val_preds.npy", val_pred)
    np.save(out / "val_targets.npy", va_t)
    np.save(out / "test_preds.npy", test_pred)
    np.save(out / "test_targets.npy", te_t)
    with open(out / "history.json", "w") as f:
        json.dump(hist, f, indent=1)
    with open(out / "metrics.json", "w") as f:
        json.dump({"args": vars(args), "best_epoch": best["epoch"], "best_which": best["which"],
                   "val": {k: v.tolist() for k, v in mv.items()},
                   "test": {k: v.tolist() for k, v in mt.items()},
                   "test_mse_ci95": ci.tolist(), "baseline_test_mse": base.tolist()}, f, indent=1)
    make_plots(out, hist, test_pred, te_t, mt)
    print(f"\nResultados guardados en {out}")


if __name__ == "__main__":
    main()
