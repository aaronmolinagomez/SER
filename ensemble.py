"""Promedia las predicciones de varias corridas (semillas) y calibra linealmente con validación.

Uso:
  python ensemble.py runs/tdnn_s0 runs/tdnn_s1 runs/tdnn_s2 ...
"""
import argparse
from pathlib import Path

import numpy as np

from ser_lib import ATTRS, LABEL_MAX, LABEL_MIN, bootstrap_ci, metrics


def show(name, preds, targets):
    m = metrics(preds, targets)
    ci = bootstrap_ci(preds, targets)
    cols = " | ".join(f"{a[:3]} {m['mse'][i]:.4f} [{ci[0, i]:.3f}-{ci[1, i]:.3f}]" for i, a in enumerate(ATTRS))
    print(f"{name:28s} MSE {cols} | medio {m['mse'].mean():.4f} | CCC medio {m['ccc'].mean():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="ensemble_test_preds.npy")
    args = ap.parse_args()

    runs = [Path(r) for r in args.runs]
    vp = [np.load(r / "val_preds.npy") for r in runs]
    tp = [np.load(r / "test_preds.npy") for r in runs]
    vt = np.load(runs[0] / "val_targets.npy")
    tt = np.load(runs[0] / "test_targets.npy")
    for r in runs[1:]:
        assert np.allclose(vt, np.load(r / "val_targets.npy")) and np.allclose(tt, np.load(r / "test_targets.npy")), \
            f"{r}: los targets no coinciden con la primera corrida"

    print("TEST (IC95 por bootstrap entre corchetes)")
    for r, p in zip(runs, tp):
        show(r.name, p, tt)
    ens_v, ens_t = np.mean(vp, axis=0), np.mean(tp, axis=0)
    show(f"ensamble de {len(runs)}", ens_t, tt)

    # calibración y = a*pred + b por atributo, ajustada en validación y aplicada a test
    cal = np.empty_like(ens_t)
    for i in range(ens_t.shape[1]):
        a, b = np.polyfit(ens_v[:, i], vt[:, i], 1)
        print(f"  calibración {ATTRS[i]:10s}: a={a:.3f} b={b:.3f}")
        cal[:, i] = a * ens_t[:, i] + b
    cal = np.clip(cal, LABEL_MIN, LABEL_MAX)
    show("ensamble + calibración", cal, tt)

    np.save(args.out, ens_t)
    np.save(Path(args.out).with_name("ensemble_calibrated_test_preds.npy"), cal)
    print(f"Guardado {args.out}")


if __name__ == "__main__":
    main()
