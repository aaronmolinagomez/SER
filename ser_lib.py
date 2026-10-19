"""Componentes compartidos del proyecto SER (TDNN para activación / valencia / dominancia)."""
import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

ATTRS = ["activation", "valence", "dominance"]
LABEL_MIN, LABEL_MAX = 1.0, 7.0


# ----------------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------------
def spec_augment(x, n_freq=2, max_f=60, n_time=2, max_t=40):
    """x: [n_mels, T] con media ~0 por banda. Máscaras en frecuencia y en tiempo (in place)."""
    n_mels, T = x.shape
    for _ in range(n_freq):
        f = random.randint(0, max_f)
        f0 = random.randint(0, n_mels - f)
        x[f0:f0 + f, :] = 0.0
    for _ in range(n_time):
        t = random.randint(0, min(max_t, T // 5))
        t0 = random.randint(0, T - t)
        x[:, t0:t0 + t] = 0.0
    return x


class FeatDataset(Dataset):
    """Log-Mel cacheados (fp16, [n_mels, T]) + targets estandarizados."""

    def __init__(self, feats, targets_z, band_scale, augment=False):
        self.feats = feats
        self.targets = targets_z
        self.band_scale = band_scale.view(-1, 1)
        self.augment = augment

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, i):
        x = self.feats[i].float() / self.band_scale
        if self.augment:
            x = spec_augment(x)
        return x, self.targets[i]


def pad_collate(batch):
    xs, ys = zip(*batch)
    lengths = torch.tensor([x.shape[1] for x in xs], dtype=torch.long)
    out = torch.zeros(len(xs), xs[0].shape[0], int(lengths.max()))
    for i, x in enumerate(xs):
        out[i, :, :x.shape[1]] = x
    return out, lengths, torch.stack(ys)


class BucketBatchSampler(Sampler):
    """Agrupa clips de largo parecido (menos padding) manteniendo el orden aleatorio entre épocas."""

    def __init__(self, lengths, batch_size, bucket_mult=20, seed=0):
        self.lengths = np.asarray(lengths)
        self.bs = batch_size
        self.bucket = batch_size * bucket_mult
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.lengths) // self.bs

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        idx = rng.permutation(len(self.lengths))
        chunks = [idx[i:i + self.bucket] for i in range(0, len(idx), self.bucket)]
        ordered = np.concatenate([c[np.argsort(self.lengths[c])] for c in chunks])
        batches = [ordered[i:i + self.bs] for i in range(0, len(ordered), self.bs)]
        batches = [b for b in batches if len(b) >= 2]  # BatchNorm de la cabeza necesita >= 2
        for j in rng.permutation(len(batches)):
            yield batches[j].tolist()


# ----------------------------------------------------------------------------
# Modelo
# ----------------------------------------------------------------------------
class MaskedBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm cuyas estadísticas de batch se calculan solo sobre frames válidos (sin padding)."""

    def forward(self, x, mask):  # x: [B, C, T], mask: [B, 1, T] (float 0/1)
        if not self.training:
            return super().forward(x)
        n = mask.sum().clamp(min=1.0)
        mean = (x * mask).sum(dim=(0, 2)) / n
        var = (((x - mean[None, :, None]) ** 2) * mask).sum(dim=(0, 2)) / n
        with torch.no_grad():
            unbiased = var * n / (n - 1.0).clamp(min=1.0)
            self.running_mean.lerp_(mean.detach(), self.momentum)
            self.running_var.lerp_(unbiased.detach(), self.momentum)
            self.num_batches_tracked += 1
        xhat = (x - mean[None, :, None]) / torch.sqrt(var[None, :, None] + self.eps)
        return xhat * self.weight[None, :, None] + self.bias[None, :, None]


class TDNNBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel, dilation, drop, residual):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel, dilation=dilation,
                              padding=dilation * (kernel - 1) // 2)  # conserva T
        self.bn = MaskedBatchNorm1d(c_out)
        self.drop = nn.Dropout(drop)
        self.res = residual and (c_in == c_out)

    def forward(self, x, mask):
        y = F.relu(self.bn(self.conv(x), mask))
        y = self.drop(y) * mask
        return y + x if self.res else y


class StatsPool(nn.Module):
    """Media + desviación estándar sobre frames válidos (statistical pooling de Snyder)."""

    def forward(self, x, mask):
        n = mask.sum(dim=2).clamp(min=1.0)
        mean = (x * mask).sum(dim=2) / n
        var = (((x - mean.unsqueeze(2)) ** 2) * mask).sum(dim=2) / n
        return torch.cat([mean, var.clamp(min=1e-5).sqrt()], dim=1)


class AttentiveStatsPool(nn.Module):
    """Pooling estadístico con atención por canal (como ECAPA), enmascarado."""

    def __init__(self, channels, att_dim=128):
        super().__init__()
        self.att = nn.Sequential(nn.Conv1d(channels, att_dim, 1), nn.Tanh(),
                                 nn.Conv1d(att_dim, channels, 1))

    def forward(self, x, mask):
        scores = self.att(x).masked_fill(mask == 0, float("-inf"))
        w = torch.softmax(scores, dim=2)
        mean = (w * x).sum(dim=2)
        var = (w * (x - mean.unsqueeze(2)) ** 2).sum(dim=2)
        return torch.cat([mean, var.clamp(min=1e-5).sqrt()], dim=1)


class TDNN(nn.Module):
    """TDNN estilo Snyder (x-vector): 5 capas de frames con contexto/dilatación + pooling + 2 densas."""

    def __init__(self, n_mels=80, width=256, emb=512, head=256, n_out=3,
                 drop_frame=0.1, drop_head=0.3, pool="asp", residual=True):
        super().__init__()
        self.blocks = nn.ModuleList([
            TDNNBlock(n_mels, width, 5, 1, drop_frame, False),   # contexto t-2..t+2
            TDNNBlock(width, width, 3, 2, drop_frame, residual),  # t-2, t, t+2
            TDNNBlock(width, width, 3, 3, drop_frame, residual),  # t-3, t, t+3
            TDNNBlock(width, width, 1, 1, drop_frame, residual),
            TDNNBlock(width, emb, 1, 1, drop_frame, False),
        ])
        self.pool = AttentiveStatsPool(emb) if pool == "asp" else StatsPool()
        self.fc1 = nn.Linear(2 * emb, head)
        self.bn = nn.BatchNorm1d(head)
        self.drop = nn.Dropout(drop_head)
        self.fc2 = nn.Linear(head, n_out)
        nn.init.normal_(self.fc2.weight, std=0.01)  # arranca prediciendo ~la media (targets estandarizados)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x, lengths):
        T = x.shape[2]
        mask = (torch.arange(T, device=x.device)[None, :] < lengths.to(x.device)[:, None])
        mask = mask.unsqueeze(1).to(x.dtype)
        for block in self.blocks:
            x = block(x, mask)
        h = self.pool(x, mask)
        h = self.drop(F.relu(self.bn(self.fc1(h))))
        return self.fc2(h)


class EMA:
    """Promedio móvil exponencial de los pesos (suaviza la validación ruidosa)."""

    def __init__(self, model, decay):
        self.decay = decay
        self.step = 0
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        d = min(self.decay, (1 + self.step) / (10 + self.step))
        for ps, pm in zip(self.shadow.parameters(), model.parameters()):
            ps.mul_(d).add_(pm.detach(), alpha=1.0 - d)
        for bs, bm in zip(self.shadow.buffers(), model.buffers()):
            bs.copy_(bm)


# ----------------------------------------------------------------------------
# Losses y métricas
# ----------------------------------------------------------------------------
def ccc_loss(pred, target):
    """1 - CCC promedio sobre atributos (varianza sesgada, consistente con la covarianza)."""
    losses = []
    for i in range(pred.shape[1]):
        p, t = pred[:, i], target[:, i]
        mp, mt = p.mean(), t.mean()
        vp = ((p - mp) ** 2).mean()
        vt = ((t - mt) ** 2).mean()
        cov = ((p - mp) * (t - mt)).mean()
        losses.append(1.0 - 2 * cov / (vp + vt + (mp - mt) ** 2 + 1e-8))
    return torch.stack(losses).mean()


def ccc_np(y, p):
    my, mp = y.mean(), p.mean()
    cov = ((y - my) * (p - mp)).mean()
    return 2 * cov / (y.var() + p.var() + (my - mp) ** 2 + 1e-12)


def metrics(preds, targets):
    return {
        "mse": ((preds - targets) ** 2).mean(axis=0),
        "mae": np.abs(preds - targets).mean(axis=0),
        "ccc": np.array([ccc_np(targets[:, i], preds[:, i]) for i in range(targets.shape[1])]),
    }


def bootstrap_ci(preds, targets, n_boot=2000, seed=0):
    """IC 95% del MSE por atributo remuestreando clips del test. Devuelve [2, n_attr]."""
    err2 = (preds - targets) ** 2
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(err2), size=(n_boot, len(err2)))
    return np.percentile(err2[idx].mean(axis=1), [2.5, 97.5], axis=0)
