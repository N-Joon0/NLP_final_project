from __future__ import annotations

import torch
import torch.nn.functional as F


def kd_per_example(z_s: torch.Tensor, z_t: torch.Tensor, T: float) -> torch.Tensor:
    return F.kl_div(F.log_softmax(z_s / T, -1), F.softmax(z_t / T, -1), reduction="none").sum(-1) * T * T


@torch.no_grad()
def difficulty_terms(z_s: torch.Tensor, z_t: torch.Tensor):
    q_t = F.softmax(z_t.float(), -1)
    q_s = F.softmax(z_s.float().detach(), -1)
    c = q_t.max(-1).values
    d = 0.5 * (q_t - q_s).abs().sum(-1)
    return c, d


def instance_weights(z_s, z_t, gamma: float, beta: float, weight_norm: str = "none", norm_mean=None):
    with torch.no_grad():
        c, d = difficulty_terms(z_s, z_t)
        w = c.pow(gamma) * (1.0 + beta * d)
        if weight_norm == "none":
            w_t = w
        elif weight_norm == "batch_mean":
            w_t = w / ((w.mean() if norm_mean is None else norm_mean) + 1e-8)
        else:
            raise ValueError(f"weight_norm 은 none|batch_mean: {weight_norm}")
    return w_t, c, d


def distill_loss(z_s, labels, z_t=None, *, lam=0.0, T=1.0, gamma=0.0, beta=0.0, weight_norm="none", norm_mean=None):
    z_s = z_s.float()
    ce = F.cross_entropy(z_s, labels, reduction="none")
    if lam == 0 or z_t is None:
        return ce.mean(), {"ce": ce.detach()}
    z_t = z_t.float()
    kd = kd_per_example(z_s, z_t, T)
    w_t, c, d = instance_weights(z_s, z_t, gamma, beta, weight_norm, norm_mean)
    loss = (ce + lam * w_t * kd).mean()
    return loss, {"ce": ce.detach(), "kd": kd.detach(), "w": w_t, "c": c, "d": d}
