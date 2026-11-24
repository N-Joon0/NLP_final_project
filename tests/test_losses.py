import torch
import torch.nn.functional as F

from dakd.losses import difficulty_terms, distill_loss, instance_weights, kd_per_example

torch.manual_seed(0)
B, K = 16, 5
zs = torch.randn(B, K, requires_grad=True)
zt = torch.randn(B, K) * 2
y = torch.randint(0, K, (B,))


def test_std_kd_equals_mean_ce_plus_lambda_kd():
    loss, _ = distill_loss(zs, y, zt, lam=0.7, T=2.0, gamma=0.0, beta=0.0, weight_norm="none")
    ref = (F.cross_entropy(zs, y, reduction="none") + 0.7 * kd_per_example(zs, zt, 2.0)).mean()
    assert abs(loss.item() - ref.item()) < 1e-6


def test_lambda_zero_is_ce():
    loss, info = distill_loss(zs, y, zt, lam=0.0, T=2.0, gamma=1.0, beta=1.0)
    assert abs(loss.item() - F.cross_entropy(zs, y).item()) < 1e-6
    assert "kd" not in info


def test_weights_have_no_grad():
    w, c, d = instance_weights(zs, zt, 1.0, 1.0, "none")
    assert not w.requires_grad and not c.requires_grad and not d.requires_grad
    zs2 = zs.detach().clone().requires_grad_()
    loss, info = distill_loss(zs2, y, zt, lam=1.0, T=2.0, gamma=1.0, beta=1.0)
    loss.backward()
    assert not info["w"].requires_grad and zs2.grad is not None


def test_batch_mean_normalization():
    w, _, _ = instance_weights(zs, zt, 1.0, 1.0, "batch_mean")
    assert abs(w.mean().item() - 1.0) < 1e-5


def test_d_range_and_identical_logits():
    _, d = difficulty_terms(zs, zt)
    assert ((d >= 0) & (d <= 1)).all()
    z = torch.randn(B, K)
    _, d0 = difficulty_terms(z, z)
    assert d0.abs().max() < 1e-6
    assert kd_per_example(z, z, 3.0).abs().max() < 1e-5


def test_weight_formula():
    c, d = difficulty_terms(zs, zt)
    w, _, _ = instance_weights(zs, zt, 2.0, 0.5, "none")
    assert torch.allclose(w, c ** 2 * (1 + 0.5 * d), atol=1e-6)
    s1, _, _ = instance_weights(zs, zt, 0.0, 0.0, "none")
    assert torch.allclose(s1, torch.ones(B))
