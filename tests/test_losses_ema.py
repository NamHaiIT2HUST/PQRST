import torch
import pytest

from pqrst.estimators.mine.losses import donsker_varadhan_loss, donsker_varadhan_loss_ema


def test_ema_first_call_initializes_from_batch():
    torch.manual_seed(0)
    t_joint = torch.randn(20)
    t_marginal = torch.randn(20)

    _, dv_bound, ma_et_new = donsker_varadhan_loss_ema(t_joint, t_marginal, ma_et=None)

    expected_mean_et = torch.exp(t_marginal).mean()
    assert torch.isclose(ma_et_new, expected_mean_et)


def test_ema_update_formula():
    torch.manual_seed(1)
    t_joint = torch.randn(20)
    t_marginal = torch.randn(20)
    ma_et_prev = torch.tensor(2.5)
    momentum = 0.01

    _, _, ma_et_new = donsker_varadhan_loss_ema(t_joint, t_marginal, ma_et_prev, momentum=momentum)

    mean_et = torch.exp(t_marginal).mean()
    expected = (1 - momentum) * ma_et_prev + momentum * mean_et
    assert torch.isclose(ma_et_new, expected)


def test_ema_dv_bound_matches_standard_loss_value():
    """dv_bound tra ve (dung de bao cao) phai KHOP gia tri DV bound thuong (chi
    khac cach tinh logsumexp vs log(mean(exp)) - toan hoc tuong duong)."""
    torch.manual_seed(2)
    t_joint = torch.randn(50)
    t_marginal = torch.randn(50)

    standard_loss = donsker_varadhan_loss(t_joint, t_marginal)  # = -dv_bound
    _, dv_bound_ema, _ = donsker_varadhan_loss_ema(t_joint, t_marginal, ma_et=None)

    assert torch.isclose(-standard_loss, dv_bound_ema, atol=1e-5)


def test_ema_loss_gradient_is_finite_and_nonzero():
    torch.manual_seed(3)
    w = torch.nn.Parameter(torch.randn(5))
    x = torch.randn(20, 5)
    t_joint = x @ w
    t_marginal = x[torch.randperm(20)] @ w

    loss, _, ma_et_new = donsker_varadhan_loss_ema(t_joint, t_marginal, ma_et=None)
    loss.backward()

    assert w.grad is not None
    assert torch.isfinite(w.grad).all()
    assert ma_et_new.requires_grad is False  # EMA khong duoc lan truyen gradient
