import numpy as np
import torch
import pytest

from pqrst.estimators.classical_fourier import FourierFeatureStatisticsNetwork
from pqrst.estimators.mine.amortized import (
    MaskedStatisticsNetwork,
    AmortizedTrainConfig,
    AmortizedTEEstimator,
    train_amortized,
)
from pqrst.data.synthetic.corpus import Window


def test_fourier_network_param_count_is_small():
    model = FourierFeatureStatisticsNetwork(n_harmonics=4)
    n_params = sum(p.numel() for p in model.parameters())
    # encode_w(4) + encode_b(4) + readout(4*2*4 + 1 = 33) = 41
    assert n_params == 41
    # Phai nho hon han T_phi chinh (25473) - dung tinh than "it tham so hon nhieu"
    n_params_main = sum(p.numel() for p in MaskedStatisticsNetwork([128, 128, 64]).parameters())
    assert n_params < n_params_main / 100


def test_fourier_network_forward_shape_and_no_nan():
    model = FourierFeatureStatisticsNetwork(n_harmonics=4)
    N = 30
    y_t = torch.randn(N, 1)
    x_lag = torch.randn(N, 1)
    y_lag = torch.randn(N, 1)
    mask = torch.ones(N, 1)
    out = model(y_t, x_lag, y_lag, mask)
    assert out.shape == (N,)
    assert torch.isfinite(out).all()


def test_fourier_network_mask_zeroes_x_lag():
    model = FourierFeatureStatisticsNetwork(n_harmonics=4)
    N = 10
    y_t = torch.randn(N, 1)
    x_lag = torch.randn(N, 1)
    y_lag = torch.randn(N, 1)
    mask_full = torch.ones(N, 1)
    mask_reduced = torch.zeros(N, 1)

    out_full = model(y_t, x_lag, y_lag, mask_full)
    out_reduced = model(y_t, x_lag, y_lag, mask_reduced)
    # Voi mask=0, x_lag bi triet tieu hoan toan truoc khi vao mang - 2 output
    # phai khac out_full (tru truong hop suy bien x_lag=0 tinh cong, rat khong
    # co the voi input ngau nhien lien tuc).
    assert not torch.allclose(out_full, out_reduced)


def test_fourier_network_gradient_flows():
    model = FourierFeatureStatisticsNetwork(n_harmonics=4)
    N = 10
    y_t = torch.randn(N, 1)
    x_lag = torch.randn(N, 1)
    y_lag = torch.randn(N, 1)
    mask = torch.ones(N, 1)
    out = model(y_t, x_lag, y_lag, mask)
    out.sum().backward()
    for p in model.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def _make_window(seed: int, n: int = 20) -> Window:
    rng = np.random.default_rng(seed)
    return Window(
        y_t=rng.standard_normal(n),
        x_lag=rng.standard_normal(n),
        y_lag=rng.standard_normal(n),
        n_samples=n,
        config_name="test_config",
        params={"a": 0.5, "b": 0.5, "c": 0.5, "noise_std": 0.5},
        te_ground_truth=0.1,
        mi_full_ground_truth=0.2,
        mi_reduced_ground_truth=0.1,
        seed=seed,
    )


def test_train_amortized_model_factory_preserves_default_architecture():
    """Khong truyen model_factory -> PHAI giu dung hanh vi cu (MaskedStatisticsNetwork
    voi hidden_dims tu config) - test hoi quy dam bao khong pha vo hanh vi cu."""
    train_windows = [_make_window(i) for i in range(4)]
    val_windows = [_make_window(100 + i) for i in range(2)]
    config = AmortizedTrainConfig(
        hidden_dims=[8, 8], max_epochs=2, patience=5,
        windows_per_batch=2, eval_n_shuffles=2, final_estimate_last_k_epochs=1,
    )
    estimator, history = train_amortized(train_windows, val_windows, config)
    assert isinstance(estimator.model, MaskedStatisticsNetwork)
    assert "train_loss_history" in history


def test_train_amortized_accepts_custom_model_factory():
    """model_factory tuy chinh -> phai dung DUNG kien truc do, khong roi ve
    MaskedStatisticsNetwork."""
    train_windows = [_make_window(i) for i in range(4)]
    val_windows = [_make_window(100 + i) for i in range(2)]
    config = AmortizedTrainConfig(
        max_epochs=2, patience=5, windows_per_batch=2,
        eval_n_shuffles=2, final_estimate_last_k_epochs=1,
    )
    estimator, history = train_amortized(
        train_windows, val_windows, config,
        model_factory=lambda: FourierFeatureStatisticsNetwork(n_harmonics=3),
    )
    assert isinstance(estimator.model, FourierFeatureStatisticsNetwork)
    assert np.isfinite(history["final_val_loss"])


def test_fourier_estimator_estimate_via_amortized_wrapper():
    """FourierFeatureStatisticsNetwork phai cam duoc thang vao AmortizedTEEstimator
    (khong can class rieng) vi estimate_te_from_window() la ham chung, model-agnostic."""
    model = FourierFeatureStatisticsNetwork(n_harmonics=4)
    config = AmortizedTrainConfig(standardize=True)
    estimator = AmortizedTEEstimator(model, config)

    rng = np.random.default_rng(0)
    x = rng.standard_normal(31)
    y = rng.standard_normal(31)
    te = estimator.estimate(x, y)
    assert isinstance(te, float)
    assert np.isfinite(te)
