import numpy as np
import torch

from pqrst.estimators.mine.amortized import MaskedStatisticsNetwork
from pqrst.estimators.classical_fourier import FourierFeatureStatisticsNetwork
from pqrst.evaluation.feature_space import (
    windows_to_tensors, collect_block_records, fit_pca, project_pca,
)


def _make_sample(seed: int, n: int = 20, coupling_c: float = 0.0):
    rng = np.random.default_rng(seed)
    y_t = rng.standard_normal(n)
    x_lag = rng.standard_normal(n)
    y_lag = rng.standard_normal(n)
    return y_t, x_lag, y_lag, {"coupling_c": coupling_c, "source": "synthetic"}


def test_windows_to_tensors_shapes():
    y_t, x_lag, y_lag, _ = _make_sample(0, n=15)
    ty_t, tx_lag, ty_lag, mask = windows_to_tensors(y_t, x_lag, y_lag, standardize=False)
    assert ty_t.shape == (15, 1)
    assert tx_lag.shape == (15, 1)
    assert ty_lag.shape == (15, 1)
    assert mask.shape == (15, 1)
    assert torch.all(mask == 1.0)


def test_masked_network_forward_blocks_matches_forward_pattern():
    """block cuoi cung PHAI la input DUNG cho lop Linear cuoi - kiem tra so luong
    khoi dung voi so lop An, va block0_input dung 4 chieu (y_t,x_lag,y_lag,mask)."""
    model = MaskedStatisticsNetwork(hidden_dims=[8, 8])
    ty_t, tx_lag, ty_lag, mask = windows_to_tensors(*_make_sample(0, n=10)[:3])
    blocks = model.forward_blocks(ty_t, tx_lag, ty_lag, mask)
    assert set(blocks.keys()) == {"block0_input", "block1", "block2"}
    assert blocks["block0_input"].shape == (10, 4)
    assert blocks["block1"].shape == (10, 8)
    assert blocks["block2"].shape == (10, 8)


def test_fourier_network_forward_blocks_has_two_blocks():
    model = FourierFeatureStatisticsNetwork(n_harmonics=4, n_directions=5)
    ty_t, tx_lag, ty_lag, mask = windows_to_tensors(*_make_sample(0, n=10)[:3])
    blocks = model.forward_blocks(ty_t, tx_lag, ty_lag, mask)
    assert set(blocks.keys()) == {"block0_input", "block1"}
    assert blocks["block0_input"].shape == (10, 4)
    assert blocks["block1"].shape == (10, 5 * 2 * 4)


def test_collect_block_records_joint_and_marginal_counts():
    model = MaskedStatisticsNetwork(hidden_dims=[8, 8])
    samples = [_make_sample(i, coupling_c=0.2 * i) for i in range(4)]

    records = collect_block_records(model, samples, include_marginal=True)
    assert len(records) == 8  # 4 samples x (joint + marginal)
    kinds = {r["kind"] for r in records}
    assert kinds == {"joint", "marginal"}
    # Nhan goc phai duoc giu nguyen trong record dau ra.
    assert all("coupling_c" in r and "source" in r for r in records)

    records_joint_only = collect_block_records(model, samples, include_marginal=False)
    assert len(records_joint_only) == 4
    assert all(r["kind"] == "joint" for r in records_joint_only)


def test_collect_block_records_activation_shape_is_per_window_vector():
    model = MaskedStatisticsNetwork(hidden_dims=[8, 8])
    samples = [_make_sample(0, n=10)]
    records = collect_block_records(model, samples, include_marginal=False)
    # Trung binh qua cua so -> 1 vector (khong con truc N mau).
    assert records[0]["block1"].shape == (8,)


def test_fit_pca_and_project_pca_roundtrip():
    model = MaskedStatisticsNetwork(hidden_dims=[8, 8])
    samples = [_make_sample(i, coupling_c=0.2 * i) for i in range(10)]
    records = collect_block_records(model, samples, include_marginal=False)

    Xp, pca, used = fit_pca(records, "block1", n_components=2)
    assert Xp.shape == (10, 2)
    assert len(used) == 10

    # Chieu LAI DUNG cac record da fit phai cho DUNG lai ket qua (round-trip).
    Xp_again = project_pca(pca, used, "block1")
    assert np.allclose(Xp, Xp_again, atol=1e-5)


def test_fit_pca_filter_fn_selects_subset():
    model = MaskedStatisticsNetwork(hidden_dims=[8, 8])
    samples = [_make_sample(i, coupling_c=0.2 * i) for i in range(5)]
    records = collect_block_records(model, samples, include_marginal=True)

    Xp, pca, used = fit_pca(records, "block1", filter_fn=lambda r: r["kind"] == "joint")
    assert len(used) == 5
    assert all(r["kind"] == "joint" for r in used)


def test_project_pca_on_different_source_does_not_refit():
    """Mo phong dung use-case 'domain gap': fit tren 1 nguon, chieu nguon KHAC vao
    CUNG khong gian - PCA object khong bi thay doi boi project_pca."""
    model = MaskedStatisticsNetwork(hidden_dims=[8, 8])
    samples_a = [_make_sample(i, coupling_c=0.2 * i) for i in range(8)]
    samples_b = [_make_sample(100 + i, coupling_c=0.9) for i in range(3)]

    records_a = collect_block_records(model, samples_a, include_marginal=False)
    records_b = collect_block_records(model, samples_b, include_marginal=False)

    Xp_a, pca, _ = fit_pca(records_a, "block0_input")
    components_before = pca.components_.copy()

    Xp_b = project_pca(pca, records_b, "block0_input")

    assert np.array_equal(pca.components_, components_before)  # khong fit lai
    assert Xp_b.shape == (3, 2)
