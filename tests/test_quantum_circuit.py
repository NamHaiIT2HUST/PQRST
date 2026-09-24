import numpy as np
import torch

from pqrst.estimators.quantum.wrapper import QuantumStatisticsNetwork
from pqrst.estimators.mine.amortized import AmortizedTEEstimator, AmortizedTrainConfig
from pqrst.evaluation.feature_space import windows_to_tensors


def _make_batch(seed: int, n: int = 10):
    rng = np.random.default_rng(seed)
    y_t = torch.tensor(rng.standard_normal(n), dtype=torch.float32).unsqueeze(-1)
    x_lag = torch.tensor(rng.standard_normal(n), dtype=torch.float32).unsqueeze(-1)
    y_lag = torch.tensor(rng.standard_normal(n), dtype=torch.float32).unsqueeze(-1)
    mask = torch.ones(n, 1)
    return y_t, x_lag, y_lag, mask


def test_param_count_matches_design():
    model = QuantumStatisticsNetwork(n_qubits=6, n_layers=4)
    n_params = sum(p.numel() for p in model.parameters())
    # encode_scale(6) + theta(4*6*2=48) + readout(6+1=7) = 61
    assert n_params == 61


def test_forward_output_shape_and_finite():
    model = QuantumStatisticsNetwork(n_qubits=6, n_layers=4)
    y_t, x_lag, y_lag, mask = _make_batch(0, n=8)
    out = model(y_t, x_lag, y_lag, mask)
    assert out.shape == (8,)
    assert torch.isfinite(out).all()


def test_gradient_flows_through_all_parameters():
    """Dieu kien thoat Pha P' (docs/NHIP2_GUIDE.md muc 3): 'gradient tinh duoc
    qua it nhat 1 buoc'."""
    model = QuantumStatisticsNetwork(n_qubits=6, n_layers=4)
    y_t, x_lag, y_lag, mask = _make_batch(0, n=8)
    out = model(y_t, x_lag, y_lag, mask)
    out.sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} khong co gradient"
        assert torch.isfinite(p.grad).all(), f"{name} co gradient khong huu han"


def test_mask_zeroes_x_lag_contribution():
    model = QuantumStatisticsNetwork(n_qubits=6, n_layers=4)
    y_t, x_lag, y_lag, _ = _make_batch(0, n=6)
    mask_full = torch.ones(6, 1)
    mask_reduced = torch.zeros(6, 1)
    out_full = model(y_t, x_lag, y_lag, mask_full)
    out_reduced = model(y_t, x_lag, y_lag, mask_reduced)
    assert not torch.allclose(out_full, out_reduced)


def test_circuit_is_not_additively_separable():
    """Test QUAN TRONG NHAT - regression truc tiep cho bug da tim thay o
    FourierFeatureStatisticsNetwork (Pha P'.0, xem classical_fourier.py va
    docs/NHIP2_GUIDE.md muc 2.1): neu entangling KHONG thuc su tron thong tin
    giua qubit encode X va qubit encode Y, T(y_t,x_lag,y_lag,mask) se la ham
    tach roi cong tinh f(y_t)+g(x_lag)+h(y_lag)+k(mask) - va theo dung bat dang
    thuc Jensen da chung minh, DV bound se bi chan tuyet doi o 0.

    Kiem tra truc tiep: doi x_lag PHAI lam output thay doi THEO CACH PHU THUOC
    vao gia tri cu the cua y_t/y_lag (khong phai 1 do lech CONG THEM giong nhau
    cho moi dong) - day la dau hieu toan hoc cua ham KHONG tach roi cong tinh."""
    torch.manual_seed(0)
    model = QuantumStatisticsNetwork(n_qubits=6, n_layers=4)
    mask = torch.ones(2, 1)

    y_t_a = torch.tensor([[-2.0], [2.0]])
    y_lag_a = torch.tensor([[-2.0], [2.0]])
    x_lag_1 = torch.tensor([[0.3], [0.3]])
    x_lag_2 = torch.tensor([[1.7], [1.7]])

    with torch.no_grad():
        out_1 = model(y_t_a, x_lag_1, y_lag_a, mask)
        out_2 = model(y_t_a, x_lag_2, y_lag_a, mask)
    diff = out_2 - out_1

    # Neu tach roi cong tinh: diff[0] == diff[1] (cung 1 do lech g(x2)-g(x1)
    # cong them vao moi dong, khong phu thuoc y_t/y_lag).
    assert not torch.isclose(diff[0], diff[1], atol=1e-4), (
        "T(y_t,x_lag,y_lag,mask) co ve TACH ROI CONG TINH - kiem tra lai entangling "
        "trong circuit.py, xem canh bao trong docs/NHIP2_GUIDE.md muc 2.1"
    )


def test_forward_blocks_matches_forward_via_readout():
    model = QuantumStatisticsNetwork(n_qubits=6, n_layers=4)
    y_t, x_lag, y_lag, mask = _make_batch(0, n=5)
    out_direct = model(y_t, x_lag, y_lag, mask)
    blocks = model.forward_blocks(y_t, x_lag, y_lag, mask)
    assert set(blocks.keys()) == {"block0_input", "block1"}
    assert blocks["block0_input"].shape == (5, 4)
    assert blocks["block1"].shape == (5, 6)
    out_from_block = model.readout(blocks["block1"]).squeeze(-1)
    assert torch.allclose(out_direct, out_from_block, atol=1e-6)


def test_plugs_into_amortized_te_estimator_without_new_class():
    """T_theta phai cam duoc thang vao AmortizedTEEstimator (dung interface, giong
    FourierFeatureStatisticsNetwork o Pha P'.0) - khong can class rieng."""
    model = QuantumStatisticsNetwork(n_qubits=6, n_layers=3)
    config = AmortizedTrainConfig(standardize=True, eval_n_shuffles=3)
    estimator = AmortizedTEEstimator(model, config)

    rng = np.random.default_rng(0)
    x = rng.standard_normal(11)
    y = rng.standard_normal(11)
    te = estimator.estimate(x, y, seed=0)
    assert isinstance(te, float)
    assert np.isfinite(te)


def test_windows_to_tensors_compatible_input():
    """T_theta phai nhan dung dinh dang tensor tu windows_to_tensors() (dung chung
    voi ca 2 kien truc khac, xem feature_space.py)."""
    model = QuantumStatisticsNetwork(n_qubits=6, n_layers=3)
    rng = np.random.default_rng(1)
    y_t = rng.standard_normal(7)
    x_lag = rng.standard_normal(7)
    y_lag = rng.standard_normal(7)
    ty_t, tx_lag, ty_lag, mask = windows_to_tensors(y_t, x_lag, y_lag, standardize=False)
    out = model(ty_t, tx_lag, ty_lag, mask)
    assert out.shape == (7,)
    assert torch.isfinite(out).all()
