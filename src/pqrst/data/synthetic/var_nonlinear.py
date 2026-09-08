"""Bo sinh du lieu VAR phi tuyen — khong co cong thuc dong cho TE, dung pseudo ground-truth.

Mo hinh de xuat (bivariate, ghep noi phi tuyen qua tanh):
    X[t] = a * X[t-1] + eps_x[t]
    Y[t] = b * Y[t-1] + c * tanh(X[t-1]) + eps_y[t]
    eps_x, eps_y ~ N(0, noise_std^2)

Khong co cong thuc dong cho TE(X->Y) o day. Ground-truth phai lay tu uoc luong hoi tu
o N rat lon (vi du N = 200_000) bang KSG hoac binning do phan giai cao, coi la
"pseudo ground-truth" - luu lai trong configs/synthetic/var_nonlinear.yaml de khong
phai tinh lai moi lan (truong pseudo_ground_truth_te).

Xem chi tiet trong docs/PHASE_P_GUIDE.md, muc 2.2.
"""

from __future__ import annotations

import numpy as np


def generate_var_nonlinear(
    n_samples: int,
    a: float,
    b: float,
    c: float,
    noise_std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sinh 1 cap chuoi (x, y) tu mo hinh VAR phi tuyen o tren.

    Khac voi generate_var_linear_gaussian, ham nay KHONG tra ve ground-truth TE
    (vi khong co cong thuc dong) - pseudo ground-truth duoc tinh rieng, mot lan,
    bang script scripts/estimate_pseudo_ground_truth.py (ban tu viet, N rat lon)
    va luu vao config YAML tuong ung.

    Args:
        n_samples: do dai chuoi thoi gian can sinh.
        a: he so tu-hoi-quy cua X.
        b: he so tu-hoi-quy cua Y.
        c: cuong do ghep noi phi tuyen X -> Y (qua tanh).
        noise_std: do lech chuan nhieu Gaussian.
        seed: seed cho RNG.

    Returns:
        (x, y): mang 1D shape (n_samples,) moi mang.

    TODO(ban tu code):
        - Cung logic burn-in nhu var_linear_gaussian.
        - Can nhac them 1 tham so 'nonlinearity' (vd: "tanh" | "square") neu muon
          thu nhieu dang phi tuyen khac nhau thay vi hard-code tanh.
    """
    rng = np.random.default_rng(seed)
    burn_in = 100
    total_samples = n_samples + burn_in
    
    x = np.zeros(total_samples)
    y = np.zeros(total_samples)
    
    eps_x = rng.normal(0, noise_std, total_samples)
    eps_y = rng.normal(0, noise_std, total_samples)
    
    for t in range(1, total_samples):
        x[t] = a * x[t-1] + eps_x[t]
        y[t] = b * y[t-1] + c * np.tanh(x[t-1]) + eps_y[t]
        
    return x[burn_in:], y[burn_in:]
