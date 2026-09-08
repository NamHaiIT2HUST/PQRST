"""Bo sinh du lieu VAR tuyen tinh Gaussian, co ground-truth TE dang cong thuc dong.

Mo hinh (bivariate VAR(1)):
    X[t] = a * X[t-1] + eps_x[t]
    Y[t] = b * Y[t-1] + c * X[t-1] + eps_y[t]
    eps_x, eps_y ~ N(0, noise_std^2), doc lap voi nhau va theo thoi gian

`c` la cuong do ghep noi X -> Y. Voi he tuyen tinh Gaussian, TE(X->Y) co cong thuc
dong tuong duong Granger causality (Geweke, 1982):
    TE(X->Y) = 0.5 * log( var(residual hoi quy Y[t] tren Y[t-1] don) /
                           var(residual hoi quy Y[t] tren Y[t-1] va X[t-1]) )

Xem chi tiet spec va ly do chon mo hinh nay trong docs/PHASE_P_GUIDE.md, muc 2.1.
"""

from __future__ import annotations

import numpy as np


def generate_var_linear_gaussian(
    n_samples: int,
    a: float,
    b: float,
    c: float,
    noise_std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Sinh 1 cap chuoi (x, y) tu mo hinh VAR(1) tuyen tinh Gaussian o tren.

    Args:
        n_samples: do dai chuoi thoi gian can sinh.
        a: he so tu-hoi-quy cua X.
        b: he so tu-hoi-quy cua Y.
        c: cuong do ghep noi X[t-1] -> Y[t]. c == 0 nghia la khong ghep noi.
        noise_std: do lech chuan cua nhieu Gaussian cong vao ca X va Y.
        seed: seed cho RNG, dam bao tai lap duoc.

    Returns:
        (x, y, te_ground_truth): x, y la mang 1D shape (n_samples,);
        te_ground_truth la gia tri TE(X->Y) tinh theo cong thuc dong o tren (don vi: nat).

    TODO(ban tu code):
        - Dung np.random.default_rng(seed) de sinh nhieu, khong dung np.random global state.
        - Can 1 buoc "burn-in" (bo vai chuc sample dau) de chuoi qua transient truoc khi lay mau,
          neu |a|, |b| gan 1.
        - te_ground_truth tinh truc tiep tu a, b, c, noise_std (khong can uoc luong tu du lieu
          da sinh) - xem cong thuc trong docstring module.
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
        y[t] = b * y[t-1] + c * x[t-1] + eps_y[t]
        
    te_ground_truth = 0.5 * np.log(1 + (c**2) / (1 - a**2))
    
    return x[burn_in:], y[burn_in:], float(te_ground_truth)
