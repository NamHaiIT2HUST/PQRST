"""Bo sinh du lieu VAR tuyen tinh Gaussian, co ground-truth TE dang cong thuc dong.

Mo hinh (bivariate VAR(1)):
    X[t] = a * X[t-1] + eps_x[t]
    Y[t] = b * Y[t-1] + c * X[t-1] + eps_y[t]
    eps_x, eps_y ~ N(0, noise_std^2), doc lap voi nhau va theo thoi gian

`c` la cuong do ghep noi X -> Y. Voi he tuyen tinh Gaussian, TE(X->Y) co cong thuc
dong tuong duong Granger causality (Geweke, 1982):
    TE(X->Y) = 0.5 * log( var(residual hoi quy Y[t] tren Y[t-1] don) /
                           var(residual hoi quy Y[t] tren Y[t-1] va X[t-1]) )

var(residual hoi quy Y[t] tren Y[t-1] va X[t-1]) = noise_std^2 (dung bang phuong sai
nhieu eps_y, vi mo hinh dung la tuyen tinh trong Y[t-1], X[t-1]).

var(residual hoi quy Y[t] tren Y[t-1] DON) KHONG duoc phep xap xi bang var(X[t-1]) =
noise_std^2/(1-a^2) - do la mot loi de mac phai (da tung co trong 1 phien ban truoc
cua file nay, gay sai so ~8% so voi mo phong Monte Carlo o cau hinh a=0.5,b=0.5,c=0.6).
Sai lam nam o cho gia dinh ngam X[t-1] doc lap voi Y[t-1], nhung dieu do CHI dung khi
a = 0: khi a != 0, Y[t-1] phu thuoc X[t-2], ma X[t-2] tuong quan voi X[t-1] qua he so
tu-hoi-quy a cua X, nen X[t-1] va Y[t-1] noi chung tuong quan voi nhau khi a != 0 va
c != 0 (dung ca 2 dieu kien nay).

Cong thuc dong CHINH XAC (giai he phuong trinh Lyapunov roi rac cho hiep phuong sai
dung cua VAR(1) 2 chieu [X, Y], roi dung Var(Y[t]|Y[t-1]) = c^2 * Var(X[t-1]|Y[t-1])
+ noise_std^2 qua luat total variance):
    sigma2 = noise_std^2
    s_xx = sigma2 / (1 - a^2)                          # Var(X) dung
    s_xy = a * c * s_xx / (1 - a*b)                     # Cov(X, Y) dung
    s_yy = (c^2*s_xx + 2*b*c*s_xy + sigma2) / (1 - b^2)  # Var(Y) dung
    var_x_given_y = s_xx - s_xy^2 / s_yy                 # Var(X[t-1] | Y[t-1])
    TE(X->Y) = 0.5 * log(1 + c^2 * var_x_given_y / sigma2)

Da verify cong thuc nay khop mo phong Monte Carlo truc tiep (N=2,000,000, hoi quy OLS
Y[t] tren [Y[t-1], X[t-1]] va tren Y[t-1] don) trong sai so < 0.1% tuong doi.

Xem chi tiet spec va ly do chon mo hinh nay trong docs/PHASE_P_GUIDE.md, muc 2.1.
"""

from __future__ import annotations

import numpy as np


def _compute_stationary_covariance(a: float, b: float, c: float, noise_std: float) -> tuple[float, float, float]:
    sigma2 = noise_std**2
    s_xx = sigma2 / (1 - a**2)
    s_xy = a * c * s_xx / (1 - a * b)
    s_yy = (c**2 * s_xx + 2 * b * c * s_xy + sigma2) / (1 - b**2)
    return s_xx, s_xy, s_yy


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
        
    s_xx, s_xy, s_yy = _compute_stationary_covariance(a, b, c, noise_std)
    sigma2 = noise_std**2
    var_x_given_y = s_xx - s_xy**2 / s_yy
    te_ground_truth = 0.5 * np.log(1 + c**2 * var_x_given_y / sigma2)

    return x[burn_in:], y[burn_in:], float(te_ground_truth)


def compute_lag1_mi_ground_truth(a: float, b: float, c: float, noise_std: float) -> float:
    s_xx, s_xy, s_yy = _compute_stationary_covariance(a, b, c, noise_std)
    cov_xlag_ynow = b * s_xy + c * s_xx
    rho = cov_xlag_ynow / np.sqrt(s_xx * s_yy)
    return -0.5 * np.log(1 - rho**2)
