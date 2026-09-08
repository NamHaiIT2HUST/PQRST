"""Mo rong MINE sang conditional MI / transfer entropy qua phan ra hieu 2 so hang.

Dong nhat thuc (DA VERIFY den do chinh xac may, sai khac ~1.4e-16 - xem
compute_var_linear_ground_truths trong var_linear_gaussian.py):

    TE(X->Y) = I(Y[t]; X[t-1] | Y[t-1])
             = I(Y[t]; X[t-1], Y[t-1])  -  I(Y[t]; Y[t-1])
             \_______ "mi_full" _______/   \__ "mi_reduced" __/

Moi so hang la 1 MI KHONG dieu kien -> uoc luong duoc bang dung co che Donsker-Varadhan
+ shuffle-batch da co tu Pha Q, khong can loss moi.

CANH BAO PHUONG SAI (quan trong, rut ra tu review Pha Q): hieu cua 2 uoc luong nhieu
thi NHIEU HON tung uoc luong thanh phan. Neu var(mi_full_hat) ~ var(mi_reduced_hat) ~ v
va 2 uoc luong doc lap thi var(te_hat) ~ 2v. Trong Pha Q, val loss da dao dong voi
do lech chuan ~0.0055 nat - hieu 2 so hang co the day con so nay len dang ke. Vi vay
muc 5 cua docs/PHASE_R_GUIDE.md BAT BUOC do va bao cao phuong sai cua tung so hang
rieng, khong chi cua TE cuoi cung.

Meo giam phuong sai (nen lam): dung CHUNG mot mang T_phi cho ca 2 so hang (co co
che mask - xem network dung o amortized.py). Khi do sai so cua 2 so hang co tuong quan
duong, va hieu cua chung triet tieu bot nhieu chung - phuong sai cua TE thap hon so voi
dung 2 mang doc lap hoan toan.
"""

from __future__ import annotations

import numpy as np
import torch


def estimate_mi_from_window(
    model: torch.nn.Module,
    a_vals: np.ndarray,
    b_vals: np.ndarray,
    n_shuffles: int,
    seed: int,
) -> float:
    """Uoc luong I(A;B) tren 1 cua so DUY NHAT bang mang T_phi DA DONG BANG (frozen).

    Day la ham cot loi cua tinh "amortized": KHONG train gi o day, chi forward pass.
    Mang da hoc tu corpus o buoc train, gio ap dung cho cua so moi.

    Args:
        model: T_phi da train, o che do eval() va trong torch.no_grad().
        a_vals: bien A, shape (N,) hoac (N, dA).
        b_vals: bien B, shape (N,) hoac (N, dB).
        n_shuffles: SO LAN shuffle de trung binh hoa uoc luong so hang marginal.
            KHONG dung 1 (day la dung loi da phat hien o Pha Q - 1 lan shuffle cho
            uoc luong Monte Carlo phuong sai rat cao). Khuyen nghi >= 20.
        seed: seed cho cac lan shuffle, de tai lap.

    Returns:
        Uoc luong DV bound cua I(A;B) tren cua so nay (don vi: nat).

    TODO(ban tu code):
        - t_joint = model(a, b)  -> shape (N,); term_joint = t_joint.mean()
        - Lap n_shuffles lan: hoan vi b, tinh t_marginal = model(a, b_perm), roi
          log_mean_exp = logsumexp(t_marginal) - log(N). Thu thap n_shuffles gia tri.
        - term_marginal = TRUNG BINH cua cac log_mean_exp thu duoc (trung binh SAU khi
          lay log tung lan - don gian va on dinh; ghi ro lua chon nay trong bao cao).
        - return float(term_joint - term_marginal)
        - Toan bo trong torch.no_grad(), model.eval().
    """
    raise NotImplementedError


def estimate_te_from_window(
    model: torch.nn.Module,
    y_t: np.ndarray,
    x_lag: np.ndarray,
    y_lag: np.ndarray,
    n_shuffles: int,
    seed: int,
) -> dict[str, float]:
    """Uoc luong TE(X->Y) tren 1 cua so qua phan ra 2 so hang o docstring module.

    Args:
        model: T_phi da train (dang shared-with-mask, xem amortized.py).
        y_t, x_lag, y_lag: 3 mang shape (N,) cua cung 1 cua so.
        n_shuffles: xem estimate_mi_from_window.
        seed: seed cho shuffle.

    Returns:
        dict 3 khoa: "mi_full", "mi_reduced", "te" (= mi_full - mi_reduced).
        Tra ve CA 2 SO HANG chu khong chi TE - de muc 5 cua PHASE_R_GUIDE.md do duoc
        phuong sai tung so hang rieng.

    TODO(ban tu code):
        - mi_full: A = y_t, B = (x_lag, y_lag) [ghep 2 cot], mask cho biet "che do day du".
        - mi_reduced: A = y_t, B = (y_lag) [cot x_lag bi zero-out], mask "che do rut gon".
        - DUNG CUNG 1 seed shuffle cho ca 2 so hang -> sai so tuong quan duong, hieu
          triet tieu bot nhieu (xem canh bao phuong sai o docstring module).
        - te = mi_full - mi_reduced. KHONG kep (clip) ve >= 0 o day; neu muon bao cao
          gia tri kep thi lam o tang phan tich va ghi ro, vi kep se lam sai lech uoc
          luong bias mot cach he thong.
    """
    raise NotImplementedError
