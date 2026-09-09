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
    # convert to tensor
    a = torch.tensor(a_vals, dtype=torch.float32)
    b = torch.tensor(b_vals, dtype=torch.float32)
    
    if a.dim() == 1: a = a.unsqueeze(-1)
    if b.dim() == 1: b = b.unsqueeze(-1)
    
    N = a.shape[0]
    
    with torch.no_grad():
        t_joint = model(a, b)
        term_joint = t_joint.mean().item()
        
        log_mean_exps = []
        rng = torch.Generator().manual_seed(seed)
        
        for _ in range(n_shuffles):
            perm = torch.randperm(N, generator=rng)
            b_perm = b[perm]
            t_marginal = model(a, b_perm)
            lme = torch.logsumexp(t_marginal, dim=0) - np.log(N)
            log_mean_exps.append(lme.item())
            
        term_marginal = np.mean(log_mean_exps)
        
    return float(term_joint - term_marginal)

def estimate_te_from_window(
    model: torch.nn.Module,
    y_t: np.ndarray,
    x_lag: np.ndarray,
    y_lag: np.ndarray,
    n_shuffles: int,
    seed: int,
) -> dict[str, float]:
    model.eval()
    
    ty_t = torch.tensor(y_t, dtype=torch.float32).unsqueeze(-1)
    tx_lag = torch.tensor(x_lag, dtype=torch.float32).unsqueeze(-1)
    ty_lag = torch.tensor(y_lag, dtype=torch.float32).unsqueeze(-1)
    
    N = ty_t.shape[0]
    mask_full = torch.ones((N, 1), dtype=torch.float32)
    mask_reduced = torch.zeros((N, 1), dtype=torch.float32)
    
    with torch.no_grad():
        # Full mode
        t_joint_full = model(ty_t, tx_lag, ty_lag, mask_full)
        term_joint_full = t_joint_full.mean().item()
        
        # Reduced mode
        t_joint_red = model(ty_t, tx_lag, ty_lag, mask_reduced)
        term_joint_red = t_joint_red.mean().item()
        
        log_mean_exps_full = []
        log_mean_exps_red = []
        
        rng = torch.Generator().manual_seed(seed)
        for _ in range(n_shuffles):
            perm = torch.randperm(N, generator=rng)
            
            # marginal full
            tx_lag_perm = tx_lag[perm]
            ty_lag_perm = ty_lag[perm]
            
            t_marg_full = model(ty_t, tx_lag_perm, ty_lag_perm, mask_full)
            log_mean_exps_full.append((torch.logsumexp(t_marg_full, dim=0) - np.log(N)).item())
            
            # marginal reduced
            t_marg_red = model(ty_t, tx_lag_perm, ty_lag_perm, mask_reduced)
            log_mean_exps_red.append((torch.logsumexp(t_marg_red, dim=0) - np.log(N)).item())
            
        term_marg_full = np.mean(log_mean_exps_full)
        term_marg_red = np.mean(log_mean_exps_red)
        
    mi_full = term_joint_full - term_marg_full
    mi_reduced = term_joint_red - term_marg_red
    te = mi_full - mi_reduced
    
    return {
        "mi_full": float(mi_full),
        "mi_reduced": float(mi_reduced),
        "te": float(te)
    }
