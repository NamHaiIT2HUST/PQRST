"""Donsker-Varadhan bound + co che shuffle-batch chuan MINE (Belghazi et al., 2018).

DV(T) = E_{P(X,Y)}[T(x,y)] - log( E_{P(X)xP(Y)}[exp(T(x,y))] )

Trong 1 batch (x, y) lay tu P(X,Y) (mau "joint"), mau tu P(X) x P(Y) (mau "marginal")
duoc xap xi bang cach GIU NGUYEN x, HOAN VI y trong batch (shuffle theo batch dim) -
day la ky thuat chuan cua MINE, khong can ve lai tu phan phoi marginal that.

Luu y on dinh so hoc: KHONG tinh log(mean(exp(T))) truc tiep (de tran so/underflow).
Dung torch.logsumexp(T, dim=0) - log(batch_size) thay the (tuong duong ve toan hoc,
on dinh hon nhieu).
"""

from __future__ import annotations

import torch


def shuffle_batch(y: torch.Tensor) -> torch.Tensor:
    """Hoan vi y theo batch dimension (dim=0) de tao mau xap xi tu P(Y) doc lap voi x.

    TODO(ban tu code):
        - Dung torch.randperm(y.shape[0], device=y.device) de lay hoan vi ngau nhien,
          roi y[perm].
        - KHONG dung numpy random / random global state - de nhat quan voi torch RNG
          (va de set torch.manual_seed dieu khien duoc ca qua trinh).
    """
    perm = torch.randperm(y.shape[0], device=y.device)
    return y[perm]


def donsker_varadhan_loss(t_joint: torch.Tensor, t_marginal: torch.Tensor) -> torch.Tensor:
    """Tinh loss can toi thieu hoa = -DV(T) (vi ta muon MAXIMIZE can duoi DV, ma
    optimizer chuan la minimize).

    Args:
        t_joint: T_phi(x, y) tren cap (x,y) THAT (tu P(X,Y)), shape (batch_size,).
        t_marginal: T_phi(x, y_shuffled) tren cap sau khi shuffle_batch(y),
            shape (batch_size,).

    Returns:
        Tensor scalar (0-dim) = -DV(T), dung lam loss cho optimizer.backward().

    TODO(ban tu code):
        - dv_bound = t_joint.mean() - (torch.logsumexp(t_marginal, dim=0) -
          torch.log(torch.tensor(t_marginal.shape[0], dtype=t_marginal.dtype)))
        - return -dv_bound
    """
    dv_bound = t_joint.mean() - (torch.logsumexp(t_marginal, dim=0) -
                                 torch.log(torch.tensor(t_marginal.shape[0], dtype=t_marginal.dtype, device=t_marginal.device)))
    return -dv_bound
