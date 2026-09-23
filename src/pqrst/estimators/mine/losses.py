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


def donsker_varadhan_loss_ema(
    t_joint: torch.Tensor,
    t_marginal: torch.Tensor,
    ma_et: torch.Tensor | None,
    momentum: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bien the ON DINH GRADIENT cua DV bound (MINE, Belghazi et al. 2018, muc 3.2 -
    "unbiasing the gradient"). Them vao khi review Pha P'.0: mo hinh Fourier features
    nho (33 tham so, chi 1 lop doc ra tuyen tinh) SUP DO ve nghiem tam thuong T=const
    ngay epoch dau (val loss ~0 suot qua trinh, thu ca tang learning rate 50 lan van
    sup do) khi train bang donsker_varadhan_loss() thuong - xem docs/NHIP2_GUIDE.md
    muc 2.1. Nguyen nhan nghi ngo: dao ham cua log(mean(exp(T_marginal))) qua 1
    minibatch nho co phuong sai cao (mau so la 1 uoc luong Monte Carlo nhieu cua chinh
    no), day gradient ve huong T=const som truoc khi mang kip hoc tin hieu thuc.

    Ky thuat (dung nguyen ban goc MINE, KHONG phai phat minh moi): giu 1 EMA
    (exponential moving average, KHONG lan truyen gradient qua no) cua
    mean(exp(T_marginal)) qua nhieu buoc train, dung EMA nay lam MAU SO CO DINH khi
    tinh dao ham (thay vi dung dung gia tri batch hien tai, nhieu hon nhieu) -
    tuong duong uoc luong lai (1/mean_et) bang (1/ema) trong cong thuc dao ham cua
    log(mean_et).

    Gia tri DV bound tra ve (de theo doi/bao cao/eval) VAN dung log(mean(exp(.)))
    THAT, khong xap xi - CHI phan gradient (loss_for_backward) duoc hieu chinh.

    Args:
        t_joint, t_marginal: xem donsker_varadhan_loss.
        ma_et: EMA hien tai cua mean(exp(t_marginal)) (tensor 0-dim, hoac None o
            buoc dau tien - se khoi tao bang chinh gia tri batch dau, KHONG can
            "warm-up" rieng).
        momentum: he so EMA, gia tri moi = (1-momentum)*ma_et_cu + momentum*batch_moi.
            0.01 (mac dinh cua MINE goc) = trung binh tren ~100 buoc gan nhat.

    Returns:
        (loss_for_backward, dv_bound_estimate.detach(), ma_et_moi.detach()) - goi
        lai ham nay o buoc sau PHAI truyen dung ma_et_moi tra ve (khong tao lai None).
    """
    et_marginal = torch.exp(t_marginal)
    mean_et = et_marginal.mean()

    if ma_et is None:
        ma_et_new = mean_et.detach()
    else:
        ma_et_new = (1 - momentum) * ma_et + momentum * mean_et.detach()

    dv_bound = t_joint.mean() - torch.log(mean_et + 1e-8)
    loss_for_backward = -(t_joint.mean() - mean_et / (ma_et_new + 1e-8))
    return loss_for_backward, dv_bound.detach(), ma_et_new
