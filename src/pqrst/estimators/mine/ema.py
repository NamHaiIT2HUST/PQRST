"""Exponential moving average (EMA) bias correction cho gradient cua so hang
log-mean-exp trong Donsker-Varadhan loss - ky thuat tuy chon tu paper MINE goc
(Belghazi et al., 2018, muc 3.2) de giam bias khi uoc luong gradient bang minibatch
nho (gradient cua log(E[exp(T)]) qua minibatch la bias, EMA giup giam bias nay).

KHONG bat buoc cho tieu chi thoat Pha Q (xem PHASE_Q_GUIDE.md muc 7) - chi lam neu
loss/MI estimate dao dong manh khi khong co EMA. Neu bo qua, xoa import file nay o
train.py va ghi ro trong bao cao la khong dung.
"""

from __future__ import annotations

import torch


class EMALogMeanExp:
    """Theo doi trung binh dong (EMA) cua mean(exp(T_marginal)) qua cac step, dung de
    thay the gradient cua logsumexp minibatch bang gradient da hieu chinh bias.

    TODO(ban tu code, chi lam neu can):
        - __init__(self, decay: float = 0.99): luu decay, self.ema_value = None.
        - update(self, batch_mean_exp: torch.Tensor) -> torch.Tensor: cap nhat
          self.ema_value = decay*self.ema_value + (1-decay)*batch_mean_exp.detach()
          (khoi tao ema_value = batch_mean_exp.detach() o lan goi dau), tra ve
          self.ema_value de dung thay logsumexp trong bien the loss co EMA.
        - Tham khao cong thuc chinh xac trong paper MINE (Belghazi et al. 2018,
          Algorithm 1) truoc khi code, dung chi dua vao mo ta ngan o day.
    """

    def __init__(self, decay: float = 0.99):
        raise NotImplementedError

    def update(self, batch_mean_exp: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
