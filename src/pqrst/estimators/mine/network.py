"""Mang thong ke T_phi cho MINE (Mutual Information Neural Estimation).

T_phi: R^dx x R^dy -> R, dung de uoc luong can duoi Donsker-Varadhan cho I(X;Y):
    I(X;Y) >= E_{P(X,Y)}[T_phi(x,y)] - log( E_{P(X)xP(Y)}[exp(T_phi(x,y))] )

Thiet ke input_dim CO THE CAU HINH (khong hardcode = 2) vi Pha R se can mo rong
sang conditional MI/TE (input them bien dieu kien Z, vd Y[t-1]) tai dung mang nay -
xem docs/PHASE_Q_GUIDE.md muc 1 va docs/ROADMAP.md, phan Pha R.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StatisticsNetwork(nn.Module):
    """MLP 2-3 lop an, nhan (x, y) da noi (concat) lam input, tra ve 1 scalar/sample.

    Spec de xuat (Pha Q, bai toan smoke test don gian - xem PHASE_Q_GUIDE.md muc 4):
        input_dim = dim(x) + dim(y)  (vd 1 + 1 = 2 cho smoke test lag-1 MI)
        hidden_dims = [64, 64]  (co the thu [128, 64] neu can)
        activation = ELU (khuyen nghi cua nhieu implementation MINE cong khai,
                           tranh dead neuron so voi ReLU khi gradient qua logsumexp)

    TODO(ban tu code):
        - __init__: dung nn.Sequential hoac list cac nn.Linear + activation, ket thuc
          bang 1 nn.Linear(..., 1) KHONG activation o lop cuoi (output la logit tho,
          khong bi chan).
        - forward(x, y): torch.cat([x, y], dim=-1) roi di qua MLP, tra ve tensor shape
          (batch_size,) (khong phai (batch_size, 1) - nho squeeze(-1)).
        - x, y truyen vao co the la tensor shape (batch_size, dx) va (batch_size, dy);
          neu dx=dy=1 (truong hop smoke test) thi shape (batch_size, 1).
    """

    def __init__(self, input_dim: int, hidden_dims: list[int] | None = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]
            
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ELU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Check if x or y are 1D, and unsqueeze if necessary
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if y.dim() == 1:
            y = y.unsqueeze(-1)
            
        xy = torch.cat([x, y], dim=-1)
        out = self.net(xy)
        return out.squeeze(-1)
