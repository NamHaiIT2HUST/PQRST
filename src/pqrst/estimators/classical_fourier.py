"""Pha P'.0 (truoc Nhip 2) - mo hinh Fourier features co dien, dung de tach bach
"loi ich tu DANG HAM Fourier" khoi "loi ich tu BAN CHAT LUONG TU" (entanglement,
khong gian Hilbert lon) TRUOC KHI dau tu vao PennyLane/mach luong tu thuc.

Co so ly thuyet (Schuld, Sweke & Meyer, Physical Review A 103, 032430, 2021):
mot mach luong tu kieu data re-uploading tuong duong ve mat toan hoc voi 1 chuoi
Fourier RIENG PHAN cua du lieu dau vao - tap tan so truy cap duoc quyet dinh boi
cach lap lai cong encode. Model duoi day mo phong DUNG cau truc do bang cach co
dien (khong dung mach luong tu nao): 1 phep encode tuyen tinh hoc duoc, roi lay
cac hoa am (harmonics) nguyen cua no, roi doc ra bang 1 lop tuyen tinh - giu so
tham so RAT nho (~vai chuc), cung bac do lon voi 1 mach luong tu nho du kien
(6-8 qubit x 3-5 lop).

Neu model nay CUNG cai thien duoc o N nho / domain gap nhu T_phi (MLP lon) thi
tin hieu tot co the den tu chinh dang ham Fourier, khong nhat thiet can luong
tu thuc - can noi ro dieu nay khi bien minh cho Nhip 2 (xem docs/NHIP2_GUIDE.md
muc 1). Neu KHONG cai thien gi, cau chuyen can luong tu thuc (entanglement)
manh hon.

Cung interface forward(y_t, x_lag, y_lag, mask) nhu MaskedStatisticsNetwork
(amortized.py) - nen cam duoc thang vao estimate_te_from_window() va
AmortizedTEEstimator ma khong can sua gi them.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FourierFeatureStatisticsNetwork(nn.Module):
    """T_phi dang Fourier features co dien - xem module docstring o tren.

    forward nhan 4 dau vao (y_t, x_lag*mask, y_lag, mask), moi cai shape
    (batch, 1), giong DUNG MaskedStatisticsNetwork.forward.
    """

    def __init__(self, n_harmonics: int = 4, encode_seed: int = 0):
        super().__init__()
        self.n_harmonics = n_harmonics
        # SUA (phat hien qua chay thu): ban dau de encode_w/encode_b la
        # nn.Parameter HOC DUOC bi SUP DO ve nghiem tam thuong T=const cua DV
        # bound (T const -> DV bound = 0 CHINH XAC voi MOI tham so - day la 1
        # saddle point luon ton tai cua chinh ham loss, khong phai bug rieng o
        # day). Model hoc duoc gan nhu ngay epoch 1 roi ket dinh o do (val loss
        # ~0.0000 suot qua trinh, xem results/figures/phase_p2_fourier_linear_loss.png).
        #
        # SUA: CO DINH encode_w/encode_b (khong train) voi gia tri da da dang tu
        # dau - dung DUNG cong thuc Random Fourier Features (Rahimi & Recht 2007):
        # encoding co dinh + CHI train lop doc ra tuyen tinh. Dam bao dac trung
        # sin/cos da khac nhau ro giua cac cua so tu epoch 0, khong con duong nao
        # de sup do ve T=const (vi muon T=const, readout phai HOC ve 0 CHINH XAC -
        # kem hap dan hon nhieu so voi truong hop encode cung hoc duoc ve 0).
        rng = torch.Generator().manual_seed(encode_seed)
        encode_w = torch.randn(4, generator=rng) * 1.5 + torch.sign(torch.randn(4, generator=rng))
        encode_b = torch.rand(4, generator=rng) * 2 * torch.pi
        self.register_buffer("encode_w", encode_w)
        self.register_buffer("encode_b", encode_b)
        # Doc ra: to hop tuyen tinh cua sin/cos tai cac hoa am 1..n_harmonics,
        # cho ca 4 chieu -> dung la 1 chuoi Fourier rieng phan cua du lieu.
        # DAY la phan DUY NHAT co tham so hoc duoc.
        self.readout = nn.Linear(4 * 2 * n_harmonics, 1)
        harmonics = torch.arange(1, n_harmonics + 1, dtype=torch.float32)
        self.register_buffer("harmonics", harmonics)

    def forward(
        self,
        y_t: torch.Tensor,
        x_lag: torch.Tensor,
        y_lag: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        x_lag_masked = x_lag * mask
        xy = torch.cat([y_t, x_lag_masked, y_lag, mask], dim=-1)  # (batch, 4)

        z = xy * self.encode_w + self.encode_b  # (batch, 4)
        angles = z.unsqueeze(-1) * self.harmonics.view(1, 1, -1)  # (batch,4,L)
        feats = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (batch,4,2L)
        feats = feats.reshape(feats.shape[0], -1)  # (batch, 4*2*L)

        out = self.readout(feats)
        return out.squeeze(-1)
