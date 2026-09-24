"""Pha P'.0 (truoc Nhip 2) - mo hinh Fourier features co dien, dung de tach bach
"loi ich tu DANG HAM Fourier" khoi "loi ich tu BAN CHAT LUONG TU" (entanglement,
khong gian Hilbert lon) TRUOC KHI dau tu vao PennyLane/mach luong tu thuc.

Co so ly thuyet (Schuld, Sweke & Meyer, Physical Review A 103, 032430, 2021):
mot mach luong tu kieu data re-uploading tuong duong ve mat toan hoc voi 1 chuoi
Fourier RIENG PHAN cua du lieu dau vao - tap tan so truy cap duoc quyet dinh boi
cach lap lai cong encode. Model duoi day mo phong DUNG cau truc do bang cach co
dien (khong dung mach luong tu nao): du lieu duoc CHIEU qua 1 ma tran co dinh
(Random Fourier Features nhieu chieu, Rahimi & Recht 2007), roi lay cac hoa am
(harmonics) nguyen cua tich vo huong do, roi doc ra bang 1 lop tuyen tinh - giu
so tham so RAT nho (~vai chuc), cung bac do lon voi 1 mach luong tu nho du kien
(6-8 qubit x 3-5 lop).

LOI DA TIM VA SUA (quan trong, ghi lai de khong tai pham o Nhip 2): ban dau
(SUA lan 1) moi chieu dau vao (y_t, x_lag, y_lag, mask) duoc encode RIENG BIET
roi moi ghep tuyen tinh o lop doc ra - khien T(y_t,x_lag,y_lag,mask) chi co the
la ham TACH ROI CONG TINH f(y_t)+g(x_lag)+h(y_lag)+k(mask), KHONG co so hang
tuong tac X-Y nao. Co the CHUNG MINH BANG TOAN (bat dang thuc Jensen ap tren
phan phoi marginal cua chinh chan duoi Donsker-Varadhan): voi T tach roi cong
tinh, gia tri toi uu TUYET DOI cua DV bound LA DUNG 0, bat ke train bao lau/
bang cach nao (khong phai bay toi uu hoa hay bat on dinh gradient - da thu ca
EMA-correction va tang windows_per_batch, van sup do dung 0, vi day la diem toi
uu THAT cua chinh lop ham bi han che). SUA lan 2 (ban hien tai): chieu ca 4
chieu dau vao CUNG LUC qua 1 huong ngau nhien truoc khi lay hoa am, tao so hang
tuong tac X-Y thuc su - xem docs/NHIP2_GUIDE.md muc 2.1 de biet chi tiet qua
trinh phat hien.

Neu model nay (da sua) CUNG cai thien duoc o N nho / domain gap nhu T_phi (MLP
lon) thi tin hieu tot co the den tu chinh dang ham Fourier, khong nhat thiet
can luong tu thuc. Neu KHONG cai thien gi, cau chuyen can luong tu thuc
(entanglement) manh hon.

Cung interface forward(y_t, x_lag, y_lag, mask) nhu MaskedStatisticsNetwork
(amortized.py) - nen cam duoc thang vao estimate_te_from_window() va
AmortizedTEEstimator ma khong can sua gi them.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FourierFeatureStatisticsNetwork(nn.Module):
    """T_phi dang Fourier features co dien - xem module docstring o tren (DAC
    BIET chu y phan "LOI DA TIM VA SUA" truoc khi doc code duoi day).

    forward nhan 4 dau vao (y_t, x_lag*mask, y_lag, mask), moi cai shape
    (batch, 1), giong DUNG MaskedStatisticsNetwork.forward.
    """

    def __init__(self, n_harmonics: int = 4, n_directions: int = 5, encode_seed: int = 0):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.n_directions = n_directions
        # CO DINH (khong train) - dung DUNG cong thuc Random Fourier Features
        # nhieu chieu: moi "huong" k la 1 to hop tuyen tinh CUA CA 4 CHIEU DAU
        # VAO (z_k = w_k . x + b_k), KHONG encode rieng tung chieu (day chinh la
        # cho da sua - xem "LOI DA TIM VA SUA" o dau file). Nho vay sin/cos(l*z_k)
        # co so hang tuong tac giua y_t, x_lag, y_lag - dieu KIEN BAT BUOC de DV
        # bound co the vuot qua 0.
        rng = torch.Generator().manual_seed(encode_seed)
        encode_w = torch.randn(n_directions, 4, generator=rng) * 1.5  # (n_directions, 4)
        encode_b = torch.rand(n_directions, generator=rng) * 2 * torch.pi  # (n_directions,)
        self.register_buffer("encode_w", encode_w)
        self.register_buffer("encode_b", encode_b)
        # Doc ra: to hop tuyen tinh cua sin/cos tai cac hoa am 1..n_harmonics,
        # cho moi huong chieu -> chuoi Fourier rieng phan cua TICH VO HUONG du
        # lieu (khong phai cua tung chieu rieng le). DAY la phan DUY NHAT co
        # tham so hoc duoc.
        self.readout = nn.Linear(n_directions * 2 * n_harmonics, 1)
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

        z = xy @ self.encode_w.T + self.encode_b  # (batch, n_directions) - TRON ca 4 chieu
        angles = z.unsqueeze(-1) * self.harmonics.view(1, 1, -1)  # (batch, n_directions, L)
        feats = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (batch, n_directions, 2L)
        feats = feats.reshape(feats.shape[0], -1)  # (batch, n_directions*2*L)

        out = self.readout(feats)
        return out.squeeze(-1)

    def forward_blocks(
        self,
        y_t: torch.Tensor,
        x_lag: torch.Tensor,
        y_lag: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Nhu forward(), tra ve activation theo khoi - dung interface voi
        MaskedStatisticsNetwork.forward_blocks() de cung dung duoc voi
        src/pqrst/evaluation/feature_space.py (Pha T.3b/Nhip 2 muc 2.2). Model nay
        chi co 1 khoi an (dac trung Fourier) truoc lop doc ra, nen dict chi co 2
        khoa: 'block0_input' (4-dim goc) va 'block1' (dac trung Fourier)."""
        x_lag_masked = x_lag * mask
        xy = torch.cat([y_t, x_lag_masked, y_lag, mask], dim=-1)
        z = xy @ self.encode_w.T + self.encode_b
        angles = z.unsqueeze(-1) * self.harmonics.view(1, 1, -1)
        feats = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        feats = feats.reshape(feats.shape[0], -1)
        return {"block0_input": xy, "block1": feats}
