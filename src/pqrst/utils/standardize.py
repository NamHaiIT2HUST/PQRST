"""Chuan hoa z-score theo TUNG CUA SO - cau noi bat buoc giua du lieu tong hop va
du lieu sinh ly that (Pha S).

VI SAO CAN (da do truc tiep, xem docs/PHASE_S_GUIDE.md muc 0):
T_phi duoc train tren du lieu tong hop co mean~0, std~0.8-1.0. Du lieu that co thang
do hoan toan khac (RR interval ~0.8+-0.1 giay; EEG band power 1-500 uV^2). Khi dua
du lieu that (thang do khac) vao mang da dong bang, output gan nhu MAT HOAN TOAN lien
he voi su that - do tuong quan voi ket qua dung chi con 0.07, trong khi bias/variance
van "dep" (vi output sup ve gan hang so). Day la kieu loi nguy hiem nhat: nhin bang
so lieu khong phat hien duoc.

CO SO LY THUYET:
TE/MI BAT BIEN duoi phep bien doi affine tung bien: TE(aX+b -> cY+d) = TE(X->Y).
Da kiem chung bang KSG: TE tren du lieu goc = 0.1697, sau khi doi thang do (x0.1+0.8)
= 0.1697 (giong het). Nghia la chuan hoa KHONG lam mat thong tin gi ve mat ly thuyet -
no chi don gian dua du lieu ve dung mien ma mang da hoc.

NGUYEN TAC SU DUNG (quan trong):
Phai ap dung CUNG MOT phep chuan hoa o CA hai noi - luc train (train_amortized) va
luc suy luan (AmortizedTEEstimator.estimate). Neu chi ap dung 1 ben se tao lech phan
phoi train/test - dung loai loi ma module nay sinh ra de khac phuc.
"""

from __future__ import annotations

import numpy as np

# Nguong duoi cho do lech chuan. Cua so gan nhu hang so (vd tin hieu bao hoa, doan
# du lieu bi mat) se co std ~ 0 -> chia cho no gay tran so. Khi do chi tru trung binh.
_STD_FLOOR = 1e-12


def standardize_window(a: np.ndarray) -> np.ndarray:
    """Z-score hoa 1 cua so: (a - mean) / std.

    Args:
        a: mang 1D (1 cua so du lieu).

    Returns:
        Mang cung shape, da chuan hoa. Neu std ~ 0 (cua so gan nhu hang so) thi chi
        tru trung binh, KHONG chia - tra ve mang toan 0 thay vi NaN/Inf.

    Luu y: chuan hoa theo TUNG cua so (khong phai theo toan bo chuoi) - vi moi cua so
    la 1 mau doc lap khi uoc luong TE, va thang do co the troi theo thoi gian trong
    du lieu sinh ly that (vd nhip tim thay doi theo giai doan giac ngu).
    """
    a = np.asarray(a, dtype=float)
    centered = a - a.mean()
    std = a.std()
    if std < _STD_FLOOR:
        return centered
    return centered / std
