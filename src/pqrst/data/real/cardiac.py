"""Kenh TIM: tu ECG (hoac annotation nhip) -> chuoi RR -> luoi thoi gian deu.

Chuoi RR (khoang cach giua 2 nhip lien tiep) la dac trung tim chuan cho nghien cuu
ghep noi tim-nao. Van de ky thuat cot loi: RR duoc lay mau theo NHIP TIM (khong deu
theo thoi gian), trong khi EEG band power lay mau DEU theo thoi gian -> phai dua ve
cung 1 luoi thoi gian truoc khi tinh TE. Xem docs/PHASE_S_GUIDE.md muc 2.1 va 2.3.
"""

from __future__ import annotations

import numpy as np


def rr_from_beat_annotations(
    beat_sample_indices: np.ndarray,
    beat_symbols: np.ndarray,
    fs: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Tinh chuoi RR tu annotation nhip da co san (KHONG can chay QRS detector).

    slpdb co san file annotation nhip (.ecg) - dung luon, bo duoc mot nguon loi lon.

    Args:
        beat_sample_indices: vi tri (chi so mau) cua tung nhip, tu wfdb annotation.
        beat_symbols: ky hieu loai nhip tuong ung ('N' = normal, 'V' = ventricular
            ectopic, 'A' = atrial premature, ...).
        fs: tan so lay mau cua ECG (Hz).

    Returns:
        (rr_times, rr_values):
            rr_times: thoi diem (giay) cua tung khoang RR - lay thoi diem nhip SAU
                cua moi cap (quy uoc phai nhat quan voi phan con lai cua pipeline).
            rr_values: do dai khoang RR (giay).

    TODO(ban tu code):
        - CHI giu nhip loai 'N' (normal) truoc khi tinh hieu. Nhip ngoai tam thu tao
          1 khoang RR ngan bat thuong + 1 khoang dai bu tru -> neu khong loc se tao
          tuong quan gia RAT manh, lam hong toan bo ket qua TE.
        - rr_values = np.diff(sample_indices_cua_nhip_N) / fs
        - Loai RR phi sinh ly: ngoai khoang [0.3, 2.0] giay (tuong ung 30-200 bpm).
    """
    raise NotImplementedError


def remove_ectopic_rr(
    rr_values: np.ndarray,
    rr_times: np.ndarray,
    threshold_ratio: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Loai cac khoang RR bat thuong con sot lai theo quy tac lech so voi trung vi cuc bo.

    Args:
        rr_values, rr_times: dau ra cua rr_from_beat_annotations.
        threshold_ratio: nguong lech tuong doi so voi trung vi cuc bo (0.2 = 20%).

    Returns:
        (rr_times_sach, rr_values_sach).

    TODO(ban tu code):
        - Tinh trung vi truot (vd cua so 5 nhip) cua rr_values.
        - Loai diem co |rr - trung_vi_cuc_bo| / trung_vi_cuc_bo > threshold_ratio.
        - BAO CAO ty le diem bi loai - neu qua cao (vd >20%) thi ban ghi do co the
          qua nhieu artifact, nen loai ca ban ghi thay vi vá tung diem.
    """
    raise NotImplementedError


def resample_rr_to_grid(
    rr_times: np.ndarray,
    rr_values: np.ndarray,
    grid_fs: float,
    t_start: float,
    t_end: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Noi suy chuoi RR (khong deu) ve luoi thoi gian DEU.

    Day la buoc dong bo hoa then chot - xem canh bao o docs/PHASE_S_GUIDE.md muc 2.3.

    Args:
        rr_times, rr_values: chuoi RR da lam sach.
        grid_fs: tan so luoi dich (khuyen nghi 4.0 Hz).
        t_start, t_end: khoang thoi gian (giay) can noi suy, PHAI dung chung mot goc
            thoi gian voi kenh EEG - neu lech goc, TE tinh ra se vo nghia.

    Returns:
        (grid_times, rr_on_grid).

    TODO(ban tu code):
        - grid_times = np.arange(t_start, t_end, 1/grid_fs)
        - Noi suy cubic hoac linear (scipy.interpolate.interp1d). Ghi ro lua chon.
        - KHONG ngoai suy (extrapolate) ngoai khoang co du lieu that - cat bo thay vi
          bia gia tri.
    """
    raise NotImplementedError
