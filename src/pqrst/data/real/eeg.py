"""Kenh NAO: tu EEG tho -> cong suat dai tan tren luoi thoi gian deu.

CHI ap dung cho slpdb va capslpdb - Fantasia va Apnea-ECG KHONG CO EEG
(da xac minh, xem docs/PHASE_S_GUIDE.md muc 1).
"""

from __future__ import annotations

import numpy as np

# Dai tan chuan trong nghien cuu giac ngu (Hz).
EEG_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}


def compute_band_power(
    eeg: np.ndarray,
    fs: float,
    band: tuple[float, float],
    grid_fs: float,
    log_transform: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Tinh cong suat 1 dai tan tren cua so truot, tra ve tren luoi thoi gian deu.

    Args:
        eeg: tin hieu EEG tho, mang 1D.
        fs: tan so lay mau cua EEG (Hz).
        band: (f_low, f_high) - xem EEG_BANDS.
        grid_fs: tan so luoi dich - PHAI giong grid_fs dung cho kenh tim (4.0 Hz)
            de hai kenh nam tren cung mot luoi thoi gian.
        log_transform: lay log cong suat. NEN de True - phan phoi cong suat lech phai
            rat manh, log dua ve gan Gaussian hon, phu hop gia dinh cua estimator.

    Returns:
        (grid_times, band_power_on_grid).

    TODO(ban tu code):
        - Dung scipy.signal.spectrogram hoac welch tren cua so truot. Do dai cua so
          nen >= vai chu ky cua tan so thap nhat trong band (vd delta 0.5Hz -> can
          >= 4 giay) - neu cua so qua ngan, uoc luong cong suat delta se rat nhieu.
        - Buoc truot = 1/grid_fs de ra dung luoi mong muon.
        - Neu log_transform: dung np.log(power + eps) voi eps nho de tranh log(0).
        - grid_times PHAI cung goc thoi gian voi kenh tim (xem cardiac.py).
    """
    raise NotImplementedError


def detect_cardiac_artifact(
    eeg: np.ndarray,
    fs: float,
    r_peak_times: np.ndarray,
    window_ms: float = 200.0,
) -> dict:
    """*** KIEM TRA BAT BUOC TRUOC KHI TIN BAT KY KET QUA SANITY CHECK NAO ***

    Phat hien nhiem nhieu truong tim (cardiac field artifact) trong EEG.

    VI SAO SONG CON (xem docs/PHASE_S_GUIDE.md muc 3.2): tin hieu dien tim rat manh va
    ro ri vao dien cuc EEG. Neu EEG bi nhiem, TE(tim->nao) se cao MOT CACH GIA TAO ->
    sanity check DAT nhung dat vi ly do hoan toan sai (ta chi dang do tin hieu tim ro
    vao chinh no, khong phai ghep noi sinh ly tim-nao). Ket qua "thanh cong" kieu nay
    con te hon that bai vi no se di thang vao bai bao.

    Args:
        eeg: tin hieu EEG tho.
        fs: tan so lay mau EEG.
        r_peak_times: thoi diem (giay) cac dinh R lay tu kenh ECG.
        window_ms: do rong cua so quanh moi dinh R de lay trung binh.

    Returns:
        dict toi thieu gom:
            "locked_average": mang trung binh EEG khoa theo dinh R,
            "peak_amplitude": bien do dinh-dinh cua locked_average,
            "baseline_amplitude": bien do tuong ung khi khoa theo thoi diem NGAU NHIEN
                (doi chung - neu khong co nhiem thi 2 gia tri nay phai tuong duong),
            "artifact_ratio": peak_amplitude / baseline_amplitude,
            "suspected": bool, True neu artifact_ratio vuot nguong.

    TODO(ban tu code):
        - Cat doan EEG quanh moi dinh R (+-window_ms/2), xep chong, lay trung binh.
          Neu EEG SACH, nhieu ngau nhien se triet tieu -> duong gan phang.
          Neu BI NHIEM, dang song QRS se hien ro.
        - Doi chung ngau nhien la BAT BUOC: khoa theo thoi diem ngau nhien (cung so
          luong) de biet muc bien do "nen". Khong co doi chung nay thi khong dien giai
          duoc con so tuyet doi.
        - Nguong de xuat: artifact_ratio > 2.0 -> nghi nhiem. Ghi ro nguong da chon.
        - Ve hinh locked_average de nguoi doc NHIN duoc - day la kiem tra ma mat nguoi
          nhin ra nhanh hon bat ky con so nao.
    """
    raise NotImplementedError
