"""Kenh TIM: tu annotation nhip -> chuoi RR -> luoi thoi gian deu.

Xem docs/PHASE_S_GUIDE.md muc 2.1 va 2.3.
"""

from __future__ import annotations

import numpy as np


def rr_from_beat_annotations(
    beat_times: np.ndarray,
    exclude_ectopic: bool = True,
    local_window_beats: int = 11,
    threshold_ratio: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Tinh chuoi RR tu thoi diem cac nhip, loc ectopic bang trung vi CUC BO.

    Args:
        beat_times: thoi diem (giay) cua tung nhip.
        exclude_ectopic: co loc nhip bat thuong hay khong.
        local_window_beats: so nhip dung de tinh trung vi cuc bo (nen la so le).
            SUA (review): ban truoc dung trung vi TOAN CUC cho ca ban ghi - sai voi
            ban ghi dai nhieu gio vi nhip tim troi theo giai doan giac ngu (vd nhanh
            hon o REM), lam trung vi toan cuc khong dai dien cho tung doan. Trung vi
            CUC BO (cua so truot vai chuc nhip quanh moi diem) thich nghi duoc voi
            xu huong troi cham, van bat duoc bat thuong dot ngot (ectopic).
        threshold_ratio: lech tuong doi so voi trung vi cuc bo de coi la ectopic.

    Returns:
        (t_rr, rr): thoi diem va do dai (giay) cua tung khoang RR con lai.
    """
    if len(beat_times) < 2:
        return np.array([]), np.array([])
    rr = np.diff(beat_times)
    t_rr = beat_times[1:]

    if exclude_ectopic and len(rr) > 0:
        local_median = _rolling_median(rr, local_window_beats)
        valid = (rr > (1 - threshold_ratio) * local_median) & (
            rr < (1 + threshold_ratio) * local_median
        )
        rr = rr[valid]
        t_rr = t_rr[valid]

    return t_rr, rr


def _rolling_median(a: np.ndarray, window: int) -> np.ndarray:
    """Trung vi cuc bo (cua so truot, can giua tai moi diem). O bien (dau/cuoi mang),
    cua so tu dong ngan lai thay vi doc ra ngoai mang."""
    n = len(a)
    half = window // 2
    out = np.empty(n)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = np.median(a[lo:hi])
    return out


def detect_r_peaks(
    ecg: np.ndarray,
    fs: float,
    bandpass: tuple[float, float] = (5.0, 15.0),
    min_rr_seconds: float = 0.3,
    threshold_factor: float = 6.0,
) -> np.ndarray:
    """Do QRS tho tu tin hieu ECG (Pan-Tompkins-lite: bandpass -> dao ham -> binh
    phuong -> trung binh truot -> tim dinh). CHI dung khi KHONG co annotation nhip
    co san (vd capslpdb - EDF, khong co file .ecg nhu slpdb).

    Da validate tren capslpdb/n1 (nguoi khoe manh, 9.6h): 42027 nhip, HR trung vi
    72.6 bpm, do lech chuan RR 0.084s, khoang trong lon nhat 1.22s - hop ly ve sinh
    ly, khong co dau hieu mat nhip hang loat.

    threshold_factor=6.0 (nguong = 6x trung vi tin hieu nang luong sau loc): chon
    thuc nghiem - dung max() lam moc de dat nguong (vd max*0.15) THAT BAI ro tren du
    lieu nay vi 1 vai dinh nang luong do chuyen dong/nhieu lam max qua lon, khien
    nguong qua cao va bo sot hang loat nhip binh thuong (quan sat truc tiep: co doan
    trong RR toi 77s). Trung vi ON DINH hon nhieu truoc outlier.
    """
    from scipy.signal import butter, filtfilt, find_peaks

    b, a = butter(2, [bandpass[0] / (fs / 2), bandpass[1] / (fs / 2)], btype="band")
    filt = filtfilt(b, a, ecg)
    deriv = np.diff(filt, prepend=filt[0])
    energy = deriv ** 2
    win = int(0.15 * fs)
    smoothed = np.convolve(energy, np.ones(win) / win, mode="same")
    min_dist = int(min_rr_seconds * fs)
    threshold = np.median(smoothed) * threshold_factor
    peaks, _ = find_peaks(smoothed, distance=min_dist, height=threshold)
    return peaks / fs


def interpolate_rr(
    t_rr: np.ndarray,
    rr: np.ndarray,
    grid_fs: float = 4.0,
    t_start: float | None = None,
    t_end: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Noi suy tuyen tinh chuoi RR (khong deu) ve luoi thoi gian DEU.

    t_start/t_end cho phep ep ve 1 khoang thoi gian CHUNG voi kenh khac (xem
    sync.align_to_common_grid) - KHONG ngoai suy ngoai [t_rr[0], t_rr[-1]] thuc te,
    np.interp tu dong giu gia tri bien ngoai khoang do nen can cat bo o noi goi.
    """
    if len(t_rr) < 2:
        return np.array([]), np.array([])
    if t_start is None:
        t_start = t_rr[0]
    if t_end is None:
        t_end = t_rr[-1]

    t_grid = np.arange(t_start, t_end, 1.0 / grid_fs)
    rr_grid = np.interp(t_grid, t_rr, rr)
    return t_grid, rr_grid
