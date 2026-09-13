"""Kenh HO HAP (Fantasia/Apnea-ECG): tin hieu RESP -> luoi thoi gian deu.

Khac han kenh EEG (Pha S, cap tim-nao): ho hap la tin hieu CHAM, dao dong gan tuan
hoan (~0.1-0.5Hz, 6-30 lan/phut) - khong can bien doi pho/cong suat dai tan nhu EEG,
loc bang thong roi noi suy ve luoi chung la du. Cung khong co van de nhiem dien tim
nhu EEG (RESP do bang cam bien co/tro khang long nguc, khong nhay voi dien truong
tim) - xem docs/PHASE_S_REVIEW.md ve quyet dinh dung Fantasia/Apnea-ECG (tim-ho hap)
lam ket qua chinh tren du lieu that, thay vi slpdb/capslpdb (tim-nao, nhiem nhieu
khong khu duoc bang 5 phuong phap da thu).
"""

from __future__ import annotations

import numpy as np


def bandpass_filter(x: np.ndarray, fs: float, band: tuple[float, float]) -> np.ndarray:
    """Loc bang thong bac 2 (Butterworth, filtfilt - 2 chieu nen khong lech pha).

    Dung chung cho CA RESP va RR: xem docs/PHASE_S_REVIEW.md ve phat hien tren
    Fantasia - loc RESP nhung KHONG loc RR cung dai tan tao bat doi xung "tri nho"
    tu tuong quan (RR ~11s, RESP ~0.65s do RR con giu nguyen thanh phan trend cham/
    LF-VLF cua HRV ma RESP da bi loc bo) - lam TE lech huong SAI so voi RSA (do
    thong ke, khong phai sinh ly). Loc CA HAI kenh cung 1 dai tan (0.1-0.5Hz) truoc
    khi tinh TE moi cong bang, va xac nhan thuc nghiem: sau khi loc RR cung dai,
    huong TE(resp->tim) > TE(tim->resp) xuat hien tro lai o 19/23 (83%) ban ghi,
    thay vi 5/23 (22%) khi chua loc RR.
    """
    from scipy.signal import butter, filtfilt

    nyq = fs / 2
    b, a = butter(2, [band[0] / nyq, band[1] / nyq], btype="band")
    return filtfilt(b, a, x)


def preprocess_respiration(
    resp: np.ndarray,
    fs: float,
    grid_fs: float = 4.0,
    bandpass: tuple[float, float] = (0.1, 0.5),
) -> tuple[np.ndarray, np.ndarray]:
    """Loc bang thong (mac dinh 0.1-0.5Hz = 6-30 lan tho/phut, trum kin dai ho hap
    binh thuong) de bo trend cham (drift cam bien) va nhieu tan so cao, roi ha mau
    ve grid_fs bang noi suy tuyen tinh.

    Returns:
        (t_grid, resp_grid): thoi diem (giay) va gia tri da loc+ha mau, cung quy uoc
        voi interpolate_rr/compute_band_power (dung truc tiep voi align_to_common_grid).

    SUA (review - chay thuc te tren 40 ban ghi Fantasia): 1 vai ban ghi (vd f2o06) co
    NaN rai rac trong tin hieu RESP goc (dut cam bien - binh thuong trong du lieu sinh
    ly THAT). `filtfilt` la loc toan tin hieu (IIR 2 chieu) - 1 mau NaN DUY NHAT lam
    NaN LAN RA TOAN BO ket qua loc (verify thuc te: 61/1.75M mau NaN dau vao ->
    28083/28083 mau NaN dau ra, tuc 100%). Gio noi suy tuyen tinh qua cac khoang NaN
    NGAN TRUOC khi loc - gia dinh dut cam bien la khoang ngan, khong phai mat du lieu
    toan bo (neu toan bo tin hieu la NaN, tra ve rong thay vi noi suy bia dat).
    """
    if len(resp) < 10:
        return np.array([]), np.array([])

    resp = np.asarray(resp, dtype=float)
    nan_mask = np.isnan(resp)
    if nan_mask.all():
        return np.array([]), np.array([])
    if nan_mask.any():
        resp = resp.copy()
        idx = np.arange(len(resp))
        resp[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], resp[~nan_mask])

    filtered = bandpass_filter(resp, fs, bandpass)

    t = np.arange(len(resp)) / fs
    t_grid = np.arange(t[0], t[-1], 1.0 / grid_fs)
    resp_grid = np.interp(t_grid, t, filtered)
    return t_grid, resp_grid
