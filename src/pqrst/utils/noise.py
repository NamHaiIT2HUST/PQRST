"""Module tao va them nhieu Gauss (Additive White Gaussian Noise - AWGN) vao tin hieu.

Dung cho cac bai kiem thu do ben (stress-test / robustness) cua cac bo uoc luong TE
(KSG vs Amortized MINE vs VQC/Hybrid) duoi cac muc SNR khac nhau (dB).
"""

from __future__ import annotations

import numpy as np


def add_gaussian_noise(
    signal: np.ndarray,
    snr_db: float | None,
    seed: int | None = None,
    use_variance: bool = True,
) -> np.ndarray:
    """Them nhieu Gauss trang dong thoi (AWGN) vao tin hieu 1D theo muc SNR (dB).

    Args:
        signal: Mang 1D tin hieu dau vao.
        snr_db: Ty le tin-tren-nhieu tinh bang decibel (dB).
            Neu snr_db la None hoac vo cung (np.isinf), tra ve ban sao tin hieu goc.
            Gia tri nho (vd 0 hoac -5 dB) bieu thi nhieu rat manh; gia tri lon (vd 20 dB) bieu thi nhieu nhe.
        seed: Seed cho RNG de dam bao tinh tai lap (reproducibility).
        use_variance: Neu True, cong suat tin hieu tinh bang phuong sai Var(signal)
            (loai bo thanh phan DC drift - chuan trong xu ly tin hieu y sinh).
            Neu False, tinh bang gia tri hieu dung RMS ^ 2 = mean(signal^2).

    Returns:
        Mang 1D cung do dai, da duoc cong them nhieu Gauss.
    """
    signal = np.asarray(signal, dtype=float)
    if snr_db is None or np.isinf(snr_db):
        return signal.copy()

    rng = np.random.default_rng(seed)

    if use_variance:
        p_signal = float(np.var(signal))
    else:
        p_signal = float(np.mean(signal**2))

    # Truong hop tin hieu gan nhu hang so (phuong sai ~ 0)
    if p_signal <= 1e-12:
        return signal.copy()

    # SNR(dB) = 10 * log10(P_signal / P_noise) => P_noise = P_signal / (10^(SNR/10))
    p_noise = p_signal / (10.0 ** (snr_db / 10.0))
    noise_std = np.sqrt(p_noise)

    noise = rng.normal(loc=0.0, scale=noise_std, size=signal.shape)
    return signal + noise
