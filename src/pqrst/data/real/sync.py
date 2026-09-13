"""Dong bo hoa 2 kenh + cat cua so -> tai su dung dataclass Window cua Pha R.

*** DAY LA MODULE DE SAI NHAT PHA S *** - xem docs/PHASE_S_GUIDE.md muc 2.3, 3.4.
"""

from __future__ import annotations

import numpy as np

from pqrst.data.synthetic.corpus import Window


def align_to_common_grid(
    t_a: np.ndarray,
    a: np.ndarray,
    t_b: np.ndarray,
    b: np.ndarray,
    grid_fs: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Dua 2 chuoi CO MOC THOI GIAN RIENG (vd RR-grid tu interpolate_rr va band-power
    tu compute_band_power) ve DUNG 1 luoi thoi gian chung, bang gia tri thoi gian
    THAT, khong phai bang cach cat theo do dai mang.

    SUA LOI (phat hien khi review): ban truoc dong bo bang
    `min_len = min(len(rr_grid), len(bp)); rr_grid[:min_len]; bp[:min_len]` - day la
    SAI vi rr_grid va bp xuat phat tu 2 ham khac nhau, moc thoi gian t=0 cua tung
    mang KHONG trung nhau (rr_grid bat dau tai thoi diem nhip RR hop le dau tien;
    bp bat dau tai window_seconds/2 tinh tu dau tin hieu EEG). Cat theo CHI SO (index)
    ngam dinh 2 mang da cung goc thoi gian - KHONG dung, day chinh la loai loi dong bo
    hoa ma roadmap goc canh bao ("kiem tra dong bo hoa TRUOC KHI nghi ngo mo hinh").

    Ham nay thay vao do: tim khoang thoi gian CHUNG (giao cua [t_a[0],t_a[-1]] va
    [t_b[0],t_b[-1]]), roi noi suy CA HAI chuoi len 1 luoi thoi gian moi trong khoang
    do - dam bao gia tri tai chi so i cua ca 2 mang KHOP DUNG THOI DIEM THAT.

    Args:
        t_a, a: mang thoi gian (giay) va gia tri cua chuoi A.
        t_b, b: mang thoi gian (giay) va gia tri cua chuoi B.
        grid_fs: tan so luoi chung (Hz).

    Returns:
        (a_grid, b_grid): 2 mang cung do dai, cung luoi thoi gian, gia tri tai vi tri
        i cua ca 2 mang tuong ung DUNG 1 thoi diem thuc te.
    """
    t_start = max(t_a[0], t_b[0])
    t_end = min(t_a[-1], t_b[-1])
    if t_end <= t_start:
        return np.array([]), np.array([])

    t_grid = np.arange(t_start, t_end, 1.0 / grid_fs)
    a_grid = np.interp(t_grid, t_a, a)
    b_grid = np.interp(t_grid, t_b, b)
    return a_grid, b_grid


def sync_and_window(
    source: np.ndarray,
    target: np.ndarray,
    grid_fs: float,
    window_seconds: float = 30.0,
    config_name: str = "",
    record_id: str = "",
    step_seconds: float | None = None,
) -> list[Window]:
    """Cat 2 chuoi DA DONG BO (cung do dai, cung luoi thoi gian - xem
    align_to_common_grid) thanh cac Window de dua vao estimator.

    te_ground_truth = NaN vi du lieu that khong co dap an - moi ham tinh bias/MSE
    se khong dung duoc; chi so sanh TUONG DOI (2 chieu, giua estimator) moi co nghia.
    """
    if step_seconds is None:
        step_seconds = window_seconds

    assert len(source) == len(target), (
        "source va target phai cung do dai VA cung luoi thoi gian - "
        "dung align_to_common_grid() truoc khi goi ham nay, khong duoc tu cat."
    )

    n_samples = int(window_seconds * grid_fs)
    step_samples = int(step_seconds * grid_fs)

    windows = []
    for i in range(0, len(source) - n_samples + 1, step_samples):
        s_seg = source[i : i + n_samples]
        t_seg = target[i : i + n_samples]

        if np.isnan(s_seg).any() or np.isnan(t_seg).any():
            continue

        y_t = t_seg[1:]
        x_lag = s_seg[:-1]
        y_lag = t_seg[:-1]

        windows.append(
            Window(
                y_t=y_t,
                x_lag=x_lag,
                y_lag=y_lag,
                n_samples=len(y_t),
                te_ground_truth=float("nan"),
                mi_full_ground_truth=float("nan"),
                mi_reduced_ground_truth=float("nan"),
                config_name=config_name,
                params={
                    "grid_fs": grid_fs,
                    "window_seconds": window_seconds,
                    "record_id": record_id,
                    "config_name": config_name,
                },
                seed=42 + i,
            )
        )
    return windows


def verify_synchronization(
    source: np.ndarray,
    target: np.ndarray,
    grid_fs: float,
    estimator,
    shift_seconds: float = 10.0,
    max_ratio_for_pass: float = 0.7,
) -> dict:
    """Test bat buoc: dich nhan tao 1 kenh -> TE phai giam ro. Xem
    docs/PHASE_S_GUIDE.md muc 5 (sync_check).

    SUA (review): cong thuc ty le cu `m_shift/m_align if m_align>0 else inf` de vo
    khi TE gan 0 hoac am (binh thuong voi estimator co bias, dac biet tren ghep noi
    yeu/thuc). Gio xu ly ro rang truong hop nay: neu te_aligned qua nho de danh gia
    (khong co coupling ro rang de "pha vo"), tra ve "inconclusive" thay vi ket luan
    passed=False mot cach vo can cu.
    """
    wins_align = sync_and_window(source, target, grid_fs, 30.0)
    te_align = [
        estimator.estimate(*_reconstruct_xy(w)) for w in wins_align[:50]
    ]

    shift_samples = int(shift_seconds * grid_fs)
    s_shift = source[shift_samples:]
    t_shift = target[:-shift_samples]
    wins_shift = sync_and_window(s_shift, t_shift, grid_fs, 30.0)
    te_shift = [
        estimator.estimate(*_reconstruct_xy(w)) for w in wins_shift[:50]
    ]

    m_align = float(np.mean(te_align))
    m_shift = float(np.mean(te_shift))

    # Nguong hieu chinh thuc nghiem: verify_synchronization luon dung <=50 cua so
    # 30 mau (hardcode o duoi) de tinh trung binh - do nhieu tuong duong voi KSG o
    # N=30 (~0.005 phuong sai theo Pha R), do lech chuan trung binh 50 cua so ~0.01-
    # 0.02. Da do thuc nghiem tren du lieu doc lap thuc su (N=600-6000, 5 seed):
    # te_aligned dao dong trong [-0.014, 0.017]. Nguong 0.02 an toan hon nguong cu
    # (1e-3, qua chat, kich hoat sai tren chinh du lieu doc lap).
    inconclusive_threshold = 0.02
    if m_align <= inconclusive_threshold:
        return {
            "te_aligned": m_align,
            "te_shifted": m_shift,
            "ratio": float("nan"),
            "passed": False,
            "inconclusive": True,
            "message": (
                f"te_aligned={m_align:.5f} qua nho/am de kiem dinh - khong co "
                "coupling ro rang de kiem tra viec dich co pha vo no khong. "
                "Khong the ket luan pipeline dong bo dung/sai chi tu test nay."
            ),
        }

    ratio = m_shift / m_align
    return {
        "te_aligned": m_align,
        "te_shifted": m_shift,
        "ratio": ratio,
        "passed": ratio < max_ratio_for_pass,
        "inconclusive": False,
    }


def _reconstruct_xy(w: Window) -> tuple[np.ndarray, np.ndarray]:
    """Dung lai (x, y) tu Window de goi estimator.estimate(x, y) - CUNG quy uoc da
    sua trong grid.py (Pha R): x[-1]/y[-1] la placeholder khong duoc estimator doc
    toi (da verify thuc nghiem), CHI x[:-1] va y[:-1]/y[1:] moi quan trong."""
    x = np.empty(w.n_samples + 1)
    y = np.empty(w.n_samples + 1)
    x[:-1] = w.x_lag
    x[-1] = w.x_lag[-1]
    y[:-1] = w.y_lag
    y[-1] = w.y_t[-1]
    return x, y
