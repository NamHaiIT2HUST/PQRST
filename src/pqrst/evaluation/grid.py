"""Danh gia so sanh 4 estimator tren luoi cau hinh - bang so lieu trung tam cua Pha R.

4 estimator: AmortizedTEEstimator (T_phi), KSG, Symbolic, Binning. Tat ca deu tuan thu
cung interface BaseTEEstimator (thiet ke tu Pha P), nen vong lap danh gia o day khong
can biet ben trong tung cai la gi - va Nhip 2 chi can them T_theta vao dict la xong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pqrst.baselines.base import BaseTEEstimator


def evaluate_estimators_on_grid(
    estimators: dict[str, BaseTEEstimator],
    test_windows: list,
    progress: bool = True,
) -> pd.DataFrame:
    """Chay moi estimator tren moi cua so kiem dinh, tra ve bang KET QUA THO (raw).

    Tra ve ket qua THO (moi dong = 1 cua so x 1 estimator), KHONG tong hop san - de
    tang phan tich phia sau tu do tinh bias/phuong sai/khoang tin cay theo bat ky cach
    nhom nao, va de audit lai tung gia tri khi nghi ngo.

    Args:
        estimators: dict {ten: estimator}.
        test_windows: list[Window] - PHAI tach bach hoan toan voi tap train.
        progress: hien thanh tien do (tqdm) - nen bat vi vong lap nay chay lau.

    Returns:
        DataFrame cac cot:
            window_id, config_name, coupling_c, noise_std, n_samples, seed,
            estimator, te_estimate, te_ground_truth, error (= estimate - ground_truth)

    TODO(ban tu code):
        - Lap qua tung cua so va tung estimator; goi estimator.estimate(x, y).
        - Voi AmortizedTEEstimator, x/y can dung dinh dang chuoi goc - luu y Window
          luu san y_t/x_lag/y_lag; can tai tao (x, y) tuong thich hoac them duong dan
          rieng cho amortized. Ghi ro lua chon trong code.
        - BAT LOI tung cai: neu 1 estimator that bai tren 1 cua so (vd KSG loi voi N=10),
          ghi te_estimate = np.nan va GHI LAI ly do, KHONG de vo ca vong lap. Bao cao
          so luong that bai theo tung estimator/tung N - ban than ty le that bai o N nho
          da la 1 ket qua dang bao cao.
    """
    raise NotImplementedError


def summarize_grid(raw: pd.DataFrame) -> pd.DataFrame:
    """Tong hop bang tho thanh bias / phuong sai / MSE theo tung o luoi va estimator.

    Args:
        raw: DataFrame tu evaluate_estimators_on_grid.

    Returns:
        DataFrame cac cot:
            config_name, coupling_c, noise_std, n_samples, estimator,
            n_valid (so cua so khong NaN), n_failed,
            bias (mean(estimate) - ground_truth),
            variance (var cua estimate qua cac cua so cung o luoi),
            mse, ci_low, ci_high (khoang tin cay 95% cua trung binh)

    TODO(ban tu code):
        - Nhom theo (config_name, coupling_c, noise_std, n_samples, estimator).
        - PHUONG SAI la chi so QUAN TRONG NHAT cua Pha R (tieu chi thoat noi ve phuong
          sai, khong phai bias) - tinh bang np.var(..., ddof=1) tren cac uoc luong hop le.
        - Bo qua NaN khi tinh (dung nanmean/nanvar hoac loc truoc), nhung PHAI bao cao
          n_failed rieng - khong duoc am tham bo di.
    """
    raise NotImplementedError


def check_exit_criterion(summary: pd.DataFrame, n_threshold: int = 30) -> dict:
    """Kiem tra TIEU CHI THOAT chinh cua Pha R mot cach dinh luong.

    Tieu chi (nguyen van tu roadmap goc): tren tap kiem dinh tong hop, T_phi co PHUONG
    SAI THAP HON KSG va Symbolic TE o vung N NHO (N < 30).

    Args:
        summary: DataFrame tu summarize_grid.
        n_threshold: nguong N "nho" (mac dinh 30 theo roadmap).

    Returns:
        dict bao cao ro rang, toi thieu gom:
            "passed": bool tong the,
            "n_cells_tested": so o luoi co N < n_threshold,
            "n_cells_amortized_beats_ksg": ...,
            "n_cells_amortized_beats_symbolic": ...,
            "detail": DataFrame so sanh phuong sai tung o.

    TODO(ban tu code):
        - Loc cac o co n_samples < n_threshold.
        - Voi moi o, so variance cua "Amortized" voi "KSG" va "Symbolic".
        - "passed" = True neu thang o DA SO o luoi (vd >= 70%) doi voi CA HAI baseline.
          Ghi ro nguong da chon trong bao cao - va neu khong dat, BAO CAO TRUNG THUC
          thay vi noi long nguong cho vua ket qua (do la p-hacking).
    """
    raise NotImplementedError
