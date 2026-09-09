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
    rows = []
    
    iterator = test_windows
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(test_windows, desc="Evaluating windows")
        except ImportError:
            pass

    for i, w in enumerate(iterator):
        # Tai tao (x, y) day du de dung voi BaseTEEstimator
        # Vi cac baseline yeu cau y phai kem y_lag (vi tri t va t-1)
        # x_lag va y_lag la o t-1; y_t la o t.
        # Mang (N+1,)
        N = w.n_samples
        x = np.empty(N + 1)
        y = np.empty(N + 1)

        # Baseline uoc luong TE bang cach doc x[:-1] (X[t-1]) va y[1:] (Y[t]),
        # dieu kien tren y[:-1] (Y[t-1]) qua history_target=1 - dung interface nay
        # phai dam bao x[:-1] == w.x_lag va y[:-1]/y[1:] == w.y_lag/w.y_t CHINH XAC.
        # Vi tri x[-1]/y[-1] khong duoc doc boi estimator (da verify thuc nghiem:
        # thay doi gia tri nay khong lam doi ket qua estimate()), nen chi can dien
        # placeholder hop le (khong dung 0.0 tuy tien de tranh diem ngoai phan phoi).
        x[:-1] = w.x_lag
        x[-1] = w.x_lag[-1]
        y[:-1] = w.y_lag
        y[-1] = w.y_t[-1]
        
        for est_name, est in estimators.items():
            try:
                te_est = est.estimate(x, y, seed=w.seed)
                error = te_est - w.te_ground_truth
            except Exception as e:
                te_est = np.nan
                error = np.nan
                
            rows.append({
                "window_id": i,
                "config_name": w.config_name,
                "coupling_c": w.params["c"],
                "noise_std": w.params["noise_std"],
                "n_samples": w.n_samples,
                "seed": w.seed,
                "estimator": est_name,
                "te_estimate": te_est,
                "te_ground_truth": w.te_ground_truth,
                "error": error
            })
            
    return pd.DataFrame(rows)


def summarize_grid(raw: pd.DataFrame) -> pd.DataFrame:
    groups = raw.groupby(["config_name", "coupling_c", "noise_std", "n_samples", "estimator"])
    
    rows = []
    for name, group in groups:
        estimates = group["te_estimate"].values
        valid_mask = ~np.isnan(estimates)
        n_total = len(estimates)
        n_valid = int(valid_mask.sum())
        n_failed = n_total - n_valid
        
        gt = group["te_ground_truth"].values[0]
        
        if n_valid > 0:
            valid_ests = estimates[valid_mask]
            bias = np.mean(valid_ests) - gt
            variance = np.var(valid_ests, ddof=1) if n_valid > 1 else 0.0
            mse = np.mean((valid_ests - gt)**2)
            
            se = np.sqrt(variance / n_valid) if n_valid > 0 else 0
            ci_low = np.mean(valid_ests) - 1.96 * se
            ci_high = np.mean(valid_ests) + 1.96 * se
        else:
            bias = np.nan
            variance = np.nan
            mse = np.nan
            ci_low = np.nan
            ci_high = np.nan
            
        rows.append({
            "config_name": name[0],
            "coupling_c": name[1],
            "noise_std": name[2],
            "n_samples": name[3],
            "estimator": name[4],
            "n_valid": n_valid,
            "n_failed": n_failed,
            "bias": bias,
            "variance": variance,
            "mse": mse,
            "ci_low": ci_low,
            "ci_high": ci_high
        })
        
    return pd.DataFrame(rows)


def check_exit_criterion(summary: pd.DataFrame, n_threshold: int = 30) -> dict:
    df_small = summary[summary["n_samples"] < n_threshold]
    
    pivot = df_small.pivot_table(
        index=["config_name", "coupling_c", "noise_std", "n_samples"],
        columns="estimator",
        values="variance"
    ).reset_index()
    
    n_cells = len(pivot)
    if n_cells == 0:
        return {"passed": False, "n_cells_tested": 0}
        
    beats_ksg = 0
    beats_sym = 0
    
    for _, row in pivot.iterrows():
        amortized_var = row.get("Amortized", np.nan)
        ksg_var = row.get("KSG", np.nan)
        sym_var = row.get("Symbolic", np.nan)
        
        if not np.isnan(amortized_var):
            if not np.isnan(ksg_var) and amortized_var < ksg_var:
                beats_ksg += 1
            if not np.isnan(sym_var) and amortized_var < sym_var:
                beats_sym += 1
                
    # Threshold is 70%
    th = 0.7
    pass_ksg = (beats_ksg / n_cells) >= th
    pass_sym = (beats_sym / n_cells) >= th
    passed = pass_ksg and pass_sym
    
    return {
        "passed": bool(passed),
        "n_cells_tested": n_cells,
        "n_cells_amortized_beats_ksg": beats_ksg,
        "n_cells_amortized_beats_symbolic": beats_sym,
        "threshold": th,
        "detail": pivot
    }
