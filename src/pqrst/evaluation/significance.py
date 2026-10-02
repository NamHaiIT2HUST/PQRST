"""Kiem dinh y nghia thong ke (significance testing) ghep cap theo cua so hoac ban ghi.

Dung cho Pha U (S0, E1) de so sanh cac estimator sat nhau:
- Hybrid vs KSG / Amortized tai diem giao cat N
- Quantum vs KSG / MLP tren tap test danh gia
- Adapted vs Unadapted tren du lieu sinh ly that
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def paired_bootstrap_test(
    a: np.ndarray | list[float],
    b: np.ndarray | list[float],
    n_bootstrap: int = 1000,
    seed: int = 42,
    alternative: str = "two-sided",
) -> dict[str, float | bool]:
    """Kiem dinh bootstrap ghep cap cho hieu so (a - b).

    Args:
        a: Mang gia tri cua phuong phap A (vd loi / MSE / TE).
        b: Mang gia tri cua phuong phap B (cung do dai voi a).
        n_bootstrap: So lan lay mau bootstrap.
        seed: Random seed cho reproducibility.
        alternative: "two-sided", "greater" (a > b), hoac "less" (a < b).

    Returns:
        dict chua mean_diff, ci_low, ci_high, p_value, wilcoxon_p.
    """
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    assert len(arr_a) == len(arr_b), f"Do dai khong khop: {len(arr_a)} vs {len(arr_b)}"
    n = len(arr_a)
    assert n > 0, "Tap du lieu rong"

    diff = arr_a - arr_b
    mean_diff = float(np.mean(diff))

    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = np.mean(diff[boot_indices], axis=1)

    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))

    # Tinh p-value bootstrap
    # Gia thuyet H0: mean(diff) = 0
    centered_boot = boot_means - mean_diff
    if alternative == "two-sided":
        p_boot = float(np.mean(np.abs(centered_boot) >= np.abs(mean_diff)))
    elif alternative == "greater":
        p_boot = float(np.mean(centered_boot >= mean_diff))
    elif alternative == "less":
        p_boot = float(np.mean(centered_boot <= mean_diff))
    else:
        raise ValueError(f"alternative khong hop le: {alternative}")

    # Wilcoxon signed-rank test lam kiem dinh phi tham so doi chung
    try:
        if alternative == "two-sided":
            alt_scipy = "two-sided"
        elif alternative == "greater":
            alt_scipy = "greater"
        else:
            alt_scipy = "less"
        _, w_p = stats.wilcoxon(arr_a, arr_b, alternative=alt_scipy)
        wilcoxon_p = float(w_p)
    except Exception:
        wilcoxon_p = float("nan")

    return {
        "n": n,
        "mean_a": float(np.mean(arr_a)),
        "mean_b": float(np.mean(arr_b)),
        "mean_diff": mean_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value_bootstrap": p_boot,
        "p_value_wilcoxon": wilcoxon_p,
        "significant_005": bool(ci_low > 0 or ci_high < 0),
    }


def compare_paired_series(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    group_col: str | None = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """So sanh ghep cap theo tung nhom (vd theo n_samples hoac config)."""
    rows = []
    if group_col is None:
        res = paired_bootstrap_test(df[col_a], df[col_b], n_bootstrap=n_bootstrap, seed=seed)
        rows.append(res)
    else:
        for grp_val, grp_df in df.groupby(group_col):
            res = paired_bootstrap_test(grp_df[col_a], grp_df[col_b], n_bootstrap=n_bootstrap, seed=seed)
            res[group_col] = grp_val
            rows.append(res)

    res_df = pd.DataFrame(rows)
    if group_col and group_col in res_df.columns:
        cols = [group_col] + [c for c in res_df.columns if c != group_col]
        res_df = res_df[cols]
    return res_df
