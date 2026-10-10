"""Module benchmark danh gia do ben truoc nhieu Gauss cong tinh (AWGN).

Kiem dinh Menh de 3 trong docs/THEORY_NOTES.md:
So sanh Bias, Variance, MSE cua cac bo uoc luong (KSG vs Amortized MINE vs Hybrid)
khi ti le tin-tren-nhieu (SNR) giam dan tu Clean (inf dB) xuong 20, 15, 10, 5, 0 dB.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pqrst.baselines.ksg import KSGTEEstimator
from pqrst.estimators.mine.amortized import AmortizedTEEstimator
from pqrst.estimators.hybrid import HybridTEEstimator
from pqrst.data.synthetic.var_linear_gaussian import (
    generate_var_linear_gaussian,
    compute_var_linear_ground_truths,
)
from pqrst.utils.noise import add_gaussian_noise


def run_noise_benchmark(
    checkpoint_path: str,
    n_windows_per_cell: int = 30,
    snr_levels: list[float | None] | None = None,
    n_samples_list: list[int] | None = None,
    seed_base: int = 42,
) -> pd.DataFrame:
    if snr_levels is None:
        snr_levels = [None, 20.0, 15.0, 10.0, 5.0, 0.0]
    if n_samples_list is None:
        n_samples_list = [20, 50, 100]

    # Khoi tao cac estimators
    ksg = KSGTEEstimator(k=4)
    amortized = AmortizedTEEstimator.load(checkpoint_path)
    hybrid = HybridTEEstimator(ksg, amortized, threshold_n=50)

    estimators = {
        "KSG": ksg,
        "Amortized": amortized,
        "Hybrid": hybrid,
    }

    # Cau hinh mo hinh sinh du lieu (VAR linear coupled)
    a, b, c, noise_std = 0.5, 0.5, 0.6, 0.5
    gt_info = compute_var_linear_ground_truths(a, b, c, noise_std)
    te_clean_gt = gt_info["te"]

    records = []
    seed_counter = seed_base

    for snr in snr_levels:
        snr_label = "Clean" if snr is None else f"{snr:.0f}dB"

        for N in n_samples_list:
            for rep in range(n_windows_per_cell):
                seed_counter += 1
                # Sinh cap chuoi sach (N+1 mau de du N mau sau lay lag)
                x, y, _ = generate_var_linear_gaussian(
                    n_samples=N + 1, a=a, b=b, c=c, noise_std=noise_std, seed=seed_counter
                )

                # Them nhieu Gauss doc lap vao x va y
                if snr is not None:
                    x_noisy = add_gaussian_noise(x, snr_db=snr, seed=seed_counter * 10 + 1)
                    y_noisy = add_gaussian_noise(y, snr_db=snr, seed=seed_counter * 10 + 2)
                else:
                    x_noisy, y_noisy = x.copy(), y.copy()

                for est_name, est in estimators.items():
                    try:
                        te_est = est.estimate(x_noisy, y_noisy, seed=seed_counter)
                    except Exception:
                        te_est = np.nan

                    records.append({
                        "snr_db": 999.0 if snr is None else snr,
                        "snr_label": snr_label,
                        "n_samples": N,
                        "rep": rep,
                        "estimator": est_name,
                        "te_estimate": te_est,
                        "te_ground_truth": te_clean_gt,
                        "error": te_est - te_clean_gt if not np.isnan(te_est) else np.nan,
                    })

    df = pd.DataFrame(records)
    return df


def summarize_noise_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    groups = df.groupby(["snr_db", "snr_label", "n_samples", "estimator"])

    for (snr_val, snr_lbl, n, est), grp in groups:
        estimates = grp["te_estimate"].dropna().values
        gt = grp["te_ground_truth"].iloc[0]
        n_valid = len(estimates)

        if n_valid > 0:
            bias = float(np.mean(estimates) - gt)
            var = float(np.var(estimates, ddof=1)) if n_valid > 1 else 0.0
            mse = float(np.mean((estimates - gt) ** 2))
        else:
            bias, var, mse = np.nan, np.nan, np.nan

        summary_rows.append({
            "snr_db": snr_val,
            "snr_label": snr_lbl,
            "n_samples": n,
            "estimator": est,
            "n_valid": n_valid,
            "mean_estimate": float(np.mean(estimates)) if n_valid > 0 else np.nan,
            "bias": bias,
            "variance": var,
            "mse": mse,
        })

    summary_df = pd.DataFrame(summary_rows)
    return summary_df


def plot_noise_robustness(summary_df: pd.DataFrame, output_fig_path: str):
    """Vẽ biểu đồ đánh giá độ bền trước nhiễu AWGN (so sánh trực diện KSG vs Amortized MINE)."""
    Path(output_fig_path).parent.mkdir(parents=True, exist_ok=True)
    n_samples_list = sorted(summary_df["n_samples"].unique())

    fig, axes = plt.subplots(1, len(n_samples_list), figsize=(15, 4.5), sharey=True)
    if len(n_samples_list) == 1:
        axes = [axes]

    colors = {"KSG": "#1f77b4", "Amortized": "#d95f02"}
    markers = {"KSG": "o", "Amortized": "s"}
    linestyles = {"KSG": "--", "Amortized": "-"}

    for idx, N in enumerate(n_samples_list):
        ax = axes[idx]
        df_n = summary_df[summary_df["n_samples"] == N]

        for est in ["KSG", "Amortized"]:
            df_est = df_n[df_n["estimator"] == est].copy()
            df_est = df_est.sort_values(by="snr_db", ascending=False)

            labels = df_est["snr_label"].tolist()
            mses = df_est["mse"].tolist()
            x_indices = list(range(len(labels)))

            ax.plot(
                x_indices,
                mses,
                label=f"{est} Estimator",
                color=colors.get(est, "gray"),
                marker=markers.get(est, "d"),
                linestyle=linestyles.get(est, "-"),
                linewidth=2.2,
                markersize=7,
            )

        ax.set_title(f"Sample size N = {N}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Noise Level (SNR)", fontsize=11)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.grid(True, linestyle="--", alpha=0.6)
        if idx == 0:
            ax.set_ylabel("Mean Squared Error (MSE)", fontsize=11)
            ax.legend(frameon=True, fontsize=10)

    plt.suptitle("Robustness to Additive Gaussian Noise (AWGN): KSG vs Amortized MINE", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_fig_path, dpi=300, bbox_inches="tight")
    plt.close()

