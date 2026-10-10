"""Module so sanh da kien truc mo hinh (Model Architecture Comparison Suite).

Phuc vu viec so sanh toan dien giua cac kien truc:
1. Classical MLP (Baseline)
2. Temporal 1D-CNN (Conv1D)
3. Fourier Feature Network (RFF)
4. Pure VQC (Quantum Data Re-uploading)
5. Hybrid Classical-Quantum (MLP + VQC)

Tieu chi so sanh:
- So luong tham so hoc duoc (Trainable Parameters)
- Thoi gian suy luan trung binh tren 1 cua so (Inference Latency in ms)
- Do chinh xac uoc luong TE (Bias, Variance, MSE tren du lieu synthetic biet ground-truth)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from pqrst.estimators.mine.amortized import (
    MaskedStatisticsNetwork,
    AmortizedTrainConfig,
    train_amortized,
)
from pqrst.estimators.mine.temporal import Conv1DStatisticsNetwork
from pqrst.estimators.classical_fourier import FourierFeatureStatisticsNetwork
from pqrst.estimators.quantum.wrapper import QuantumStatisticsNetwork
from pqrst.estimators.quantum.hybrid_vqc import HybridClassicalQuantumStatisticsNetwork
from pqrst.estimators.mine.conditional import estimate_te_from_window
from pqrst.utils.standardize import standardize_window


def count_parameters(model: nn.Module) -> int:
    """Dem tong so tham so co the huan luyen (trainable parameters)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_model_inference(
    model: nn.Module,
    test_windows: list,
    n_shuffles: int = 20,
    seed: int = 42,
    standardize: bool = True,
) -> dict[str, float]:
    """Do toc do suy luan va sai so uoc luong TE tren tap cua so test."""
    model.eval()

    estimates = []
    latencies_ms = []

    for w in test_windows:
        y_t, x_lag, y_lag = w.y_t, w.x_lag, w.y_lag
        if standardize:
            y_t = standardize_window(y_t)
            x_lag = standardize_window(x_lag)
            y_lag = standardize_window(y_lag)

        t0 = time.perf_counter()
        res = estimate_te_from_window(
            model=model,
            y_t=y_t,
            x_lag=x_lag,
            y_lag=y_lag,
            n_shuffles=n_shuffles,
            seed=seed,
        )
        t1 = time.perf_counter()

        estimates.append(res["te"])
        latencies_ms.append((t1 - t0) * 1000.0)

    estimates = np.array(estimates)
    gts = np.array([w.te_ground_truth for w in test_windows])

    bias = float(np.mean(estimates - gts))
    var = float(np.var(estimates, ddof=1)) if len(estimates) > 1 else 0.0
    mse = float(np.mean((estimates - gts) ** 2))
    avg_latency = float(np.mean(latencies_ms))

    return {
        "n_params": count_parameters(model),
        "avg_latency_ms": avg_latency,
        "mean_te_estimate": float(np.mean(estimates)),
        "bias": bias,
        "variance": var,
        "mse": mse,
    }


def plot_architecture_comparison(
    comparison_df: pd.DataFrame, output_fig_path: str
):
    """Ve bieu do so sanh cac kien truc mo hinh theo Params, Latency va MSE."""
    Path(output_fig_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    models = comparison_df["model_name"].tolist()
    x = np.arange(len(models))

    # 1. So luong tham so (log scale neu chenh lech lon)
    params = comparison_df["n_params"].tolist()
    bars1 = axes[0].bar(x, params, color="#2b5c8f", width=0.55)
    axes[0].set_title("So luong tham so (Parameters)", fontsize=12, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=20, ha="right")
    axes[0].set_ylabel("Parameters Count")
    axes[0].grid(axis="y", linestyle="--", alpha=0.6)
    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2.0, yval, f"{int(yval)}", ha="center", va="bottom", fontsize=9)

    # 2. Do tre suy luan (ms/cua so)
    latencies = comparison_df["avg_latency_ms"].tolist()
    bars2 = axes[1].bar(x, latencies, color="#e27c38", width=0.55)
    axes[1].set_title("Thoi gian suy luan (Latency per Window)", fontsize=12, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=20, ha="right")
    axes[1].set_ylabel("Milliseconds (ms)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.6)
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, yval, f"{yval:.1f}ms", ha="center", va="bottom", fontsize=9)

    # 3. Sai so MSE
    mses = comparison_df["mse"].tolist()
    bars3 = axes[2].bar(x, mses, color="#2a9d8f", width=0.55)
    axes[2].set_title("Sai so toan phuong trung binh (MSE)", fontsize=12, fontweight="bold")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=20, ha="right")
    axes[2].set_ylabel("MSE (nats^2)")
    axes[2].grid(axis="y", linestyle="--", alpha=0.6)
    for bar in bars3:
        yval = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width() / 2.0, yval, f"{yval:.4f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle("So sanh Da kien truc: MLP vs 1D-CNN vs VQC vs MLP+VQC", fontsize=14, y=1.03)
    plt.tight_layout()
    plt.savefig(output_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
