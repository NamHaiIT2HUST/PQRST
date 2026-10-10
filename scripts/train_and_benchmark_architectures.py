"""Script huan luyen va so sanh toan dien cac kien truc mo hinh:
1. Classical MLP (128-128-64)
2. Small MLP (16-16)
3. Temporal 1D-CNN (Conv1D)
4. Pure VQC (Quantum Data Re-uploading)
5. Hybrid Classical-Quantum (MLP + VQC)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch

from pqrst.estimators.mine.amortized import (
    MaskedStatisticsNetwork,
    AmortizedTrainConfig,
    train_amortized,
)
from pqrst.estimators.mine.temporal import Conv1DStatisticsNetwork
from pqrst.estimators.quantum.wrapper import QuantumStatisticsNetwork
from pqrst.estimators.quantum.hybrid_vqc import HybridClassicalQuantumStatisticsNetwork
from pqrst.evaluation.model_comparison import (
    count_parameters,
    benchmark_model_inference,
    plot_architecture_comparison,
)
from pqrst.data.synthetic.corpus import generate_corpus, split_corpus_by_seed


def train_temporal_conv1d(train_windows: list, val_windows: list, epochs: int = 20) -> Conv1DStatisticsNetwork:
    print(f"--- Training Temporal 1D-CNN ({epochs} epochs) ---")
    cfg = AmortizedTrainConfig(
        max_epochs=epochs,
        learning_rate=1e-3,
        windows_per_batch=16,
        patience=epochs,
        final_estimate_last_k_epochs=5,
        standardize=True,
    )
    est, hist = train_amortized(
        train_windows=train_windows,
        val_windows=val_windows,
        config=cfg,
        model_factory=lambda: Conv1DStatisticsNetwork(hidden_channels=[16, 32]),
    )
    return est.model


def train_hybrid_vqc(train_windows: list, val_windows: list, epochs: int = 10) -> HybridClassicalQuantumStatisticsNetwork:
    print(f"--- Training Hybrid Classical-Quantum (MLP + VQC) ({epochs} epochs) ---")
    cfg = AmortizedTrainConfig(
        max_epochs=epochs,
        learning_rate=5e-3,
        windows_per_batch=16,
        patience=epochs,
        final_estimate_last_k_epochs=3,
        standardize=True,
    )
    est, hist = train_amortized(
        train_windows=train_windows,
        val_windows=val_windows,
        config=cfg,
        model_factory=lambda: HybridClassicalQuantumStatisticsNetwork(
            n_qubits=4, n_layers=2, encoder_hidden_dim=16, readout_hidden_dim=8
        ),
    )
    return est.model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train_windows", type=int, default=60)
    parser.add_argument("--n_test_windows", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--output_csv", default="results/tables/model_architecture_comparison.csv")
    parser.add_argument("--output_fig", default="results/figures/model_architecture_comparison.png")
    args = parser.parse_args()

    Path("results/checkpoints").mkdir(parents=True, exist_ok=True)
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    Path("results/figures").mkdir(parents=True, exist_ok=True)

    print("Sinh du lieu corpus mo phong cho huan luyen va danh gia...")
    # Sinh tap du lieu nhe cho training & validation
    corpus = generate_corpus(
        coupling_values=[0.6],
        noise_values=[0.5],
        n_values=[50],
        n_windows_per_cell=args.n_train_windows + args.n_test_windows,
        a=0.5,
        b=0.5,
        base_seed=1000,
    )

    train_all = corpus[: args.n_train_windows]
    test_windows = corpus[args.n_train_windows :]
    train_windows, val_windows = split_corpus_by_seed(train_all, val_fraction=0.2, split_seed=42)

    # 1. Classical MLP (Load checkpoint da co)
    print("Nap Classical MLP (128-128-64)...")
    ckpt_mlp = torch.load("results/checkpoints/phi_amortized_standardized.pt", weights_only=True)
    cfg_mlp = AmortizedTrainConfig(**ckpt_mlp["config"])
    model_mlp = MaskedStatisticsNetwork(cfg_mlp.hidden_dims)
    model_mlp.load_state_dict(ckpt_mlp["state_dict"])

    # 2. Small MLP (Load checkpoint da co)
    print("Nap Small MLP (16-16)...")
    ckpt_small = torch.load("results/checkpoints/phi_amortized_small.pt", weights_only=True)
    cfg_small = AmortizedTrainConfig(**ckpt_small["config"])
    model_small = MaskedStatisticsNetwork(cfg_small.hidden_dims)
    model_small.load_state_dict(ckpt_small["state_dict"])

    # 3. Pure VQC (Load checkpoint da co)
    print("Nap Pure VQC (6 Qubits, 4 Layers)...")
    ckpt_vqc = torch.load("results/checkpoints/theta_quantum_400.pt", weights_only=True)
    model_vqc = QuantumStatisticsNetwork(
        n_qubits=ckpt_vqc["n_qubits"], n_layers=ckpt_vqc["n_layers"]
    )
    model_vqc.load_state_dict(ckpt_vqc["state_dict"])

    # 4. Temporal 1D-CNN (Train nhanh va luu checkpoint)
    model_conv1d = train_temporal_conv1d(train_windows, val_windows, epochs=args.epochs)
    torch.save(model_conv1d.state_dict(), "results/checkpoints/phi_temporal_conv1d.pt")

    # 5. Hybrid Classical-Quantum (MLP + VQC) (Train nhanh va luu checkpoint)
    model_hybrid_vqc = train_hybrid_vqc(train_windows, val_windows, epochs=max(args.epochs // 2, 5))
    torch.save(model_hybrid_vqc.state_dict(), "results/checkpoints/theta_hybrid_vqc.pt")

    # Danh gia va do benchmarks tat ca mo hinh
    print("\nBat dau danh gia va so sanh da kien truc...")
    models_to_test = {
        "Classical MLP": (model_mlp, "Classical Deep NN"),
        "Small MLP": (model_small, "Classical Lightweight"),
        "Temporal 1D-CNN": (model_conv1d, "Classical Temporal CNN"),
        "Pure VQC": (model_vqc, "Quantum Re-uploading"),
        "Hybrid MLP+VQC": (model_hybrid_vqc, "Hybrid Quantum-Classical"),
    }

    comparison_records = []
    for name, (m, arch_type) in models_to_test.items():
        print(f"Evaluating {name}...")
        metrics = benchmark_model_inference(
            model=m,
            test_windows=test_windows,
            n_shuffles=20,
            seed=42,
            standardize=True,
        )
        rec = {
            "model_name": name,
            "architecture_type": arch_type,
            "n_params": metrics["n_params"],
            "avg_latency_ms": metrics["avg_latency_ms"],
            "mean_te_estimate": metrics["mean_te_estimate"],
            "bias": metrics["bias"],
            "variance": metrics["variance"],
            "mse": metrics["mse"],
        }
        comparison_records.append(rec)

    df_comp = pd.DataFrame(comparison_records)
    df_comp.to_csv(args.output_csv, index=False)
    print(f"\nDa luu bang so sanh kien truc vao: {args.output_csv}")
    print(df_comp.to_string(index=False))

    plot_architecture_comparison(df_comp, args.output_fig)
    print(f"Da xuat bieu do so sanh vao: {args.output_fig}")


if __name__ == "__main__":
    main()
