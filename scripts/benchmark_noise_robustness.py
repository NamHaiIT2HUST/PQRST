"""CLI entry point benchmark danh gia do ben truoc nhieu Gauss cong tinh (AWGN)."""

from __future__ import annotations

import argparse
from pathlib import Path

from pqrst.evaluation.noise_benchmark import (
    run_noise_benchmark,
    summarize_noise_benchmark,
    plot_noise_robustness,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results/checkpoints/phi_amortized_standardized.pt")
    parser.add_argument("--n_windows", type=int, default=30)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output_csv", default="results/tables/noise_robustness_benchmark.csv")
    parser.add_argument("--output_fig", default="results/figures/noise_robustness_mse.png")
    args = parser.parse_args()

    n_windows = 5 if args.quick else args.n_windows
    snr_levels = [None, 20.0, 10.0, 0.0] if args.quick else [None, 20.0, 15.0, 10.0, 5.0, 0.0]
    n_samples = [20, 50] if args.quick else [20, 50, 100]

    print(f"Chay Gaussian Noise Benchmark (quick={args.quick}, n_windows={n_windows})...")
    raw_df = run_noise_benchmark(
        checkpoint_path=args.checkpoint,
        n_windows_per_cell=n_windows,
        snr_levels=snr_levels,
        n_samples_list=n_samples,
    )

    summary_df = summarize_noise_benchmark(raw_df)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.output_csv, index=False)
    print(f"Da luu ket qua vao: {args.output_csv}")

    plot_noise_robustness(summary_df, args.output_fig)
    print(f"Da luu bieu do danh gia vao: {args.output_fig}")


if __name__ == "__main__":
    main()
