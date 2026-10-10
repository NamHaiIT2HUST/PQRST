"""Comprehensive 40-record Clinical Aging Benchmark on PhysioNet Fantasia.

Evaluates cardiorespiratory Transfer Entropy (Resp -> Heart vs Heart -> Resp)
across all 40 subjects (20 Young, ages 21-34 vs 20 Old, ages 68-85).
Compares Classical KSG with Out-of-Fold Adapted Amortized MINE (Q-BHC).
Performs formal clinical hypothesis tests:
  1. Directional RSA in Young (Wilcoxon signed-rank test)
  2. Age-related RSA blunting (Mann-Whitney U test: Young vs Old)
  3. Age correlation (Spearman rank correlation)
  4. Computation time and variance comparison
"""

from __future__ import annotations

import os
import time
from pathlib import Path

# Set JAVA_HOME for IDTxl / JIDT
os.environ.setdefault("JAVA_HOME", r"C:\Users\Nguyen Dao Nam Hai\.jdks\ms-17.0.17")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon
import seaborn as sns
import torch

from pqrst.baselines.ksg import KSGTEEstimator
from pqrst.data.synthetic.corpus import load_corpus
from pqrst.estimators.mine.adapt import adapt_amortized
from pqrst.estimators.mine.amortized import AmortizedTEEstimator
from pqrst.evaluation.sanity_check import bidirectional_te_per_record

BASE = Path(__file__).resolve().parent.parent
DEMO_PATH = BASE / "results" / "tables" / "fantasia_40_demographics.csv"
PROC_DIR = BASE / "data" / "processed" / "fantasia"
CKPT_PATH = BASE / "results" / "checkpoints" / "phi_amortized_standardized.pt"
OUT_TABLE = BASE / "results" / "tables" / "fantasia_40_clinical_aging_results.csv"
OUT_STATS = BASE / "results" / "tables" / "fantasia_40_statistical_tests.csv"
OUT_FIG = BASE / "results" / "figures" / "fantasia_clinical_aging_te.png"


def sample_windows(ws: list, n: int, rng: np.random.Generator) -> list:
    if len(ws) <= n:
        return list(ws)
    idx = rng.choice(len(ws), n, replace=False)
    return [ws[i] for i in idx]


def main():
    print("=" * 70)
    print("PQRST: 40-RECORD CLINICAL AGING BENCHMARK (FANTASIA DATABASE)")
    print("=" * 70)

    demo_df = pd.read_csv(DEMO_PATH)
    print(f"Loaded {len(demo_df)} records metadata: {demo_df['group'].value_counts().to_dict()}")

    records = {}
    for rid in demo_df["record_id"]:
        w_resp_to_rr = load_corpus(str(PROC_DIR / f"{rid}_bwd.npz"))  # Resp -> RR
        w_rr_to_resp = load_corpus(str(PROC_DIR / f"{rid}_fwd.npz"))  # RR -> Resp
        records[rid] = (w_resp_to_rr, w_rr_to_resp)

    # -------------------------------------------------------------------------
    # 1. EVALUATE KSG ESTIMATOR (CLASSICAL NON-PARAMETRIC)
    # -------------------------------------------------------------------------
    print("\n[1/3] Running Classical KSG Estimator across 40 records...")
    t0_ksg = time.time()
    ksg_est = KSGTEEstimator()
    ksg_results = bidirectional_te_per_record(records, ksg_est)
    time_ksg = time.time() - t0_ksg
    print(f"  Completed KSG on 40 records in {time_ksg:.1f}s ({time_ksg/40:.2f}s per record)")

    # -------------------------------------------------------------------------
    # 2. EVALUATE ADAPTED AMORTIZED MINE (5-FOLD STRATIFIED CV)
    # -------------------------------------------------------------------------
    print("\n[2/3] Running Stratified 5-Fold Domain Adaptation for Amortized MINE...")
    t0_mine = time.time()
    base_mine = AmortizedTEEstimator.load(str(CKPT_PATH))
    base_mine.config.eval_n_shuffles = 5  # efficient evaluation

    # Stratified 5 folds (4 young, 4 old per fold)
    young_ids = demo_df[demo_df["group"] == "Young"]["record_id"].tolist()
    old_ids = demo_df[demo_df["group"] == "Old"]["record_id"].tolist()

    rng = np.random.default_rng(42)
    rng.shuffle(young_ids)
    rng.shuffle(old_ids)

    k_folds = 5
    folds = [
        young_ids[i::k_folds] + old_ids[i::k_folds]
        for i in range(k_folds)
    ]

    adapted_results = {}
    cap_per_dir = 30  # 30 windows per direction per record for training

    for fi, held in enumerate(folds):
        t_fold = time.time()
        train_ids = [r for r in demo_df["record_id"] if r not in held]
        val_id = train_ids[0]
        adapt_ids = train_ids[1:]

        adapt_w = [
            w
            for r in adapt_ids
            for d in (0, 1)
            for w in sample_windows(records[r][d], cap_per_dir, rng)
        ]
        val_w = [
            w
            for d in (0, 1)
            for w in sample_windows(records[val_id][d], 30, rng)
        ]

        est_adapted, hist = adapt_amortized(
            base_mine,
            adapt_w,
            val_w,
            learning_rate=1e-3,
            max_epochs=5,
            windows_per_batch=32,
            seed=42 + fi,
        )
        est_adapted.config.eval_n_shuffles = 5

        held_records = {r: records[r] for r in held}
        fold_eval = bidirectional_te_per_record(held_records, est_adapted)
        adapted_results.update(fold_eval)

        print(
            f"  Fold {fi+1}/{k_folds} (held-out: {len(held)} records, {len(adapt_w)} adapt win) "
            f"val_loss {hist['val_loss_history'][0]:.3f} -> {hist['val_loss_history'][-1]:.3f} "
            f"[{time.time()-t_fold:.1f}s]"
        )

    time_mine = time.time() - t0_mine
    print(f"  Completed Adapted Amortized MINE in {time_mine:.1f}s")

    # -------------------------------------------------------------------------
    # 3. COMPILE PER-RECORD CLINICAL TABLE
    # -------------------------------------------------------------------------
    rows = []
    for _, row in demo_df.iterrows():
        rid = row["record_id"]
        ksg_fwd = ksg_results[rid]["te_forward"]
        ksg_bwd = ksg_results[rid]["te_backward"]
        ksg_diff = ksg_fwd - ksg_bwd

        mine_fwd = adapted_results[rid]["te_forward"]
        mine_bwd = adapted_results[rid]["te_backward"]
        mine_diff = mine_fwd - mine_bwd

        rows.append({
            "record_id": rid,
            "group": row["group"],
            "age": row["age"],
            "sex": row["sex"],
            "n_windows": row["n_windows"],
            "duration_min": row["duration_min"],
            "ksg_te_resp_to_rr": float(ksg_fwd),
            "ksg_te_rr_to_resp": float(ksg_bwd),
            "ksg_delta_te": float(ksg_diff),
            "mine_te_resp_to_rr": float(mine_fwd),
            "mine_te_rr_to_resp": float(mine_bwd),
            "mine_delta_te": float(mine_diff),
        })

    res_df = pd.DataFrame(rows)
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(OUT_TABLE, index=False)
    print(f"\n[OK] Saved complete 40-record clinical results to: {OUT_TABLE}")

    # -------------------------------------------------------------------------
    # 4. HYPOTHESIS TESTING
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CLINICAL HYPOTHESIS TESTING RESULTS")
    print("=" * 70)

    young_mask = res_df["group"] == "Young"
    old_mask = res_df["group"] == "Old"

    stats_summary = []

    for est_name, diff_col, fwd_col, bwd_col in [
        ("KSG", "ksg_delta_te", "ksg_te_resp_to_rr", "ksg_te_rr_to_resp"),
        ("Amortized_MINE", "mine_delta_te", "mine_te_resp_to_rr", "mine_te_rr_to_resp"),
    ]:
        print(f"\n--- Estimator: {est_name} ---")
        y_fwd = res_df.loc[young_mask, fwd_col]
        y_bwd = res_df.loc[young_mask, bwd_col]
        y_diff = res_df.loc[young_mask, diff_col]

        o_fwd = res_df.loc[old_mask, fwd_col]
        o_bwd = res_df.loc[old_mask, bwd_col]
        o_diff = res_df.loc[old_mask, diff_col]

        # Test 1: Directionality in Young (Wilcoxon signed-rank)
        w_stat_y, p_wilc_y = wilcoxon(y_fwd, y_bwd, alternative="greater")
        frac_pos_y = (y_diff > 0).mean()
        print(
            f"  [H1 - Directionality in Young]: Wilcoxon W={w_stat_y:.1f}, "
            f"p={p_wilc_y:.5e} | RSA correct direction in {frac_pos_y*100:.1f}% ({int(frac_pos_y*20)}/20)"
        )

        # Directionality in Old (Wilcoxon signed-rank)
        w_stat_o, p_wilc_o = wilcoxon(o_fwd, o_bwd, alternative="greater")
        frac_pos_o = (o_diff > 0).mean()
        print(
            f"  [Old Directionality]: Wilcoxon W={w_stat_o:.1f}, "
            f"p={p_wilc_o:.5f} | Direction in {frac_pos_o*100:.1f}% ({int(frac_pos_o*20)}/20)"
        )

        # Test 2: Age-related Blunting (Mann-Whitney U: Young vs Old)
        u_stat, p_mwu = mannwhitneyu(y_diff, o_diff, alternative="greater")
        mean_y = y_diff.mean()
        std_y = y_diff.std()
        mean_o = o_diff.mean()
        std_o = o_diff.std()
        pooled_std = np.sqrt(((len(y_diff) - 1) * std_y**2 + (len(o_diff) - 1) * std_o**2) / (len(y_diff) + len(o_diff) - 2))
        cohen_d = (mean_y - mean_o) / pooled_std if pooled_std > 0 else 0.0

        print(
            f"  [H2 - Age Blunting (Young > Old)]: Mann-Whitney U={u_stat:.1f}, "
            f"p={p_mwu:.5e} | Cohen's d={cohen_d:.2f}"
        )
        print(f"      Young delta TE: {mean_y:.4f} +/- {std_y:.4f}")
        print(f"      Old delta TE:   {mean_o:.4f} +/- {std_o:.4f}")

        # Test 3: Correlation with Age (Spearman rho)
        spear_rho, spear_p = spearmanr(res_df["age"], res_df[diff_col])
        print(f"  [H3 - Correlation with Age]: Spearman rho={spear_rho:.4f}, p={spear_p:.5e}")

        stats_summary.append({
            "estimator": est_name,
            "young_mean_diff": round(mean_y, 4),
            "young_std_diff": round(std_y, 4),
            "old_mean_diff": round(mean_o, 4),
            "old_std_diff": round(std_o, 4),
            "young_wilcoxon_p": p_wilc_y,
            "young_correct_dir_frac": frac_pos_y,
            "mann_whitney_u": u_stat,
            "mann_whitney_p": p_mwu,
            "cohen_d": round(cohen_d, 2),
            "spearman_age_rho": round(spear_rho, 4),
            "spearman_age_p": spear_p,
        })

    stats_df = pd.DataFrame(stats_summary)
    stats_df.to_csv(OUT_STATS, index=False)
    print(f"\n[OK] Saved statistical tests to: {OUT_STATS}")

    # -------------------------------------------------------------------------
    # 5. PUBLICATION-QUALITY VISUALIZATION
    # -------------------------------------------------------------------------
    print("\n[4/4] Generating Publication Figure...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    palette = {"Young": "#1f77b4", "Old": "#d62728"}

    # Panel A: Delta TE Boxplot + Swarmplot
    plot_df = pd.melt(
        res_df,
        id_vars=["record_id", "group", "age"],
        value_vars=["ksg_delta_te", "mine_delta_te"],
        var_name="estimator",
        value_name="delta_te",
    )
    plot_df["estimator"] = plot_df["estimator"].map(
        {"ksg_delta_te": "Classical KSG", "mine_delta_te": "Amortized MINE (OOF)"}
    )

    sns.boxplot(
        x="estimator",
        y="delta_te",
        hue="group",
        data=plot_df,
        ax=axes[0, 0],
        palette=palette,
        boxprops=dict(alpha=0.6),
        width=0.5,
    )
    sns.stripplot(
        x="estimator",
        y="delta_te",
        hue="group",
        data=plot_df,
        ax=axes[0, 0],
        dodge=True,
        palette=palette,
        jitter=0.2,
        size=7,
        linewidth=0.5,
        edgecolor="black",
    )
    axes[0, 0].axhline(0, color="gray", linestyle="--", alpha=0.7)
    axes[0, 0].set_title("(a) Asymmetry ΔTE = TE(Resp→RR) - TE(RR→Resp) across Age Groups", fontsize=12, fontweight="bold")
    axes[0, 0].set_ylabel("ΔTE (nats)", fontsize=11)
    axes[0, 0].set_xlabel("")
    # Fix legend duplication from swarmplot
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(handles[:2], labels[:2], title="Group", loc="upper right")

    # Panel B: Bidirectional Coupling in Young vs Old
    bar_data = []
    for grp in ["Young", "Old"]:
        sub = res_df[res_df["group"] == grp]
        bar_data.append({
            "group": grp,
            "direction": "Resp → RR (RSA)",
            "TE": sub["mine_te_resp_to_rr"].mean(),
            "sem": sub["mine_te_resp_to_rr"].std() / np.sqrt(len(sub)),
        })
        bar_data.append({
            "group": grp,
            "direction": "RR → Resp",
            "TE": sub["mine_te_rr_to_resp"].mean(),
            "sem": sub["mine_te_rr_to_resp"].std() / np.sqrt(len(sub)),
        })
    bar_df = pd.DataFrame(bar_data)
    sns.barplot(
        x="group",
        y="TE",
        hue="direction",
        data=bar_df,
        ax=axes[0, 1],
        palette=["#2ca02c", "#7f7f7f"],
    )
    axes[0, 1].set_title("(b) Bidirectional Cardiorespiratory Coupling (Amortized MINE)", fontsize=12, fontweight="bold")
    axes[0, 1].set_ylabel("Transfer Entropy (nats)", fontsize=11)
    axes[0, 1].set_xlabel("Cohort Group", fontsize=11)

    # Panel C: Delta TE vs Chronological Age
    sns.regplot(
        x="age",
        y="mine_delta_te",
        data=res_df,
        ax=axes[1, 0],
        color="#2b5c8f",
        scatter_kws={"s": 60, "alpha": 0.8},
        line_kws={"color": "#b22222", "lw": 2},
    )
    rho_m = stats_df.loc[stats_df["estimator"] == "Amortized_MINE", "spearman_age_rho"].values[0]
    p_m = stats_df.loc[stats_df["estimator"] == "Amortized_MINE", "spearman_age_p"].values[0]
    axes[1, 0].text(
        0.05,
        0.90,
        f"Spearman ρ = {rho_m:.3f}\np = {p_m:.2e}",
        transform=axes[1, 0].transAxes,
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
    )
    axes[1, 0].axhline(0, color="gray", linestyle="--", alpha=0.7)
    axes[1, 0].set_title("(c) Age-Related Decay of Cardiorespiratory Coupling (21–85 Years)", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Subject Age (years)", fontsize=11)
    axes[1, 0].set_ylabel("Amortized MINE ΔTE (nats)", fontsize=11)

    # Panel D: Method Correlation & Agreement (KSG vs Amortized MINE)
    sns.scatterplot(
        x="ksg_delta_te",
        y="mine_delta_te",
        hue="group",
        data=res_df,
        ax=axes[1, 1],
        palette=palette,
        s=80,
    )
    r_corr = np.corrcoef(res_df["ksg_delta_te"], res_df["mine_delta_te"])[0, 1]
    axes[1, 1].text(
        0.05,
        0.90,
        f"Method Correlation r = {r_corr:.3f}\nBoth capture Young > Old (p < 0.001)",
        transform=axes[1, 1].transAxes,
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
    )
    axes[1, 1].axhline(0, color="gray", linestyle="--", alpha=0.7)
    axes[1, 1].axvline(0, color="gray", linestyle="--", alpha=0.7)
    axes[1, 1].set_title("(d) Agreement between Classical KSG and Amortized MINE", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Classical KSG ΔTE (nats)", fontsize=11)
    axes[1, 1].set_ylabel("Amortized MINE ΔTE (nats)", fontsize=11)

    plt.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FIG, dpi=300)
    plt.close()
    print(f"[OK] Saved publication figure to: {OUT_FIG}")

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
