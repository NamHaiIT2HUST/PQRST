"""Chay 3 baseline (KSG, symbolic TE, binning) tren 3 bo du lieu tong hop, so voi
ground-truth, in bang sai so ra console va luu vao results/tables/.

Cach chay (sau khi cai xong moi truong, xem docs/PHASE_P_GUIDE.md):
    python scripts/validate_baselines.py

Day la tieu chi thoat chinh cua Pha P - xem checklist trong docs/PHASE_P_GUIDE.md muc 5.

TODO(ban tu code) - phac thao cac buoc:
    1. Doc configs/synthetic/*.yaml va configs/baselines/*.yaml (dung PyYAML).
    2. Voi moi cau hinh sinh du lieu (vd trong var_linear_gaussian.yaml["configs"]):
       - Sinh n_repetitions cap (x, y) voi seed khac nhau (dung pqrst.utils.seeding.make_rng
         hoac truyen seed truc tiep vao ham sinh).
       - Voi moi baseline (KSGTEEstimator, SymbolicTEEstimator, BinningTEEstimator):
         goi estimate(x, y), thu thap tat ca gia tri uoc luong.
       - Tinh bias, mse (pqrst.evaluation.metrics) so voi ground-truth (hoac
         pseudo_ground_truth_te cho var_nonlinear/periodic_coupling).
    3. Gop ket qua thanh 1 bang (pandas.DataFrame): cot = [dataset, config_name, baseline,
       bias, mse, n_repetitions].
    4. In bang ra console (df.to_string()) va luu results/tables/phase_p_baseline_validation.csv.
    5. (tuy chon) ve boxplot sai so moi baseline, luu
       results/figures/phase_p_baseline_validation.png.
"""

from __future__ import annotations


import argparse
import pandas as pd
import yaml
from pathlib import Path
import numpy as np

from pqrst.data.synthetic.var_linear_gaussian import generate_var_linear_gaussian
from pqrst.data.synthetic.var_nonlinear import generate_var_nonlinear
from pqrst.data.synthetic.periodic_coupling import generate_periodic_coupling
from pqrst.baselines.ksg import KSGTEEstimator
from pqrst.baselines.binning import BinningTEEstimator
from pqrst.baselines.symbolic_te import SymbolicTEEstimator
from pqrst.evaluation.metrics import bias, mse
from pqrst.utils.seeding import make_rng


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    # 1. Doc configs
    base_dir = Path(__file__).parent.parent
    config_dir = base_dir / "configs"
    
    syn_var_lin = load_yaml(config_dir / "synthetic" / "var_linear_gaussian.yaml")
    syn_var_nonlin = load_yaml(config_dir / "synthetic" / "var_nonlinear.yaml")
    syn_periodic = load_yaml(config_dir / "synthetic" / "periodic_coupling.yaml")
    
    ksg_cfg = load_yaml(config_dir / "baselines" / "ksg.yaml")
    binning_cfg = load_yaml(config_dir / "baselines" / "binning.yaml")
    symbolic_cfg = load_yaml(config_dir / "baselines" / "symbolic_te.yaml")
    
    baselines = {
        "KSG": KSGTEEstimator(**ksg_cfg),
        "Binning": BinningTEEstimator(**binning_cfg),
        "Symbolic": SymbolicTEEstimator(**symbolic_cfg),
    }
    
    datasets = {
        "VAR_Linear_Gaussian": {
            "configs": syn_var_lin["configs"],
            "generator": generate_var_linear_gaussian,
            "reps": syn_var_lin["n_repetitions"],
            "n_samples": 1000  # Default N if not specified for Phase P validation
        },
        "VAR_Nonlinear": {
            "configs": syn_var_nonlin["configs"],
            "generator": generate_var_nonlinear,
            "reps": syn_var_nonlin["n_repetitions"],
            "n_samples": 1000
        },
        "Periodic_Coupling": {
            "configs": syn_periodic["configs"],
            "generator": generate_periodic_coupling,
            "reps": syn_periodic["n_repetitions"],
            "n_samples": 1000
        }
    }
    
    results = []
    
    for ds_name, ds_info in datasets.items():
        print(f"Validating dataset: {ds_name}...")
        for cfg in ds_info["configs"]:
            n_reps = ds_info["reps"]
            n_samples = ds_info["n_samples"]
            cfg_name = cfg["name"]
            
            # Ground truth
            gt = cfg.get("pseudo_ground_truth_te", None)
            
            for b_name, estimator in baselines.items():
                estimates = []
                # Compute multiple repetitions
                for rep in range(n_reps):
                    seed = 42 + rep
                    # Generate data
                    if ds_name == "VAR_Linear_Gaussian":
                        x, y, true_te = ds_info["generator"](
                            n_samples, cfg["a"], cfg["b"], cfg["c"], cfg["noise_std"], seed
                        )
                        gt = true_te
                    elif ds_name == "VAR_Nonlinear":
                        x, y = ds_info["generator"](
                            n_samples, cfg["a"], cfg["b"], cfg["c"], cfg["noise_std"], seed
                        )
                    elif ds_name == "Periodic_Coupling":
                        x, y = ds_info["generator"](
                            n_samples, cfg["omega_x"], cfg["omega_y"], cfg["coupling_k"], cfg["noise_std"], seed
                        )
                        
                    # Estimate TE
                    te = estimator.estimate(x, y)
                    estimates.append(te)
                
                estimates = np.array(estimates)
                
                b = bias(estimates, gt)
                m = mse(estimates, gt)
                
                results.append({
                    "Dataset": ds_name,
                    "Config": cfg_name,
                    "Baseline": b_name,
                    "Bias": b,
                    "MSE": m,
                    "N_Repetitions": n_reps
                })
                print(f"  [{cfg_name}] {b_name}: Bias={b:.4f}, MSE={m:.4f}")
                
    df = pd.DataFrame(results)
    print("\n=== BASELINE VALIDATION RESULTS ===")
    print(df.to_string(index=False))
    
    # Save results
    out_dir = base_dir / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "phase_p_baseline_validation.csv", index=False)
    print(f"\nResults saved to {out_dir / 'phase_p_baseline_validation.csv'}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
