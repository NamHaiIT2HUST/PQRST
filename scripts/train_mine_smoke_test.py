"""Smoke test cho MINE (Pha Q): train T_phi tren bai toan MI(X[t-1];Y[t]) don gian
nhat co the (VAR tuyen tinh Gaussian, 1 cau hinh, khong quet luoi), so voi ground-truth
dang cong thuc dong, ve loss curve. Day la tieu chi thoat chinh cua Pha Q - xem
docs/PHASE_Q_GUIDE.md muc 7.

Cach chay (sau khi implement xong cac stub trong src/pqrst/estimators/mine/):
    python scripts/train_mine_smoke_test.py

TODO(ban tu code) - phac thao cac buoc:
    1. Doc configs/mine/smoke_test.yaml (PyYAML).
    2. Sinh du lieu: goi generate_var_linear_gaussian() N_REALIZATIONS lan, moi lan
       seed = base_seed + i, n_samples = n_samples_per_realization. LAY (x[t-1], y[t])
       tu MOI realization (bo phan tu dau/cuoi cho khop chi so) roi GHEP LAI thanh 1
       mang lon (khong dung 1 chuoi lien tuc dai - ghep nhieu realization ngan giup
       giam tuong quan trong batch, xem PHASE_Q_GUIDE.md muc 4).
    3. Tinh mi_ground_truth = compute_lag1_mi_ground_truth(a, b, c, noise_std) (mot
       lan, khong phu thuoc N).
    4. Goi train_mine(x_pool, y_pool, TrainConfig(**cfg["train"])) -> TrainResult.
    5. In ra: final_val_mi_estimate, mi_ground_truth, sai so tuong doi. So sanh voi
       acceptance_relative_error trong config, in PASS/FAIL ro rang (khong chi in so).
    6. Ve loss curve (train_loss_history va val_loss_history tren cung 1 hinh, truc x
       = epoch) bang matplotlib, luu results/figures/phase_q_mine_smoke_test_loss.png.
    7. Luu ket qua so (mi_estimate, mi_ground_truth, sai so, config dung) ra
       results/tables/phase_q_mine_smoke_test.json hoac .csv de audit lai duoc.
"""

from __future__ import annotations


import yaml
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from pqrst.data.synthetic.var_linear_gaussian import generate_var_linear_gaussian, compute_lag1_mi_ground_truth
from pqrst.estimators.mine.train import train_mine, TrainConfig

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main() -> None:
    base_dir = Path(__file__).parent.parent
    config_path = base_dir / "configs" / "mine" / "smoke_test.yaml"
    cfg = load_yaml(config_path)
    
    data_cfg = cfg["data"]
    n_realizations = data_cfg["n_realizations"]
    n_samples_per_realization = data_cfg["n_samples_per_realization"]
    base_seed = data_cfg["base_seed"]
    a, b, c, noise_std = data_cfg["a"], data_cfg["b"], data_cfg["c"], data_cfg["noise_std"]
    
    x_pool = []
    y_pool = []
    
    for i in range(n_realizations):
        seed = base_seed + i
        x, y, _ = generate_var_linear_gaussian(
            n_samples=n_samples_per_realization,
            a=a, b=b, c=c, noise_std=noise_std, seed=seed
        )
        # Lay x[t-1] va y[t]
        x_lag = x[:-1]
        y_now = y[1:]
        x_pool.append(x_lag)
        y_pool.append(y_now)
        
    x_pool = np.concatenate(x_pool, axis=0)
    y_pool = np.concatenate(y_pool, axis=0)
    
    mi_ground_truth = compute_lag1_mi_ground_truth(a, b, c, noise_std)
    
    train_cfg = TrainConfig(**cfg["train"])
    result = train_mine(x_pool, y_pool, train_cfg)
    
    mi_estimate = result.final_val_mi_estimate
    rel_error = abs(mi_estimate - mi_ground_truth) / mi_ground_truth
    
    print(f"MI Estimate (Validation): {mi_estimate:.4f}")
    print(f"MI Ground Truth:         {mi_ground_truth:.4f}")
    print(f"Relative Error:          {rel_error:.2%}")
    
    acc_threshold = cfg["acceptance_relative_error"]
    if rel_error < acc_threshold:
        print(f"RESULT: PASS (error < {acc_threshold:.0%})")
    else:
        print(f"RESULT: FAIL (error >= {acc_threshold:.0%})")
        
    # Plot loss curve
    out_dir_fig = base_dir / "results" / "figures"
    out_dir_fig.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(result.train_loss_history, label='Train Loss')
    plt.plot(result.val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Donsker-Varadhan Loss')
    plt.title('MINE Smoke Test Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(out_dir_fig / "phase_q_mine_smoke_test_loss.png")
    
    # Save results
    out_dir_tab = base_dir / "results" / "tables"
    out_dir_tab.mkdir(parents=True, exist_ok=True)
    
    res_dict = {
        "mi_estimate": float(mi_estimate),
        "mi_ground_truth": float(mi_ground_truth),
        "relative_error": float(rel_error),
        "config": cfg,
        "pass": bool(rel_error < acc_threshold)
    }
    
    with open(out_dir_tab / "phase_q_mine_smoke_test.json", "w") as f:
        json.dump(res_dict, f, indent=4)

if __name__ == "__main__":
    main()
