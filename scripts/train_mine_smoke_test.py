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


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
