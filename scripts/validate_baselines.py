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


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
