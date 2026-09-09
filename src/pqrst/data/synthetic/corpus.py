"""Sinh kho ngu lieu tong hop (corpus) da cau hinh cho Pha R.

Khac biet cot loi so voi Pha Q: Pha Q train tren DUNG 1 cau hinh; Pha R train tren
NHIEU cau hinh tron lan (coupling khac nhau, noise khac nhau, N khac nhau) de mang
T_phi hoc duoc mot ham ti-so-mat-do TONG QUAT, ap dung duoc cho cua so MOI ma khong
can train lai. Day chinh la y nghia cua chu "amortized" trong ten de tai (AQNE-TE) -
xem docs/PHASE_R_GUIDE.md muc 1.

Mot "cua so" (window) = 1 doan ngan gom N mau lien tiep tu 1 realization, kem metadata
(cau hinh sinh ra no + ground-truth TE/MI da biet). Corpus = tap hop nhieu cua so.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Window:
    """1 cua so du lieu + metadata day du de danh gia sau nay.

    Attributes:
        y_t:      Y[t],     shape (N,) - bien dich tai thoi diem t
        x_lag:    X[t-1],   shape (N,) - bien nguon tre 1 buoc
        y_lag:    Y[t-1],   shape (N,) - bien dich tre 1 buoc (bien DIEU KIEN cho TE)
        n_samples: N (do dai cua so, = len(y_t))
        config_name: ten cau hinh sinh ra cua so nay (vd "coupling_0.6_noise_0.5")
        params: dict tham so day du (a, b, c, noise_std) de tai lap
        te_ground_truth: TE(X->Y) that, cong thuc dong (chi co voi VAR tuyen tinh)
        mi_full_ground_truth: I(Y[t]; X[t-1], Y[t-1]) that
        mi_reduced_ground_truth: I(Y[t]; Y[t-1]) that
        seed: seed da dung, de tai lap
    """

    y_t: np.ndarray
    x_lag: np.ndarray
    y_lag: np.ndarray
    n_samples: int
    config_name: str
    params: dict
    te_ground_truth: float
    mi_full_ground_truth: float
    mi_reduced_ground_truth: float
    seed: int


def generate_corpus(
    coupling_values: list[float],
    noise_values: list[float],
    n_values: list[int],
    n_windows_per_cell: int,
    a: float,
    b: float,
    base_seed: int,
) -> list[Window]:
    """Sinh corpus quet toan bo luoi (coupling x noise x N).

    Args:
        coupling_values: danh sach gia tri c (cuong do ghep noi) can quet.
        noise_values: danh sach gia tri noise_std can quet.
        n_values: danh sach do dai cua so N can quet (vd [10,20,30,50,100,200]).
        n_windows_per_cell: so cua so sinh cho MOI o luoi (moi cua so 1 seed khac nhau).
        a, b: he so tu-hoi-quy, giu co dinh trong Pha R (chi quet c va noise).
        base_seed: seed goc; seed cua tung cua so PHAI duy nhat va tai lap duoc.

    Returns:
        List[Window], do dai = len(coupling)*len(noise)*len(n)*n_windows_per_cell.

    TODO(ban tu code):
        - Voi moi o luoi (c, noise_std, N), lap n_windows_per_cell lan:
            * Tinh seed duy nhat: PHAI la ham don anh (injective) cua (chi so o luoi,
              chi so lap) - vd dung 1 bo dem tang dan tu base_seed. TUYET DOI KHONG
              dung cung 1 seed cho 2 cua so khac nhau (se tao ban sao trung lap, lam
              hong tinh doc lap cua tap danh gia).
            * Goi generate_var_linear_gaussian(n_samples=N+1, a, b, c, noise_std, seed)
              - luu y sinh N+1 mau vi sau khi lay lag se con dung N mau.
            * Trich y_t = y[1:], x_lag = x[:-1], y_lag = y[:-1] (cung do dai N).
            * Tinh ground-truth bang compute_var_linear_ground_truths(a,b,c,noise_std)
              (xem var_linear_gaussian.py) - CHI phu thuoc tham so, khong phu thuoc N,
              nen tinh 1 lan cho moi o luoi roi tai dung, dung tinh lai moi cua so.
        - Tra ve list[Window].
    """
    windows = []
    from pqrst.data.synthetic.var_linear_gaussian import generate_var_linear_gaussian, compute_var_linear_ground_truths
    
    seed_counter = base_seed
    
    for c in coupling_values:
        for noise_std in noise_values:
            # Tinh ground truth 1 lan cho moi cau hinh
            gt_dict = compute_var_linear_ground_truths(a, b, c, noise_std)
            te_gt, mi_full_gt, mi_reduced_gt = gt_dict["te"], gt_dict["mi_full"], gt_dict["mi_reduced"]
            config_name = f"c_{c}_noise_{noise_std}"
            
            for N in n_values:
                for _ in range(n_windows_per_cell):
                    current_seed = seed_counter
                    seed_counter += 1
                    
                    x, y, _ = generate_var_linear_gaussian(
                        n_samples=N+1,
                        a=a, b=b, c=c, noise_std=noise_std, seed=current_seed
                    )
                    
                    y_t = y[1:]
                    x_lag = x[:-1]
                    y_lag = y[:-1]
                    
                    win = Window(
                        y_t=y_t,
                        x_lag=x_lag,
                        y_lag=y_lag,
                        n_samples=N,
                        config_name=config_name,
                        params={"a": a, "b": b, "c": c, "noise_std": noise_std},
                        te_ground_truth=te_gt,
                        mi_full_ground_truth=mi_full_gt,
                        mi_reduced_ground_truth=mi_reduced_gt,
                        seed=current_seed
                    )
                    windows.append(win)
                    
    return windows

def split_corpus_by_seed(
    corpus: list[Window], val_fraction: float, split_seed: int
) -> tuple[list[Window], list[Window]]:
    rng = np.random.default_rng(split_seed)
    indices = rng.permutation(len(corpus))
    
    val_size = int(len(corpus) * val_fraction)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    
    val_windows = [corpus[i] for i in val_idx]
    train_windows = [corpus[i] for i in train_idx]
    
    def get_cells(windows):
        return set((w.config_name, w.n_samples) for w in windows)
        
    train_cells = get_cells(train_windows)
    val_cells = get_cells(val_windows)
    
    missing_in_val = train_cells - val_cells
    if missing_in_val:
        print(f"WARNING: val missing cells: {missing_in_val}")
        
    missing_in_train = val_cells - train_cells
    if missing_in_train:
        print(f"WARNING: train missing cells: {missing_in_train}")
        
    return train_windows, val_windows

def save_corpus(corpus: list[Window], path: str) -> None:
    import json
    from pathlib import Path
    
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    
    y_t_list, x_lag_list, y_lag_list = [], [], []
    metadata = []
    
    for i, w in enumerate(corpus):
        y_t_list.append(w.y_t)
        x_lag_list.append(w.x_lag)
        y_lag_list.append(w.y_lag)
        
        metadata.append({
            "idx": i,
            "n_samples": w.n_samples,
            "config_name": w.config_name,
            "params": w.params,
            "te_ground_truth": w.te_ground_truth,
            "mi_full_ground_truth": w.mi_full_ground_truth,
            "mi_reduced_ground_truth": w.mi_reduced_ground_truth,
            "seed": w.seed
        })
        
    npz_path = p.with_suffix(".npz")
    json_path = p.with_suffix(".json")
    
    # Store objects with ragged shape using object arrays
    np.savez_compressed(
        npz_path, 
        y_t=np.array(y_t_list, dtype=object), 
        x_lag=np.array(x_lag_list, dtype=object), 
        y_lag=np.array(y_lag_list, dtype=object)
    )
    
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

def load_corpus(path: str) -> list[Window]:
    import json
    from pathlib import Path
    
    p = Path(path)
    npz_path = p.with_suffix(".npz")
    json_path = p.with_suffix(".json")
    
    with np.load(npz_path, allow_pickle=True) as data:
        y_t_list = data["y_t"]
        x_lag_list = data["x_lag"]
        y_lag_list = data["y_lag"]
        
    with open(json_path, "r") as f:
        metadata = json.load(f)
        
    windows = []
    for i, m in enumerate(metadata):
        w = Window(
            y_t=y_t_list[i].astype(float),
            x_lag=x_lag_list[i].astype(float),
            y_lag=y_lag_list[i].astype(float),
            n_samples=m["n_samples"],
            config_name=m["config_name"],
            params=m["params"],
            te_ground_truth=m["te_ground_truth"],
            mi_full_ground_truth=m["mi_full_ground_truth"],
            mi_reduced_ground_truth=m["mi_reduced_ground_truth"],
            seed=m["seed"]
        )
        windows.append(w)
        
    return windows
