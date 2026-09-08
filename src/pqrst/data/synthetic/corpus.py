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
    raise NotImplementedError


def split_corpus_by_seed(
    corpus: list[Window], val_fraction: float, split_seed: int
) -> tuple[list[Window], list[Window]]:
    """Chia corpus thanh train/validation TACH BACH theo cua so (khong theo mau).

    QUAN TRONG: phai chia theo CUA SO nguyen ven, khong duoc tron mau cua cung 1 cua so
    vao ca 2 tap - neu khong se ro ri thong tin (leakage) va lam ket qua danh gia lac
    quan gia tao.

    Args:
        corpus: toan bo corpus.
        val_fraction: ty le cua so danh cho validation.
        split_seed: seed cho phep hoan vi, de tai lap.

    Returns:
        (train_windows, val_windows).

    TODO(ban tu code):
        - Hoan vi ngau nhien chi so cua so (np.random.default_rng(split_seed)), cat
          theo val_fraction.
        - Kiem tra ca 2 tap deu con day du cac o luoi (moi (c, noise, N) deu xuat hien
          o ca train lan val) - neu khong, canh bao; vi neu 1 o luoi chi nam o val thi
          do la bai toan "generalize sang cau hinh moi", khac voi bai toan dang danh gia.
    """
    raise NotImplementedError


def save_corpus(corpus: list[Window], path: str) -> None:
    """Luu corpus ra dia (de notebook sau tai lai ma khong phai sinh lai).

    TODO(ban tu code):
        - Dinh dang de xuat: 1 file .npz chua cac mang da xep chong + 1 file .json
          chua metadata (config_name, params, ground truths, seed) theo dung thu tu.
          Hoac dung pickle neu don gian hon - nhung .npz + .json de audit hon.
        - Ghi ro so cua so, tong so mau, va cac o luoi vao metadata de kiem tra lai.
    """
    raise NotImplementedError


def load_corpus(path: str) -> list[Window]:
    """Doc lai corpus da luu boi save_corpus.

    TODO(ban tu code): dao nguoc chinh xac logic cua save_corpus.
    """
    raise NotImplementedError
