"""Cong cu chan doan khong gian dac trung (PCA theo tung khoi mang) - hinh thuc
hoa tu notebooks/phase_t_04_pca_feature_analysis.ipynb (Pha T.3b) de tai dung
NGUYEN XI cho T_theta o Nhip 2 (xem docs/NHIP2_GUIDE.md muc 2.2) ma khong phai
viet lai.

Model-agnostic bang duck-typing: bat ky model co method
`forward_blocks(y_t, x_lag, y_lag, mask) -> dict[str, torch.Tensor]` deu dung
duoc voi module nay (da co san o MaskedStatisticsNetwork va
FourierFeatureStatisticsNetwork - xem forward_blocks() cua tung class).

3 muc dich da xac nhan huu ich o Pha T.3b:
1. PCA to theo nhan (vd cuong do ghep noi c thuc) - kiem tra mang hoc dung
   cau truc du lieu khong.
2. PCA joint vs marginal - kiem tra mang phan biet dung 2 loai mau ma chan
   duoi Donsker-Varadhan yeu cau.
3. Chieu du lieu MOI (vd du lieu that) vao PCA da fit tren du lieu khac (vd
   synthetic) KHONG fit lai - hinh "domain gap".
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from sklearn.decomposition import PCA

from pqrst.utils.standardize import standardize_window


def windows_to_tensors(
    y_t: np.ndarray, x_lag: np.ndarray, y_lag: np.ndarray, standardize: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Chuyen 1 cua so (3 mang 1D cung do dai) thanh tensor (N,1) + mask_full
    (N,1) toan 1 - dung dinh dang input cho forward()/forward_blocks() cua ca
    MaskedStatisticsNetwork va FourierFeatureStatisticsNetwork."""
    if standardize:
        y_t = standardize_window(y_t)
        x_lag = standardize_window(x_lag)
        y_lag = standardize_window(y_lag)
    N = len(y_t)
    ty_t = torch.tensor(y_t, dtype=torch.float32).unsqueeze(-1)
    tx_lag = torch.tensor(x_lag, dtype=torch.float32).unsqueeze(-1)
    ty_lag = torch.tensor(y_lag, dtype=torch.float32).unsqueeze(-1)
    mask_full = torch.ones((N, 1), dtype=torch.float32)
    return ty_t, tx_lag, ty_lag, mask_full


def collect_block_records(
    model: torch.nn.Module,
    samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, dict]],
    standardize: bool = False,
    include_marginal: bool = True,
    shuffle_seed: int = 0,
) -> list[dict]:
    """Voi moi cua so trong `samples` (list of (y_t, x_lag, y_lag, labels)), chay
    forward_blocks() cua `model` (frozen, torch.no_grad()) tren cap "joint" (dung
    thu tu that) va, neu include_marginal=True, tren cap "marginal" (xao tron
    x_lag,y_lag CUNG mot hoan vi, giong dung cach train_amortized() tinh DV bound
    - xem amortized.py). Trung binh activation qua cua so (moi cua so N mau -> 1
    diem dai dien), tra ve list[dict] voi cac khoa la ten khoi (vd 'block0_input',
    'block1', ...) cong them 'kind' ('joint'/'marginal') va toan bo `labels` da
    truyen vao (vd {'coupling_c': 0.4, 'source': 'synthetic'}).

    labels trong `samples` KHONG bi doc/sua - chi copy nguyen vao record dau ra,
    nen goi co the dua vao bat ky nhan can dung de to mau/loc sau nay (coupling_c,
    record_id, source, ...).
    """
    model.eval()
    rng = torch.Generator().manual_seed(shuffle_seed)
    records: list[dict] = []

    with torch.no_grad():
        for y_t, x_lag, y_lag, labels in samples:
            ty_t, tx_lag, ty_lag, mask_full = windows_to_tensors(y_t, x_lag, y_lag, standardize)
            N = ty_t.shape[0]

            blocks_joint = model.forward_blocks(ty_t, tx_lag, ty_lag, mask_full)
            rec_joint = {k: v.detach().numpy().mean(axis=0) for k, v in blocks_joint.items()}
            rec_joint.update(labels)
            rec_joint["kind"] = "joint"
            records.append(rec_joint)

            if include_marginal:
                perm = torch.randperm(N, generator=rng)
                blocks_marg = model.forward_blocks(ty_t, tx_lag[perm], ty_lag[perm], mask_full)
                rec_marg = {k: v.detach().numpy().mean(axis=0) for k, v in blocks_marg.items()}
                rec_marg.update(labels)
                rec_marg["kind"] = "marginal"
                records.append(rec_marg)

    return records


def fit_pca(
    records: list[dict],
    block_name: str,
    filter_fn: Callable[[dict], bool] | None = None,
    n_components: int = 2,
    random_state: int = 0,
) -> tuple[np.ndarray, PCA, list[dict]]:
    """Fit PCA tren activation cua 1 khoi (`block_name`), tren cac record thoa
    `filter_fn` (vd chi 'kind'=='joint') - mac dinh dung tat ca record. Tra ve
    (diem da chieu (n_records, n_components), doi tuong PCA da fit (de tai dung
    voi project_pca cho du lieu khac), va list record da loc - de lay lai nhan
    (vd coupling_c) theo DUNG thu tu voi diem tra ve)."""
    used = [r for r in records if filter_fn(r)] if filter_fn is not None else list(records)
    X = np.stack([r[block_name] for r in used])
    pca = PCA(n_components=n_components, random_state=random_state)
    Xp = pca.fit_transform(X)
    return Xp, pca, used


def project_pca(pca: PCA, records: list[dict], block_name: str) -> np.ndarray:
    """Chieu activation cua 1 khoi tu `records` (thuong la 1 nguon du lieu KHAC -
    vd du lieu that) vao khong gian PCA DA FIT SAN (`pca`, tu fit_pca tren du
    lieu khac, vd synthetic) - KHONG fit lai. Dung de ve hinh domain-gap."""
    X = np.stack([r[block_name] for r in records])
    return pca.transform(X)
