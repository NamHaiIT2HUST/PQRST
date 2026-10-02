"""Pha U - thich nghi mien (domain adaptation) khong nhan cho T_phi.

Y tuong: chan duoi Donsker-Varadhan KHONG can nhan TE that (chi can cap joint va
marginal xao tron trong tung cua so), nen co the tinh chinh mang da train tren du
lieu mo phong bang CHINH cac cua so du lieu that - khong can biet dap an.

QUY TAC CHONG ROI RI (bat buoc): thich nghi chi tren cua so cua cac BAN GHI TRAIN;
danh gia tren cac ban ghi GIU LAI (chia theo ban ghi, KHONG theo cua so - cua so
cung ban ghi tuong quan manh). Xem docs/FLOW_TRIEN_KHAI_PHA_U.md buoc S1.
"""
from __future__ import annotations

import copy
import dataclasses

import numpy as np

from pqrst.estimators.mine.amortized import (
    AmortizedTEEstimator,
    train_amortized,
)


def make_record_folds(record_ids: list[str], k: int, seed: int = 0) -> list[list[str]]:
    """Chia danh sach ban ghi thanh k nhom (fold) ngau nhien, co dinh theo seed,
    moi ban ghi thuoc dung 1 fold."""
    if k < 2 or k > len(record_ids):
        raise ValueError("k phai trong [2, so ban ghi]")
    rng = np.random.default_rng(seed)
    ids = list(record_ids)
    rng.shuffle(ids)
    return [ids[i::k] for i in range(k)]


def adapt_amortized(
    base: AmortizedTEEstimator,
    adapt_windows: list,
    val_windows: list,
    learning_rate: float = 1e-4,
    max_epochs: int = 5,
    windows_per_batch: int = 32,
    final_estimate_last_k_epochs: int = 2,
    seed: int = 42,
) -> tuple[AmortizedTEEstimator, dict]:
    """Tinh chinh 1 ban SAO cua `base` (khong sua base) bang DV loss tren
    `adapt_windows`; `val_windows` chi dung theo doi/dung som (PHAI khac cac ban
    ghi giu lai de danh gia). Dung lai train_amortized (model_factory nap trong so
    cua base) nen loss, shuffle-trong-cua-so, weight averaging deu giong het luc
    train goc."""
    cfg = dataclasses.replace(
        base.config,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        patience=max_epochs,
        windows_per_batch=windows_per_batch,
        final_estimate_last_k_epochs=final_estimate_last_k_epochs,
        seed=seed,
    )
    base_model = base.model

    def factory():
        return copy.deepcopy(base_model)

    return train_amortized(adapt_windows, val_windows, cfg, model_factory=factory)
