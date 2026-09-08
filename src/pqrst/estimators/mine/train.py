"""Vong lap huan luyen MINE: Adam + Donsker-Varadhan loss + shuffle-batch, co
early stopping tren tap validation tach bach. Xem spec day du trong
docs/PHASE_Q_GUIDE.md muc 2-3.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from pqrst.estimators.mine.network import StatisticsNetwork
from pqrst.estimators.mine.losses import donsker_varadhan_loss, shuffle_batch


@dataclass
class TrainConfig:
    """Sieu tham so huan luyen. Doc gia tri thuc te tu configs/mine/smoke_test.yaml,
    KHONG hard-code trong code goi ham train_mine."""

    hidden_dims: list[int] = field(default_factory=lambda: [64, 64])
    learning_rate: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 200
    patience: int = 15          # so epoch lien tiep val loss khong cai thien -> dung
    val_fraction: float = 0.2   # ty le du lieu danh cho validation (tach bach voi train)
    seed: int = 42


@dataclass
class TrainResult:
    """Ket qua tra ve sau khi train xong - dung de danh gia va ve loss curve."""

    model: StatisticsNetwork
    train_loss_history: list[float]
    val_loss_history: list[float]
    final_val_mi_estimate: float  # DV bound danh gia tren tap validation, KHONG phai train


def train_mine(x: np.ndarray, y: np.ndarray, config: TrainConfig) -> TrainResult:
    """Huan luyen 1 StatisticsNetwork T_phi tren du lieu (x, y) da cho, tra ve model
    + lich su loss + uoc luong MI cuoi cung tren tap validation.

    Args:
        x, y: mang 1D hoac 2D (n_samples, dim) - cap mau tu P(X,Y). KHONG phai chuoi
            thoi gian lien tuc can giu thu tu - ham nay coi moi hang la 1 sample doc
            lap, viec dam bao cac sample it tuong quan (vd ghep nhieu realization
            ngan, seed khac nhau) la trach nhiem cua noi goi (script), khong phai
            cua ham nay.
        config: xem TrainConfig.

    Returns:
        TrainResult.

    TODO(ban tu code) - phac thao cac buoc:
        1. torch.manual_seed(config.seed) de tai lap duoc.
        2. Chuyen x, y sang torch.Tensor float32, dam bao shape (n, dim) (unsqueeze
           neu dang la 1D).
        3. Chia train/val theo config.val_fraction (vd dung sklearn.model_selection
           hoac tu chia bang random permutation index - KHONG chia theo thu tu neu
           du lieu la ghep nhieu realization, de tranh leak cau truc).
        4. Khoi tao StatisticsNetwork(input_dim=x.shape[-1]+y.shape[-1],
           hidden_dims=config.hidden_dims), optimizer Adam(lr=config.learning_rate).
        5. Vong lap epoch: voi moi batch (shuffle train set moi epoch, chia thanh cac
           batch kich thuoc batch_size):
             - t_joint = model(x_batch, y_batch)
             - y_shuffled = shuffle_batch(y_batch)
             - t_marginal = model(x_batch, y_shuffled)
             - loss = donsker_varadhan_loss(t_joint, t_marginal)
             - optimizer.zero_grad(); loss.backward(); optimizer.step()
           Cuoi moi epoch: tinh val loss TUONG TU tren toan bo tap validation (khong
           backward, dung torch.no_grad()), luu vao train_loss_history/val_loss_history.
        6. Early stopping: neu val loss khong cai thien sau config.patience epoch lien
           tiep thi dung, giu lai model tai epoch tot nhat (khong phai epoch cuoi).
        7. final_val_mi_estimate: gia tri DV bound (= -loss) tot nhat tren tap
           validation - day chinh la uoc luong MI cuoi cung, dung de so sanh voi
           ground-truth trong script smoke test.
    """
    raise NotImplementedError
