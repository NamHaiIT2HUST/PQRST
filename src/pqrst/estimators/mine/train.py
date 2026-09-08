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
    torch.manual_seed(config.seed)
    
    # 2. Convert to tensors
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)
        
    if x.dim() == 1:
        x = x.unsqueeze(-1)
    if y.dim() == 1:
        y = y.unsqueeze(-1)
        
    n_samples = x.shape[0]
    
    # 3. Train/val split
    perm = torch.randperm(n_samples)
    val_size = int(n_samples * config.val_fraction)
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]
    
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    
    # 4. Initialize model and optimizer
    input_dim = x.shape[-1] + y.shape[-1]
    model = StatisticsNetwork(input_dim=input_dim, hidden_dims=config.hidden_dims)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    train_loss_history = []
    val_loss_history = []
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    # 5. Training loop
    for epoch in range(config.max_epochs):
        model.train()
        
        # Shuffle train data for batches
        train_perm = torch.randperm(x_train.shape[0])
        x_train_shuffled = x_train[train_perm]
        y_train_shuffled = y_train[train_perm]
        
        batch_losses = []
        for i in range(0, x_train.shape[0], config.batch_size):
            x_batch = x_train_shuffled[i:i+config.batch_size]
            y_batch = y_train_shuffled[i:i+config.batch_size]
            
            # Skip very small last batch to maintain reliable bounds
            if x_batch.shape[0] < max(2, config.batch_size // 10):
                continue
                
            t_joint = model(x_batch, y_batch)
            y_shuffled = shuffle_batch(y_batch)
            t_marginal = model(x_batch, y_shuffled)
            
            loss = donsker_varadhan_loss(t_joint, t_marginal)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_losses.append(loss.item())
            
        epoch_train_loss = np.mean(batch_losses)
        train_loss_history.append(epoch_train_loss)
        
        # Validation step
        model.eval()
        with torch.no_grad():
            t_joint_val = model(x_val, y_val)
            y_val_shuffled = shuffle_batch(y_val)
            t_marginal_val = model(x_val, y_val_shuffled)
            val_loss = donsker_varadhan_loss(t_joint_val, t_marginal_val).item()
            
        val_loss_history.append(val_loss)
        
        # 6. Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # clone state dict
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= config.patience:
            break
            
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    # 7. Final val MI estimate
    final_val_mi_estimate = -best_val_loss
    
    return TrainResult(
        model=model,
        train_loss_history=train_loss_history,
        val_loss_history=val_loss_history,
        final_val_mi_estimate=final_val_mi_estimate
    )
