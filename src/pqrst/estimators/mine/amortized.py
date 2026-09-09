"""Bo uoc luong TE AMORTIZED: train 1 lan tren corpus da cau hinh, roi DONG BANG va
ap dung cho cua so moi ma khong train lai. Day la dong gop khoa hoc chinh cua de tai.

Vi sao amortized co the thang KSG o N nho (luan diem cot loi cua ca du an):
KSG phai uoc luong lai tu dau tren tung cua so - voi N < 30 diem thi gan nhu khong du
thong tin, nen phuong sai rat lon. Nguoc lai, T_phi da hoc mot ham ti-so-mat-do TONG
QUAT tu hang chuc nghin cua so o buoc train; khi gap cua so ngan moi, no chi can ap
dung tri thuc da hoc thay vi suy ra tu 20 diem du lieu. Do la "kien thuc tien nghiem
hoc duoc" (learned prior) ma phuong phap phi tham so nhu KSG khong co.

Diem noi kien truc quan trong: class duoi day ke thua BaseTEEstimator (dinh nghia tu
Pha P) - nho vay no cam thang vao dung ma so sanh da co, khong phai sua code danh gia.
Nhip 2 (luong tu) se them 1 class tuong tu voi T_theta thay cho T_phi.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from pqrst.baselines.base import BaseTEEstimator


class MaskedStatisticsNetwork(nn.Module):
    """T_phi dung chung cho CA 2 so hang cua phan ra conditional MI.

    Input = concat([y_t, x_lag, y_lag, mask]) voi mask la co nhi phan bao che do:
        mask = 1.0 -> che do "full":    dung ca x_lag va y_lag
        mask = 0.0 -> che do "reduced": x_lag bi zero-out, chi dung y_lag
    => input_dim = 1 (y_t) + 1 (x_lag) + 1 (y_lag) + 1 (mask) = 4

    Vi sao dung chung 1 mang thay vi 2 mang rieng:
        1. Dung y roadmap goc ("dung chung T_phi").
        2. Giam phuong sai cua hieu 2 so hang (sai so tuong quan duong, triet tieu bot).
        3. Nhip 2 chi phai hoan doi DUNG 1 mang sang mach luong tu, khong phai 2.

    PHUONG AN DU PHONG neu train khong on dinh: dung 2 mang StatisticsNetwork rieng
    (input_dim=3 cho full, input_dim=2 cho reduced). Neu phai dung phuong an nay, GHI RO
    trong bao cao va do lai phuong sai TE de so sanh voi phuong an dung chung.

    TODO(ban tu code):
        - Tai su dung y het cach dung MLP cua StatisticsNetwork (network.py): cac lop
          Linear + ELU, lop cuoi Linear(...,1) khong activation, forward tra ve shape
          (batch,) sau squeeze(-1).
        - forward(y_t, x_lag, y_lag, mask): zero-out x_lag o dau mask==0 TRUOC khi
          concat (nhan element-wise voi mask), roi noi ca mask vao lam dac trung.
    """

    def __init__(self, hidden_dims: list[int] | None = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128, 64]
            
        # Input dim is 4: y_t, x_lag, y_lag, mask
        in_dim = 4
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ELU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        y_t: torch.Tensor,
        x_lag: torch.Tensor,
        y_lag: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # Zero-out x_lag where mask is 0
        x_lag_masked = x_lag * mask
        
        xy = torch.cat([y_t, x_lag_masked, y_lag, mask], dim=-1)
        out = self.net(xy)
        return out.squeeze(-1)


@dataclass
class AmortizedTrainConfig:
    """Sieu tham so train amortized. Doc tu configs/mine/amortized.yaml."""

    hidden_dims: list[int] = field(default_factory=lambda: [128, 128, 64])
    learning_rate: float = 1e-3
    windows_per_batch: int = 32     # so CUA SO moi buoc cap nhat (khong phai so mau)
    max_epochs: int = 100
    patience: int = 15
    eval_n_shuffles: int = 20       # so lan shuffle khi danh gia (KHONG dung 1 - xem Pha Q)
    final_estimate_last_k_epochs: int = 10  # trung binh k epoch cuoi, KHONG "best-of-all"
    seed: int = 42


class AmortizedTEEstimator(BaseTEEstimator):
    """TE(X->Y) uoc luong bang mang T_phi da train san tren corpus (frozen khi dung).

    Cach dung (sau khi da train xong va nap checkpoint):
        est = AmortizedTEEstimator.load("results/checkpoints/phi_amortized.pt")
        te = est.estimate(x, y)   # cung chu ky voi KSG/binning/symbolic tu Pha P

    TODO(ban tu code):
        - __init__(self, model, config): luu model (da eval()), config.
        - estimate(self, x, y, **kwargs): tu x, y (2 chuoi thoi gian 1D cung do dai),
          trich y_t=y[1:], x_lag=x[:-1], y_lag=y[:-1], roi goi
          estimate_te_from_window(...) tu conditional.py, tra ve khoa "te".
          -> Nho vay class nay TUONG THICH HOAN TOAN voi ma so sanh da co tu Pha P.
        - save(self, path) / load(cls, path): luu+nap state_dict cua model KEM
          config (de biet hidden_dims ma dung lai dung kien truc). Dung torch.save voi
          1 dict {"state_dict":..., "config":...}.
    """

    def __init__(self, model: MaskedStatisticsNetwork, config: AmortizedTrainConfig):
        self.model = model
        self.config = config
        self.model.eval()

    def estimate(self, x: np.ndarray, y: np.ndarray, **kwargs) -> float:
        from pqrst.estimators.mine.conditional import estimate_te_from_window
        y_t = y[1:]
        x_lag = x[:-1]
        y_lag = y[:-1]
        
        seed = kwargs.get("seed", 42)
        n_shuffles = self.config.eval_n_shuffles
        
        res = estimate_te_from_window(
            model=self.model,
            y_t=y_t,
            x_lag=x_lag,
            y_lag=y_lag,
            n_shuffles=n_shuffles,
            seed=seed
        )
        return res["te"]

    def save(self, path: str) -> None:
        import dataclasses
        torch.save({
            "state_dict": self.model.state_dict(),
            "config": dataclasses.asdict(self.config)
        }, path)

    @classmethod
    def load(cls, path: str) -> "AmortizedTEEstimator":
        data = torch.load(path, weights_only=True)
        config = AmortizedTrainConfig(**data["config"])
        model = MaskedStatisticsNetwork(config.hidden_dims)
        model.load_state_dict(data["state_dict"])
        return cls(model, config)


def train_amortized(
    train_windows: list,
    val_windows: list,
    config: AmortizedTrainConfig,
) -> tuple[AmortizedTEEstimator, dict]:
    """Train T_phi tren corpus da cau hinh (danh sach Window tu corpus.py).

    KHAC BIET COT LOI so voi train_mine cua Pha Q:
        - Pha Q: 1 batch = mot mo mau ROI RAC lay tu 1 cau hinh duy nhat.
        - Pha R: 1 batch = nhieu CUA SO tron tu NHIEU cau hinh khac nhau; DV bound duoc
          tinh RIENG TRONG TUNG CUA SO roi lay trung binh cac cua so lam loss.
          Ly do: shuffle-batch phai dien ra TRONG cung 1 cua so (cung 1 phan phoi), neu
          tron mau tu nhieu cau hinh roi shuffle chung thi so hang marginal se sai -
          DAY LA LOI DE MAC NHAT CUA CA PHA R, doc ky truoc khi code.

    Args:
        train_windows, val_windows: list[Window] tu split_corpus_by_seed.
        config: xem AmortizedTrainConfig.

    Returns:
        (estimator, history) voi history chua train_loss_history, val_loss_history,
        va cac chi so chan doan can cho bao cao.

    TODO(ban tu code) - phac thao:
        1. torch.manual_seed(config.seed).
        2. Khoi tao MaskedStatisticsNetwork(config.hidden_dims) + Adam.
        3. Moi epoch: xao tron thu tu train_windows, chia thanh nhom
           config.windows_per_batch cua so. Voi moi nhom:
             - Voi TUNG cua so trong nhom: tinh DV bound cho ca 2 che do (full va
               reduced) - shuffle TRONG cua so do.
             - loss cua nhom = -(trung binh cac DV bound cua ca 2 che do tren cac cua so).
               (Train CA 2 che do cung luc de mang hoc tot ca 2 so hang.)
             - backward + step.
        4. Cuoi moi epoch: danh gia tren val_windows (torch.no_grad), dung
           config.eval_n_shuffles lan shuffle - KHONG dung 1 lan.
        5. Early stopping theo val loss voi config.patience.
        6. MODEL DEPLOY CUOI CUNG = TRUNG BINH THAM SO (weight averaging) cua
           config.final_estimate_last_k_epochs epoch CUOI DA CHAY (khong phai epoch
           co val loss thap nhat trong toan bo qua trinh). Day la sua loi
           selection-bias da phat hien khi review ca Pha Q va lan review dau cua Pha R
           (xem docs/PHASE_Q_REPORT.md muc 4 va docs/PHASE_R_REPORT.md muc 4) - dung
           quay lai kieu "best-of-all-epochs".
    """
    torch.manual_seed(config.seed)

    model = MaskedStatisticsNetwork(hidden_dims=config.hidden_dims)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    from collections import deque
    from pqrst.estimators.mine.losses import shuffle_batch, donsker_varadhan_loss

    def prep_window(w):
        ty_t = torch.tensor(w.y_t, dtype=torch.float32).unsqueeze(-1)
        tx_lag = torch.tensor(w.x_lag, dtype=torch.float32).unsqueeze(-1)
        ty_lag = torch.tensor(w.y_lag, dtype=torch.float32).unsqueeze(-1)
        return ty_t, tx_lag, ty_lag

    train_loss_history = []
    val_loss_history = []

    best_val_loss = float('inf')
    patience_counter = 0

    # Sua loi selection-bias phat hien khi review (best-of-all-epochs khong khop spec):
    # thay vi luu 1 state_dict "tot nhat", giu 1 buffer k state_dict CUA k EPOCH GAN
    # NHAT (khong quan tam epoch nao "tot nhat"), roi TRUNG BINH THAM SO (weight
    # averaging) cua ca k epoch do de lam model deploy cuoi cung. Early stopping
    # (best_val_loss/patience) van dung de quyet dinh DUNG LUC NAO, nhung khong con
    # quyet dinh model nao duoc deploy.
    k_tail = config.final_estimate_last_k_epochs
    tail_state_dicts = deque(maxlen=k_tail)
    
    # Eval logic to avoid duplication
    def evaluate(windows, n_shuffles):
        model.eval()
        losses = []
        with torch.no_grad():
            rng = torch.Generator().manual_seed(config.seed)
            for w in windows:
                ty_t, tx_lag, ty_lag = prep_window(w)
                N = ty_t.shape[0]
                
                mask_full = torch.ones((N, 1), dtype=torch.float32)
                mask_red = torch.zeros((N, 1), dtype=torch.float32)
                
                for mask in (mask_full, mask_red):
                    t_joint = model(ty_t, tx_lag, ty_lag, mask)
                    
                    lmes = []
                    for _ in range(n_shuffles):
                        perm = torch.randperm(N, generator=rng)
                        t_marg = model(ty_t, tx_lag[perm], ty_lag[perm], mask)
                        lmes.append(torch.logsumexp(t_marg, dim=0) - np.log(N))
                        
                    loss_w = t_joint.mean() - torch.stack(lmes).mean()
                    losses.append(-loss_w.item())
        return np.mean(losses)
        
    for epoch in range(config.max_epochs):
        model.train()
        rng = np.random.default_rng(config.seed + epoch)
        perm = rng.permutation(len(train_windows))
        
        batch_losses = []
        for i in range(0, len(train_windows), config.windows_per_batch):
            idx = perm[i:i+config.windows_per_batch]
            
            optimizer.zero_grad()
            group_losses = []
            
            for j in idx:
                w = train_windows[j]
                ty_t, tx_lag, ty_lag = prep_window(w)
                N = ty_t.shape[0]
                
                mask_full = torch.ones((N, 1), dtype=torch.float32)
                mask_red = torch.zeros((N, 1), dtype=torch.float32)
                
                for mask in (mask_full, mask_red):
                    t_joint = model(ty_t, tx_lag, ty_lag, mask)
                    
                    # 1 shuffle for train is standard, or config.eval_n_shuffles?
                    # The prompt says "danh gia tren val_windows... KHONG dung 1 lan shuffle". For train, 1 is fine to keep it fast, but let's just use 1.
                    perm_idx = torch.randperm(N)
                    t_marg = model(ty_t, tx_lag[perm_idx], ty_lag[perm_idx], mask)
                    
                    loss_w = donsker_varadhan_loss(t_joint, t_marg)
                    group_losses.append(loss_w)
                    
            if not group_losses:
                continue
                
            loss = torch.stack(group_losses).mean()
            loss.backward()
            optimizer.step()
            
            batch_losses.append(loss.item())
            
        train_loss_history.append(np.mean(batch_losses))
        
        val_loss = evaluate(val_windows, config.eval_n_shuffles)
        val_loss_history.append(val_loss)

        tail_state_dicts.append({k: v.cpu().clone() for k, v in model.state_dict().items()})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            break

    # Deploy = TRUNG BINH THAM SO cua k epoch cuoi da chay (khong phai best-of-all-epochs).
    # Neu train dung < k_tail epoch (vd fail som), dung tat ca epoch da co.
    averaged_state = {}
    for key in tail_state_dicts[0].keys():
        stacked = torch.stack([sd[key].float() for sd in tail_state_dicts], dim=0)
        averaged_state[key] = stacked.mean(dim=0)
    model.load_state_dict(averaged_state)

    k = config.final_estimate_last_k_epochs
    final_val_loss = np.mean(val_loss_history[-k:]) if len(val_loss_history) >= k else np.mean(val_loss_history)

    estimator = AmortizedTEEstimator(model, config)
    history = {
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "final_val_loss": final_val_loss,
        "n_epochs_averaged": len(tail_state_dicts),
    }

    return estimator, history
