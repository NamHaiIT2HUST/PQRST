# Pha Q — Hướng dẫn chi tiết (Nhịp 1: MINE cổ điển)

Mục tiêu pha này (nhắc lại từ roadmap): **vòng lặp huấn luyện mạng thống kê `T_φ` qua bound Donsker–Varadhan chạy đúng**, trên đúng 1 cấu hình tổng hợp đơn giản nhất — chưa quét lưới, chưa mở rộng sang conditional MI/TE (đó là Pha R). Theo mentor, đây là bước **ít rủi ro nhất dự án** vì công thức MINE (Belghazi et al., 2018) đã được cộng đồng dùng và kiểm chứng rộng rãi từ 2018.

Input từ Pha P: bộ sinh `generate_var_linear_gaussian` và interface `BaseTEEstimator` (Pha Q chưa dùng `BaseTEEstimator` — `T_φ` sẽ được bọc vào interface đó ở Pha R, khi cần so sánh trực tiếp với 3 baseline).

---

## 0. Cấu trúc thư mục đã dựng sẵn cho Pha Q

```
src/pqrst/estimators/mine/
├── network.py     # StatisticsNetwork (T_phi) - STUB
├── losses.py      # donsker_varadhan_loss, shuffle_batch - STUB
├── train.py        # TrainConfig, TrainResult, train_mine() - STUB
└── ema.py          # EMA bias correction - STUB, TUY CHON (khong bat buoc)
configs/mine/
└── smoke_test.yaml # sieu tham so + cau hinh du lieu cho bai toan smoke test
scripts/
└── train_mine_smoke_test.py  # entry point chinh - STUB
tests/
└── test_mine.py     # 7 test skeleton (pytest.skip) - STUB
```

Đã bổ sung thêm 1 hàm vào `src/pqrst/data/synthetic/var_linear_gaussian.py`:
`compute_lag1_mi_ground_truth(a, b, c, noise_std)` — ground-truth cho bài toán smoke test (xem mục 4).

Tất cả các file trên là **stub**: docstring có spec/công thức/chữ ký hàm đầy đủ, thân hàm `raise NotImplementedError`. Đó là phần bạn code.

---

## 1. Kiến trúc mạng `T_φ` (`network.py`)

`T_φ: R^dx × R^dy → R`, MLP nhận `(x, y)` đã nối (concat) làm input, trả về 1 số vô hướng mỗi sample.

**Điểm thiết kế quan trọng:** `input_dim` phải **cấu hình được**, không hard-code `= 2`. Lý do: ở Pha R, mạng này sẽ được dùng lại cho conditional MI (input thêm biến điều kiện Z, ví dụ `Y[t-1]`), nên interface cần đủ tổng quát ngay từ bây giờ để không phải viết lại kiến trúc mạng ở Pha R.

Spec đề xuất cho bài toán smoke test (Pha Q, input_dim=2):
- `hidden_dims = [64, 64]`
- activation `ELU` (khuyến nghị phổ biến trong các implementation MINE công khai — tránh dead neuron tốt hơn ReLU khi gradient truyền qua `logsumexp`)
- Lớp cuối: `nn.Linear(hidden_dims[-1], 1)`, **không** activation (output là logit thô, không bị chặn khoảng giá trị)

## 2. Loss Donsker–Varadhan + shuffle-batch (`losses.py`)

```
DV(T) = E_{P(X,Y)}[T(x,y)] - log( E_{P(X)×P(Y)}[exp(T(x,y))] )
```

- Mẫu "joint" = `(x, y)` thật trong batch.
- Mẫu "marginal" = `(x, shuffle_batch(y))` — **hoán vị `y` theo batch dimension**, giữ nguyên `x`. Đây là kỹ thuật chuẩn của MINE, không cần lấy mẫu thật từ phân phối marginal.
- **Ổn định số học:** dùng `torch.logsumexp(t_marginal, dim=0) - log(batch_size)` thay vì `log(mean(exp(t_marginal)))` trực tiếp — tránh tràn số/underflow khi `T` lớn.
- Loss cần tối thiểu hoá = `-DV(T)` (vì ta muốn *maximize* cận dưới DV, nhưng optimizer chuẩn là minimize).

## 3. Vòng lặp huấn luyện (`train.py`)

- Optimizer Adam, `lr` mặc định `1e-3` (đọc từ config, không hard-code).
- Chia **train/validation tách bạch** (`val_fraction` trong config) — **early stopping dựa trên val loss**, không dựa trên train loss (tránh overfit).
- Giữ lại model tại epoch có val loss tốt nhất, không phải epoch cuối cùng.
- `final_val_mi_estimate` = giá trị DV bound tốt nhất **trên tập validation** (không phải train) — đây là số dùng để so với ground-truth, vì đánh giá trên chính tập train sẽ lạc quan (optimistic) do mạng học "nhớ" batch.

## 4. Bài toán smoke test cụ thể — vì sao không dùng luôn TE

Roadmap Pha R mới là lúc "mở rộng loss sang conditional MI qua `I(A;B|C) = I(A;B,C) − I(A;C)`". Pha Q cố tình **đơn giản hoá xuống bài toán MI không điều kiện** — đúng tinh thần "ít rủi ro nhất, có nhiều tài liệu tham khảo" mà mentor nhấn mạnh, vì MI không điều kiện là bài toán benchmark kinh điển của MINE.

**Bài toán chọn: `MI(X[t-1]; Y[t])`** trên chính hệ VAR tuyến tính Gaussian đã có từ Pha P (cấu hình `coupled_low_noise`: `a=0.5, b=0.5, c=0.6, noise_std=0.5`) — tái dùng đúng hệ thống của cả dự án thay vì bịa một bài toán đồ chơi tách rời.

**Ground-truth đã verify bằng Monte Carlo** (N=2.000.000, so khớp thực nghiệm trong sai số <0.3%) — công thức đã nằm sẵn trong docstring của `compute_lag1_mi_ground_truth()`:
```
cov_xlag_ynow = b * s_xy + c * s_xx      # Cov(X[t-1], Y[t])
rho = cov_xlag_ynow / sqrt(s_xx * s_yy)
MI = -0.5 * log(1 - rho^2)
```
(`s_xx, s_xy, s_yy` là hiệp phương sai dừng đã giải Lyapunov, giống hệt logic trong `generate_var_linear_gaussian`.)

**Lưu ý quan trọng về lấy mẫu huấn luyện — tránh tương quan trong batch:**
MINE giả định các sample trong 1 batch để `shuffle_batch` hoạt động đúng là **độc lập** (hoặc ít nhất không quá tương quan với nhau). Nếu lấy `(x[t-1], y[t])` từ **một** chuỗi thời gian dài liên tục, các sample tại các `t` gần nhau sẽ tương quan mạnh (do `X`, `Y` có tự-tương-quan), có thể làm chậm hội tụ hoặc tăng phương sai gradient. **Giải pháp:** sinh **nhiều thực hiện (realization) độc lập, ngắn** (seed khác nhau) rồi **ghép lại** thành 1 pool mẫu, thay vì 1 chuỗi dài duy nhất — xem `configs/mine/smoke_test.yaml` (`n_realizations: 50 × n_samples_per_realization: 2000` ≈ 100.000 cặp mẫu) và TODO chi tiết trong `scripts/train_mine_smoke_test.py`.

## 5. EMA bias correction (`ema.py`) — tuỳ chọn

Kỹ thuật từ paper MINE gốc (mục 3.2) để giảm bias gradient của `log(E[exp(T)])` khi ước lượng qua minibatch nhỏ. **Không bắt buộc cho tiêu chí thoát Pha Q** — chỉ làm nếu loss/MI estimate dao động mạnh khi không có nó. Nếu bỏ qua, ghi rõ trong báo cáo là không dùng, không cần xoá file (để lại làm việc tương lai nếu cần).

## 6. Test (`tests/test_mine.py`)

7 test skeleton đã có, thay `pytest.skip(...)` bằng assertion thật:
- `StatisticsNetwork`: đúng shape output, reproducible với cùng seed.
- `losses`: `shuffle_batch` đúng là hoán vị, loss hữu hạn, loss phản ánh đúng chiều của DV bound.
- `train_mine`: loss giảm dần trên bài toán Gaussian tương quan đơn giản; MI estimate khớp ground-truth trên chính bài toán VAR linear Gaussian (test chậm, tương đương quy mô script chính — có thể giảm quy mô/đánh dấu slow nếu cần, nhưng phải đủ lớn để có ý nghĩa thống kê).

## 7. Checklist thoát Pha Q

- [ ] `StatisticsNetwork`, `donsker_varadhan_loss`, `shuffle_batch`, `train_mine` implement xong, đúng chữ ký hàm trong stub.
- [ ] `pytest tests/test_mine.py` pass (không còn skip).
- [ ] `python scripts/train_mine_smoke_test.py` chạy xong:
  - Loss giảm ổn định (không NaN/Inf, không dao động không kiểm soát) — nhìn `results/figures/phase_q_mine_smoke_test_loss.png`.
  - `|MI_estimate − MI_ground_truth| / MI_ground_truth < 0.20` (ngưỡng trong `configs/mine/smoke_test.yaml`, nới hơn Pha P vì MINE có phương sai cao hơn KSG ở cùng N — đây là quan sát bình thường, không phải lỗi).
- [ ] Kết quả số lưu lại được (`results/tables/phase_q_mine_smoke_test.json` hoặc `.csv`) để audit.
- [ ] Nếu dùng EMA (mục 5): ghi rõ trong báo cáo; nếu không dùng: cũng ghi rõ là quyết định có chủ đích.

**Nếu sau khi thử 2-3 cấu hình siêu tham số cơ bản (learning rate, kiến trúc mạng, batch size) vẫn không hội tụ:** nhiều khả năng lỗi nằm ở bước chuẩn hoá dữ liệu hoặc cơ chế `shuffle_batch`, không phải bản thân thuật toán MINE — kiểm tra 2 chỗ đó trước (đúng như lưu ý gốc của mentor trong `docs/PQRST_Roadmap_goc.md`).

Khi checklist trên xong, báo lại để review trước khi sang Pha R (mở rộng conditional MI/TE, kiểm định đầy đủ trên lưới cấu hình).
