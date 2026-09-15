# Tổng quan chi tiết tiến độ dự án PQRST (Q-BHC / AQNE-TE)

> Tài liệu này tổng hợp ĐẦY ĐỦ nội dung từ `PHASE_P_REPORT.md`, `PHASE_Q_REPORT.md`,
> `PHASE_R_REPORT.md`, `PHASE_S_REPORT.md`, `PHASE_S_REVIEW.md`, `PHASE_T_REPORT.md`
> thành 1 báo cáo tiến độ liền mạch — dùng để báo cáo mentor, không cần mở từng
> file riêng. Số liệu, bảng, hình đều lấy trực tiếp từ các báo cáo gốc (không suy
> diễn thêm).
>
> Khác `ROADMAP.md` (kế hoạch gốc, không đổi theo thời gian) — file này là ảnh chụp
> tiến độ THỰC TẾ, cập nhật mỗi khi xong 1 việc lớn.
>
> Nếu cần làm rõ RIÊNG input/output/research gap (không lẫn với tiến độ), xem
> [`PROBLEM_STATEMENT.md`](PROBLEM_STATEMENT.md).

**Cập nhật lần cuối:** 2026-09-15.

---

## Mục lục

1. [Cấu trúc dự án](#1-cấu-trúc-dự-án)
2. [Pha P — Khởi động](#2-pha-p--khởi-động-✅-đạt)
3. [Pha Q — MINE cổ điển](#3-pha-q--mine-cổ-điển-✅-đạt)
4. [Pha R (+R2) — Amortized MINE](#4-pha-r-r2--amortized-mine-✅-hạ-tầng-đạt-tiêu-chí-thoát-chưa-đạt-nhưng-có-câu-chuyện)
5. [Pha S — Dữ liệu thật](#5-pha-s--dữ-liệu-thật-✅-đạt-qua-đổi-hướng)
6. [Pha T — Hybrid + hiệu chỉnh bias](#6-pha-t--hybrid--hiệu-chỉnh-bias-✅-code-đạt-bản-thảo-chưa-làm)
7. [Tiếp theo: bản thảo và Nhịp 2](#7-tiếp-theo-bản-thảo-và-nhịp-2)
8. [Đánh giá quy mô còn lại](#8-đánh-giá-quy-mô-còn-lại)
9. [Việc linh tinh chưa xử lý](#9-việc-linh-tinh-chưa-xử-lý)

---

## 1. Cấu trúc dự án

Dự án có **2 "Nhịp"**, mỗi nhịp gồm 5 pha:

- **Nhịp 1 (cổ điển)** — P → Q → R → S → T: dùng mạng nơ-ron `T_φ` (MLP) làm
  "statistics network" cho ước lượng transfer entropy (TE) kiểu MINE.
- **Nhịp 2 (lượng tử)** — P′ → Q′ → R′ → S′ → T′: **tái sử dụng gần như toàn bộ**
  hạ tầng dữ liệu/loss/đánh giá của Nhịp 1, chỉ hoán đổi `T_φ` bằng mạch lượng tử
  `T_θ` (data re-uploading circuit) tại 1 điểm nối duy nhất trong
  `src/pqrst/estimators/`.

Luận điểm khoa học trung tâm: **liệu MINE amortized (cổ điển hoặc lượng tử) có
thắng được các baseline TE cổ điển (KSG/binning/symbolic) ở kích thước mẫu N nhỏ**
— vùng mà baseline cổ điển thường yếu, và là vùng thực tế nhất khi làm việc với
cửa sổ dữ liệu sinh lý ngắn.

---

## 2. Pha P — Khởi động ✅ ĐẠT

**Mục tiêu:** dựng hạ tầng ước lượng TE làm nền cho toàn dự án — bộ sinh dữ liệu
tổng hợp có đáp án đúng (ground truth) + 3 baseline cổ điển + interface chung.

### 2.1. Đã làm (5 việc, 5 commit đầu + 3 commit sửa lỗi)

| # | Việc | File chính |
|---|---|---|
| 1 | Hàm tiện ích dùng chung (seed RNG, bias, MSE) | `src/pqrst/utils/seeding.py`, `src/pqrst/evaluation/metrics.py` |
| 2 | 3 bộ sinh dữ liệu tổng hợp: VAR tuyến tính Gaussian, VAR phi tuyến, ghép nối tuần hoàn (periodic) | `src/pqrst/data/synthetic/*.py` |
| 3 | 3 baseline TE cổ điển qua IDTxl: KSG, binning, symbolic | `src/pqrst/baselines/*.py` |
| 4 | Script đối chiếu baseline vs ground-truth | `scripts/validate_baselines.py` |
| 5 | Test tự động | `tests/*.py` |

**Vì sao quan trọng:** bộ sinh dữ liệu có ground-truth là "đáp án" để chấm điểm
MỌI estimator TE xuyên suốt cả 2 Nhịp. 3 baseline là đối chứng cho luận điểm khoa
học chính. Interface `BaseTEEstimator` chung là quyết định kiến trúc quan trọng
nhất — nhờ nó Pha Q/Nhịp 2 chỉ cần thêm 1 class mới, không sửa lại code cũ.

### 2.2. Kỹ thuật

- **VAR tuyến tính Gaussian**: `X[t]=aX[t-1]+ε`, `Y[t]=bY[t-1]+cX[t-1]+ε`. TE(X→Y)
  có công thức đóng qua giải phương trình Lyapunov cho hiệp phương sai dừng.
- **VAR phi tuyến / periodic coupling**: không có công thức đóng → pseudo
  ground-truth = KSG chạy ở N=200.000 (`estimate_pseudo_ground_truth.ipynb`), case
  "uncoupled" dùng đúng giá trị lý thuyết `0.0`.
- **3 baseline** qua IDTxl (`JidtKraskovTE`, `JidtDiscreteTE` cho binning và symbolic
  sau khi tự mã hoá ordinal-pattern theo Staniek & Lehnertz 2008).
- Môi trường: Python 3.11 + IDTxl/JPype (cần JVM) + `numpy<2` lúc đó (IDTxl dùng
  `np.math.factorial` đã gỡ ở numpy 2.0 — pin này sau đó được bỏ ở Pha S khi xác
  nhận IDTxl vẫn chạy đúng ở numpy 2.4.6).

### 2.3. LỖI PHÁT HIỆN: công thức ground-truth TE sai

```
Sai (bản đầu):  TE = 0.5·log(1 + c²/(1−a²))
Đúng (đã sửa):  giải Lyapunov cho hiệp phương sai dừng [X,Y], rồi
                TE = 0.5·log(1 + c²·Var(X[t-1]|Y[t-1])/σ²)
```

**Nguyên nhân:** công thức sai ngầm giả định X[t-1] độc lập với Y[t-1] — chỉ đúng
khi a=0. Vì mọi config dự án đều có a=0.5, giả định này sai.

**Xác minh — 2 lần độc lập, khớp nhau:**

| Nguồn | Cách tính | TE (nats) |
|---|---|---|
| Claude (review 1) | Monte Carlo N=2.000.000 | 0.18153 |
| Claude | Công thức Lyapunov (giải tích) | 0.18139 |
| Agent coding (review độc lập, seed khác) | Monte Carlo N=2.000.000 | 0.18146 |
| — | Công thức sai (bản đầu) | 0.19602 |

3 số Monte Carlo/giải tích độc lập lệch nhau <0.1%; công thức sai lệch ~8%. Đã sửa,
thêm test hồi quy `test_ground_truth_matches_ksg_at_large_n`.

### 2.4. Kết quả (N=1000, 50 lần lặp/cấu hình)

| Dataset | Config | KSG bias | Binning bias | Symbolic bias |
|---|---|---|---|---|
| VAR Linear Gaussian | coupled | 0.0056 | 0.0465 | −0.0764 |
| VAR Linear Gaussian | uncoupled | 0.0036 | 0.1085 | 0.0313 |
| VAR Nonlinear | coupled_low_noise | 0.0093 | 0.0695 | −0.0423 |
| VAR Nonlinear | uncoupled | 0.0036 | 0.1085 | 0.0313 |
| Periodic Coupling | coupled | −0.0321 | −0.1880 | −0.2609 |
| Periodic Coupling | uncoupled | 0.0286 | 0.1278 | 0.0181 |

KSG bám sát ground-truth ở mọi config (bias ≤0.03 nat). Binning/Symbolic bias lớn
hơn nhiều (tới 0.19–0.26 nat) — tín hiệu tốt cho luận điểm dự án (baseline cổ điển
yếu ở N nhỏ, đúng thứ MINE/lượng tử cần vượt qua).

**Checklist thoát:** ✅ tất cả 7 mục (môi trường, IDTxl+JVM, 3 bộ sinh, 3 baseline,
validate script, quyết định symbolic TE, không commit file lớn/bí mật).

---

## 3. Pha Q — MINE cổ điển ✅ ĐẠT

**Mục tiêu:** `T_φ` (MLP) + Donsker-Varadhan (DV) loss, smoke test trên 1 cấu hình
đơn giản.

### 3.1. Đã làm

| # | Việc | File |
|---|---|---|
| 1 | Ground-truth MI(X[t-1];Y[t]) công thức đóng | `var_linear_gaussian.py` |
| 2 | `StatisticsNetwork` (MLP), `donsker_varadhan_loss`, `shuffle_batch` | `estimators/mine/network.py`, `losses.py` |
| 3 | `train_mine` (Adam + early stopping) | `estimators/mine/train.py` |
| 4 | Script smoke test | `scripts/train_mine_smoke_test.py` |
| 5 | Test (7 test) | `tests/test_mine.py` |

Bài toán: MI(X[t-1];Y[t]) trên `a=0.5,b=0.5,c=0.6,noise_std=0.5`, ground-truth
`0.2197 nats`. Dữ liệu train = 50 realization × 2000 mẫu.

### 3.2. Kết quả

- `pytest tests/test_mine.py`: 20/20 pass.
- Smoke test: MI estimate `0.2364`, ground-truth `0.2197`, sai số **7.60%** (ngưỡng
  20%) — PASS. Stress-test 3 seed khác: sai số 0.80–5.03%.

![Loss huấn luyện smoke test MINE](../results/figures/phase_q_mine_smoke_test_loss.png)

### 3.3. Phát hiện phương pháp luận (không phải bug): selection bias ở "best epoch"

Nhìn đồ thị trên: train loss hội tụ nhanh rồi phẳng quanh -0.21 đến -0.217. **Val
loss dao động rất mạnh** (-0.2165 đến -0.2364, std≈0.0055 nat — lớn ngang chính sai
số 7.6% đang báo cáo). Early stopping chọn epoch 4 làm "best" — đúng điểm thấp nhất
NGẪU NHIÊN trên đường val loss.

**Nếu lấy trung bình 10 epoch cuối** (sau hội tụ) thay vì "best-of-20": MI trung
bình ≈0.2260, sai số chỉ **2.86%** — sát ground-truth HƠN số đang báo cáo.

**Nguyên nhân:** validation mỗi epoch chỉ dùng 1 lần shuffle để ước lượng số hạng
marginal — ước lượng Monte Carlo phương sai cao vốn có của DV bound, không phải
lỗi code. Chọn "val loss thấp nhất trong 20 epoch" = chọn giá trị may mắn nhất
trong 20 lần đo nhiễu (selection bias kinh điển).

**Không chặn Pha Q** vì tiêu chí đạt ở cả 2 cách tính, và Pha R đã có kế hoạch
bootstrap ≥500 lần + quét N sẽ tự khắc phục. **Khuyến nghị cho Pha R:** dùng trung
bình vài epoch cuối, không "best-of-all-epochs" đơn lẻ — *khuyến nghị này sau đó
được phát hiện là CHƯA thực sự áp dụng ở Pha R (xem mục 4.3)*.

**Checklist thoát:** ✅ tất cả 6 mục.

---

## 4. Pha R (+R2) — Amortized MINE ✅ hạ tầng đạt, tiêu chí thoát chưa đạt nhưng có câu chuyện

**Mục tiêu:** mở rộng conditional MI/TE, sinh kho ngữ liệu đầy đủ, quét lưới N ×
coupling × noise, so `T_φ` với baseline bằng bootstrap.

### 4.1. Thiết kế đã xác nhận đúng

| Quyết định | Đánh giá |
|---|---|
| `MaskedStatisticsNetwork` dùng chung 1 mạng (mask=full/reduced) | ✅ Đúng |
| Shuffle marginal dùng CHUNG 1 permutation cho `mi_full` và `mi_reduced` (tương quan dương, giảm phương sai hiệu) | ✅ Đúng, đã verify: var(mi_full)≈0.012, var(mi_reduced)≈0.007, var(te)≈0.0036 — giảm mạnh so với tổng 2 số hạng (0.019) |
| `eval_n_shuffles=20` (sửa lỗi Pha Q) | ✅ Đúng |
| Batch nhiều cửa sổ, DV bound tính riêng từng cửa sổ | ✅ Đúng |

### 4.2. LỖI CHẶN: dữ liệu đầu vào bị lệch chỉ số trong `grid.py`

```python
# TRƯỚC (sai):
x[:-1] = w.x_lag
x[1:] = w.x_lag   # <-- ghi đè dòng trên, làm x lệch 1 chỉ số
```

**Hậu quả:** mọi giá trị `x` vào **cả 4 estimator** trong 72.000 phép đo bị lệch —
tương đương đo TE(X[t-2]→Y[t]) chứ không phải TE(X[t-1]→Y[t]).

**Verify:** KSG gọi trực tiếp = 0.1074; dựng lại đúng (đã sửa) = 0.1074 ✅; dựng lại
theo code lỗi = 0.1156 ❌. Đã sửa (`x[-1]=w.x_lag[-1]`), thêm 2 test hồi quy
(`TestGridReconstruction`), chạy lại toàn bộ 72.000 phép đo (414 giây).

### 4.3. Phát hiện: "trung bình 10 epoch cuối" không khớp code thật

Báo cáo gốc (agent coding viết) tuyên bố checkpoint = trung bình 10 epoch cuối,
nhưng thực tế `train_amortized()` deploy `best_model_state` (đúng 1 epoch val loss
thấp nhất — cùng cơ chế gây selection bias ở Pha Q, khuyến nghị mục 3.3 chưa được
áp dụng thật). Ít nghiêm trọng hơn Pha Q vì `evaluate()` dùng cùng 1 seed cố định
mọi epoch (std val loss 10 epoch cuối = 0.00247, thấp hơn 0.0055 ở Pha Q).

**Đã sửa:** `train_amortized` giờ deploy **trung bình tham số (weight averaging)**
thật của k epoch cuối, có test xác nhận toán học. *(Checkpoint chính lúc đó chưa
train lại với logic mới — sau này ở Pha S, checkpoint được train lại từ đầu với
chuẩn hoá, dùng đúng logic weight-averaging này.)*

### 4.4. Kết quả chính — tiêu chí thoát và điểm giao cắt N≈30

Huấn luyện `MaskedStatisticsNetwork` chính hội tụ mượt, không dấu hiệu chọn-điểm-nhiễu như Pha Q:

![Loss huấn luyện Amortized (mạng chính)](../results/figures/phase_r_amortized_training_loss.png)

| | Báo cáo gốc (lỗi) | Sau khi sửa |
|---|---|---|
| Thắng KSG (N<30) | 6/30 (20%) | **9/30 (30%)** |
| Thắng Symbolic (N<30) | 29/30 | 28/30 |
| Thắng Binning (N<30) | — | 30/30 |
| **Đạt tiêu chí (≥70% cả 2)** | ❌ | ❌ **Chưa đạt** |

![Phương sai theo N](../results/figures/phase_r_variance_vs_n.png)

| N | Var(Amortized) | Var(KSG) | Ai thắng |
|---|---|---|---|
| 10 | 0.0117 | 0.0037 | KSG (rõ) |
| 20 | 0.0064 | 0.0054 | KSG (nhẹ) |
| **30** | **0.0042** | **0.0053** | **Amortized** |
| 50 | 0.0026 | 0.0044 | Amortized |
| 100 | 0.0014 | 0.0029 | Amortized (gấp 2) |
| 200 | 0.0007 | 0.0018 | Amortized (gấp 2.6) |

**Amortized thắng rõ và cách biệt ngày càng lớn từ N=30 trở lên** — tiêu chí gốc
(N<30) rơi đúng vào vùng Amortized yếu nhất. Không phải "thua toàn diện" — là 1
điểm giao cắt rõ ràng.

![Bias theo N](../results/figures/phase_r_bias_vs_n.png)

KSG nhất quán (bias co về 0: -0.078→-0.008). Amortized bias không về 0
(-0.065→-0.044, giữ nguyên từ N=50) — mạng đóng băng không "học thêm", bias bị
chặn bởi năng lực mô hình lúc train.

### 4.5. Ablation mạng nhỏ — giả thuyết "ít tham số" bị BÁC BỎ

Kiểm định trước khi đầu tư Nhịp 2: nếu Amortized thua ở N nhỏ do overfit (mạng lớn
học quá khớp), thì mạng ít tham số hơn (dù cổ điển hay lượng tử) có thể tổng quát
tốt hơn.

Huấn luyện mạng nhỏ hội tụ tốt (val loss std=0.0005, thấp hơn cả mạng chính):

![Loss huấn luyện mạng nhỏ (ablation)](../results/figures/phase_r_ablation_small_loss.png)

![So sánh phương sai mạng lớn vs nhỏ](../results/figures/phase_r_ablation_variance_comparison.png)

| | Mạng LỚN `[128,128,64]` | Mạng NHỎ `[16,16]` (~70 lần ít tham số) |
|---|---|---|
| Bias @ N=10 | -0.0651 | -0.0635 |
| Variance @ N=10 | 0.01168 | 0.01237 (nhích cao hơn) |
| Thắng KSG (N<30) | 9/30 | 9/30 |

**Kết luận: giảm ~70 lần tham số không cải thiện gì.** 2 đường variance gần như
song song, cùng cắt KSG ở N≈30, cùng thắng đúng 9/30 ô. Giả thuyết "ít tham số =
tổng quát tốt ở N nhỏ" — lý do chính để kỳ vọng mạch lượng tử giúp được — **bị bác
bỏ bởi thực nghiệm**. Vấn đề nhiều khả năng nằm ở nhiễu Monte Carlo vốn có của
`logsumexp` khi quá ít điểm dữ liệu, không nằm ở dung lượng mô hình.

### 4.6. Thử nghiệm phi tuyến (periodic coupling) — giả thuyết BỊ BÁC BỎ theo chiều ngược

Giả thuyết: KSG chỉ mạnh trên Gaussian tuyến tính (sân nhà của k-NN); trên dữ liệu
phi tuyến/tuần hoàn (gần thực tế tim-não hơn), Amortized có thể có lợi thế hơn.

Huấn luyện trên corpus periodic coupling hội tụ tốt (val loss std=0.0042), không lỗi NaN trong 28.800 phép đo đánh giá:

![Loss huấn luyện periodic coupling](../results/figures/phase_r2_periodic_training_loss.png)

![Phương sai theo N — periodic coupling](../results/figures/phase_r2_periodic_variance_vs_n.png)

| | VAR tuyến tính Gaussian | Periodic coupling |
|---|---|---|
| Thắng KSG (N<30) | 9/30 (30%) | **0/16 (0%)** |
| Thắng Symbolic (N<30) | 28/30 | 7/16 |
| Thắng Binning (N<30) | 30/30 | 14/16 |

**Kết quả NGƯỢC kỳ vọng: KSG mạnh HƠN trên dữ liệu phi tuyến.** Amortized thua ở
TOÀN BỘ 16 ô N<30 — tệ hơn cả kết quả tuyến tính. Điểm giao cắt vẫn tồn tại (từ
N≈50) nhưng mức thua ở N nhỏ nặng hơn. **Diễn giải:** lợi thế KSG ở N nhỏ là đặc
tính chung của phương pháp k-NN (thích nghi cục bộ) khi ít dữ liệu, không đặc thù
riêng phân phối Gaussian. 2 kiểm định độc lập (ablation + periodic) đều chỉ về
cùng 1 nguyên nhân: **nút thắt nằm ở chính công thức DV bound/logsumexp khi có quá
ít điểm**, không nằm ở kiến trúc mạng hay loại dữ liệu.

### 4.7. Khuyến nghị đã áp dụng cho các bước sau

- **(a) Dùng estimator lai (hybrid): KSG cho N<30-50, Amortized cho N lớn hơn** —
  khuyến nghị mạnh, không cần nghiên cứu thêm.
- Điểm giao cắt N≈30-50 là hiện tượng ổn định qua 3 lần thử nghiệm độc lập (mạng
  lớn, mạng nhỏ, dữ liệu phi tuyến) — đủ để công bố.
- **Cảnh báo quan trọng cho Nhịp 2:** cả 2 vòng kiểm định đều làm suy yếu lý do "ít
  tham số sẽ giúp ở N nhỏ" — mạch lượng tử (cũng ít tham số) nhiều khả năng gặp
  đúng giới hạn tương tự. Nếu triển khai Nhịp 2, nên đặt kỳ vọng ở vùng N vừa/lớn.
- **Phát hiện này quay lại giải thích trực tiếp vì sao Amortized FAIL trên dữ liệu
  thật ở Pha S** (xem mục 5.5) — không phải hiện tượng mới, là hệ quả đã được cảnh
  báo trước.

---

## 5. Pha S — Dữ liệu thật ✅ ĐẠT (qua đổi hướng)

**Mục tiêu gốc:** TE tim-não trên `slpdb`/`capslpdb`. **Thực tế:** đổi hướng sang
TE tim-hô hấp (Fantasia/Apnea-ECG) sau khi phát hiện vấn đề chặn không sửa được.

### 5.1. Điều kiện tiên quyết (S0): chuẩn hoá bất biến thang đo

Checkpoint `T_φ` gốc chỉ cho tương quan **0.07** với đáp án khi gặp thang đo dữ
liệu thật (khác thang đo dữ liệu tổng hợp lúc train). Đã retrain với
`standardize_window` (z-score từng cửa sổ) áp dụng cả lúc train và lúc suy luận.

**Kết quả sau retrain:** MSE (N≥50) cũ 0.00509 → mới 0.00446 (tốt hơn, không tệ
đi); **tương quan bất biến thang đo: 1.00000** (so với 0.07 trước đó). Checkpoint
`phi_amortized_standardized.pt` dùng cho toàn bộ Pha S.

### 5.2. TE tim-não (slpdb/capslpdb) — LOẠI KHỎI KẾT QUẢ CHÍNH

**Phát hiện:** dry-run `slp01a` → `artifact_ratio=36.49` (ngưỡng 2.0). Mở rộng
toàn bộ **18/18 bản ghi slpdb đều vượt ngưỡng** (7.89–60.35).

**Xác minh đây là nhiễu điện THẬT, không phải heartbeat-evoked potential (HEP)**
(nếu nhầm sẽ vô tình xóa mất chính tín hiệu cần đo):

| | Đỉnh lệch (so R-peak) | Độ rộng |
|---|---|---|
| EEG khóa-theo-R-peak (`slp01a`) | -8ms | 12ms |
| QRS của chính ECG | -8ms | 16ms |

Trùng khớp gần hoàn toàn với QRS — HEP thật đỉnh muộn hơn nhiều (~200-300ms), rộng
hơn nhiều.

**5 phương pháp khử nhiễu đã thử — đều KHÔNG đạt:**

| # | Phương pháp | Kết quả | Vì sao thất bại |
|---|---|---|---|
| 1 | Template trung bình khóa-R-peak | Không ổn định — vài bản ghi giảm mạnh, vài bản ghi TĂNG | Hình dạng nhiễm thật khác nhau giữa các nhịp nhiều hơn 1 template cố định |
| 2 | + Căn chỉnh jitter (Woody's method) | Không sửa được | Không phải do lệch pha |
| 3 | + Vút Tukey (chống bước nhảy biên) | Cải thiện nhóm "template nhọn" nhưng không bản ghi nào đạt ngưỡng | Nhóm "tăng" có template GẦN-HẰNG-SỐ, trừ tạo bước nhảy mới bị phát hiện là nhiễu |
| 4 | Hồi quy tuyến tính EEG theo ECG thật | Tệ hơn cả template (1 bản ghi: 15.4→99.5) | Quan hệ ECG↔nhiễu không ổn định suốt đêm |
| 5 | ICA (9 kênh capslpdb), `find_bads_ecg` | Không tách được thành phần nào | artifact_ratio từng thành phần 3.2–11.6, lan trải đều, không tập trung 1 nguồn |

`capslpdb` (9 kênh differential, đáng ra ít nhạy nhiễu hơn) **cũng nhiễm cả 9 kênh**
(5.76–26.63) — kênh tốt nhất vẫn gấp ~3x ngưỡng.

**Quyết định:** không tiếp tục đầu tư khử nhiễu — ghi nhận làm phát hiện phương
pháp luận (đóng góp, không phải thất bại). Chuyển kết quả chính sang tim-hô hấp.

### 5.3. TE tim-hô hấp (Fantasia + Apnea-ECG) — KẾT QUẢ CHÍNH

**3 lỗi tìm và sửa trong chính hướng mới, trước khi có kết quả tin được:**

**Lỗi #1 — NaN lan toàn tín hiệu qua `filtfilt`:** vài bản ghi (vd `f2o06`) có NaN
rải rác (đứt cảm biến, bình thường ở dữ liệu thật). `filtfilt` là lọc IIR 2 chiều
TOÀN tín hiệu: 61/1.75 triệu mẫu NaN đầu vào → **100% NaN đầu ra**. Sửa: nội suy
qua khoảng NaN ngắn TRƯỚC khi lọc. Kết quả: PASS 20→23/40.

**Lỗi #2 — bất đối xứng lọc bằng-thông RR/RESP:** TE cho sai hướng lần đầu (chỉ
5/23 đúng hướng RSA). Điều tra bằng đo thời gian tự tương quan tích hợp:

| Tín hiệu | τ trung bình | Số bản ghi RR dài hơn RESP |
|---|---|---|
| RR (chưa lọc) | 11.37s | **23/23 (100%)** |
| RESP (đã lọc 0.1–0.5Hz) | 0.65s | — |

RESP đã lọc bằng-thông, RR thì không → RR giữ trend chậm (LF/VLF HRV), "trí nhớ"
tự tương quan chênh ~17 lần → TE lệch hướng (hiệu ứng thống kê, không phải sinh lý
thật). **Sửa:** lọc CẢ RR và RESP cùng dải 0.1–0.5Hz.

| | Trước sửa | Sau sửa |
|---|---|---|
| Số bản ghi đúng hướng RSA | 5/23 (22%) | 19/23 (83%) |
| TE gộp: hiệu số | -0.0095 | **+0.0278** |

**Lỗi #3 (chuẩn bị Q1/Q2) — pseudo-replication:** sanity check đầu tiên gộp TẤT CẢ
cửa sổ mọi bản ghi thành 1 tập (N=4049 "ảo") rồi bootstrap — lỗi thống kê kinh
điển (cửa sổ cùng 1 bản ghi tương quan, không độc lập). **Sửa:** tính TE trung bình
MỖI bản ghi, kiểm định trên N=số_bản_ghi (đơn vị mẫu ĐÚNG) + Wilcoxon signed-rank.

### 5.4. Kết quả cuối cùng (thống kê đúng phương pháp)

| Estimator | TE(hô hấp→tim) | TE(tim→hô hấp) | CI 95% hiệu số | Wilcoxon p | Kết luận |
|---|---|---|---|---|---|
| **KSG (Fantasia, N=17)** | 0.1109 | 0.0929 | **[0.0039, 0.0353]** | 0.0116 | ✅ PASS |
| Amortized (Fantasia, N=17) | -0.2474 | -0.0704 | [-0.2119, -0.1434] | 1.0000 | ❌ FAIL |

**Đối chứng độc lập thứ 2 (Apnea-ECG) + gộp meta-analysis:**

| | N | TE(hô hấp→tim) | TE(tim→hô hấp) | CI 95% | Wilcoxon p |
|---|---|---|---|---|---|
| Apnea-ECG riêng | 6 | 0.1362 | 0.1185 | [-0.0038, 0.0455] (chứa 0) | 0.2188 |
| **Gộp Fantasia + Apnea-ECG** | **23** | 0.1175 | 0.0996 | **[0.0057, 0.0311]** | **0.0046** |

Apnea-ECG (cấu trúc khác Fantasia: ECG và hô hấp ở 2 file WFDB riêng — `a01`=ECG,
`a01r`=hô hấp) một mình chưa đủ mạnh (N=6) nhưng cùng chiều (4/6 bản ghi). Gộp 2
nguồn ĐỘC LẬP THỐNG KÊ THẬT (đối tượng khác, thiết bị khác) cho **bằng chứng mạnh
nhất toàn Pha S**: 18/23 (78%) bản ghi đúng hướng.

**Kiểm tra độ nhạy tham số (9 tổ hợp bandpass × độ dài cửa sổ):**

| Bandpass (Hz) | Cửa sổ (s) | N pass | Hiệu số | Wilcoxon p | PASS |
|---|---|---|---|---|---|
| (0.15, 0.4) | 20 | 15 | 0.0313 | 0.0062 | ✅ |
| (0.15, 0.4) | 30 | 15 | 0.0258 | 0.0319 | ✅ |
| (0.15, 0.4) | 45 | 15 | 0.0220 | 0.0206 | ❌ |
| (0.1, 0.5) *mặc định* | 20 | 17 | 0.0220 | 0.0008 | ✅ |
| (0.1, 0.5) | 30 | 17 | 0.0180 | 0.0116 | ✅ |
| (0.1, 0.5) | 45 | 17 | 0.0159 | 0.0153 | ❌ |
| (0.05, 0.6) | 20 | 19 | 0.0108 | 0.0180 | ✅ |
| (0.05, 0.6) | 30 | 19 | 0.0070 | 0.1467 | ❌ |
| (0.05, 0.6) | 45 | 19 | 0.0055 | 0.2706 | ❌ |

**Chiều kết quả ổn định TUYỆT ĐỐI 9/9 (100%)**. Ý nghĩa thống kê đạt 5/9, giảm dần
CÓ GIẢI THÍCH ĐƯỢC (cửa sổ dài hơn/bandwidth rộng hơn đều pha loãng tín hiệu cụ
thể) — hành vi hợp lý của hiệu ứng THẬT, không phải kết quả giả.

![TE hai chiều, khoảng tin cậy 95%](../results/figures/phase_s_bidirectional_te.png)

### 5.5. Vì sao Amortized FAIL trên dữ liệu thật (điều tra, có nguyên nhân rõ)

```
mi_full (X_lag+Y_lag dự đoán Y_t)     ≈ 0.08
mi_reduced (chỉ Y_lag dự đoán Y_t)    ≈ 0.40   <- LỚN HƠN mi_full ~5 lần (vô lý về lý thuyết)
TE = mi_full - mi_reduced             ≈ -0.33
```

RR/RESP sau lọc bằng-thông là tín hiệu rất mượt, gần tuần hoàn — quá khứ tự dự
đoán tương lai cực tốt. `MaskedStatisticsNetwork` huấn luyện trên corpus
VAR-tuyến-tính (KHÔNG mượt/tuần hoàn) — lệch chuẩn nặng khi gặp dạng tín hiệu này.

**Khớp chính xác với phát hiện Pha R2** (mục 4.6): amortized yếu hơn KSG trên dữ
liệu phi tuyến/periodic ở N nhỏ. Đây KHÔNG phải hiện tượng mới của Pha S — là hệ
quả trực tiếp, có thể dự đoán trước, của phát hiện đã có từ trước. Cách sửa đúng
(retrain với corpus pha `periodic_coupling`) là việc lớn, quyết định KHÔNG làm
trong phạm vi Pha S (xem mục 7 — cân đối chi phí/lợi ích).

### 5.6. Toàn bộ 13 lỗi đã tìm và sửa trong Pha S

| # | File | Lỗi | Cách phát hiện |
|---|---|---|---|
| 1 | `sync.py` | Đồng bộ theo độ dài mảng thay vì mốc thời gian thật | Review code |
| 2 | `eeg.py` | `np.trapz` bị gỡ khỏi numpy 2.x, crash 100% | Chạy thật lần đầu |
| 3 | `eeg.py` | `detect_cardiac_artifact` không có đối chứng ngẫu nhiên thật | Review + test |
| 4 | `cardiac.py` | Lọc ectopic dùng trung vị TOÀN CỤC, sai với bản ghi dài trôi nhịp tim | Review |
| 5 | Notebook 02/03 | Tên hàm import sai sau khi đổi tên trong `.py` | Chạy thật lần đầu |
| 6 | `download.py` | Không xác minh file thật sau khi tải — báo "thành công" nhưng 0 file | Kiểm tra đĩa |
| 7 | `download.py` | `wfdb.dl_database` không báo tiến độ — tải chậm nhìn như treo | Đo tốc độ mạng |
| 8 | `download.py` | `capslpdb` là EDF thuần, không `.hea` — thất bại hoàn toàn, im lặng | Đo HTTP trực tiếp |
| 9 | `download.py` | Báo động giả "thiếu file" cho 8 bản ghi Apnea-ECG (`*er`) | Kiểm tra đĩa+header |
| 10 | `respiration.py` | NaN đứt cảm biến lan toàn tín hiệu qua `filtfilt` | Chạy 40 bản ghi |
| 11 | Pipeline TE | Bất đối xứng lọc bằng-thông RR/RESP làm TE lệch hướng | Đo tự tương quan |
| 12 | `sync.py` | `verify_synchronization` hardcode `0.7` thay vì đọc config | Review code |
| 13 | `sanity_check.py` | Pseudo-replication — gộp cửa sổ làm đơn vị mẫu | Rà soát trước công bố |

**Checklist thoát:** ✅ ĐẠT (KSG PASS, 2 nguồn độc lập, độ nhạy tham số ổn định).

---

## 6. Pha T — Hybrid + hiệu chỉnh bias ✅ code đạt (bản thảo chưa làm)

> Chi tiết đầy đủ: [`PHASE_T_REPORT.md`](PHASE_T_REPORT.md). Kế hoạch/ước tính
> thời gian gốc: [`PHASE_T_GUIDE.md`](PHASE_T_GUIDE.md).

**T.1 — Tổng hợp kết quả chính:** 1 notebook (`phase_t_01_main_results.ipynb`) đọc
lại toàn bộ số liệu đã tính từ Pha R/R2/S (không tính lại), tái xác nhận **khớp
100%** với số liệu real-data đã công bố ở `PHASE_S_REPORT.md` mục 5.4 (N=23, CI
`[0.0057,0.0311]`, Wilcoxon p=0.0046) — chạy độc lập lần 2, không lệch số.

![Main synthetic results](../results/figures/phase_t_main_variance_vs_n.png)

**T.2 — `HybridTEEstimator`** (`src/pqrst/estimators/hybrid.py`): KSG cho N<50,
Amortized cho N≥50. Phát hiện + sửa 1 bug thật lúc validate: ngưỡng ban đầu 30
(đúng cho tuyến tính) khiến Hybrid chọn nhầm Amortized tại N=30 trên dữ liệu
periodic (variance KSG 0.00636 < Amortized 0.00717 ở đó) — đã sửa ngưỡng chung
thành 50, an toàn trên cả 2 loại dữ liệu. Test hồi quy thật (`test_hybrid.py`,
4 test) đã tự verify không vô nghĩa (test cũ từng luôn-đúng bất kể ngưỡng, đã
thay bằng assertion có thể fail thật).

**T.3 — Hiệu chỉnh bias (tuỳ chọn, đã làm thêm):** leave-one-config-out trên
`Amortized` (và cả `KSG` để so sánh công bằng) — `Amortized_calibrated` thắng MSE
liên tục N=20–100 trên Linear VAR, và thắng **mọi N** (10→200) trên Periodic
Coupling, kể cả sau khi calibrate KSG. Bias của Amortized ổn định theo N (dễ hiệu
chỉnh); bias của KSG trên dữ liệu phi tuyến biến động mạnh giữa config (hiệu
chỉnh làm nó TỆ hơn ở hầu hết N trên periodic) — sự khác biệt này là 1 luận điểm
đáng đưa vào Discussion. **Chỉ là ablation synthetic, không áp dụng lên dữ liệu
thật, không ảnh hưởng tiêu chí thoát.**

![Bias calibration MSE](../results/figures/phase_t_bias_calibration_mse.png)

**Test suite:** `test_hybrid.py` + `test_calibration.py` — 7/7 PASS (đã tự chạy
lại với `JAVA_HOME` set để test tích hợp KSG+Amortized chạy thật).

**Checklist thoát Pha T:** Nhóm A (tổng hợp, bắt buộc) ✅ · Nhóm B (tuỳ chọn) ✅ ·
Nhóm D (test) ✅ · Nhóm C (bản thảo T.6 + review T.7) — **chưa làm, gác lại theo
quyết định báo cáo mentor trước.**

---

## 7. Tiếp theo: bản thảo và Nhịp 2

### 7.1. Viết bản thảo (T.6, T.7) — CHƯA BẮT ĐẦU

Toàn bộ số liệu/hình/lập luận đã đầy đủ từ `PHASE_P/Q/R/S/T_REPORT.md` — phần
việc còn lại là **tổ chức lại thành văn bản mạch lạc** (Abstract→Conclusion),
không phải tìm thêm bằng chứng mới. Đây là việc của chủ dự án; Claude hỗ trợ soát
câu chữ/dịch từng đoạn khi được yêu cầu. Sau khi có bản thảo: T.7 review đối
chiếu số liệu 100% khớp báo cáo gốc.

**Cổng quyết định sau Pha T:** kết quả hiện tại (Amortized thắng rõ N≥30 trên dữ
liệu tuyến tính, thua KSG trên phi tuyến/dữ liệu thật thô — nhưng thắng cả MSE
sau hiệu chỉnh bias trên synthetic) là "kết quả có câu chuyện, ngang bằng" theo
phân loại roadmap gốc → **vẫn đáng sang Nhịp 2**, hạ kỳ vọng phần câu chuyện phi
tuyến.

### 7.2. Nhịp 2 (P′→T′, lượng tử) — CHƯA BẮT ĐẦU

Chưa có dòng code nào. Tái dùng nguyên `src/pqrst/data/`, `data/processed/`,
pipeline đánh giá — chỉ viết `src/pqrst/estimators/quantum/wrapper.py`
(`T_theta(x,y)->scalar`, cùng chữ ký `T_phi`) rồi cắm vào `train.py` đã có.

**Cảnh báo quan trọng đã rút ra từ Pha R (mục 4.7):** ablation mạng nhỏ + thử
nghiệm periodic coupling đều làm suy yếu lý do "ít tham số sẽ giúp ở N nhỏ" — mạch
lượng tử (cũng ít tham số) nhiều khả năng gặp giới hạn tương tự (nhiễu Monte Carlo
vốn có của DV bound ở N nhỏ, không phải vấn đề dung lượng mô hình). Nên đặt kỳ
vọng Nhịp 2 ở vùng N vừa/lớn, không phải N<30.

Có cổng dự phòng: ≥2 cấu hình không hội tụ (nghi barren plateau) → có thể dừng
Nhịp 2, dùng Nhịp 1 làm bản thảo hoàn chỉnh — không phải rủi ro chí mạng.

---

## 8. Đánh giá quy mô còn lại

| Mục tiêu | Còn lại | Ước tính |
|---|---|---|
| Nộp Q1/Q2 chỉ dựa trên Nhịp 1 | Code đã xong 100% — chỉ còn viết bản thảo (T.6) + review (T.7) | Gần cán đích |
| Làm đủ cả 2 Nhịp (đúng scope gốc) | Toàn bộ Nhịp 2 (P′-T′) — 1 chu kỳ đầy đủ, tái dùng hạ tầng nên rẻ hơn xây từ đầu | Còn khá nhiều (~2-3 tháng theo ước tính gốc roadmap) |

---

## 9. Việc linh tinh chưa xử lý

- `results/figures/*` và `results/tables/*` đang bị `.gitignore` loại — mọi hình
  tham chiếu trong các báo cáo (kể cả Pha T) hiện KHÔNG có trong lịch sử git. Cần
  quyết định: force-add đích danh các file được tham chiếu, hoặc chấp nhận chỉ
  xem local (đã có sẵn lệnh `git add` chỉ định file cụ thể vẫn add được dù bị
  ignore, sẽ có cảnh báo nhưng không phải lỗi).

---

## Phụ lục: bản đồ file báo cáo chi tiết

| Pha | Báo cáo chi tiết | Notebook chính |
|---|---|---|
| P | [`PHASE_P_REPORT.md`](PHASE_P_REPORT.md) | — (script `validate_baselines.py`) |
| Q | [`PHASE_Q_REPORT.md`](PHASE_Q_REPORT.md) | `train_mine_smoke_test.py` |
| R (+R2) | [`PHASE_R_REPORT.md`](PHASE_R_REPORT.md) | `phase_r_01..05_*.ipynb`, `phase_r2_periodic_01..04_*.ipynb` |
| S | [`PHASE_S_REPORT.md`](PHASE_S_REPORT.md), [`PHASE_S_REVIEW.md`](PHASE_S_REVIEW.md) | `phase_s_00..05_*.ipynb` |
| T | [`PHASE_T_REPORT.md`](PHASE_T_REPORT.md) | `phase_t_01_main_results.ipynb`, `phase_t_02_hybrid_validation.ipynb`, `phase_t_03_bias_calibration.ipynb` |
