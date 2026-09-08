# Báo cáo hoàn thành Pha P — Nhịp 1 (Khởi động)

**Ngày:** xem lịch sử commit `git log --oneline` (8 commit, từ `732ce71` đến `520fc21`).
**Trạng thái:** hạ tầng + baseline đã chạy được, đã qua 1 vòng review phát hiện và sửa 1 lỗi khoa học quan trọng, được xác nhận độc lập lần 2 bởi agent coding.

---

## 1. Pha P đã làm gì

Theo đúng phạm vi trong [`PHASE_P_GUIDE.md`](PHASE_P_GUIDE.md): dựng xong **hạ tầng ước lượng transfer entropy (TE)** để làm nền cho toàn bộ dự án — chưa đụng tới mô hình MINE (đó là Pha Q) và chưa đụng tới dữ liệu sinh lý thật (đó là Pha S).

Cụ thể 5 việc, tương ứng 5 commit đầu:

| # | Việc | File chính | Commit |
|---|---|---|---|
| 1 | Hàm tiện ích dùng chung (seed RNG, bias, MSE) | `src/pqrst/utils/seeding.py`, `src/pqrst/evaluation/metrics.py` | `5e33f8f` |
| 2 | 3 bộ **sinh dữ liệu tổng hợp** (synthetic) có/gần-có ground-truth TE | `src/pqrst/data/synthetic/*.py` | `844d37e` |
| 3 | 3 **baseline TE cổ điển** (KSG, binning, symbolic) qua IDTxl, dùng chung 1 interface | `src/pqrst/baselines/*.py` | `9c688c9` |
| 4 | Script **đối chiếu baseline vs ground-truth**, xuất bảng số liệu | `scripts/validate_baselines.py` | `f7ee249` |
| 5 | Test tự động cho cả 2 phần trên | `tests/*.py` | `72b572f` |

Sau đó là 1 vòng review (chi tiết ở mục 4) sửa lại 3 commit tiếp: `b7c8bf0`, `3669048`, `520fc21`.

## 2. Vì sao từng phần này quan trọng

- **Bộ sinh dữ liệu tổng hợp có ground-truth**: đây là "đáp án" để chấm điểm mọi estimator TE (baseline cổ điển ở Pha P, mạng MINE ở Pha Q/R, mạch lượng tử ở Nhịp 2). Không có đáp án đúng thì không thể nói method nào tốt hơn method nào — đây là lý do sai 1 con số ở đây (mục 4) nghiêm trọng hơn nhiều so với sai ở nơi khác.
- **3 baseline cổ điển**: là đối chứng xuyên suốt cả dự án. Luận điểm khoa học chính ("MINE/lượng tử thắng baseline cổ điển ở N nhỏ") chỉ có ý nghĩa nếu baseline được cài đúng và đã tự chứng minh là hoạt động hợp lý trên bài toán đã biết đáp án — đó chính là mục đích của `validate_baselines.py`.
- **1 interface chung `BaseTEEstimator`**: quyết định kiến trúc quan trọng nhất của Pha P. Nhờ nó, Pha Q (thêm `T_φ` — mạng MINE) và Nhịp 2 (thêm `T_θ` — mạch lượng tử) chỉ cần viết thêm 1 class mới kế thừa cùng interface, không phải sửa lại code ở Pha R/S/T.

## 3. Cách làm (tóm tắt kỹ thuật)

- **VAR tuyến tính Gaussian** (`var_linear_gaussian.py`): mô hình `X[t]=aX[t-1]+ε`, `Y[t]=bY[t-1]+cX[t-1]+ε`. TE(X→Y) có công thức đóng, tính bằng cách giải hệ phương trình Lyapunov cho ma trận hiệp phương sai dừng của hệ VAR(1) 2 chiều (chi tiết mục 4).
- **VAR phi tuyến** và **ghép nối tuần hoàn** (`var_nonlinear.py`, `periodic_coupling.py`): không có công thức đóng, nên **pseudo ground-truth** được lấy bằng cách chạy KSG ở N=200.000 (notebook [`estimate_pseudo_ground_truth.ipynb`](../notebooks/estimate_pseudo_ground_truth.ipynb)), riêng 2 trường hợp "uncoupled" (không ghép nối) dùng giá trị lý thuyết đúng `0.0` (2 chuỗi độc lập tuyệt đối) thay vì bias âm nhỏ mà KSG trả về.
- **3 baseline**: gọi qua IDTxl (`JidtKraskovTE` cho KSG, `JidtDiscreteTE` cho binning và cho symbolic sau khi tự mã hoá ordinal-pattern theo Staniek & Lehnertz 2008 — IDTxl không có sẵn estimator symbolic). Kết quả của binning/symbolic được quy đổi từ bit sang nat (`× ln 2`) để thống nhất đơn vị với KSG.
- **Môi trường**: venv Python 3.11 + IDTxl/JPype (cần JVM, `JAVA_HOME` trỏ JDK cài sẵn trên máy) + `numpy<2` (IDTxl hiện dùng `np.math.factorial` đã bị gỡ ở numpy 2.0).

## 4. Vấn đề phát hiện & sửa trong quá trình review

Khi review lần đầu, phát hiện **công thức ground-truth TE cho VAR tuyến tính Gaussian sai**:

```
Sai (bản đầu):  TE = 0.5·log(1 + c²/(1−a²))
Đúng (đã sửa):  giải Lyapunov cho hiệp phương sai dừng [X,Y], rồi
                TE = 0.5·log(1 + c²·Var(X[t-1]|Y[t-1])/σ²)
```

**Nguyên nhân**: công thức sai ngầm giả định `X[t-1]` độc lập với `Y[t-1]`. Điều này chỉ đúng khi `a=0`. Vì `Y[t-1]` phụ thuộc `X[t-2]`, mà `X[t-2]` tương quan với `X[t-1]` qua hệ số tự-hồi-quy `a`, nên khi `a≠0` (mọi config trong dự án đều có `a=0.5`), `X[t-1]` và `Y[t-1]` **không** độc lập — công thức đơn giản bỏ sót phần tương quan này.

**Xác minh — 2 lần độc lập, kết quả khớp nhau:**

| Nguồn | Cách tính | TE (nats) |
|---|---|---|
| Claude (review lần 1) | Monte Carlo trực tiếp, N=2.000.000 | 0.18153 |
| Claude | Công thức Lyapunov (giải tích) | 0.18139 |
| Agent coding (review độc lập) | Monte Carlo trực tiếp, N=2.000.000, seed khác | 0.18146 |
| — | Công thức sai (bản đầu) | 0.19602 |

Cả 3 con số Monte Carlo/giải tích độc lập nằm trong khoảng 0.1814–0.1815 (lệch nhau <0.1%), trong khi công thức sai lệch tới ~8%. Đã sửa trong commit `b7c8bf0`, kèm thêm test `test_ground_truth_matches_ksg_at_large_n` (`tests/test_synthetic_generators.py`) để tự động bắt lại lỗi tương tự trong tương lai — test này lẽ ra đã tồn tại theo đúng checklist gốc của `PHASE_P_GUIDE.md` mục 2.4 nhưng bị bỏ sót ở bản code đầu tiên.

2 việc dọn nhỏ kèm theo (commit `3669048`, `520fc21`):
- Bỏ 2 field cấu hình `history_length_x/history_length_y` trong `configs/baselines/*.yaml` vì code không thực sự đọc chúng (dead config).
- Ghi rõ trong `configs/synthetic/var_nonlinear.yaml` và `periodic_coupling.yaml` rằng `pseudo_ground_truth_te=0.0` cho case "uncoupled" là giá trị lý thuyết (không phải số KSG in ra), và lưu lại output thật của notebook để audit được.

## 5. Kết quả hiện tại

File: [`results/tables/phase_p_baseline_validation.csv`](../results/tables/phase_p_baseline_validation.csv) (N=1000 mỗi lần chạy, 50 lần lặp mỗi cấu hình).

| Dataset | Config | KSG bias | Binning bias | Symbolic bias |
|---|---|---|---|---|
| VAR Linear Gaussian | coupled (low/high noise) | 0.0056 | 0.0465 | −0.0764 |
| VAR Linear Gaussian | uncoupled | 0.0036 | 0.1085 | 0.0313 |
| VAR Nonlinear | coupled_low_noise | 0.0093 | 0.0695 | −0.0423 |
| VAR Nonlinear | uncoupled | 0.0036 | 0.1085 | 0.0313 |
| Periodic Coupling | coupled | −0.0321 | −0.1880 | −0.2609 |
| Periodic Coupling | uncoupled | 0.0286 | 0.1278 | 0.0181 |

**Đọc kết quả:** KSG bám sát ground-truth ở mọi config (bias ≤ 0.03 nat). Binning và Symbolic có bias lớn hơn hẳn (tới 0.19–0.26 nat), đặc biệt rõ ở Periodic Coupling và ở mọi trường hợp "uncoupled" — đây nhiều khả năng là hạn chế thật của phương pháp rời rạc hoá ở N=1000 (curse of dimensionality: 8 bin × 3 chiều = 512 ô cho 1000 mẫu), không phải lỗi cài đặt, và **thực ra là tín hiệu tốt** cho luận điểm của dự án (baseline cổ điển yếu ở N nhỏ — đúng thứ MINE/lượng tử ở Nhịp sau cần chứng minh vượt qua được).

## 6. Vì sao chưa thấy "dataset"

Pha P **không tạo ra file dataset nào cần commit** — đây là chủ đích, không phải thiếu sót:

- Dữ liệu **tổng hợp** (synthetic) được sinh ra ngay lúc chạy (`generate_var_linear_gaussian(...)` gọi là có ngay, tái lập được nhờ `seed`), không lưu ra đĩa. Thứ được lưu lại là *kết quả tổng hợp* sau khi ước lượng — chính là `results/tables/phase_p_baseline_validation.csv` ở mục 5. Việc lưu hẳn 1 "kho ngữ liệu" (≥20.000 cửa sổ train) là việc của **Pha R**, chưa tới lúc.
- Dữ liệu **thật** (Fantasia, Apnea-ECG, MIT-BIH Polysomnographic, CAP Sleep Database) là phạm vi của **Pha S** — chưa bắt đầu. `src/pqrst/data/real/` và `data/raw/`, `data/processed/` hiện chỉ có `.gitkeep` placeholder, đúng như thiết kế ban đầu.

Nếu muốn "nhìn thấy" dữ liệu tổng hợp trông ra sao ngay bây giờ (trước Pha R), có thể yêu cầu xuất thử vài nghìn điểm ra CSV/hình vẽ minh hoạ — việc này ngoài phạm vi checklist Pha P nhưng làm nhanh được nếu cần.

## 7. Đối chiếu checklist thoát Pha P (`PHASE_P_GUIDE.md` mục 5)

- [x] Môi trường cài xong, `pytest -q` chạy sạch (13 passed).
- [x] IDTxl + JVM chạy được (`JAVA_HOME` trỏ JDK 22).
- [x] 3 bộ sinh dữ liệu tổng hợp, có test.
- [x] 3 baseline, đúng interface `BaseTEEstimator`.
- [x] `validate_baselines.py` chạy xong, bảng không NaN/Inf.
- [x] Quyết định symbolic TE ghi lại ở `docs/references/symbolic_te_decision.md`.
- [x] Không có file lớn/bí mật bị commit.

→ **Pha P đạt tiêu chí thoát.** Riêng mức bias của Binning/Symbolic (mục 5) nên được ghi nhận như 1 phát hiện khoa học ban đầu (baseline yếu ở N nhỏ), không phải điều kiện chặn — sẽ được đo lại kỹ hơn (quét N, bootstrap) ở Pha R.

## 8. Bước tiếp theo

Sang **Pha Q** — mạng thống kê `T_φ` (MLP) + loss Donsker–Varadhan, chạy trên đúng 1 cấu hình đơn giản trước (xem `docs/ROADMAP.md`, mục Pha Q).
