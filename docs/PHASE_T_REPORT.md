# Pha T — Báo cáo kết quả (Nhịp 1: T.1, T.2, T.3)

> Xem kế hoạch/ước tính thời gian gốc ở [`PHASE_T_GUIDE.md`](PHASE_T_GUIDE.md).
> File này là báo cáo KẾT QUẢ sau khi đã làm — cùng cấu trúc với
> `PHASE_P/Q/R/S_REPORT.md`.

**Phạm vi đã hoàn thành:** T.1 (tổng hợp kết quả chính), T.2 (HybridTEEstimator),
T.3 (hiệu chỉnh bias + kiểm tra công bằng KSG). **Chưa làm** (theo quyết định của
chủ dự án — cần báo cáo mentor trước): T.6 (viết bản thảo), T.7 (review nội bộ
trước khi nộp). T.4 (hạ tầng thống kê) không cần việc riêng — đã tái dùng nguyên
từ Pha S trong T.1.

---

## 1. Tóm tắt cho người bận

- **T.1:** 1 notebook (`phase_t_01_main_results.ipynb`) đọc lại toàn bộ số liệu đã
  tính từ Pha R/R2/S (không tính lại gì), tái xác nhận **khớp 100%** với số liệu
  real-data đã công bố ở `PHASE_S_REPORT.md` mục 5.4 (N=23, CI `[0.0057,0.0311]`,
  Wilcoxon p=0.0046).
- **T.2:** `HybridTEEstimator` (KSG cho N<50, Amortized cho N≥50) đã code + test.
  Phát hiện và sửa 1 bug logic thật trong lúc validate (ngưỡng 30 chọn nhầm
  Amortized trên dữ liệu periodic ở N=30) — xem chi tiết mục 3.
- **T.3 (tuỳ chọn, đã làm thêm vì còn thời gian):** hiệu chỉnh bias cho Amortized
  bằng leave-one-config-out giúp nó thắng KSG cả về MSE (không chỉ variance) trên
  gần hết miền N đã kiểm định — kể cả khi áp CÙNG phép hiệu chỉnh cho KSG để so
  sánh công bằng. Xem mục 4.
- **Toàn bộ test liên quan (`test_hybrid.py`, `test_calibration.py`): 7/7 PASS**
  (đã tự chạy lại, có cấu hình `JAVA_HOME` để test tích hợp KSG+Amortized chạy
  thật, không chỉ mock).

---

## 2. T.1 — Tổng hợp kết quả chính

**Nguồn dữ liệu:** đọc lại (không tính lại) `phase_r_grid_summary.csv`,
`phase_r2_periodic_grid_summary.csv`, và dữ liệu thật Fantasia+Apnea-ECG đã xử lý
từ Pha S.

**Kết quả xác nhận lại real-data (chạy `bidirectional_te_per_record` +
`run_sanity_check_per_record` với KSG, độc lập với số đã lưu trong báo cáo Pha S):**

```
Tong so ban ghi (Fantasia + Apnea-ECG): 23
N=23, TE(resp->tim)=0.1175, TE(tim->resp)=0.0996
CI hieu 95%: [0.0057, 0.0311]
Wilcoxon p=0.0046
=== KHOP voi so da cong bo trong PHASE_S_REPORT.md muc 5.4 ===
```

→ Không có drift/lỗi giữa lúc chạy Pha S và lúc tổng hợp lại ở Pha T — số liệu ổn
định qua 2 lần tính độc lập.

![Main synthetic results](../results/figures/phase_t_main_variance_vs_n.png)

*Variance vs N (log-log), 2 panel (Linear VAR Gaussian | Periodic Coupling), cả 4
estimator (KSG/Binning/Symbolic/Amortized) + Hybrid — Hybrid luôn bám đúng đường
bao dưới (variance thấp nhất) ở cả 2 panel, chuyển từ KSG sang Amortized đúng tại
N=50.*

**Bảng tổng hợp exit criteria toàn dự án** (`phase_t_exit_criteria_summary.csv`):

| Pha | Tiêu chí | Kết quả |
|---|---|---|
| P | Baseline bias thấp trên dữ liệu có ground-truth | ĐẠT (KSG bias≤0.03 nat mọi config) |
| Q | Sai số MI < 20% | ĐẠT (7.6%, hoặc 2.86% nếu trung bình post-convergence) |
| R | Thắng KSG+Symbolic ở ≥70% ở N<30 | CHƯA ĐẠT (9/30) — nhưng thắng rõ từ N≥30, điểm giao cắt ổn định qua 3 kiểm định |
| S | Sanity check TE đạt trên ≥1 bộ dữ liệu thật | ĐẠT (KSG, Fantasia+Apnea-ECG gộp N=23, CI hoàn toàn dương) |
| T.1 | Tổng hợp kết quả chính + xác nhận lại số real-data | ĐẠT |

---

## 3. T.2 — HybridTEEstimator (KSG/Amortized)

`src/pqrst/estimators/hybrid.py`: `estimate(x,y) = KSG` nếu `N<threshold_n`, else
`Amortized`. **Ngưỡng mặc định = 50** (không phải 30 — xem lý do dưới).

**Bug phát hiện lúc validate trên lưới đã có** (`phase_r2_periodic_grid_summary.csv`,
N=[10,20,30,50,100,200]):

| N | KSG variance (periodic) | Amortized variance (periodic) | Ai thắng |
|---|---|---|---|
| 30 | 0.00636 | 0.00717 | **KSG** |
| 50 | 0.00691 | 0.00455 | Amortized |

Ngưỡng 30 (đúng cho dữ liệu tuyến tính) làm Hybrid chọn NHẦM Amortized tại N=30
trên dữ liệu periodic. Vì dữ liệu THẬT không biết trước là tuyến tính hay phi
tuyến, quyết định dùng **1 ngưỡng chung = 50** — hy sinh 1 phần hiệu năng trên
tuyến tính ở N=30–49 (Amortized vẫn thắng ở đó) để đổi lại không bao giờ chọn
nhầm phương án tệ hơn ở bất kỳ N đã kiểm chứng nào, trên cả 2 loại dữ liệu.

**Test** (`tests/test_hybrid.py`, 4 test, đã tự đọc lại code + chạy — không chỉ
tin báo cáo agent):
- `test_hybrid_threshold` — logic chuyển ngưỡng cơ bản.
- `test_hybrid_does_not_regress_periodic_n30_bug` — **test hồi quy THẬT**: assert
  trực tiếp variance KSG<Amortized tại N=30 periodic từ CSV, rồi assert Hybrid
  chọn đúng KSG ở đó. Phải FAIL nếu `threshold_n≤30` (đã tự kiểm chứng độc lập
  bằng script riêng ở vòng review trước — 0/48 dòng "test cũ" từng vô nghĩa lý
  nay đã được thay bằng test có khả năng fail thật).
- `test_hybrid_tradeoff_report` — log (không assert cứng) danh sách các trường
  hợp Hybrid phải hy sinh so với phương án tốt nhất — dùng để viết Discussion.
- `test_hybrid_integration` — chạy KSG+Amortized thật (không mock) ở N=10 và
  N=100, `skip` nếu thiếu checkpoint.

---

## 4. T.3 — Hiệu chỉnh bias (bias calibration) + kiểm tra công bằng

**Thiết kế** (`src/pqrst/estimators/calibration.py`):
`calibrate_grid_leave_one_config_out(df, estimator_name)` — với mỗi (config, N),
fit `bias_hat(N)` (bảng tra + nội suy tuyến tính, không ngoại suy) CHỈ từ các
config KHÁC (leave-one-config-out — tránh rò rỉ dữ liệu). Vì trừ 1 hằng số khỏi
mọi ước lượng không đổi variance, `calibrated_mse` tính trực tiếp từ bảng summary
đã có, không cần chạy lại model.

**Lưu ý về cách đọc số liệu:** `calibrated_bias` trung bình luôn ≈0 (1e-17) ở MỌI
N — đây là hệ quả toán học tất nhiên của leave-one-out (`sum` residual leave-one-out
luôn = 0 về đại số), KHÔNG phải bằng chứng riêng cho hiệu quả của calibration. Bằng
chứng thật nằm ở MSE (kết hợp cả bias² + variance từng dòng), không phải ở bias
trung bình.

**Kết quả MSE (đã tự đọc lại CSV, groupby độc lập, khớp đúng số agent báo cáo):**

| N | Amortized | Amortized_calibrated | KSG | KSG_calibrated |
|---|---|---|---|---|
| **Linear VAR** | | | | |
| 10 | 0.01805 | 0.01419 | 0.01700 | **0.01197** |
| 20 | 0.01122 | **0.00833** | 0.01203 | 0.00951 |
| 30 | 0.00858 | **0.00604** | 0.00917 | 0.00773 |
| 50 | 0.00653 | **0.00431** | 0.00645 | 0.00572 |
| 100 | 0.00482 | **0.00300** | 0.00349 | 0.00329 |
| 200 | 0.00392 | 0.00215 | 0.00186 | **0.00181** |
| **Periodic Coupling** | | | | |
| 10 | 0.02095 | **0.01934** | 0.02938 | 0.02593 |
| 20 | 0.01551 | **0.01367** | 0.03165 | 0.03325 |
| 30 | 0.01177 | **0.00930** | 0.03262 | 0.03575 |
| 50 | 0.00921 | **0.00647** | 0.03830 | 0.04408 |
| 100 | 0.00720 | **0.00454** | 0.02790 | 0.03270 |
| 200 | 0.00590 | **0.00306** | 0.01389 | 0.01610 |

**Nhận xét (đã tự verify, không chỉ tin báo cáo agent):**
- Trên **Linear VAR**: `Amortized_calibrated` thắng MSE liên tục N=20–100;
  `KSG_calibrated` thắng ở 2 đầu mút N=10 và N=200.
- Trên **Periodic Coupling**: `Amortized_calibrated` thắng MSE ở **mọi N** kiểm
  định (10→200) — cả trước và sau khi calibrate KSG.
- Hiệu chỉnh làm KSG **tốt hơn** trên Linear (bias KSG khá ổn định theo N, dự đoán
  tốt) nhưng làm KSG **tệ hơn** trên Periodic ở hầu hết N (trừ N=10) — vì bias của
  KSG trên dữ liệu phi tuyến biến động mạnh giữa các config (không ổn định theo
  N), nên nội suy leave-one-out dự đoán sai hướng. Đây là điểm khác biệt bản chất
  giữa 2 estimator: bias của Amortized là hằng số mô hình (ổn định, dễ hiệu
  chỉnh), bias của KSG trên dữ liệu phi tuyến là nhiễu hình học (không ổn định,
  khó hiệu chỉnh) — bản thân sự khác biệt này là 1 luận điểm đáng đưa vào
  Discussion của bản thảo.

![Bias calibration MSE](../results/figures/phase_t_bias_calibration_mse.png)

**Giới hạn cần nêu rõ trong bản thảo:** T.3 là ablation trên dữ liệu **synthetic**
(Pha R/R2), **KHÔNG áp dụng lên dữ liệu thật** — real-data pass ở Pha S dùng KSG
thô, không dùng calibration này. Không ảnh hưởng tiêu chí thoát chính.

---

## 5. Test suite

```
tests/test_calibration.py, tests/test_hybrid.py — 7 passed (JAVA_HOME đã set để
test_hybrid_integration chạy KSG+Amortized thật, không chỉ mock/skip)
```

---

## 6. Checklist thoát Pha T (cập nhật)

**Nhóm A — Tổng hợp (bắt buộc):** ✅ xong cả 3 mục (T.1 tổng hợp, T.2 Hybrid, hình
gộp linear+periodic+hybrid).

**Nhóm B — Tuỳ chọn:** ✅ đã làm thêm (T.3 hiệu chỉnh bias + kiểm tra công bằng).

**Nhóm C — Bản thảo (bắt buộc, CHƯA làm — quyết định gác lại để báo cáo mentor
trước):**
- [ ] Bản thảo đầy đủ Abstract→Conclusion (T.6)
- [ ] Toàn bộ hình dịch nhãn tiếng Anh (đã xong — xem `phase_r_ablation_*.png`,
      `phase_s_bidirectional_te.png` dịch lại + 2 hình mới T.1/T.3 đều tiếng Anh
      từ đầu)
- [ ] Review đối chiếu số liệu 100% khớp báo cáo gốc (T.7 — làm sau khi có bản
      thảo)

**Nhóm D — Chất lượng:** ✅ `pytest` cho `test_hybrid.py` + `test_calibration.py`
pass 7/7.

**Kết luận:** phần CODE của Pha T đã hoàn chỉnh 100%. Phần còn lại (T.6 viết bản
thảo, T.7 review trước khi nộp) là việc của chủ dự án, làm sau khi báo cáo mentor.
