# Pha T — Hướng dẫn chi tiết (Nhịp 1: Kết quả & Bản thảo checkpoint)

> **Khác các pha trước:** không có bug lớn nào cần lo (P/Q/R/S đã validate kỹ,
> 20+ lỗi đã tìm và sửa). Pha T chủ yếu là **tổng hợp** (không phải tính toán mới)
> và **viết** — hạ tầng thống kê (bootstrap, permutation test, kiểm định theo đơn
> vị bản ghi) đã có sẵn từ Pha S, tái dùng được toàn bộ.

**Input:** Pha S xong — sanity check ĐẠT trên dữ liệu thật (KSG, Fantasia+Apnea-ECG
gộp N=23, CI `[0.0057,0.0311]`, Wilcoxon p=0.0046).

**Tiêu chí thoát (nguyên văn roadmap gốc):** bộ kết quả hoàn chỉnh (bảng + hình) +
bản thảo checkpoint có thể nộp độc lập (Entropy / Frontiers in Network Physiology).

**⚠️ Điều chỉnh phạm vi so với `ROADMAP.md` gốc:** roadmap viết "quét N trên 4 bộ
dữ liệu × 4 phương pháp" — giả định trước khi biết kết quả Pha S. Thực tế: TE
tim-não (`slpdb`/`capslpdb`) đã bị loại khỏi kết quả chính (nhiễm nhiễu điện tim
không khử được). Phạm vi dữ liệu thật của Pha T chỉ còn **2 bộ** (Fantasia +
Apnea-ECG, đã gộp), cộng dữ liệu tổng hợp (VAR tuyến tính + phi tuyến/periodic) từ
Pha R/R2.

---

## 0. Ước tính thời gian tổng — tóm tắt cho người bận

| Việc | Bản chất | Thời gian |
|---|---|---|
| T.1 Tổng hợp thí nghiệm chính | **Tổng hợp** số liệu đã tính từ Pha R/R2/S, không tính lại từ đầu | 1–2 ngày |
| T.2 Formalize estimator lai (hybrid) | Code mới nhưng nhỏ, tái dùng lưới đã có | 1 ngày |
| T.3 Hiệu chỉnh bias (tuỳ chọn) | Code mới nhỏ | 0.5–1 ngày (bỏ được) |
| T.4 Hạ tầng thống kê | **Đã có sẵn** từ Pha S, chỉ cần tích hợp | 0.5 ngày |
| T.5 Hình/bảng cho bản thảo | Chỉnh lại hình đã có + vài hình mới | 1–2 ngày |
| T.6 Viết bản thảo | **Phần lớn nhất, biến động nhiều nhất** | 1.5–4 tuần |
| T.7 Review nội bộ trước khi nộp | Đối chiếu số liệu bài viết vs báo cáo/notebook | 1–2 ngày |
| **Tổng cộng** | | **~2–5 tuần**, phần code chỉ ~1 tuần |

**Vì sao phần code nhẹ hơn ước tính gốc (2-3 tuần cho CẢ Pha T):** `bootstrap_ci`,
`permutation_test_te`, `run_sanity_check_per_record` đã viết và test đầy đủ ở Pha S
— không cần viết `src/pqrst/evaluation/bootstrap.py`/`permutation_test.py` mới như
roadmap gốc dự tính. Việc chính còn lại là **tổng hợp + viết**, không phải xây hạ
tầng.

---

## 1. T.1 — Tổng hợp thí nghiệm chính (KHÔNG tính toán lại từ đầu)

Mục tiêu: 1 script/notebook duy nhất tái tạo được TOÀN BỘ bảng/hình kết quả chính
cho bản thảo, đọc từ dữ liệu **đã tính sẵn** — không chạy lại 72.000+28.800 phép đo
synthetic (Pha R/R2) hay pipeline dữ liệu thật (Pha S).

**Nguồn dữ liệu có sẵn, chỉ cần đọc và tổng hợp:**
- `results/tables/phase_r_grid_summary.csv` — lưới VAR tuyến tính (N×coupling×noise, 4 estimator)
- `results/tables/phase_r2_periodic_grid_summary.csv` — lưới periodic coupling
- `results/tables/phase_r_ablation_small_summary.csv` — ablation mạng nhỏ
- `data/processed/fantasia/*.npz` + `quality_report.csv`, `data/processed/apnea-ecg/quality_report.csv` — dữ liệu thật đã tiền xử lý

**Việc cần làm:** viết `scripts/run_main_experiment.py` (hoặc 1 notebook tổng hợp
`notebooks/phase_t_01_main_results.ipynb`):
1. Đọc 2 file grid summary (linear + periodic), gộp thành 1 bảng duy nhất có cột
   `dataset_type` (linear/periodic) — đây là bảng chính "Amortized vs 3 baseline
   qua N, cả 2 loại dữ liệu".
2. Đọc kết quả thật (Fantasia+Apnea-ECG, đã có số liệu chính xác trong
   `PHASE_S_REPORT.md` mục 5.4) — KHÔNG cần chạy lại pipeline, chỉ cần đọc lại
   `data/processed/*/quality_report.csv` + windows đã lưu, gọi
   `bidirectional_te_per_record` + `run_sanity_check_per_record` (đã có, nhanh).
3. Xuất bảng tổng hợp cuối + 1-2 hình mới (nếu cần) theo chuẩn bản thảo (xem T.5).

**Thời gian:** 1–2 ngày — chủ yếu là viết code đọc+gộp CSV/npz đã có, không phải
tính toán nặng.

---

## 2. T.2 — Formalize estimator lai (hybrid KSG/Amortized)

Pha R (mục 4.7, `PHASE_R_REPORT.md`) khuyến nghị mạnh: **dùng KSG cho N<30-50,
Amortized cho N≥30-50** — điểm giao cắt ổn định qua 3 kiểm định độc lập (mạng lớn,
mạng nhỏ, dữ liệu phi tuyến). Hiện tại đây chỉ là 1 khuyến nghị bằng lời, chưa có
class code thật.

**Việc cần làm:**
```python
# src/pqrst/estimators/hybrid.py
class HybridTEEstimator(BaseTEEstimator):
    """TE = KSG neu N < threshold, Amortized neu N >= threshold. Nguong mac dinh
    30 - diem giao cat da xac nhan on dinh qua 3 kiem dinh doc lap (xem
    PHASE_R_REPORT.md muc 4.4-4.6)."""
    def __init__(self, amortized_estimator, threshold_n: int = 30):
        ...
    def estimate(self, x, y, **kwargs) -> float:
        n = len(x) - 1  # so mau cua so (tru placeholder)
        return self.ksg.estimate(x, y) if n < self.threshold_n else self.amortized.estimate(x, y)
```
Validate: áp ngay lên lưới ĐÃ CÓ (`phase_r_grid_summary.csv` + `phase_r2_periodic_grid_summary.csv`)
— không cần chạy lại, chỉ cần chọn giá trị Amortized hoặc KSG theo N cho mỗi dòng
đã có sẵn, rồi tính lại bias/variance của "Hybrid". Thêm test đơn giản xác nhận
logic chuyển ngưỡng đúng.

**Thời gian:** 1 ngày (code nhỏ, validate trên dữ liệu có sẵn, không train/chạy lại gì).

---

## 3. T.3 — Hiệu chỉnh bias (tuỳ chọn, có thể bỏ)

Pha R (mục 6.1) đề xuất: bias của Amortized khá ổn định theo N → fit 1 hồi quy đơn
giản `bias(N)` trên tập validation, trừ đi lúc suy luận → cải thiện MSE (không chỉ
variance) ở N≥30. **Đây là việc tuỳ chọn** — không ảnh hưởng tiêu chí thoát (kết
quả real-data đã PASS mà không cần calibration này), chỉ làm câu chuyện synthetic
"đẹp" hơn (Amortized thắng cả MSE, không chỉ variance).

**Quyết định:** làm nếu còn thời gian sau T.1/T.2, bỏ nếu cần rút ngắn tiến độ —
không chặn việc nộp bản thảo.

**Thời gian nếu làm:** 0.5–1 ngày.

---

## 4. T.4 — Hạ tầng thống kê: ĐÃ CÓ, chỉ cần tích hợp

Roadmap gốc yêu cầu viết `src/pqrst/evaluation/bootstrap.py` và `permutation_test.py`
mới — **không cần nữa**, đã có từ Pha S:

| Hạ tầng | File | Trạng thái |
|---|---|---|
| Bootstrap CI (≥500 lần) | `sanity_check.bootstrap_ci` | ✅ Có, test đầy đủ |
| Permutation test | `sanity_check.permutation_test_te` | ✅ Có |
| Kiểm định ĐÚNG đơn vị mẫu (theo bản ghi, không pseudo-replication) | `sanity_check.run_sanity_check_per_record` + `bidirectional_te_per_record` | ✅ Có, Wilcoxon kèm |

**Việc cần làm:** chỉ gọi lại các hàm này trong `run_main_experiment.py` (T.1) cho
cả nhánh synthetic (nếu muốn CI cho bảng grid, hiện tại `phase_r_grid_summary.csv`
có thể đã có sẵn từ trước — kiểm tra trước khi tính lại) và nhánh dữ liệu thật (đã
có số từ Pha S, chỉ cần đọc lại).

**Thời gian:** 0.5 ngày (kiểm tra + gọi hàm, không viết mới).

---

## 5. T.5 — Hình & bảng cho bản thảo

**Đã có sẵn** (`results/figures/*.png`, dùng được trực tiếp hoặc chỉnh nhẹ):
`phase_q_mine_smoke_test_loss.png`, `phase_r_variance_vs_n.png`, `phase_r_bias_vs_n.png`,
`phase_r_ablation_variance_comparison.png`, `phase_r2_periodic_variance_vs_n.png`,
`phase_s_bidirectional_te.png`.

**Cần làm mới:**
- 1 bảng/hình gộp linear + periodic + hybrid (từ T.1/T.2) — chưa có bản gộp nào.
- Nếu nộp tạp chí tiếng Anh: **dịch nhãn trục/tiêu đề hình sang tiếng Anh** (hiện
  toàn bộ hình đang có chữ tiếng Việt "khong dau" trong code vẽ) — cần chạy lại
  đúng cell vẽ hình với label mới, không cần tính lại số liệu.

**Thời gian:** 1–2 ngày (chủ yếu là dịch nhãn + vẽ 1-2 hình gộp mới).

---

## 6. T.6 — Viết bản thảo (việc lớn nhất, thời gian biến động nhiều nhất)

**Cấu trúc đề xuất** (Entropy/Frontiers in Network Physiology — cả 2 đều nhận bài
phương pháp luận + thực nghiệm về ước lượng thông tin trên tín hiệu sinh lý):

| Mục | Nội dung | Nguồn |
|---|---|---|
| Abstract + Introduction | Động lực: ước lượng TE amortized, MINE, ghép nối tim-hô hấp/tim-não | Tổng hợp |
| Methods | Bộ sinh dữ liệu (Pha P), baseline (Pha P), `MaskedStatisticsNetwork` + chuẩn hoá (Pha Q/R/S0), pipeline dữ liệu thật + kiểm định (Pha S) | `PHASE_P/Q/R/S_REPORT.md` |
| Results | Bảng/hình từ T.1, kết quả real-data (KSG PASS, 2 nguồn độc lập), estimator lai | T.1, T.2, T.5 |
| Discussion — hạn chế (2 mục quan trọng, đã điều tra kỹ, KHÔNG che giấu) | (a) Amortized yếu trên dữ liệu phi tuyến/dao động ở N nhỏ (Pha R2 + Pha S mục 5.5); (b) nhiễm nhiễu điện tim trong PSG EEG không khử được bằng 5 phương pháp chuẩn (Pha S mục 5.2) | `PHASE_R_REPORT.md` mục 4.6, `PHASE_S_REPORT.md` mục 5.2/5.5 |
| Conclusion + Future work | Hướng cải thiện N nhỏ (control-variate/bound khác), Nhịp 2 (lượng tử) | `PHASE_R_REPORT.md` mục 4.7 |

**Vì sao thời gian biến động lớn (1.5–4 tuần):** phụ thuộc tốc độ viết của bạn,
số vòng góp ý với mentor, và có dịch/viết trực tiếp tiếng Anh hay không — đây là
yếu tố tôi không kiểm soát/ước tính chính xác được. Tư liệu (số liệu, hình, lập
luận) đã ĐẦY ĐỦ và chính xác từ 4 báo cáo pha — phần việc chính là **tổ chức lại
thành văn bản mạch lạc**, không phải tìm thêm bằng chứng mới.

**Gợi ý cách rút ngắn:** viết Methods + Results trước (dữ liệu đã có sẵn, ít cần
sáng tạo văn phong) — thường nhanh hơn — rồi mới viết Introduction/Discussion
(cần liên hệ tài liệu rộng hơn, tốn thời gian đọc/trích dẫn hơn).

---

## 7. T.7 — Review nội bộ trước khi nộp

Đối chiếu MỌI số liệu trong bản thảo với đúng file gốc (báo cáo/notebook/CSV) —
tránh lệch số do gõ tay/nhớ nhầm khi viết. Việc này quan trọng vì dự án đã có
lịch sử nhiều lỗi tinh vi (lệch chỉ số, pseudo-replication...) — sai số liệu ở
bản thảo cuối là rủi ro cao nhất, dễ tránh nhất.

**Thời gian:** 1–2 ngày.

---

## 8. Checklist thoát Pha T

**Nhóm A — Tổng hợp (bắt buộc):**
- [ ] `run_main_experiment.py` đọc + gộp bảng linear/periodic/real-data thành 1 bộ kết quả chính
- [ ] `HybridTEEstimator` implement + validate trên lưới đã có
- [ ] Hình gộp linear+periodic+hybrid mới

**Nhóm B — Tuỳ chọn (không chặn):**
- [ ] Hiệu chỉnh bias cho Amortized (bỏ được nếu thiếu thời gian)

**Nhóm C — Bản thảo (bắt buộc):**
- [ ] Bản thảo đầy đủ Abstract→Conclusion
- [ ] Toàn bộ hình dịch nhãn (nếu nộp tạp chí tiếng Anh)
- [ ] Review đối chiếu số liệu 100% khớp báo cáo gốc

**Nhóm D — Chất lượng:**
- [ ] `pytest -q` toàn bộ pass (bao gồm test mới cho `HybridTEEstimator`)

---

## 9. Phân công

| Việc | Ai làm |
|---|---|
| Viết code (T.1, T.2, T.3) | Agent coding — không commit/push |
| Chạy notebook tổng hợp (nhẹ, không cần chạy lại pipeline nặng) | Chủ dự án hoặc agent — không tốn nhiều thời gian máy |
| Viết bản thảo | **Chủ dự án** (có thể nhờ Claude soát câu chữ/dịch từng đoạn nếu cần) |
| Review code + số liệu bản thảo | Claude (reviewer) |
| Commit / push / nộp bài | **Chủ dự án** |

---

## 10. Bước tiếp theo

Xong Pha T → **cổng chuyển Nhịp** (nguyên văn roadmap gốc):
- Kết quả tốt → sang Nhịp 2 với câu chuyện mạnh.
- Kết quả ngang bằng (đúng tình huống hiện tại: Amortized thắng rõ N≥30 tuyến
  tính, thua KSG ở N nhỏ/phi tuyến/dữ liệu thật) → **vẫn đáng sang Nhịp 2**, hạ kỳ
  vọng phần câu chuyện phi tuyến.
- Hết thời gian/nguồn lực → dừng ở đây, nộp bản thảo checkpoint — điểm dừng an
  toàn, không phải thất bại.
