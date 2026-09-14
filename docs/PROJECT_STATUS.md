# Tổng quan tiến độ dự án PQRST (Q-BHC / AQNE-TE)

> File này là bản tóm tắt tiến độ TOÀN DỰ ÁN, cập nhật theo thời gian — khác
> `ROADMAP.md` (kế hoạch gốc, không đổi) và các `PHASE_*_REPORT.md` (chi tiết từng
> pha). Xem file này trước để biết đang ở đâu, còn gì phải làm.

**Cập nhật lần cuối:** 2026-09-14, ngay sau khi hoàn tất Pha S (bao gồm sửa lỗi
thống kê, kiểm tra độ nhạy, đối chứng độc lập).

---

## 1. Cấu trúc dự án

2 "Nhịp", mỗi nhịp 5 pha (P→T ở Nhịp 1 cổ điển, P′→T′ ở Nhịp 2 lượng tử — tái dùng
gần như toàn bộ hạ tầng Nhịp 1, chỉ hoán đổi `T_φ` (MLP) bằng `T_θ` (mạch lượng tử)
tại 1 điểm nối).

---

## 2. Đã hoàn thành

### Pha P — Khởi động ✅
3 bộ sinh dữ liệu tổng hợp (VAR tuyến tính Gaussian, phi tuyến, periodic coupling),
3 baseline TE (KSG/binning/symbolic), script validate. 1 lỗi công thức ground-truth
(Lyapunov covariance) đã phát hiện và sửa, verify bằng Monte Carlo 2M mẫu.
→ [`PHASE_P_REPORT.md`](PHASE_P_REPORT.md)

### Pha Q — MINE cổ điển ✅
`StatisticsNetwork`, Donsker-Varadhan loss, `shuffle_batch`, `train_mine`. Smoke
test MI(X[t-1];Y[t]) đạt sai số 0.8–7.6% (ngưỡng 20%), 20/20 test pass. Phát hiện
(không chặn): best-of-all-epochs gây lệch lạc quan (7.6% vs 2.86% thật).
→ [`PHASE_Q_REPORT.md`](PHASE_Q_REPORT.md)

### Pha R (+ R2) — Amortized MINE ✅
`MaskedStatisticsNetwork` (1 mạng chung cho `mi_full`/`mi_reduced`, giảm phương sai
qua tương quan dương), corpus 27k/4k/18k cửa sổ, đánh giá lưới đầy đủ N×coupling×
noise. 1 lỗi CHẶN (`grid.py` lệch chỉ số, ảnh hưởng cả 4 estimator) đã sửa, chạy lại
72.000 phép đo. Mở rộng R2: dữ liệu phi tuyến/periodic coupling.

**Phát hiện cốt lõi:** Amortized thắng KSG rõ và cách biệt tăng dần từ N≈30 trở lên
(dữ liệu tuyến tính); nhưng **yếu hơn KSG trên dữ liệu phi tuyến/periodic ở N nhỏ**
— phát hiện này quay lại ảnh hưởng trực tiếp tới việc đọc kết quả ở Pha S.
→ [`PHASE_R_REPORT.md`](PHASE_R_REPORT.md)

### Pha S — Dữ liệu thật ✅ (vừa xong)
**Đổi hướng giữa pha:** kế hoạch gốc dùng TE tim-não (slpdb/capslpdb), nhưng phát
hiện **100% bản ghi nhiễm nhiễu điện tim thật** trong kênh EEG (xác minh bằng hình
dạng sóng, không phải heartbeat-evoked potential) — thử 5 phương pháp khử nhiễu có
cơ sở khoa học đều không đạt. Chuyển kết quả chính sang **TE tim-hô hấp**
(Respiratory Sinus Arrhythmia) trên Fantasia + Apnea-ECG.

**Kết quả cuối (đã qua thống kê đúng phương pháp — không pseudo-replication):**
- Fantasia (N=17 bản ghi, đơn vị mẫu = bản ghi): KSG PASS, CI hiệu số `[0.0039,
  0.0353]`, Wilcoxon p=0.0116.
- Apnea-ECG (N=6, đối chứng độc lập thứ 2): cùng chiều, chưa đủ mạnh riêng lẻ.
- **Gộp cả 2 bộ (N=23, kiểu meta-analysis): CI `[0.0057, 0.0311]`, Wilcoxon
  p=0.0046** — bằng chứng mạnh nhất, độc lập 2 nguồn dữ liệu.
- Kiểm tra độ nhạy (9 tổ hợp bandpass × độ dài cửa sổ): **chiều ổn định tuyệt đối
  9/9**, ý nghĩa thống kê giảm dần có giải thích được (không phải dấu hiệu giả).
- Amortized MINE FAIL trên dữ liệu thật (TE âm) — khớp với phát hiện Pha R2 (yếu
  trên dữ liệu phi tuyến/dao động ở N nhỏ), ghi nhận là hạn chế đã biết.

**13 lỗi thực sự tìm và sửa** trong quá trình chạy thật (đồng bộ hoá, tải dữ liệu,
NaN, bất đối xứng lọc tín hiệu, thống kê pseudo-replication...).
→ [`PHASE_S_REPORT.md`](PHASE_S_REPORT.md), [`PHASE_S_REVIEW.md`](PHASE_S_REVIEW.md)

---

## 3. Đang làm / Tiếp theo

### Pha T — Kết quả & bản thảo checkpoint (Nhịp 1) — CHƯA BẮT ĐẦU
Theo `ROADMAP.md`, ước tính gốc 2-3 tuần:
- [ ] `run_main_experiment.py`: quét N trên các bộ dữ liệu × phương pháp. **Cần
      điều chỉnh phạm vi so với kế hoạch gốc**: chỉ Fantasia + Apnea-ECG dùng được
      trên dữ liệu thật (tim-não bị loại ở Pha S) + dữ liệu tổng hợp từ Pha R.
- [x] Bootstrap + permutation test — **đã có sẵn** từ Pha S
      (`sanity_check.bootstrap_ci`, `permutation_test_te`, `run_sanity_check_per_record`),
      tái dùng được, không cần viết lại.
- [ ] Viết bản thảo (LaTeX/Overleaf) — tổng hợp 4 báo cáo Pha P/Q/R/S thành 1 câu
      chuyện mạch lạc. Đây là phần việc lớn nhất còn lại của Pha T.
- **Cổng quyết định sau Pha T** (theo roadmap gốc): kết quả hiện tại (Amortized
  thắng rõ ở N≥30 synthetic tuyến tính, thua KSG trên phi tuyến/dữ liệu thật) là
  "kết quả có câu chuyện, ngang bằng" → **vẫn đáng sang Nhịp 2**, hạ kỳ vọng phần
  nào cho câu chuyện phi tuyến.

### Nhịp 2 (P′→T′, lượng tử) — CHƯA BẮT ĐẦU
Chưa có dòng code nào. Theo thiết kế, tái dùng nguyên `src/pqrst/data/`,
`data/processed/`, pipeline đánh giá — chỉ viết `src/pqrst/estimators/quantum/wrapper.py`
(`T_theta(x, y) -> scalar`, cùng chữ ký với `T_phi`) rồi cắm vào `train.py` đã có.
Có cổng dự phòng: ≥2 cấu hình không hội tụ (nghi barren plateau) → có thể dừng
Nhịp 2, dùng Nhịp 1 làm bản thảo hoàn chỉnh — không phải rủi ro chí mạng.

---

## 4. Đánh giá quy mô còn lại

| Mục tiêu | Còn lại | Ước tính |
|---|---|---|
| Nộp Q1/Q2 chỉ dựa trên Nhịp 1 | Chủ yếu viết bản thảo, không còn nhiều code lớn | Gần cán đích |
| Làm đủ cả 2 Nhịp (đúng scope gốc) | Toàn bộ Nhịp 2 (P′-T′) — 1 chu kỳ đầy đủ, tái dùng hạ tầng nên rẻ hơn xây từ đầu | Còn khá nhiều (~2-3 tháng theo ước tính gốc roadmap) |

---

## 5. Việc linh tinh chưa xử lý

- `verify_lyapunov.py` ở gốc repo — file scratch từ rất lâu, chưa track, đã hỏi
  nhiều lần chưa quyết định giữ hay xoá.
