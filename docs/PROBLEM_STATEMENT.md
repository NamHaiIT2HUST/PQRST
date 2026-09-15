# Định nghĩa vấn đề: Input, Output, và Khoảng trống nghiên cứu (Research Gap)

> Tài liệu này viết riêng để làm rõ 3 câu hỏi mentor thường hỏi đầu tiên: **đầu vào
> là gì, đầu ra là gì, và khoảng trống nghiên cứu cụ thể mà đề tài lấp là gì.**
> Không lặp lại toàn bộ tiến độ (xem `PROJECT_STATUS.md` cho việc đó) — file này
> chỉ tập trung làm RÕ khung vấn đề bằng số liệu + hình thật, không suy diễn thêm.

**Câu hỏi khoa học trung tâm:** liệu một estimator transfer entropy (TE) học sẵn
một lần (amortized, kiểu MINE) có thể **thắng các estimator TE cổ điển
(KSG/binning/symbolic) ở kích thước cửa sổ N nhỏ** — đúng vùng mà tín hiệu sinh lý
thực tế (ECG, hô hấp, EEG) hay gặp phải do cửa sổ ghi ngắn/không dừng?

---

## 1. Input — chính xác là gì (3 cấp độ)

### 1.1. Dữ liệu thô (raw)

| Nguồn | Loại tín hiệu | Định dạng | Vai trò |
|---|---|---|---|
| Synthetic — VAR tuyến tính Gaussian (Pha P) | 2 chuỗi thời gian sinh từ mô hình có **TE lý thuyết biết trước** | mảng số, sinh bằng code | Kiểm định estimator có đúng không (ground truth) |
| Synthetic — Periodic coupling phi tuyến (Pha R2) | Tương tự, nhưng ghép nối **phi tuyến/tuần hoàn** | mảng số | Kiểm định estimator trên bài toán khó hơn VAR |
| Fantasia (PhysioNet) | ECG (250 Hz) + hô hấp (respiratory belt, cùng file WFDB) | `.dat/.hea/.ecg` (wfdb) | Dữ liệu THẬT — cặp tim–hô hấp |
| Apnea-ECG (PhysioNet) | ECG + hô hấp (2 file WFDB riêng cho cùng 1 lần ghi) | `.dat/.hea/.qrs` (wfdb) | Dữ liệu THẬT — đối chứng độc lập thứ 2 |
| slpdb / capslpdb (PhysioNet) | ECG + EEG (PSG) | wfdb | **Đã loại khỏi phân tích chính** — xem mục 5, Gap 3 |

### 1.2. Sau tiền xử lý — cấp độ đưa vào pipeline

Với dữ liệu thật, input thực sự của toàn bộ hệ thống là **2 chuỗi thời gian đã
đồng bộ trên cùng 1 lưới thời gian rời rạc**:

- **X(t) = RR interval** (khoảng cách giữa 2 nhịp tim liên tiếp, giây) — suy ra từ
  annotation nhịp ECG, loại nhịp lỗi (ectopic), nội suy tuyến tính, lọc bằng-thông
  0.1–0.5 Hz (dải nhịp hô hấp người lớn, 6–30 lần/phút).
- **Y(t) = Respiration** (tín hiệu cảm biến cơ/trở kháng lồng ngực) — cùng lọc
  bằng-thông 0.1–0.5 Hz.
- **Lưới thời gian chung: 4 Hz** — cả 2 chuỗi được nội suy về đúng lưới này để mỗi
  chỉ số mẫu tương ứng đúng 1 thời điểm thật (đây từng là 1 bug đã sửa — xem
  `PHASE_S_REPORT.md` mục 5, lỗi số 12).

![Input signals](../results/figures/phase_t_input_output_signals.png)

*Hình trên: X(t) (RR, đỏ) và Y(t) (hô hấp, xanh) sau lọc, trên bản ghi thật
Fantasia `f1o01`, 3 phút đầu. Chú ý 2 chuỗi dao động cùng tần số (nhịp hô hấp) —
đây chính là hiện tượng sinh lý cần đo hướng ghép nối (RSA). Panel dưới minh hoạ
đơn vị chia cửa sổ thật dùng cho ước lượng TE: **cửa sổ 30 giây không chồng lấn =
120 mẫu ở 4 Hz.**

### 1.3. Đơn vị đưa vào estimator (input thật của hàm `estimate(x, y)`)

Với MỖI cửa sổ N mẫu, estimator nhận vào **2 vector con trễ 1 bước** để ước lượng
TE theo 1 hướng:

```
TE(X->Y) can:  Y_t (dai N-1, tu mau 2..N)
               Y_lag = Y[0:N-1]  (Y qua khu)
               X_lag = X[0:N-1]  (X qua khu, "nguon" thong tin dang xet)
estimate(x, y) trong code = estimate(X_lag, {Y_lag, Y_t})
```

Đổi vai trò X↔Y để có hướng ngược lại (`TE(Y->X)`) — mỗi cửa sổ sinh ra **2 số**
(1 cho mỗi hướng).

---

## 2. Output — chính xác là gì (3 cấp độ)

### 2.1. Mức thấp nhất: 1 số thực (nats) cho 1 cửa sổ, 1 hướng

`TE(X→Y)` — lượng thông tin về `Y` trong tương lai được giải thích thêm bởi quá
khứ của `X`, sau khi đã biết quá khứ của chính `Y`. Đơn vị: nats (log tự nhiên).

### 2.2. Mức bản ghi (record): tổng hợp qua nhiều cửa sổ

Với dữ liệu thật, 1 bản ghi (vd `f1o01`, ~2 giờ) chia thành nhiều cửa sổ 30s →
lấy **trung bình TE qua các cửa sổ** cho mỗi hướng → mỗi bản ghi cho ra đúng
**2 số**: `TE_forward` (hô hấp→tim), `TE_backward` (tim→hô hấp).

### 2.3. Mức tập dữ liệu: kết luận thống kê

Trên N bản ghi độc lập → **bootstrap CI 95% của hiệu số** `TE_forward - TE_backward`
+ **Wilcoxon signed-rank p-value** (kiểm định phi tham số, đơn vị mẫu = bản ghi,
không phải cửa sổ — tránh pseudo-replication, xem Gap 4).

**Kết quả THẬT đã đạt (KSG, gộp Fantasia+Apnea-ECG, N=23 bản ghi):**

| | TE(hô hấp→tim) | TE(tim→hô hấp) | CI 95% hiệu số | Wilcoxon p |
|---|---|---|---|---|
| **KSG** | 0.1175 | 0.0996 | **[0.0057, 0.0311]** | **0.0046** |

→ CI hoàn toàn dương, p<0.01: **hô hấp→tim mạnh hơn tim→hô hấp**, đúng hướng RSA
đã biết trong y văn (Grossman & Taylor 2007). Đây là câu trả lời cho câu hỏi
**sinh lý học** — khác với câu hỏi **chất lượng estimator** ở mục 2.4.

![Output: bidirectional TE](../results/figures/phase_s_bidirectional_te.png)

### 2.4. Với dữ liệu synthetic: output là chỉ số CHẤT LƯỢNG estimator

Vì synthetic biết trước TE lý thuyết, output ở đây không phải câu trả lời sinh lý
mà là **bias, variance, MSE** của mỗi estimator so với giá trị thật — dùng để so
sánh 4 phương pháp (KSG/Binning/Symbolic/Amortized) theo N.

![Output: estimator quality vs N](../results/figures/phase_t_main_variance_vs_n.png)

---

## 3. Sơ đồ tổng thể input → xử lý → output

```
                     ┌──────────────────────────┐
   Du lieu THAT  --->│ ECG (nhip) + Ho hap (tho) │
   (Fantasia/         └──────────────┬───────────┘
    Apnea-ECG)                       │ trich RR, loc bang-thong, dong bo luoi 4Hz
                                      v
                     ┌──────────────────────────┐
   Du lieu           │  X(t)=RR(t), Y(t)=RESP(t)│ <-- INPUT muc 1.2
   SYNTHETIC   ------>│  (2 chuoi da dong bo)    │
   (VAR/periodic,                                │
    biet TE thuc)                      chia cua so N mau (30s = 120 mau)
                                      v
                     ┌──────────────────────────┐
                     │  estimator.estimate(X,Y) │ <-- KSG / Binning / Symbolic /
                     │  (KHONG BIET truoc N/loai │     Amortized(MINE) / Hybrid
                     │   dong luc hoc)           │
                     └──────────────┬───────────┘
                                     │ 1 so (nats) / cua so / huong
                                     v
              trung binh qua cua so cung 1 ban ghi  ->  TE_forward, TE_backward
                                     │
                                     v
        bootstrap CI + Wilcoxon (N ban ghi doc lap) ->  KET LUAN THONG KE cuoi
```

---

## 4. Research Gap — khoảng trống cụ thể đề tài lấp

**Gap 1 — Estimator cổ điển yếu ở N nhỏ, chưa có lựa chọn thay thế được kiểm định
kỹ trên dữ liệu sinh lý thật.** KSG/binning/symbolic đều là estimator k-NN/tần
suất — cần N đủ lớn để ước lượng mật độ ổn định. Với tín hiệu sinh lý thực tế
(cửa sổ ngắn để bắt được động lực học không-dừng, hoặc dữ liệu wearable/thời gian
thực), N thường chỉ 10–50 mẫu. MINE amortized (train 1 mạng `T_φ` 1 lần, suy luận
tức thời cho mọi cửa sổ mới) hứa hẹn variance thấp hơn ở đúng vùng này — nhưng
**mức độ thắng/thua chính xác theo N chưa được đo trước khi có đề tài này.**

**Gap 2 — Điểm giao cắt (crossover N) giữa 2 lớp phương pháp phụ thuộc vào loại
động lực học, chưa có công cụ chọn tự động.** Đề tài đo trực tiếp: điểm giao cắt
≈N=30 (tuyến tính) nhưng ≈N=50 (phi tuyến/periodic) — khác nhau đủ để 1 ngưỡng
sai (30 áp cho cả 2 loại) làm sai quyết định ở đúng N=30–49 trên dữ liệu phi
tuyến (bug thật đã bắt được — xem `PHASE_T_REPORT.md` mục 3). → xây
`HybridTEEstimator` với ngưỡng an toàn chung = 50.

**Gap 3 — Domain generalization: model học trên corpus synthetic có tổng quát
hoá lên dữ liệu sinh lý thật không? Đây là câu hỏi thường bị BỎ QUA trong các
nghiên cứu MINE/amortized ước lượng thông tin (thường chỉ báo cáo kết quả trên
synthetic).** Đề tài kiểm định trực tiếp và tìm ra: KSG tổng quát tốt (PASS,
mục 2.3), Amortized (chưa hiệu chỉnh) **THẤT BẠI hoàn toàn** trên dữ liệu thật
(`TE(hô hấp→tim)=-0.2474` — âm, sai cả hướng, Wilcoxon p=1.0000). Điều tra tận
gốc (không chỉ ghi nhận) cho thấy nguyên nhân: `mi_reduced` (chỉ dùng quá khứ của
chính Y) lớn hơn `mi_full` (dùng cả X) ~5 lần trên dữ liệu thật — mạng học được
biểu diễn không tổng quát hoá đúng ra ngoài phân phối synthetic đã train. **Đây
là 1 phát hiện khoa học có giá trị** (giới hạn thật của amortized MINE khi lệch
domain), không phải lỗi cần giấu.

**Gap 4 — Thống kê đúng đơn vị mẫu khi kiểm định hướng ghép nối tim–hô hấp trên
dữ liệu thật.** Nhiều nghiên cứu RSA gộp mọi cửa sổ như mẫu độc lập (pseudo-
replication) — thổi phồng độ tin cậy giả. Đề tài sửa: đơn vị kiểm định = BẢN GHI
(N=23), không phải cửa sổ, cho kết quả CI rộng hơn nhưng **đáng tin cậy hơn** và
vẫn loại trừ 0 hoàn toàn.

**Gap 5 (mở ra Nhịp 2, hướng tương lai) — mạch lượng tử ít tham số có giải quyết
được yếu điểm N nhỏ của amortized không?** Ablation mạng nhỏ cổ điển (`[16,16]`
so với `[128,128,64]`) đã kiểm định giả thuyết "ít tham số → tổng quát tốt hơn ở
N nhỏ" TRƯỚC khi đầu tư vào lượng tử (xem `PHASE_R_REPORT.md` mục 6–8) — kết quả
định hướng cho việc mạch lượng tử (cũng ít tham số) có đáng làm hay không.

---

## 5. Bảng tóm tắt 1 trang cho mentor

| | Input | Output | Câu hỏi trả lời |
|---|---|---|---|
| **Synthetic (P, R, R2)** | 2 chuỗi mô phỏng, TE lý thuyết biết trước | bias/variance/MSE theo N, theo estimator | Estimator nào chính xác nhất ở N nào? |
| **Thật (S, T)** | RR + hô hấp đồng bộ, N=23 bản ghi (Fantasia+Apnea-ECG) | TE_forward, TE_backward mỗi bản ghi → CI + p-value gộp | Hướng ghép nối tim–hô hấp nào chiếm ưu thế? (Trả lời: hô hấp→tim, CI [0.0057,0.0311], p=0.0046) |
| **Research gap chính** | — | — | Amortized MINE thắng variance ở N≥50 nhưng KHÔNG tổng quát hoá được lên dữ liệu thật (thất bại có điều tra nguyên nhân) — KSG vẫn là lựa chọn an toàn cho dữ liệu thật hiện tại; Hybrid + calibration là 2 hướng cải thiện đã kiểm định trên synthetic |
