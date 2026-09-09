# Pha S — Hướng dẫn chi tiết (Nhịp 1: Dữ liệu sinh lý thật)

> **Đây là lần đầu dự án chạm dữ liệu thật.** Pha P/Q/R đều chạy trên dữ liệu tổng hợp do chính ta sinh ra — biết trước đáp án, sạch, không artifact. Pha S đảo ngược hoàn toàn: không có ground-truth, dữ liệu bẩn, và **lỗi tiền xử lý thường ngụy trang thành phát hiện khoa học**. Roadmap gốc xếp rủi ro Pha S ở mức **Trung bình** với ghi chú quan trọng: *"Nếu sanity check không đạt, kiểm tra pipeline đồng bộ hoá TRƯỚC khi nghi ngờ mô hình."*

**Input:** Pha R xong — `T_φ` đã đóng băng, đã biết rõ vùng hoạt động (thắng KSG ở N≥50, thua ở N<30).

**Tiêu chí thoát (nguyên văn roadmap gốc):** sanity check `TE(tim→não) > TE(não→tim)` đạt trên **ít nhất một** bộ dữ liệu có nhãn rõ.

---

## 0. ⛔ ĐIỀU KIỆN TIÊN QUYẾT — phải làm TRƯỚC khi chạm dữ liệu thật

**Đây là phát hiện quan trọng nhất khi chuẩn bị Pha S. Bỏ qua mục này thì toàn bộ kết quả Pha S sẽ vô nghĩa nhưng *nhìn vẫn có vẻ ổn*.**

### 0.1. Vấn đề: `T_φ` hiện tại không bất biến theo thang đo

Tôi đã đo trực tiếp trên checkpoint hiện có (`phi_amortized.pt`), 300 cửa sổ test N=50/100:

| Đầu vào | bias | variance | **corr với kết quả đúng** |
|---|---|---|---|
| Nguyên gốc (như Pha R) | -0.0173 | 0.00085 | 1.00 (mốc) |
| Z-score hoá | -0.0751 | 0.00334 | **0.87** |
| Đổi thang đo `×0.1 + 0.8` (mô phỏng RR interval) | 0.0055 | 0.00010 | **0.07** ⚠️ |

**Đọc bảng này cho kỹ.** Dòng cuối — mô phỏng đúng thang đo của RR interval thật — có bias và variance **đẹp nhất bảng**, nhưng tương quan với đáp án đúng chỉ **0.07**, tức là gần như **không còn liên hệ gì với sự thật**. Con số "đẹp" đó là do output của mạng bị *sụp về gần hằng số*, không phải do chính xác. Nếu cứ thế chạy Pha S, bảng kết quả sẽ trông hoàn toàn bình thường và **không ai phát hiện ra là rác**.

**Nguyên nhân:** `T_φ` được train trên dữ liệu tổng hợp có `mean≈0, std≈0.8–1.0`. Dữ liệu thật hoàn toàn khác thang đo:
- RR interval: ~0.8 ± 0.1 giây (mean cách xa 0)
- EEG band power: 1–500 µV², chỉ dương, lệch phải mạnh

→ Mạng đóng băng nhận đầu vào **ngoài phân phối** (out-of-distribution) và cho ra kết quả vô nghĩa.

### 0.2. Cơ sở lý thuyết của cách khắc phục

TE/MI **bất biến dưới phép biến đổi affine từng biến riêng** — `TE(aX+b → cY+d) = TE(X→Y)`. Đã kiểm chứng bằng KSG:

| | TE |
|---|---|
| Ground truth | 0.1814 |
| KSG dữ liệu gốc | 0.1697 |
| KSG sau đổi thang đo `×0.1+0.8` | **0.1697** (giống hệt) |
| KSG sau z-score | 0.1726 |

KSG tôn trọng đúng tính chất lý thuyết này. `T_φ` thì không — **không phải vì mất mát thông tin, mà vì lệch phân phối train/test**. Nghĩa là chuẩn hoá **không làm mất gì về mặt lý thuyết**, hoàn toàn hợp lệ.

### 0.3. Việc phải làm (S0)

**Train lại `T_φ` trên corpus đã chuẩn hoá z-score theo từng cửa sổ**, rồi áp dụng cùng phép chuẩn hoá đó cho dữ liệu thật. Khi train và test cùng phân phối, không còn degradation.

- Thêm chuẩn hoá **per-window z-score** (dùng `pqrst.utils.standardize.standardize_window`) vào **cả** `train_amortized()` **và** `AmortizedTEEstimator.estimate()` — bật/tắt qua 1 cờ trong config để đảm bảo train/inference **luôn nhất quán**.
- Train lại → checkpoint mới `phi_amortized_standardized.pt` (**~2 giờ**, theo đo thực tế ở ablation trước).
- **Xác minh bắt buộc trước khi qua S1:**
  1. Trên tập test tổng hợp, bản chuẩn hoá đạt bias/variance **không tệ hơn đáng kể** bản gốc ở N≥50.
  2. Đưa cùng dữ liệu nhưng đổi thang đo (`×0.1+0.8`) → kết quả **gần như không đổi** (corr > 0.99). Đây là bằng chứng đã bất biến thang đo — chính là thứ đang thiếu.

Chỉ khi cả 2 điều kiện đạt mới được sang S1. Nếu không đạt → dừng, báo lại, đừng chạy tiếp lên dữ liệu thật.

---

## 1. Bốn bộ dữ liệu — sự thật quan trọng đã xác minh

**Đã kiểm tra trực tiếp trang PhysioNet của cả 4 bộ (không dựa vào trí nhớ):**

| Bộ dữ liệu | Kênh có thật | Có EEG? | Dung lượng | Định dạng |
|---|---|---|---|---|
| **MIT-BIH Polysomnographic** (`slpdb`) | ECG + **EEG** + hô hấp, 18 bản ghi, >80h | ✅ **CÓ** | 632 MB | WFDB `.dat/.hea` + `.ecg`/`.st` |
| **CAP Sleep** (`capslpdb`) | ≥3 kênh **EEG** + ECG + EOG + EMG + hô hấp, 108 bản ghi | ✅ **CÓ** | **40.1 GB** | EDF |
| **Fantasia** | ECG + hô hấp + huyết áp (nửa số bản ghi), 40 người, 250 Hz | ❌ **KHÔNG** | 293 MB | WFDB |
| **Apnea-ECG** | ECG (chỉ 8/70 bản ghi có thêm hô hấp + SpO2), 100 Hz | ❌ **KHÔNG** | 581 MB | WFDB |

### ⚠️ Hệ quả quan trọng: chỉ 2/4 bộ làm được tim–não

**Fantasia và Apnea-ECG KHÔNG có EEG** → **không thể** dùng cho luận điểm tim–não. Một agent code không biết điều này có thể mất nhiều ngày tìm kênh EEG không tồn tại.

Roadmap gốc thực ra đã đúng khi chỉ yêu cầu sanity check trên `slpdb` và `capslpdb`. Vai trò thật của 4 bộ:

| Bộ | Vai trò trong bài báo |
|---|---|
| `slpdb` | **Chính** — TE tim↔não, sanity check |
| `capslpdb` | **Chính** — TE tim↔não, tái lập độc lập trên nhóm bệnh nhân khác |
| Fantasia | **Phụ** — ghép nối tim↔hô hấp; so trẻ/già (hiệu ứng tuổi tác lên ghép nối) |
| Apnea-ECG | **Phụ** — tim↔hô hấp trên 8 bản ghi có hô hấp; phần còn lại chỉ dùng cho động lực học tim |

Phải **nói thẳng điều này trong bài báo** — không được để người đọc hiểu nhầm là đã kiểm chứng tim–não trên cả 4 bộ.

### Về dung lượng CAP Sleep (40 GB)

Máy còn 162 GB nên tải được, nhưng **không cần tải hết**. Khuyến nghị: tải trước **16 bản ghi người khoẻ mạnh** (`n1`–`n16`) — đủ cho sanity check, và nhóm khoẻ mạnh là đối chứng sạch nhất. `wfdb` cho phép tải từng bản ghi riêng, không bắt tải cả bộ.

---

## 2. Tiền xử lý — nơi lỗi ẩn nấp nhiều nhất

### 2.1. Kênh tim: ECG → chuỗi RR

- **`slpdb` có sẵn annotation nhịp (`.ecg`)** — dùng luôn, **không cần** tự chạy QRS detector. Đây là món quà lớn: bỏ được một nguồn lỗi.
- CAP Sleep (EDF) không có annotation nhịp → cần dò QRS (`neurokit2` hoặc `wfdb.processing.xqrs_detect`).
- **Lọc nhịp ngoại tâm thu (ectopic):** chỉ giữ nhịp loại `N` (normal). Một nhịp ngoại tâm thu tạo ra 1 khoảng RR ngắn bất thường + 1 khoảng dài bù trừ → nếu không lọc sẽ tạo tương quan giả rất mạnh. Quy tắc phổ biến: loại RR lệch >20% so với trung vị cục bộ, rồi nội suy.

### 2.2. Kênh não: EEG → công suất dải tần

- Tính công suất các dải `delta (0.5–4Hz)`, `theta (4–8)`, `alpha (8–13)`, `beta (13–30)` trên cửa sổ trượt.
- **Nên lấy `log` công suất** — phân phối công suất lệch phải rất mạnh; log giúp gần Gaussian hơn, hợp với giả định của estimator.

### 2.3. ⚠️ Đồng bộ hoá — vấn đề kỹ thuật khó nhất

RR là chuỗi **không đều theo nhịp tim**; EEG band power là chuỗi **đều theo thời gian**. Muốn tính TE, hai chuỗi phải nằm trên **cùng một lưới thời gian**.

Cách chuẩn: nội suy RR về lưới đều (khuyến nghị **4 Hz**), rồi tính EEG band power trên đúng lưới đó.

**Đây chính xác là chỗ roadmap cảnh báo "kiểm tra đồng bộ hoá trước khi nghi ngờ mô hình".** Lỗi lệch thời gian ở đây sẽ tạo ra TE giả theo hướng bất kỳ. Bắt buộc:
- Ghi rõ mốc thời gian gốc (t=0) của từng kênh, kiểm tra chúng thực sự cùng gốc.
- Vẽ 2 chuỗi chồng lên nhau ở vài đoạn ngẫu nhiên, **nhìn bằng mắt** để xác nhận khớp.
- Test tự động: dịch nhân tạo 1 kênh đi vài giây → TE phải **giảm rõ**. Nếu không đổi thì pipeline đang không thực sự đo quan hệ thời gian.

### 2.4. Chọn độ dài cửa sổ N — nối trực tiếp với kết quả Pha R

Pha R đã xác lập (ổn định qua 3 thử nghiệm): **`T_φ` thắng KSG ở N≥50, thua ở N<30**.

Với lưới 4 Hz:

| Cửa sổ thời gian | N (mẫu) | Vùng |
|---|---|---|
| 12.5 s | 50 | ranh giới |
| **30 s** | **120** | ✅ **Amortized thắng** |
| 60 s | 240 | ✅ Amortized thắng |

**Khuyến nghị: cửa sổ 30 giây → N=120.** Lý do đẹp: 30 giây đúng bằng **1 epoch chấm giai đoạn giấc ngủ tiêu chuẩn** — vừa khớp nhãn có sẵn của `slpdb`/`capslpdb`, vừa nằm gọn trong vùng `T_φ` mạnh. Đây là chỗ kết quả Pha R trở nên trực tiếp hữu dụng.

---

## 3. Sanity check — và cái bẫy phải tránh

### 3.1. Yêu cầu

Tính TE hai chiều trên `slpdb` và `capslpdb`, kiểm tra `TE(tim→não) > TE(não→tim)`.

### 3.2. ⚠️ BẪY: nhiễu trường tim (cardiac field artifact)

**Đây là cái bẫy nguy hiểm nhất của cả Pha S.** Tín hiệu điện tim rất mạnh và **rò rỉ vào điện cực EEG** (cardiac field artifact / ballistocardiogram). Nếu EEG bị nhiễm nhiễu tim:

→ `TE(tim→não)` sẽ cao **một cách giả tạo** → sanity check **ĐẠT** → nhưng đạt **vì lý do hoàn toàn sai**: ta chỉ đang đo tín hiệu tim rò vào chính nó, không phải ghép nối sinh lý tim–não.

Một kết quả "thành công" kiểu này còn tệ hơn thất bại, vì nó sẽ đi thẳng vào bài báo.

**Bắt buộc kiểm tra trước khi công bố bất kỳ kết quả sanity check nào:**
1. **Trung bình EEG khoá theo nhịp R (R-peak locked average):** cắt các đoạn EEG quanh mỗi đỉnh R rồi lấy trung bình. Nếu xuất hiện dạng sóng QRS rõ → EEG **đang bị nhiễm** nhiễu tim.
2. **Phổ EEG:** tìm đỉnh tại tần số tim (~1–1.5 Hz) và các hài của nó.
3. Nếu phát hiện nhiễm: hoặc loại kênh/bản ghi đó, hoặc khử nhiễu (ICA), và **ghi rõ trong báo cáo**.

### 3.3. Kiểm định ý nghĩa thống kê

TE ước lượng luôn ra số dương (do bias), nên `TE(A→B) > TE(B→A)` một chút **chưa chắc có ý nghĩa**. Bắt buộc:
- **Permutation/surrogate test:** xáo trộn phá vỡ quan hệ thời gian (giữ nguyên phân phối biên) → dựng phân phối null → tính p-value.
- Báo cáo TE kèm **khoảng tin cậy**, không chỉ 1 con số.

### 3.4. Nếu sanity check KHÔNG đạt

Theo đúng thứ tự roadmap gốc yêu cầu — **kiểm tra pipeline trước khi nghi ngờ mô hình**:
1. Đồng bộ hoá thời gian (mục 2.3)
2. Lọc ectopic (mục 2.1)
3. Nhiễu tim (mục 3.2)
4. Độ dài cửa sổ có nằm trong vùng `T_φ` mạnh không (mục 2.4)
5. Đối chiếu với KSG — nếu **cả hai** estimator cùng cho kết quả ngược chiều kỳ vọng, nhiều khả năng là **dữ liệu/pipeline**, không phải `T_φ`.

Và nếu sau tất cả vẫn không đạt: **báo cáo trung thực**. Roadmap xếp tình huống này ở mức rủi ro Trung bình, không phải thất bại của dự án.

---

## 4. Checklist thoát Pha S

**Nhóm A — Điều kiện tiên quyết (BẮT BUỘC trước tiên):**
- [ ] Thêm chuẩn hoá z-score per-window vào `train_amortized` + `estimate` (chung 1 cờ config)
- [ ] Train lại → `phi_amortized_standardized.pt`
- [ ] Xác minh: hiệu năng trên synthetic không tệ đi ở N≥50
- [ ] Xác minh: bất biến thang đo (corr > 0.99 khi đổi `×0.1+0.8`)

**Nhóm B — Dữ liệu & tiền xử lý:**
- [ ] Cài `wfdb` (+ `mne` hoặc `pyedflib` cho EDF của CAP)
- [ ] Tải `slpdb` (632MB), Fantasia (293MB), Apnea-ECG (581MB), CAP subset `n1`–`n16`
- [ ] RR từ annotation nhịp (`slpdb`), lọc ectopic
- [ ] EEG band power (log), cùng lưới 4 Hz
- [ ] **Kiểm tra đồng bộ**: vẽ chồng + test dịch thời gian nhân tạo
- [ ] Cắt cửa sổ 30 s (N=120) → tái dùng `Window` từ Pha R

**Nhóm C — Sanity check:**
- [ ] **Kiểm tra nhiễu tim trên EEG** (R-peak locked average) — làm TRƯỚC khi tin kết quả
- [ ] TE hai chiều trên `slpdb` và `capslpdb`, cả `T_φ` lẫn KSG
- [ ] Permutation test + khoảng tin cậy
- [ ] Đạt trên ít nhất 1 bộ → thoát Pha S

**Nhóm D — Chất lượng:**
- [ ] `pytest -q` toàn bộ pass
- [ ] Không commit dữ liệu thật vào git (`data/raw/` đã trong `.gitignore`)

---

## 5. Phân công

| Việc | Ai làm |
|---|---|
| Viết code (A, B, C) | Agent coding — **không commit, không push** |
| Tải dữ liệu + chạy notebook nặng | **Chủ dự án** (agent chỉ đưa lệnh) |
| Review code + kết quả | Claude (reviewer) |
| Commit / push | **Chủ dự án** |

---

## 6. Bước tiếp theo

Xong Pha S → **Pha T**: thí nghiệm chính (quét N × 4 bộ × 4 phương pháp), bootstrap ≥500, permutation test, và **bản thảo checkpoint có thể nộp độc lập** — điểm dừng an toàn đầu tiên của cả dự án.
