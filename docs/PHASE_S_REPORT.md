# Báo cáo Pha S — Dữ liệu sinh lý thật: đổi hướng, lỗi phát hiện, và kết quả cuối

> Pha S là pha khó nhất từ đầu dự án: khác các pha trước (dữ liệu tổng hợp, biết đáp
> án đúng), dữ liệu thật không có ground truth, và lộ ra nhiều lớp vấn đề chỉ xuất
> hiện khi chạy thật (không phải lỗi logic thuần túy). Báo cáo này ghi lại toàn bộ
> quá trình: từ kế hoạch ban đầu (TE tim-não trên slpdb/capslpdb), phát hiện chặn
> (nhiễm nhiễu điện tim không khử được), đổi hướng sang TE tim-hô hấp (Fantasia), và
> 2 lỗi thêm tìm ra trong chính hướng mới trước khi có kết quả cuối đáng tin.

---

## 1. Tóm tắt cho người bận

- **Tiêu chí thoát Pha S: ĐẠT** — nhưng bằng con đường khác kế hoạch gốc.
- **TE tim-não (slpdb/capslpdb) — LOẠI KHỎI KẾT QUẢ CHÍNH.** Toàn bộ 18/18 bản ghi
  slpdb và cả 9 kênh của capslpdb đều nhiễm nhiễu điện tim thật (đã xác minh bằng
  hình dạng sóng, không phải heartbeat-evoked potential). Đã thử **5 phương pháp
  khử nhiễu** có cơ sở khoa học (template trung bình, +căn chỉnh jitter Woody, +vút
  Tukey, hồi quy tuyến tính theo ECG thật, ICA 9 kênh) — **không phương pháp nào
  đưa được về dưới ngưỡng**. Đây là phát hiện phương pháp luận thật, được ghi nhận
  làm hạn chế/đóng góp, không phải bug.
- **TE tim-hô hấp (Fantasia) — KẾT QUẢ CHÍNH, ĐẠT với KSG.** Sau khi sửa 2 lỗi (NaN
  lan trong `filtfilt`, và bất đối xứng lọc bằng-thông giữa RR/RESP) **và sửa 1 lỗi
  thống kê** (pseudo-replication — xem mục 3.3), KSG xác nhận đúng hướng RSA đã biết
  trong tài liệu (hô hấp→tim > tim→hô hấp) trên **N=17 bản ghi** (đơn vị mẫu ĐÚNG,
  không phải cửa sổ), CI 95% của hiệu số `[0.0039, 0.0353]`, Wilcoxon p=0.0116.
- **Đối chứng độc lập thứ 2 (Apnea-ECG, 6 bản ghi)** cùng chiều (không đủ mạnh để có
  ý nghĩa riêng do N nhỏ) — **gộp cả 2 bộ (N=23) cho kết quả mạnh nhất**: CI
  `[0.0057, 0.0311]`, Wilcoxon p=0.0046, 18/23 (78%) bản ghi đúng hướng.
- **Kiểm tra độ nhạy tham số (9 tổ hợp bandpass × độ dài cửa sổ):** chiều kết quả
  **ổn định tuyệt đối 9/9 (100%)** — ý nghĩa thống kê giảm dần có giải thích được
  khi cửa sổ dài hơn/băng thông rộng hơn (pha loãng tín hiệu), không phải dấu hiệu
  kết quả giả. Xem mục 3.4.
- **Amortized MINE FAIL trên dữ liệu thật** (TE âm, `-0.25` và `-0.07`) — do mạng bị
  lệch chuẩn (miscalibrated) trên tín hiệu dao động gần tuần hoàn, khớp với phát
  hiện Pha R2 (amortized yếu hơn KSG trên dữ liệu phi tuyến/periodic ở N nhỏ). Ghi
  nhận là hạn chế đã biết, không retrain (quyết định của chủ đề tài, xem mục 6).

---

## 2. Quyết định đổi hướng: vì sao bỏ slpdb/capslpdb làm nguồn EEG chính

### 2.1. Phát hiện: 100% bản ghi nhiễm nhiễu điện tim

Chạy dry-run trên `slp01a` (bản ghi đầu tiên) phát hiện `artifact_ratio=36.49`
(ngưỡng 2.0) — nghi nhiễm nặng. Mở rộng ra **toàn bộ 18/18 bản ghi slpdb**: tất cả
đều vượt ngưỡng, dao động 7.89–60.35.

**Xác minh đây là nhiễu điện THẬT, không phải heartbeat-evoked potential (HEP)** —
HEP là đáp ứng thần kinh thật với nhịp tim, nếu nhầm sẽ vô tình xóa mất chính tín
hiệu cần đo:

| | Đỉnh lệch (so với R-peak) | Độ rộng |
|---|---|---|
| EEG khóa-theo-R-peak (`slp01a`) | -8ms | 12ms |
| QRS của chính ECG (khóa theo R-peak của nó) | -8ms | 16ms |

Đỉnh và độ rộng **trùng khớp gần như hoàn toàn với QRS** — HEP thật đạt đỉnh muộn
hơn nhiều (~200-300ms) và rộng hơn nhiều. Kết luận: đây là điện tim rò rỉ trực tiếp
vào kênh EEG, không phải HEP.

### 2.2. 5 phương pháp khử nhiễu đã thử — đều không đạt

| # | Phương pháp | Kết quả | Vì sao thất bại |
|---|---|---|---|
| 1 | Template trung bình khóa-R-peak, trừ theo tỷ lệ từng nhịp | Không ổn định — 1 số bản ghi giảm mạnh, 1 số TĂNG | Biên độ/hình dạng nhiễm thật khác nhau giữa các nhịp nhiều hơn 1 template cố định mô tả được |
| 2 | + Căn chỉnh jitter (Woody's method) | Không sửa được — vẫn tăng đúng những bản ghi cũ | Không phải do lệch pha |
| 3 | + Vút Tukey (chống bước nhảy biên) | Cải thiện nhóm "template nhọn" nhưng **không bản ghi nào đạt ngưỡng** | Điều tra: nhóm "tăng" có template GẦN-HẰNG-SỐ (không phải xung nhọn) — trừ tạo bước nhảy mới bị phát hiện như nhiễu |
| 4 | Hồi quy tuyến tính EEG theo ECG thật (FIR ngắn, toàn bản ghi) | **Tệ hơn cả template** (1 bản ghi: 15.4→99.5) | Quan hệ ECG↔nhiễu không ổn định suốt đêm (trở kháng da, hô hấp, cử động) — 1 bộ lọc cố định cho cả bản ghi không khớp |
| 5 | ICA (9 kênh EEG differential của `capslpdb`), `find_bads_ecg` (correlation + CTPS) | Không tách được thành phần nào | Đo trực tiếp artifact_ratio trên từng thành phần ICA: 3.2–11.6, **lan trải đều**, không tập trung vào 1 nguồn độc lập như ICA giả định |

**capslpdb** (9 kênh EEG differential, đáng ra ít nhạy nhiễu hơn referential đơn
kênh) **cũng nhiễm ở cả 9 kênh** (5.76–26.63) — kênh tốt nhất vẫn gấp ~3x ngưỡng.

### 2.3. Quyết định cuối

Không tiếp tục đầu tư vào khử nhiễu (rủi ro cao, đã đủ bằng chứng đây là bài toán
khó — cần ICA đa kênh mật độ cao hoặc bộ lọc thích nghi cục bộ phức tạp, ngoài
phạm vi hợp lý). Chuyển kết quả chính trên dữ liệu thật sang **ghép nối tim-hô hấp
(Fantasia)** — sạch, đã xác lập rõ trong tài liệu sinh lý học (Respiratory Sinus
Arrhythmia). `slpdb`/`capslpdb` giữ lại trong `notebooks/phase_s_02_preprocess.ipynb`
như minh chứng phát hiện phương pháp luận.

---

## 3. Kết quả chính: TE tim-hô hấp trên Fantasia

### 3.1. Lỗi #1 — NaN lan toàn tín hiệu qua `filtfilt`

Một vài bản ghi (vd `f2o06`) có NaN rải rác trong tín hiệu RESP gốc (đứt cảm biến —
bình thường trong dữ liệu thật). `scipy.signal.filtfilt` là lọc IIR 2 chiều toàn
tín hiệu: **61/1.75 triệu mẫu NaN đầu vào → 100% mẫu NaN đầu ra**. Đã sửa bằng nội
suy tuyến tính qua các khoảng NaN ngắn TRƯỚC khi lọc
(`respiration.preprocess_respiration`). Kết quả: số bản ghi PASS tăng từ 20 → 23/40.

### 3.2. Lỗi #2 (nghiêm trọng hơn) — bất đối xứng lọc bằng-thông giữa RR và RESP

**Phát hiện:** chạy sanity check lần đầu, TE cho **sai hướng** so với RSA đã biết
(chỉ 5/23 bản ghi đúng hướng "hô hấp→tim", TE gộp âm `-0.0095`).

**Điều tra bằng đo thời gian tự tương quan tích hợp** (integrated autocorrelation
time) của RR và RESP trên lưới 4Hz đã dùng để tính TE:

| Tín hiệu | τ trung bình (giây) | Số bản ghi RR có τ dài hơn RESP |
|---|---|---|
| RR (chưa lọc) | 11.37s | **23/23 (100%)** |
| RESP (đã lọc 0.1–0.5Hz) | 0.65s | — |

RESP đã được lọc bằng-thông (0.1-0.5Hz) để cô lập dao động hô hấp, loại bỏ trend
chậm — nhưng **RR thì không**, nên vẫn giữ nguyên thành phần biến thiên chậm
(LF/VLF của HRV). Bất đối xứng "trí nhớ" tự tương quan (chênh ~17 lần) làm TE lệch
hướng — hiệu ứng thống kê, không phải sinh lý thật.

**Sửa:** thêm `respiration.bandpass_filter()`, áp dụng **cùng dải 0.1–0.5Hz cho CẢ
RR và RESP** trước khi đồng bộ/tính TE.

**Xác minh sau khi sửa** (kiểm định trực tiếp, không phải suy diễn):

| | Trước sửa | Sau sửa |
|---|---|---|
| Số bản ghi đúng hướng RSA (từng bản ghi riêng) | 5/23 (22%) | 19/23 (83%) |
| TE gộp: hiệu số (hô hấp→tim − tim→hô hấp) | -0.0095 | **+0.0278** |

Số bản ghi PASS kiểm định đồng bộ giảm 23→17 sau khi sửa — đây là **dấu hiệu tốt**:
sau khi lọc đúng dải tần quan tâm, vài bản ghi có ghép nối yếu không còn "vô tình"
qua kiểm tra nhờ nội dung phổ rộng còn sót lại từ RR chưa lọc.

### 3.3. Lỗi #3 (chuẩn bị cho Q1/Q2) — gộp cửa sổ làm đơn vị mẫu (pseudo-replication)

**Phát hiện khi rà soát lại cho mục tiêu công bố:** phiên bản đầu tiên của sanity
check gộp TẤT CẢ cửa sổ của TẤT CẢ bản ghi thành 1 tập (`N=4049`) rồi bootstrap/kiểm
định trên đó. Đây là lỗi thống kê kinh điển ("pseudo-replication" / giả lập số
mẫu): các cửa sổ trong CÙNG 1 bản ghi tương quan với nhau (cùng 1 người, cùng 1 đêm
ghi), không độc lập — coi chúng là "4049 mẫu độc lập" phản ánh SAI độ không chắc
chắn thực sự. Đây là loại lỗi reviewer thống kê/sinh lý học sẽ bắt ngay trong 1 bài
Q1/Q2.

**Sửa:** thêm `bidirectional_te_per_record`/`run_sanity_check_per_record` — tính 1
giá trị TE trung bình cho MỖI bản ghi, kiểm định trên N=số_bản_ghi (đơn vị mẫu
ĐÚNG), kèm kiểm định Wilcoxon signed-rank (phi tham số, phù hợp N nhỏ) làm kiểm
định độc lập thứ 2.

### 3.4. Kết quả sanity check chính thức (theo đơn vị bản ghi, N=17)

| Estimator | TE(hô hấp→tim) | TE(tim→hô hấp) | CI 95% của hiệu số | Wilcoxon p | Kết luận |
|---|---|---|---|---|---|
| **KSG** | 0.1109 | 0.0929 | **[0.0039, 0.0353]** | 0.0116 | ✅ **PASS** |
| Amortized | -0.2474 | -0.0704 | [-0.2119, -0.1434] | 1.0000 | ❌ FAIL |

CI theo đơn vị bản ghi rộng hơn CI (sai) theo cửa sổ trước đó (`[0.0162,0.0200]`) —
đúng như dự kiến khi sửa một giả định độc lập bị vi phạm, nhưng **vẫn hoàn toàn
loại trừ 0** — kết luận PASS đứng vững dưới phương pháp thống kê đúng.

**=== KẾT LUẬN CHUNG: ĐẠT (KSG xác nhận đúng hướng RSA, thống kê đúng phương pháp) ===**

### 3.5. Kiểm tra độ nhạy tham số (chuẩn bị cho Q1/Q2)

Chạy lại toàn bộ pipeline trên 40 bản ghi Fantasia với 9 tổ hợp (3 dải bandpass ×
3 độ dài cửa sổ) — xem `notebooks/phase_s_04_sensitivity_analysis.ipynb`:

| Bandpass (Hz) | Cửa sổ (s) | N pass | Hiệu số | CI 95% | Wilcoxon p | PASS |
|---|---|---|---|---|---|---|
| (0.15, 0.4) | 20 | 15 | 0.0313 | [0.0138, 0.0491] | 0.0062 | ✅ |
| (0.15, 0.4) | 30 | 15 | 0.0258 | [0.0037, 0.0496] | 0.0319 | ✅ |
| (0.15, 0.4) | 45 | 15 | 0.0220 | [-0.0037, 0.0491] | 0.0206 | ❌ |
| (0.1, 0.5) *(mặc định)* | 20 | 17 | 0.0220 | [0.0111, 0.0347] | 0.0008 | ✅ |
| (0.1, 0.5) | 30 | 17 | 0.0180 | [0.0039, 0.0353] | 0.0116 | ✅ |
| (0.1, 0.5) | 45 | 17 | 0.0159 | [-0.0012, 0.0353] | 0.0153 | ❌ |
| (0.05, 0.6) | 20 | 19 | 0.0108 | [0.0010, 0.0236] | 0.0180 | ✅ |
| (0.05, 0.6) | 30 | 19 | 0.0070 | [-0.0068, 0.0249] | 0.1467 | ❌ |
| (0.05, 0.6) | 45 | 19 | 0.0055 | [-0.0116, 0.0275] | 0.2706 | ❌ |

**Chiều kết quả ổn định TUYỆT ĐỐI: 9/9 (100%) hiệu số đều dương** (đúng hướng RSA).
Ý nghĩa thống kê đạt 5/9 tổ hợp — **giảm dần CÓ GIẢI THÍCH ĐƯỢC**, không ngẫu
nhiên: hiệu số giảm đơn điệu khi cửa sổ dài hơn (20s→30s→45s, cả 3 dải bandpass) và
khi bandpass rộng hơn (0.15-0.4 → 0.1-0.5 → 0.05-0.6) — cả 2 xu hướng đều pha loãng
tín hiệu ghép nối cụ thể bằng nội dung ngoài dải/ngoài cửa sổ quan tâm. Đây là hành
vi hợp lý của 1 hiệu ứng THẬT có kích thước vừa phải, không phải dấu hiệu kết quả
giả/ngẫu nhiên.

### 3.6. Đối chứng độc lập thứ 2: Apnea-ECG + gộp meta-analysis

`notebooks/phase_s_05_apnea_ecg_replication.ipynb` — 8 bản ghi Apnea-ECG có kênh hô
hấp (`a01-a04, b01, c01-c03`), cấu trúc khác Fantasia (ECG và hô hấp ở 2 file WFDB
riêng cho cùng 1 lần ghi — `a01`=ECG+`.qrs`, `a01r`=hô hấp, không annotation nhịp).
6/8 bản ghi PASS kiểm tra đồng bộ.

| | N | TE(hô hấp→tim) | TE(tim→hô hấp) | CI 95% | Wilcoxon p |
|---|---|---|---|---|---|
| Apnea-ECG riêng | 6 | 0.1362 | 0.1185 | [-0.0038, 0.0455] (chứa 0) | 0.2188 |
| **Gộp Fantasia + Apnea-ECG** | **23** | 0.1175 | 0.0996 | **[0.0057, 0.0311]** | **0.0046** |

Apnea-ECG một mình chưa đủ mạnh (N=6 nhỏ) nhưng **cùng chiều** với Fantasia (4/6 bản
ghi đúng hướng). Vì đây là 2 nguồn dữ liệu **độc lập thống kê thực sự** (đối tượng
khác, thiết bị ghi khác), gộp thành 1 kiểm định chung trên N=23 (kiểu meta-analysis)
là cách tổng hợp đúng — cho **bằng chứng mạnh nhất trong toàn bộ Pha S**: CI hoàn
toàn dương, Wilcoxon p=0.0046, 18/23 (78%) bản ghi đúng hướng.

---

## 4. Vì sao Amortized FAIL trên dữ liệu thật (điều tra, không chỉ ghi nhận)

Phân rã trực tiếp `mi_full`/`mi_reduced` trên vài cửa sổ thật:

```
mi_full (X_lag+Y_lag dự đoán Y_t)     ≈ 0.08
mi_reduced (chỉ Y_lag dự đoán Y_t)    ≈ 0.40   <- LỚN HƠN mi_full ~5 lần
TE = mi_full - mi_reduced             ≈ -0.33
```

Về lý thuyết `mi_reduced` không thể lớn hơn `mi_full` (thêm biến điều kiện không
làm giảm thông tin thật). RR/RESP sau khi lọc bằng-thông là tín hiệu **rất mượt,
gần tuần hoàn** — quá khứ của nó (`y_lag`) tự dự đoán tương lai (`y_t`) cực tốt
(ngoại suy 1 sóng sin từ vài điểm là bài toán dễ). `MaskedStatisticsNetwork` được
huấn luyện trên corpus VAR-tuyến-tính (nhiễu ngẫu nhiên, KHÔNG mượt/tuần hoàn như
vậy) — bị lệch chuẩn nặng khi gặp tín hiệu tự-đoán-được-cao dạng này.

**Khớp với phát hiện đã có từ Pha R2** (thử nghiệm `periodic_coupling`): amortized
yếu hơn KSG trên dữ liệu phi tuyến/dao động ở N nhỏ. Đây là hạn chế nhất quán,
không phải hiện tượng mới riêng của Pha S.

**Cách sửa đúng** (không làm trong phạm vi Pha S theo quyết định của chủ đề tài):
retrain `T_phi` với corpus pha trộn thêm `periodic_coupling` (đã có sẵn generator từ
Pha R2) + chuẩn hóa — việc lớn, tốn thời gian sinh corpus + train + validate lại từ
đầu, không đảm bảo hết hoàn toàn vì dao động THẬT khác dao động TỔNG HỢP.

---

## 5. Danh sách lỗi đã sửa trong Pha S (tổng hợp)

| # | File | Lỗi | Cách phát hiện |
|---|---|---|---|
| 1 | `sync.py` | Đồng bộ theo độ dài mảng (`min_len`) thay vì mốc thời gian thật | Review code trước khi chạy |
| 2 | `eeg.py` | `np.trapz` bị gỡ khỏi numpy 2.x, crash 100% | Chạy thật lần đầu |
| 3 | `eeg.py` | `detect_cardiac_artifact` không có đối chứng ngẫu nhiên thật | Review + viết test |
| 4 | `cardiac.py` | Lọc ectopic dùng trung vị TOÀN CỤC, sai với bản ghi dài trôi nhịp tim | Review |
| 5 | Notebook 02/03 | Tên hàm import sai sau khi đổi tên trong `.py`, biến không tồn tại | Chạy thật lần đầu |
| 6 | `download.py` | Không xác minh file thật sau khi tải — báo "thành công" nhưng 0 file trên đĩa | Kiểm tra đĩa trực tiếp |
| 7 | `download.py` | `wfdb.dl_database` không báo tiến độ — tải chậm (~56KB/s) nhìn như treo | Đo tốc độ mạng trực tiếp |
| 8 | `download.py` | `capslpdb` là EDF thuần, không có `.hea` — `wfdb.dl_database` thất bại hoàn toàn, im lặng | Đo trực tiếp HTTP (404 vs 200) |
| 9 | `download.py`/`list_local_records` | Báo động giả "thiếu file" cho 8 bản ghi Apnea-ECG (`*er`) vốn không có `.dat` theo thiết kế | Kiểm tra đĩa + đọc header |
| 10 | `respiration.py` | NaN đứt cảm biến lan toàn tín hiệu qua `filtfilt` | Chạy 40 bản ghi Fantasia, phát hiện `ratio=nan` |
| 11 | Pipeline TE tim-hô hấp | Bất đối xứng lọc bằng-thông RR/RESP làm TE lệch hướng (mục 3.2) | Đo tự tương quan sau khi thấy hướng TE sai |
| 12 | `sync.py` | `verify_synchronization` hardcode `0.7` thay vì đọc từ config | Review code |
| 13 | `sanity_check.py` | Gộp cửa sổ làm đơn vị mẫu bootstrap — pseudo-replication, phản ánh sai độ tin cậy (mục 3.3) | Rà soát lại trước khi chuẩn bị công bố |

---

## 6. Khuyến nghị cho Pha T / viết bài báo

1. **Khung câu chuyện Q1/Q2 đề xuất:** MINE amortized + so sánh KSG validated đầy đủ
   trên dữ liệu tổng hợp (Pha P-R) → xác nhận trên dữ liệu THẬT, **2 nguồn độc lập**
   (Fantasia + Apnea-ECG, TE tim-hô hấp, khớp RSA đã biết, kiểm tra độ nhạy tham số
   đầy đủ) → 2 hạn chế được điều tra kỹ và báo cáo trung thực: (a) nhiễm điện tim
   trong PSG EEG không khử được bằng 5 phương pháp chuẩn (đóng góp phương pháp luận,
   hiếm nghiên cứu cùng chủ đề kiểm tra kỹ vậy), (b) amortized cần cải thiện để tổng
   quát hóa trên dữ liệu phi tuyến/dao động thật.
2. **Không nên báo cáo TE tim-não trên slpdb/capslpdb** như kết quả dương — rủi ro
   bị nghi ngờ toàn bộ bài nếu reviewer có nền EEG/sleep research phát hiện vấn đề
   nhiễm nhiễu đã biết trong tài liệu.
3. **Đã hoàn thành cho Q1/Q2** (không còn là "cân nhắc"): thống kê đúng theo đơn vị
   bản ghi (không pseudo-replication), kiểm tra độ nhạy 9 tổ hợp tham số, đối chứng
   độc lập Apnea-ECG + gộp meta-analysis 2 nguồn (N=23, bằng chứng mạnh nhất). Xem
   mục 3.3–3.6.
4. **Nếu muốn theo đuổi TE tim-não sau này:** cần dữ liệu EEG mật độ cao hơn (nhiều
   kênh độc lập thật, không phải vài kênh differential lân cận) để ICA khả thi, hoặc
   đầu tư bộ lọc thích nghi cục bộ (theo dõi trôi dạt theo thời gian) — cả 2 đều là
   dự án con riêng, ngoài phạm vi Pha S.
