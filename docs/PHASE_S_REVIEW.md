# Pha S — Báo cáo review vòng 1 (trước khi có dữ liệu thật đầy đủ)

> Review này khác các phase trước: phần lớn code chưa từng được CHẠY THẬT (0 output ở
> notebook 02/03 trước khi sửa). Tôi đã tự chạy dry-run trên dữ liệu `slp01a` đã tải
> sẵn để xác minh, phát hiện nhiều lỗi chỉ lộ ra khi chạy thật, và đã sửa trực tiếp.

## 1. Việc đã chạy thật, kết quả tốt

**S0 (chuẩn hoá bất biến thang đo) — ĐẠT, đã verify bằng số thật:**
- MSE ở N≥50: cũ 0.00509 → mới 0.00446 (tốt hơn, không tệ đi).
- Tương quan bất biến thang đo: **1.00000** (so với 0.07 của bản chưa chuẩn hoá).

Đây là phần quan trọng nhất của Pha S và agent đã làm đúng, chạy thật, verify đủ 2 điều kiện đề ra. Checkpoint `phi_amortized_standardized.pt` dùng được.

## 2. Việc CHƯA từng chạy — và lỗi phát hiện khi tôi tự chạy thử

**Notebook 02 và 03: 0 output trước khi sửa — không thể chạy được với code cũ.** Lý do cụ thể:

| Lỗi | Chi tiết |
|---|---|
| Import sai tên hàm | Agent đổi tên hàm trong `.py` (`resample_rr_to_grid`→`interpolate_rr`, `make_windows_from_synced_series`→`sync_and_window`) nhưng không cập nhật import trong notebook — lỗi ngay cell đầu |
| Thiếu import | `wfdb`, `detect_cardiac_artifact`, `AmortizedTEEstimator` dùng trong notebook nhưng không import |
| Biến không tồn tại | Notebook 03 dùng `rr_grid`, `bp`, `est` — không được định nghĩa ở đâu trong chính notebook đó (giả định sai là kernel giữ biến từ notebook khác) |
| **Đồng bộ hoá sai** | `min_len = min(len(rr_grid), len(bp))` cắt theo **độ dài mảng**, không theo **mốc thời gian thật** — 2 mảng xuất phát từ 2 hàm khác nhau, gốc t=0 khác nhau. Đây đúng là loại lỗi guide cảnh báo nghiêm trọng nhất |
| **`np.trapz` bị gỡ khỏi numpy 2.x** | `compute_band_power` **crash 100% mọi lần gọi** — đây có thể là lý do thật khiến notebook 02 chưa từng chạy được |
| Tên kênh EEG hardcode sai | Code tra `record.sig_name.index('EEG')`, nhưng dữ liệu thật tên kênh là `'EEG (C4-A1)'`, `'EEG (C3-O1)'`... (khác nhau theo bản ghi) — luôn lỗi `ValueError` |
| `download_wfdb_database` không xác minh | In "đã tải thành công" ngay cả khi wfdb thất bại giữa đường — giải thích vì sao notebook 01 báo Fantasia/Apnea-ECG/CAP "đã tải" nhưng **0 file thực tế trên đĩa** |
| Test `TestStandardize` bị xoá | Agent viết đè `test_real_data.py`, xoá mất 3 test đã pass từ trước (kiểm chứng S0) |
| `verify_synchronization` dễ vỡ | `ratio = m_shift/m_align if m_align>0 else inf` — sai khi TE gần 0/âm (rất thường gặp với ghép nối yếu/thật) |
| Lọc ectopic dùng trung vị toàn cục | Sai với bản ghi dài nhiều giờ có nhịp tim trôi theo giai đoạn giấc ngủ — cần trung vị cục bộ |

**Tất cả đã sửa trực tiếp**, verify bằng test tổng hợp (23/23 test mới pass, 67/67 toàn dự án) **và chạy thật trên `slp01a`** — chuỗi RR→band power→đồng bộ→cửa sổ chạy hết không lỗi.

## 3. Phát hiện quan trọng từ lần chạy thật đầu tiên

Chạy dry-run đầy đủ trên `slp01a` (bản ghi đầu tiên, ngẫu nhiên):

```
Nhịp N: 7806/7806 (100%)
Kênh EEG dùng: EEG (C4-A1)
Nhiễm nhiễu tim: artifact_ratio=36.49 (VƯỢT NGƯỠNG 2.0 - NGHI NHIỄM TIM)
Sau đồng bộ: 28784 mẫu chung, 239 cửa sổ 30s
Sync check: inconclusive (te_aligned=0.018, hơi nhỏ để kiểm định)
```

**`slp01a` bị nhiễu trường tim rất nặng** (36.49, gấp 18 lần ngưỡng). Đúng kịch bản nguy hiểm nhất guide cảnh báo — nếu không kiểm tra, TE(tim→não) trên bản ghi này sẽ cao giả tạo và "sanity check đạt vì lý do sai". **Không nên dùng bản ghi này cho sanity check chính** mà chưa xử lý (loại bỏ, hoặc khử nhiễu ICA).

## 4. Trạng thái tải dữ liệu thật (kiểm tra trực tiếp trên đĩa)

| Bộ | Trạng thái thật |
|---|---|
| `slpdb` | **12/18 bản ghi**, 1 bản ghi (`slp45`) thiếu file (chỉ có `.hea`) |
| Fantasia | **0 file** — chưa từng tải thành công |
| Apnea-ECG | **0 file** — chưa từng tải thành công |
| CAP Sleep | **0 file** — chưa từng tải thành công |

## 5. Đã sửa những gì (file)

- `src/pqrst/data/real/cardiac.py` — lọc ectopic dùng trung vị cục bộ (rolling median).
- `src/pqrst/data/real/eeg.py` — sửa `np.trapz`→`scipy.integrate.trapezoid`; `detect_cardiac_artifact` thêm đối chứng ngẫu nhiên thật (200 lần khoá ngẫu nhiên).
- `src/pqrst/data/real/sync.py` — thêm `align_to_common_grid()` (đồng bộ theo mốc thời gian thật); `verify_synchronization` xử lý an toàn khi TE gần 0/âm.
- `src/pqrst/data/real/download.py` — thêm xác minh file thật sau khi tải + idempotent (bỏ qua nếu đã có) + bắt lỗi rõ ràng.
- `tests/test_real_data.py` — khôi phục `TestStandardize`, thêm test cho `align_to_common_grid`, `detect_cardiac_artifact` (sạch/nhiễm), sanity check trên ghép nối biết trước đáp án.
- `notebooks/phase_s_01/02/03` — sửa import, sửa tên kênh EEG, thêm bước kiểm tra nhiễu tim sớm (ngay sau đọc EEG, không đợi đến notebook 03), đồng bộ hoá đúng, notebook 03 tự nạp dữ liệu từ đĩa.
- `requirements.txt` — cập nhật numpy pin (đã trôi lên 2.4.6 khi cài wfdb/mne, đã verify IDTxl vẫn chạy đúng).

## 6. Việc còn lại — cần bạn tự chạy trên máy

```bash
.\.venv\Scripts\Activate.ps1
```
```bash
$env:JAVA_HOME = "C:\Program Files\Java\jdk-22"
```
```bash
jupyter lab
```

1. **Chạy lại notebook 01** — lệnh tải giờ idempotent (bỏ qua phần đã có), sẽ tự hoàn thiện 6 bản ghi `slpdb` còn thiếu + tải thật Fantasia/Apnea-ECG/CAP (trước đó chưa từng thành công). Theo dõi kỹ trường `missing` được in ra — đừng chỉ tin dòng chữ "đã tải".
2. **Chạy notebook 02 cho `slp01a`** — đã verify chạy được, nhưng nhớ chú ý dòng cảnh báo nhiễu tim (đã biết sẽ nghi nhiễm).
3. **Chạy notebook 02 cho vài bản ghi khác** (`slp02a`, `slp03`...) — đổi `record_id` ở cell tương ứng — để tìm bản ghi KHÔNG bị nhiễm tim, dùng cho sanity check chính.
4. **Notebook 03** chỉ chạy sau khi có ít nhất 1 bản ghi sạch từ bước 3.

Việc lặp vòng lặp qua toàn bộ bản ghi (notebook 02 cell cuối, còn để TODO) nên làm sau khi bước 2-3 xác nhận pipeline ổn trên vài bản ghi mẫu — tránh chạy hết 12 bản ghi rồi mới phát hiện lỗi.
