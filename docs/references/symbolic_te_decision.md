# Quyết định: thư viện/cách làm cho Symbolic Transfer Entropy

> Mục 1.3 và 3 của [`PHASE_P_GUIDE.md`](../PHASE_P_GUIDE.md) yêu cầu xác nhận việc này
> trước khi code `src/pqrst/baselines/symbolic_te.py`. Điền vào đây khi đã quyết định,
> đây là 1 mục trong checklist thoát Pha P.

## Trạng thái
- [x] Đã kiểm tra IDTxl có estimator symbolic TE sẵn không (đọc doc/API bản đang cài).
- [x] Đã quyết định phương án cuối cùng.

## Phương án đã xét

| Phương án | Ưu điểm | Nhược điểm |
|---|---|---|
| IDTxl có sẵn (nếu có) | Nhất quán với 2 baseline kia, ít code tự viết | Đã kiểm tra, IDTxl không có sẵn JidtSymbolicTE |
| Thư viện khác (vd `pyinform`) | Đã kiểm định, ít bug | Thêm 1 dependency, có thể khác quy ước đơn vị (nat/bit) |
| Tự code (ordinal pattern + TE rời rạc) | Kiểm soát hoàn toàn, hiểu rõ từng bước, tái sử dụng được JidtDiscreteTE của IDTxl | Tốn thời gian, rủi ro bug tự tạo |

## Quyết định cuối cùng
Sử dụng phương án: **Tự code (ordinal pattern + TE rời rạc sử dụng JidtDiscreteTE)**. 

## Lý do
IDTxl không có sẵn estimator chuyên dụng cho Symbolic TE. Việc tự code bước mã hóa ordinal pattern khá đơn giản (chỉ vài dòng NumPy), sau đó có thể tận dụng luôn `JidtDiscreteTE` của IDTxl để tính toán TE trên chuỗi ký hiệu. Phương án này vừa không phải thêm dependency mới như `pyinform`, vừa đảm bảo sự nhất quán trong việc sử dụng engine JIDT (cho ra kết quả đơn vị, cấu hình giống nhau) cho cả 3 baseline.
