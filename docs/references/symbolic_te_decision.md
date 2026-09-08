# Quyết định: thư viện/cách làm cho Symbolic Transfer Entropy

> Mục 1.3 và 3 của [`PHASE_P_GUIDE.md`](../PHASE_P_GUIDE.md) yêu cầu xác nhận việc này
> trước khi code `src/pqrst/baselines/symbolic_te.py`. Điền vào đây khi đã quyết định,
> đây là 1 mục trong checklist thoát Pha P.

## Trạng thái
- [ ] Đã kiểm tra IDTxl có estimator symbolic TE sẵn không (đọc doc/API bản đang cài).
- [ ] Đã quyết định phương án cuối cùng.

## Phương án đã xét

| Phương án | Ưu điểm | Nhược điểm |
|---|---|---|
| IDTxl có sẵn (nếu có) | Nhất quán với 2 baseline kia, ít code tự viết | Chưa xác nhận có tồn tại |
| Thư viện khác (vd `pyinform`) | Đã kiểm định, ít bug | Thêm 1 dependency, có thể khác quy ước đơn vị (nat/bit) |
| Tự code (ordinal pattern + TE rời rạc) | Kiểm soát hoàn toàn, hiểu rõ từng bước | Tốn thời gian, rủi ro bug tự tạo |

## Quyết định cuối cùng
*(điền sau khi đọc doc IDTxl thực tế)*

## Lý do
*(điền)*
