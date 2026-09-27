# Flow triển khai Pha U (nâng cấp hướng Q1) — từng bước làm

Tài liệu vận hành nội bộ (không gửi mentor). Chi tiết lý do/thời gian ở `KE_HOACH_NANG_CAP_Q1.md`.

## Nguyên tắc làm việc (giữ nguyên từ trước)

- Claude viết code + test + tài liệu; **chủ dự án tự chạy notebook** và tự commit/push (Claude chỉ in lệnh).
- Mọi việc chạy lâu: Claude tự kiểm chứng ở quy mô rất nhỏ trước, rồi mới giao chạy bản đầy đủ.
- Kết quả notebook: Claude đọc trực tiếp CSV/hình (không chỉ tin output), báo trung thực kể cả kết quả xấu.
- Sau mỗi bước: cập nhật tài liệu tương ứng, `pytest` toàn bộ phải pass.
- Không tuyên bố vượt quá số liệu (đặc biệt với lượng tử: không nói "quantum advantage").
- Tiền tố notebook mới: `phase_u_XX_*.ipynb`; báo cáo pha: `docs/PHASE_U_REPORT.md`.

## Sơ đồ phụ thuộc

```
S0 (E1,E4) ──► S1 (A1 thăm dò) ──► G1 ──┬─ tốt ──► S2 (A2) ──► S3 (A3,A4) ─┐
                                         └─ xấu ──► (giữ định vị Q2)          │
S1b (kiểm tra mã TREET) ──► S4 (C) ──────────────────────────────────────────┤
S5 (B toán) ── song song ─────────────────────────────────────────────────────┤
S6 (D mở rộng) ──────────────────────────────────────────────────────────────┤
S7 (E2,E3 tái lập, git) ─────────────────────────────────────────────────────┴─► S8 viết bản thảo ─► S9 rà soát
```

## S0 — Chuẩn bị (E1, E4), ~2–3 ngày

| | |
|---|---|
| Việc | Kiểm định ý nghĩa thống kê cho các so sánh sát nhau; rà test/tài liệu; cố định phiên bản thư viện |
| Đầu vào | CSV grid đã có (`phase_r*_grid_raw.csv`, `phase_t_*`), `phase_p2_quantum_*_eval.csv` |
| Code | `src/pqrst/evaluation/significance.py` (bootstrap ghép cặp theo cửa sổ), test kèm theo |
| Notebook | `phase_u_00_significance.ipynb` (nhẹ) |
| Ra | `results/tables/phase_u_significance.csv`; bổ sung vào báo cáo |
| Lưu ý | CSV `*_raw` cần có ước lượng theo cửa sổ; nếu quantum chưa lưu theo cửa sổ thì lưu lại khi chạy |
| Thoát | Mọi so sánh chính có p/CI; pytest pass |

## S1 — A1: thăm dò thích nghi trên dữ liệu thật, ~3–5 ngày

| | |
|---|---|
| Việc | Tinh chỉnh `T_φ` (checkpoint `phi_amortized_standardized.pt`) bằng DV loss trên cửa sổ thật của các bản ghi train; đánh giá trên bản ghi giữ lại |
| Chia dữ liệu | Theo **bản ghi** (không theo cửa sổ). Bắt đầu: 1 lần chia (vd 16 train / 7 test), sau đó k-fold ở S3 |
| Code | `src/pqrst/estimators/mine/adapt.py` (hàm `adapt_on_windows`, tái dùng `donsker_varadhan_loss`, `feature_space.py`), test kèm theo |
| Notebook | `phase_u_01_domain_gap_probe.ipynb` (dùng lại pipeline tiền xử lý Pha S, nạp lại cửa sổ từ `data/processed/`) |
| Đo | (1) TE hai chiều mỗi bản ghi giữ lại; (2) tương quan với KSG theo bản ghi; (3) hình PCA trước/sau |
| Chạy | Claude thử quy mô nhỏ; chủ dự án chạy đầy đủ (ước ~1–2 giờ) |
| **G1** | Đúng hướng RSA trên bản ghi giữ lại VÀ đồng thuận với KSG rõ hơn trước thích nghi → đi S2. Ngược lại → dừng hướng A, ghi kết quả âm, chuyển S4–S7 với định vị Q2 |

## S1b — Kiểm tra mã nguồn TREET/TENDE, ~1 ngày

Tra repo công khai và điều khoản; quyết định S4 chạy trực tiếp hay so gián tiếp. Ghi vào `KE_HOACH_NANG_CAP_Q1.md`.

## S2 — A2: domain randomization (chỉ nếu qua G1), ~1 tuần

| | |
|---|---|
| Việc | Bộ sinh mô phỏng gần tín hiệu thật hơn (AR dao động dải hẹp, ghép nối, innovation không Gaussian) → corpus mới → huấn luyện lại `T_φ` |
| Code | `src/pqrst/data/synthetic/oscillatory.py`, mở rộng `generate_corpus_*`; test ground-truth (TE thật tính giải tích hoặc KSG N lớn, có test hồi quy như Pha P) |
| Notebook | `phase_u_02_corpus_randomized.ipynb`, `phase_u_03_train_randomized.ipynb` (nặng: dự kiến ~1–3 giờ, chạy qua đêm, tắt sleep) |
| Thoát | Đánh giá trên bản ghi giữ lại như S1; so sánh: gốc / thích nghi / randomized / cả hai |

## S3 — A3 + A4: đo gap định lượng và kiểm chứng k-fold, ~1 tuần

| | |
|---|---|
| Việc | Số đo khoảng cách miền trong không gian đặc trưng (vd khoảng cách phân phối giữa cụm mô phỏng và thật) trước/sau; k-fold trên 23 bản ghi |
| Code | mở rộng `feature_space.py` (hàm đo khoảng cách) + test |
| Notebook | `phase_u_04_kfold_real.ipynb` |
| Ra | Bảng: hướng TE, CI, Wilcoxon theo bản ghi, so với KSG; hình gap trước/sau |

## S4 — C: so sánh trực diện TREET, ~1–2 tuần

| | |
|---|---|
| Việc | Cài quá trình chuyển mạch phi tuyến (công thức TE thật lấy từ bài, kiểm chứng số bằng KSG N lớn trước khi tin) |
| Code | `src/pqrst/data/synthetic/switching.py` + test ground-truth |
| Notebook | `phase_u_05_treet_benchmark.ipynb` |
| Đo | KSG, mạng học sẵn (huấn luyện trên họ này), Hybrid ở N=10–200; TREET nếu chạy được, không thì đối chiếu số công bố |
| Thoát | Bảng MSE/bias/variance theo N kèm CI |

## S5 — B: toán học, ~1 tuần (song song)

Viết `docs/THEORY_NOTES.md`: Mệnh đề 1 (tách rời cộng tính → DV ≤ 0), Mệnh đề 2 (phương sai hiệu hai nhánh theo ρ). Kèm 1 notebook/script kiểm chứng số (`phase_u_06_theory_check.ipynb`): công thức phương sai khớp số đo từ Pha Q′. Claude soạn, chủ dự án + mentor rà.

## S6 — D: mở rộng bằng chứng, ~2 tuần

| Việc | Ghi chú |
|---|---|
| Đa seed ≥5 + CI | Chạy lại các thí nghiệm mô phỏng chính với seed khác; notebook `phase_u_07_multiseed.ipynb` (nặng, chạy đêm) |
| Họ dữ liệu mới | Thêm 1–2 bộ sinh (phi tuyến khác, không Gaussian) + ground-truth có test |
| Trễ k>1 | Thêm tuỳ chọn độ dài lịch sử vào mạng/corpus/đánh giá; kiểm tra kết luận N-crossover còn không |
| Lâm sàng | Kiểm tra cỡ mẫu nhóm (Fantasia trẻ/già; Apnea-ECG nhóm) **trước**; nếu đủ thì `phase_u_08_group_comparison.ipynb`, kiểm định giữa nhóm theo bản ghi. Nếu không đủ cỡ mẫu thì ghi nhận là giới hạn, không ép |

## S7 — E2, E3: tái lập và quản lý kết quả, ~2–3 ngày

`scripts/reproduce_all.py` (hoặc notebook tổng) tái tạo mọi bảng/hình bài báo từ dữ liệu đã lưu; quyết định `.gitignore` cho `results/` (force-add đích danh các file được bài dùng, hoặc chuyển sang lưu trữ ngoài kèm checksum); cố định `requirements.txt` bản.

## S8 — Viết bản thảo (T.6), ~3–4 tuần

Khung: Introduction (khoảng trống N nhỏ) → Related Work (TREET/TENDE) → Methods (estimator, Hybrid, hiệu chỉnh, chẩn đoán PCA, thích nghi nếu có) → Results (mô phỏng, đối chiếu TREET, dữ liệu thật, lâm sàng nếu có) → Discussion (domain gap, hạn chế trung thực) → Phụ lục (lượng tử, dequantization, mệnh đề toán). Chủ dự án viết, Claude soát câu chữ/dịch từng đoạn.

## S9 — Rà soát cuối (T.7), ~1 tuần

Đối chiếu 100% số liệu bản thảo với CSV/báo cáo gốc; chạy lại `reproduce_all`; kiểm tra tài liệu tham khảo và trạng thái công bố của mọi bài trích dẫn.

## Bảng checklist theo dõi

| Bước | Trạng thái |
|---|---|
| S0 kiểm định thống kê | ☐ |
| S1 thăm dò A1 → G1 | ☐ |
| S1b kiểm tra mã TREET | ☐ |
| S2 randomization (nếu qua G1) | ☐ |
| S3 gap định lượng + k-fold | ☐ |
| S4 so sánh TREET | ☐ |
| S5 toán | ☐ |
| S6 mở rộng | ☐ |
| S7 tái lập + git | ☐ |
| S8 bản thảo | ☐ |
| S9 rà soát | ☐ |

## Quy tắc quyết định

- **G1 xấu:** không đầu tư tiếp A; bài định vị Q2 (benchmark + chẩn đoán); vẫn làm B, C, D, E.
- **Bất kỳ kết quả thí nghiệm nào trái kỳ vọng:** ghi nhận trung thực, đưa nguyên nhân (đo, không đoán) vào báo cáo pha; không sửa số liệu để khớp câu chuyện.
- **Việc nặng (>1 giờ máy):** luôn có notebook riêng, có bước tự kiểm tra sớm (in cảnh báo nếu dấu hiệu hỏng) để dừng sớm được.
