# Nhật ký Hoạt động và Tiến độ (Diary)

**Cập nhật lần cuối:** 2026-10-10

## Định hướng từ Mentor
> "Kì này anh muốn các bạn trong nhóm Signal Processing và System có thể hỗ trợ lẫn nhau. Anh muốn các bạn học thêm từ nhau nhiều thứ và cố gắng có được các kết quả phục vụ cho đồ án tốt nghiệp sắp tới vì 80% các bạn đang năm 4 rồi. 
> 
> Anh muốn nhóm phải thật giỏi để có thể phát triển các thiết bị tại biên.
> 
> Các bạn cập nhật đường dẫn nhật kí lên cho anh để thầy và anh theo dõi được tiến độ. Ngoài ra anh muốn tất cả các bạn sẽ quản lý 1 nhóm vi xử lý hoặc xử lý tín hiệu số để tập leader, cũng như hỗ trợ các bạn làm nghiên cứu."

---

## 1. Những việc ĐÃ LÀM (Done)
Dựa trên trạng thái dự án thực tế:
- **Hạ tầng thuật toán (Pha P, Q, R):** Đã xây dựng hoàn thiện nền tảng đánh giá Transfer Entropy (TE) bằng học sâu (Amortized MINE). Đã kiểm chứng được mô hình có khả năng vượt qua các phương pháp truyền thống ở cửa sổ dữ liệu lớn (N > 30).
- **Kiểm định dữ liệu sinh lý thực tế (Pha S):** Đã thực nghiệm và đánh giá thành công trên các bộ dữ liệu thật (Fantasia, Apnea-ECG) cho bài toán tương tác tim - hô hấp.
- **Tích hợp giải pháp lai (Pha T):** Đã lập trình thành công module `HybridTEEstimator` kết hợp KSG và Amortized MINE.
- **Nghiên cứu mô hình lượng tử (Nhịp 2):** Đã chạy thử nghiệm mạch lượng tử học máy (Quantum Machine Learning) và đưa ra kết luận rõ ràng về khả năng ứng dụng so với cổ điển.
- **Pha U - Nâng cấp Q1/Q2 (Phần 1 - Đã xong 2026-10-10):**
  - Đã chính thức hóa các Mệnh đề toán học (Mệnh đề 1, 2, 3, 4) trong `docs/THEORY_NOTES.md`.
  - Đã xây dựng module tiện ích tiêm nhiễu Gauss `src/pqrst/utils/noise.py` và bộ test đạt 123/123 test PASS.
  - Đã hoàn thành thực nghiệm đánh giá độ bền trước nhiễu Gauss (AWGN từ Clean đến 0 dB): Amortized MINE và Hybrid chứng minh MSE thấp hơn KSG từ 1.5 - 2.8 lần ở vùng $N \ge 50$.
  - Đã xuất bản báo cáo chi tiết Pha U tại `docs/PHASE_U_REPORT.md` kèm hình vẽ chất lượng cao tại `results/figures/noise_robustness_mse.png`.
- **Pha U - Nâng cấp Q1/Q2 (Phần 2 - Đã xong 2026-10-10):**
  - Đã xây dựng thành công mô hình lai Quantum-Classical `HybridClassicalQuantumStatisticsNetwork` (MLP + VQC 4-qubit) tại `src/pqrst/estimators/quantum/hybrid_vqc.py`.
  - Đã xây dựng mô hình Temporal 1D-CNN `Conv1DStatisticsNetwork` tại `src/pqrst/estimators/mine/temporal.py`.
  - Đã xây dựng suite kiểm thử và đánh giá đa kiến trúc so sánh: Classical MLP, Small MLP, Temporal 1D-CNN, Pure VQC và Hybrid MLP+VQC (`results/tables/model_architecture_comparison.csv` và `results/figures/model_architecture_comparison.png`).
  - Đã chạy và nhúng kết quả vào notebook `notebooks/phase_u_06_model_architecture_comparison.ipynb`.

- **Pha U - Nâng cấp Q1/Q2 (Phần 3 - Đã xong 2026-10-10):**
  - Giải quyết triệt để vấn đề 17/40 bản ghi: Bộ lọc cũ loại nhầm 17/20 bản ghi người cao tuổi do hiểu lầm hiện tượng suy giảm RSA sinh lý tự nhiên là lỗi đồng bộ.
  - Phục hồi và tiền xử lý toàn bộ 40/40 bản ghi Fantasia (20 Trẻ: 21–34 tuổi vs 20 Già: 68–85 tuổi, 10 Nam / 10 Nữ mỗi nhóm) với 9.443 cửa sổ.
  - Hoàn thành benchmark lâm sàng trên toàn bộ 40 bản ghi: Amortized MINE (thích nghi 5-fold OOF) và KSG chứng minh hiện tượng lão hóa tim mạch (RSA blunting) với ý nghĩa thống kê cao (Wilcoxon $p < 0.001$, Mann-Whitney $U = 297.0, p = 0.0045$, Cohen's $d = 0.67$).
  - Xác nhận giảm phương sai 2.2 lần trên dữ liệu thật (STD nhóm trẻ $0.0184$ so với $0.0406$ của KSG).
  - Xuất bản đồ thị chất lượng cao 4 panel `results/figures/fantasia_clinical_aging_te.png` và notebook `notebooks/phase_u_07_fantasia_clinical_aging.ipynb` đã chạy hoàn tất.
- **Pha U - Đóng gói Bản thảo Bài báo Q1/Q2 (Phần 4 - Đã xong 2026-10-10):**
  - Hoàn thiện bản thảo bài báo khoa học chi tiết tại `docs/PAPER_MANUSCRIPT_DRAFT.md` (chuẩn bị cho IEEE Transactions on Biomedical Engineering / TNNLS).
  - Tích hợp toàn bộ cấu trúc: Abstract, Introduction, 4 Mệnh đề toán học, Kiến trúc mô hình & UDA, 3 bài benchmark lớn, Thảo luận so sánh TREET/TENDE và Kết luận.
  - Test suite hoàn thành 128/128 tests PASS (100%).

## 2. Những việc ĐANG LÀM và HƯỚNG TỚI (Doing / To-do)
- **Tập trung Đồ án Tốt nghiệp & Bản thảo LaTeX:** Các bạn sinh viên năm 4 sử dụng kết quả, thuật toán và dữ liệu đã đạt được để đóng gói thành đồ án tốt nghiệp và biên dịch vào file `docs/paper/main.tex`.
- **Họp báo cáo Mentor & Thầy:** Trình bày kết quả giải quyết 5 điểm phản hồi và xin ý kiến rà soát bản thảo trước khi submit.
- **Phát triển năng lực quản lý (Leadership):** Các thành viên bắt đầu đảm nhận vai trò leader, quản lý các nhóm nhỏ mảng vi xử lý hoặc xử lý tín hiệu số để hỗ trợ các nghiên cứu mới.
- **Hướng tới Edge AI (Thiết bị tại biên):** Nâng cao chuyên môn nhóm, tối ưu hoá các mô hình TE nhỏ gọn để có khả năng tích hợp và chạy trực tiếp trên các thiết bị phần cứng tại biên.
- **Cập nhật tiến độ định kỳ:** Duy trì file nhật ký này để thầy và mentor có thể theo dõi sát sao tiến độ nghiên cứu của toàn nhóm.
