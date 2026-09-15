## 1. Thông số model

Mạng `T_φ`: MLP 4→128→128→64→1, ELU. **25.473 tham số.** Có thêm bản thu nhỏ
`[16,16]` (369 tham số) để kiểm định giả thuyết "ít tham số". Huấn luyện: Adam
lr=0.001, tối đa 100 epoch, trên 27.000 cửa sổ mô phỏng, test trên 7.200 cửa sổ
khác seed.

## 2. Hình chỉ rõ "gap"

Chiếu dữ liệu tim–hô hấp thật vào đúng không gian đặc trưng mạng đã học từ
dữ liệu mô phỏng (không train lại, không fit lại PCA):

![Domain gap](../results/figures/phase_t_pca_domain_gap.png)

Dữ liệu thật (tam giác đỏ) nằm tách biệt hoàn toàn khỏi vùng dữ liệu mô phỏng ở
mọi lớp — đây là gap: dữ liệu thật rơi ra ngoài "vùng hiểu biết" của mạng, giải
thích bằng hình học cho việc mạng thất bại trên dữ liệu thật ở Pha S.

## 3. Model có phù hợp với phân phối dữ liệu không — PCA theo từng khối

Tô theo cường độ ghép nối thật (dữ liệu mô phỏng):

![PCA theo coupling](../results/figures/phase_t_pca_blocks_by_coupling.png)

Từ khối 1 trở đi các điểm tách dần theo đúng giá trị ghép nối thật — mạng học
được biểu diễn phản ánh đúng cấu trúc dữ liệu, không học mù.

## 4. Góc nhìn toán học

Tô theo joint (cặp thật) vs marginal (cặp xáo trộn) — đúng 2 loại mẫu mà chặn
dưới Donsker-Varadhan (nền tảng lý thuyết của MINE) yêu cầu mạng phải phân biệt
được:

![PCA joint vs marginal](../results/figures/phase_t_pca_blocks_joint_vs_marginal.png)

Tách biệt rõ ngay từ khối 1 — mạng học đúng đối tượng toán học của bài toán,
không chỉ "ra số tốt hơn".