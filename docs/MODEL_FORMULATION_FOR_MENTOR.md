# ĐỀ XUẤT KIẾN TRÚC MÔ HÌNH VÀ CƠ SỞ TOÁN HỌC (Dành cho Mentor)

*Tài liệu này mô tả chi tiết luồng dữ liệu, kiến trúc mạng và hàm mục tiêu (loss function) của phương pháp Amortized TE + UDA, nhằm hỗ trợ quá trình công thức hóa toán học cho bài báo.*

---

## 1. Bài toán và Ký hiệu (Problem Formulation)
Chúng ta cần ước lượng Transfer Entropy (TE) từ chuỗi thời gian $X$ sang chuỗi thời gian $Y$.
- Cho 2 chuỗi thời gian liên tục: $X = \{x_1, x_2, ..., x_t\}$ và $Y = \{y_1, y_2, ..., y_t\}$.
- Xét tại thời điểm $t$, với độ trễ (lag) là $k$:
  - Trạng thái hiện tại của biến mục tiêu: $Y_t = y_t$
  - Lịch sử của biến mục tiêu: $Y_{lag} = \{y_{t-k}, ..., y_{t-1}\}$
  - Lịch sử của biến nguồn: $X_{lag} = \{x_{t-k}, ..., x_{t-1}\}$

Theo định nghĩa của Shannon Information Theory, TE được biểu diễn qua Conditional Mutual Information (CMI):
$$TE_{X \rightarrow Y} = I(Y_t ; X_{lag} | Y_{lag}) = I(Y_t ; X_{lag}, Y_{lag}) - I(Y_t ; Y_{lag})$$

**Mục tiêu của Mô hình:** Thay vì dùng KSG (k-NN) để ước lượng 2 cụm Mutual Information (MI) này, chúng ta sử dụng Neural Network (Dựa trên MINE) để tối ưu hóa trực tiếp cận dưới (lower bound) của MI.

---

## 2. Kiến trúc Mạng Đề xuất (Amortized MINE with Shared Network)
Để giải quyết vấn đề phương sai (variance) quá lớn khi kích thước mẫu $N$ nhỏ ($N \le 200$), chúng ta không dùng mạng MINE tách biệt truyền thống, mà đề xuất kiến trúc **Shared Network** (Dùng chung trọng số).

### 2.1. Cấu trúc mạng tính $I(Y_t ; X_{lag}, Y_{lag})$ (Ký hiệu là $MI_{full}$)
Mạng này (gọi là $T_{\theta}$) nhận 2 nhánh đầu vào:
1.  **Nhánh Mục tiêu (Target Branch):** Nhận $Y_t$. Đi qua hàm biến đổi $f_{\theta_1}$.
2.  **Nhánh Lịch sử (History Branch):** Nhận concatenated vector $[X_{lag}, Y_{lag}]$. Đi qua hàm biến đổi $g_{\theta_2}$.

*Điểm mấu chốt (Novelty):* Trong các lớp ẩn (hidden layers) của $f_{\theta_1}$ và $g_{\theta_2}$, chúng ta ép chúng **chia sẻ chung một phần trọng số** (Shared Weights). Việc này ép các đặc trưng (features) của cả 2 nhánh phóng chiếu vào cùng một không gian ẩn (Shared Latent Space), giúp giảm triệt để sự độc lập giả mạo (spurious independence) trên mẫu nhỏ $\rightarrow$ Từ đó giảm phương sai (Variance Reduction).

Đầu ra của 2 nhánh được gộp lại (ví dụ: qua dot product hoặc concat) để xuất ra 1 giá trị vô hướng (scalar), đại diện cho mức độ tương quan.

### 2.2. Cấu trúc mạng tính $I(Y_t ; Y_{lag})$ (Ký hiệu là $MI_{reduced}$)
Tương tự như trên, nhưng mạng này (gọi là $T_{\phi}$) chỉ nhận:
1. Nhánh Mục tiêu: $Y_t$
2. Nhánh Lịch sử: $Y_{lag}$ (Không có $X_{lag}$).

Giá trị TE cuối cùng: $TE_{X \rightarrow Y} = MI_{full} - MI_{reduced}$

---

## 3. Hàm Mục Tiêu (Loss Function)
Chúng ta tối ưu hóa mạng nơ-ron bằng cách cực đại hóa cận dưới **Donsker-Varadhan (DV bound)**.
Với một tập dữ liệu gồm các mẫu (samples) khớp (Joint distribution $\mathbb{P}$) và các mẫu không khớp/hoán vị ngẫu nhiên (Marginal distribution $\mathbb{Q}$):

$$Loss = - \left( \mathbb{E}_{\mathbb{P}}[T(Joint)] - \log \mathbb{E}_{\mathbb{Q}}[e^{T(Marginal)}] \right)$$

Mô hình (Amortized) được huấn luyện một lần duy nhất trên lượng lớn dữ liệu giả lập (Synthetic Data) với hàm loss này.

---

## 4. Unsupervised Domain Adaptation (UDA)
Khi áp dụng mô hình (đã pre-train) vào dữ liệu thật (Ví dụ: Fantasia), phân phối dữ liệu bị lệch (Domain Gap).

**Cơ chế UDA:**
- Ngay tại thời điểm suy diễn (Test-time), ta lấy các mẫu dữ liệu thật (chưa biết chiều nhân quả / không có nhãn).
- Cho các mẫu dữ liệu này chạy qua mạng, tính lại hàm DV Loss như ở Mục 3.
- Tiến hành cập nhật (Backpropagation) một vài bước (epochs) với learning rate nhỏ để fine-tune các trọng số của lớp Shared Network.
- *Nhận xét:* Vì DV Loss chỉ đo Mutual Information dựa trên Joint và Marginal distributions (tự sinh ra bằng cách shuffle dữ liệu), quá trình này hoàn toàn **Unsupervised** (không cần ground-truth của bài toán). Nhờ đó, mô hình tự uốn nắn theo phân phối sinh lý thật của bệnh nhân mà không bị Overfitting vào nhãn.
