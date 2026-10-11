# TỔNG HỢP VÀ PHÂN TÍCH TOÁN HỌC CHUYÊN SÂU: TREET & TENDE
**Tài liệu phân tích đối sánh dành cho Mentor và Nhóm Nghiên cứu PQRST**
*Biên soạn: SPARC Laboratory — ĐHBK Hà Nội*

---

## LỜI MỞ ĐẦU: VÌ SAO TOÁN CỦA HAI BÀI NÀY LẠI "XỊN"?

Hai bài báo được Mentor chỉ ra đại diện cho **hai trường phái toán học đỉnh cao nhất hiện nay** (2025–2026) trong bài toán ước lượng Transfer Entropy (TE):
1. **TREET (IEEE Access 2025 / arXiv:2402.06919)**: Trường phái **Lý thuyết Thông tin Cổ điển & Giải tích Hàm kết hợp Causal Attention** (Ben-Gurion University). Bài báo sử dụng công cụ toán học rất nặng: Biểu diễn biến phân Donsker-Varadhan (DV), Định lý Công thái học Birkhoff tổng quát (Generalized Birkhoff Ergodic Theorem), Định lý Xấp xỉ Phổ quát cho Causal Transformer, và Bổ đề Biểu diễn Hàm (Functional Representation Lemma).
2. **TENDE (AISTATS 2026 / arXiv:2510.14096)**: Trường phái **Phương trình Vi phân Ngẫu nhiên (SDE) & Khớp Hàm Điểm (Score-based Diffusion)** (KAUST & EURECOM). Bài báo sử dụng mối liên hệ vi phân hình học giữa hàm điểm (Score functions) và phân kỳ Kullback-Leibler (KL divergence), giúp **thoát khỏi hoàn toàn định lý bất khả McAllester-Stratos** vốn là "cơn ác mộng" của các mô hình cận biến phân như MINE.

Dưới đây là bản tóm tắt chi tiết, chuẩn xác từng công thức toán học, kèm theo phân tích bản chất trực giác và đối sánh trực diện với mô hình Amortized MINE của nhóm chúng ta.

---

# PHẦN 1: BÀI BÁO 1 — TREET
### *(Transfer Entropy Estimation via Transformers — Luxembourg, Tsur, Permuter 2025)*

---

### 1.1. Định nghĩa toán học & Mối liên hệ với Directed Information (DI)

TREET định nghĩa Transfer Entropy giữa hai quá trình dừng $\mathbb{X}$ và $\mathbb{Y}$ với bậc nhớ $(k, l)$ như sau:
$$\mathrm{TE}_{\mathbb{X} \to \mathbb{Y}}(k, l) := I(X_{t-l}^t; Y_t \mid Y_{t-k}^{t-1})$$

> **Điểm toán học khác biệt:** Định nghĩa chuẩn của Schreiber (2000) chỉ lấy quá khứ $X_{t-l}^{t-1}$, nhưng TREET đưa cả hiện tại $X_t$ vào chuỗi nguồn. Khi quá trình độc lập cùng phân phối (i.i.d.), công thức này thu về Mutual Information $I(X_t; Y_t)$, giúp thống nhất lý thuyết thông tin.

Khi $k = l$, TREET thiết lập cầu nối với **Directed Information Rate** $I(\mathbb{X} \to \mathbb{Y})$:
$$I(X^n \to Y^n) = \sum_{i=0}^{n-1} I(X^i; Y_i \mid Y^{i-1}) = \sum_{i=0}^{n-1} \mathrm{TE}_{\mathbb{X} \to \mathbb{Y}}(i)$$
$$I(\mathbb{X} \to \mathbb{Y}) = \lim_{n \to \infty} \frac{1}{n} I(X^n \to Y^n) = \lim_{n \to \infty} I(X^n; Y_n \mid Y^{n-1})$$

**Bổ đề 1 (Tính tương đương dưới điều kiện Markov):**
Nếu hai quá trình thỏa mãn tính chất xích Markov:
$$Y_l - Y^{l-1} - Y_{-\infty}^{-1} \quad \text{và} \quad Y_l - (X^l, Y^{l-1}) - (X_{-\infty}^{-1}, Y_{-\infty}^{-1})$$
Thì với mọi $m \ge l$:
$$\mathrm{TE}_{\mathbb{X} \to \mathbb{Y}}(m) = \mathrm{TE}_{\mathbb{X} \to \mathbb{Y}}(l) = I(\mathbb{X} \to \mathbb{Y})$$
*Ý nghĩa:* Transfer Entropy với độ dài ngữ cảnh hữu hạn $l$ chính là nghiệm xấp xỉ chính xác của thông tin định hướng trên chuỗi dài vô hạn.

---

### 1.2. Biểu diễn TE qua Hiệu hai Phân kỳ KL với Phân phối Tham chiếu (Lemma 2)

Điểm "sáng tạo" nhất về mặt giải tích của TREET là đưa thêm một **phân phối tham chiếu liên tục tùy ý $\widetilde{Y}$** (thường chọn phân phối đều Uniform) để phân rã Transfer Entropy:

**Bổ đề 2 (Biểu diễn TE qua hiệu 2 phân kỳ KL):**
$$\mathrm{TE}_{\mathbb{X} \to \mathbb{Y}}(l) = \mathcal{D}_{Y^l \mid Y^{l-1} X^l \parallel \widetilde{Y}^l} - \mathcal{D}_{Y^l \mid Y^{l-1} \parallel \widetilde{Y}^l}$$
Trong đó:
$$\mathcal{D}_{Y^l \mid Y^{l-1} \parallel \widetilde{Y}^l} := D_{\mathrm{KL}}\left(P_{Y_l \mid Y^{l-1}} \,\big\|\, P_{\widetilde{Y}} \;\Big|\; P_{Y^{l-1}}\right) = \int \log \frac{P(y_l \mid y^{l-1})}{P_{\widetilde{Y}}(y_l)} P(x^l, y^l) d(x^l, y^l)$$
$$\mathcal{D}_{Y^l \mid Y^{l-1} X^l \parallel \widetilde{Y}^l} := D_{\mathrm{KL}}\left(P_{Y_l \mid Y^{l-1} X^l} \,\big\|\, P_{\widetilde{Y}} \;\Big|\; P_{Y^{l-1} X^l}\right) = \int \log \frac{P(y_l \mid x^l, y^{l-1})}{P_{\widetilde{Y}}(y_l)} P(x^l, y^l) d(x^l, y^l)$$

**Tại sao cách làm này thông minh?**
Khi trừ hai số hạng cho nhau:
$$\mathcal{D}_{Y^l \mid Y^{l-1} X^l \parallel \widetilde{Y}^l} - \mathcal{D}_{Y^l \mid Y^{l-1} \parallel \widetilde{Y}^l} = \int \log \frac{P(y_l \mid x^l, y^{l-1})}{P(y_l \mid y^{l-1})} P(x^l, y^l) d(x^l, y^l) = h(Y_l \mid Y^{l-1}) - h(Y_l \mid X^l, Y^{l-1}) = \mathrm{TE}$$
Số hạng tham chiếu $P_{\widetilde{Y}}(y_l)$ bị **triệt tiêu hoàn toàn**! Phân phối $\widetilde{Y}$ đóng vai trò như một "điện thế mốc" (ground reference), cho phép chuyển bài toán tính Transfer Entropy có điều kiện phức tạp thành việc ước lượng 2 phân kỳ KL độc lập.

---

### 1.3. Cận biến phân Donsker-Varadhan (DV) trong TREET

Với mỗi số hạng phân kỳ KL, TREET áp dụng biểu diễn biến phân Donsker-Varadhan:
$$\mathcal{D}_{Y^l \mid Y^{l-1} \parallel \widetilde{Y}^l} = \sup_{g_y \in \mathcal{G}_{\mathrm{ctf}}^Y} \left( \mathbb{E}\left[g_y(Y^l)\right] - \log \mathbb{E}\left[e^{g_y(\widetilde{Y}_l, Y^{l-1})}\right] \right)$$
$$\mathcal{D}_{Y^l \mid Y^{l-1} X^l \parallel \widetilde{Y}^l} = \sup_{g_{xy} \in \mathcal{G}_{\mathrm{ctf}}^{XY}} \left( \mathbb{E}\left[g_{xy}(Y^l, X^l)\right] - \log \mathbb{E}\left[e^{g_{xy}(\widetilde{Y}_l, Y^{l-1}, X^l)}\right] \right)$$

Trên mẫu thực nghiệm $D_n$, nghiệm tối ưu lý thuyết đạt được tại tỉ số mật độ logarit thực sự:
$$f_{y, l}^*(Y^l) = \log \frac{d P_{Y^l}}{d(P_{Y^{l-1}} \otimes P_{\widetilde{Y}})} = \log P(y_l \mid y^{l-1}) - \log P_{\widetilde{Y}}(y_l)$$

---

### 1.4. Mạng Fixed-Past Causal Attention (FPCA)

Để đảm bảo tính nhân quả (không nhìn trước tương lai) và giữ ngữ cảnh lịch sử cố định có độ dài $l$, TREET thiết kế mặt nạ chú ý dạng băng Toeplitz (Toeplitz-like banded mask $M' \in \mathbb{R}^{L \times L}$):
$$M'_{i,j} = \begin{cases} 1 & \text{nếu } 0 \le i - j < l \\ -\infty & \text{ngược lại} \end{cases}$$
Toán tử chú ý FPCA:
$$\mathrm{FPCA}(X) = \mathrm{softmax}\left( \frac{QK^\top}{\sqrt{d_k}} \odot M' \right) V$$

Để tính số hạng mẫu tham chiếu $\mathbb{E}[e^{g_y(\widetilde{Y}_l, Y^{l-1})}]$, TREET sửa đổi cơ chế Attention thành **Modified FPCA**: Giữ nguyên toàn bộ Key và Value của quá khứ $Y^{l-1}$, chỉ thay thế Query, Key, Value của thời điểm hiện tại $t$ bằng mẫu tham chiếu $\widetilde{Y}_t$.
*Độ phức tạp:* Giảm xuống $\mathcal{O}(L \cdot l \cdot d_o)$ thay vì $\mathcal{O}(L^2)$ của Transformer thông thường.

---

### 1.5. Chứng minh Toán học: Tính Vững Mạnh (Strong Consistency — Theorem 3)

TREET chứng minh rằng ước lượng $\widehat{\mathrm{TE}}(D_n; l) \xrightarrow{P\text{-a.s.}} \mathrm{TE}(l)$ khi cỡ mẫu $n \to \infty$ qua 3 bước:
1. **Bước 1 (Biểu diễn):** Chuyển TE thành hiệu hai thế năng DV (Bổ đề 2).
2. **Bước 2 (Công thái học - Birkhoff Ergodic Theorem):** Áp dụng Định lý Birkhoff tổng quát:
   $$\frac{1}{n} \sum_{i=1}^n g_i(T^i \omega) \xrightarrow{n \to \infty} \mathbb{E}[g] \quad (P\text{-a.s.})$$
   Vì quá trình dừng và ergodic, trung bình mẫu hội tụ tới kỳ vọng toán học với sai số $< \epsilon/8$.
3. **Bước 3 (Xấp xỉ hàm phổ quát - Universal Approximation):** Dùng tính chất Lipschitz của hàm mũ $e^x$ trên đoạn $(-\infty, M]$:
   $$\frac{1}{n}\sum_{i=1}^n \left| e^{f^*(y_i)} - e^{g(y_i)} \right| \le e^M \frac{1}{n}\sum_{i=1}^n \left| f^*(y_i) - g(y_i) \right|$$
   Kết hợp với định lý xấp xỉ phổ quát của Transformer, chặn trên sai số toàn phần $\le \epsilon/2$.

---

# PHẦN 2: BÀI BÁO 2 — TENDE
### *(Transfer Entropy Neural Diffusion Estimation — Galeano Muñoz, Bounoua, Franzese, Michiardi, Filippone — AISTATS 2026)*

---

### 2.1. Đặt vấn đề: "Cơn ác mộng" của Cận Biến phân (MINE/TREET)

TENDE mở đầu bằng việc trích dẫn bài báo toán học nổi tiếng của **McAllester & Stratos (AISTATS 2020)**:
> *"Mọi bộ ước lượng thông tin tương hỗ dựa trên cận dưới biến phân phân kỳ (như Donsker-Varadhan / MINE) đều yêu cầu cỡ mẫu tăng theo cấp số nhân $n = \mathcal{O}(e^I)$ để hội tụ khi giá trị thông tin $I$ lớn."*

Để giải quyết triệt để vấn đề này, TENDE **từ bỏ hoàn toàn cận biến phân DV** và xây dựng bộ ước lượng dựa trên **Mô hình Khuếch tán Hàm điểm (Score-based Diffusion Models)**.

---

### 2.2. Phương trình Vi phân Ngẫu nhiên (VP-SDE) & Hàm Điểm (Score Function)

TENDE sử dụng phương trình vi phân ngẫu nhiên bảo toàn phương sai (Variance Preserving SDE):
$$dX_t = f_t X_t dt + g_t dW_t, \quad t \in [0, T]$$
Với $f_t = -\frac{1}{2}\beta(t)$, $g_t = \sqrt{\beta(t)}$, $\beta(t) = \beta_{\min} + (\beta_{\max} - \beta_{\min})t$.
Nghiệm chuyển tiếp $X_t \mid X_0 = x$ có dạng phân phối Gauss giải tích đóng:
$$p_{0t}(x_t \mid x_0) = \mathcal{N}\left(x_t; \; k_t x_0, \; \sigma_t^2 I\right), \quad k_t = \exp\left(\int_0^t f_s ds\right), \; \sigma_t^2 = 1 - k_t^2$$

**Định nghĩa Hàm Điểm (Score Function):**
$$S_{p_X}(x) := \nabla_x \log p_X(x)$$
Mạng nơ-ron $S(x; \theta)$ học hàm điểm bằng kỹ thuật **Denoising Score Matching** (Vincent 2011, Song 2020):
$$\mathcal{L}(\theta) = \int_0^T g_t^2 \, \mathbb{E}_{x \sim p, \, \tilde{x} \sim p_{0t}}\left[ \left\| S(\tilde{x}; \theta) - \nabla_{\tilde{x}} \log p_{0t}(\tilde{x} \mid x) \right\|^2 \right] dt$$

---

### 2.3. Cầu nối Toán học Đột phá: Phân kỳ KL và Hàm Điểm

**Định lý Cốt lõi (Bounoua 2024, Franzese 2023):**
Phân kỳ KL giữa hai phân phối liên tục bất kỳ $p$ và $q$ qua quá trình khuếch tán được tính chính xác bằng:
$$D_{\mathrm{KL}}(p \parallel q) = \int_0^T \frac{g_t^2}{2} \, \mathbb{E}_{x \sim p_t}\left[ \left\| S_{p_t}(x) - S_{q_t}(x) \right\|^2 \right] dt + D_{\mathrm{KL}}(p_T \parallel q_T)$$

Khi thời gian khuếch tán $T$ đủ lớn, $p_T$ và $q_T$ đều co về phân phối dừng chuẩn $\mathcal{N}(0, I)$ nhờ tính chất co của nửa nhóm khuếch tán (Logarithmic Sobolev Inequality - Collet & Malrieu 2008), do đó $D_{\mathrm{KL}}(p_T \parallel q_T) \to 0$.

Bộ ước lượng phân kỳ KL của TENDE:
$$e(p, q) = \int_0^T \frac{g_t^2}{2} \, \mathbb{E}_{x \sim p_t}\left[ \left\| S_{p_t}(x; \theta_1^*) - S_{q_t}(x; \theta_2^*) \right\|^2 \right] dt \simeq D_{\mathrm{KL}}(p \parallel q)$$

---

### 2.4. Tại sao Toán của TENDE Miễn nhiễm với Cận McAllester-Stratos?

TENDE phân tích sai số ước lượng cực kỳ chặt chẽ (Mục A.3 trong phụ lục):
$$e(p, q) - D_{\mathrm{KL}}(p \parallel q) = d - D_{\mathrm{KL}}(p_T \parallel q_T)$$
Trong đó sai số hàm điểm $d$ có dạng:
$$d = \int_0^T \frac{g_t^2}{2} \, \mathbb{E}_{x \sim p_t} \left[ \|\epsilon_t^p(x) - \epsilon_t^q(x)\|^2 + 2\langle S_{p_t}(x) - S_{q_t}(x), \; \epsilon_t^p(x) - \epsilon_t^q(x) \rangle \right] dt$$
Với $\epsilon_t^p(x) = S_{p_t}(x; \theta^*) - S_{p_t}(x)$ là sai số xấp xỉ của mạng.

> **HAI TÍNH CHẤT TOÁN HỌC ĐẮT GIÁ:**
> 1. **$d$ không xác định dấu (có thể dương hoặc âm):** $e(p, q)$ **không phải là cận dưới hay cận trên** của phân kỳ KL $\to$ Thoát khỏi hoàn toàn định lý bất khả của McAllester & Stratos!
> 2. **Cơ chế tự triệt tiêu sai số đồng pha (Common-Mode Error Cancellation):** Nếu mạng nơ-ron mắc cùng một sai số hệ thống trên cả hai phân phối $\epsilon_t^p(x) = \epsilon_t^q(x)$, thì sai số hiệu $d \equiv 0$ tuyệt đối, bất kể độ lớn của sai số thành phần!

---

### 2.5. Ba Công thức Ước lượng Transfer Entropy của TENDE

TENDE chuyển Transfer Entropy về Thông tin Tương hỗ Có điều kiện (CMI):
$$\mathrm{TE}_{X \to Y}(k, \ell) = I(Y_t; X_{t-k} \mid Y_{t-\ell})$$
Đặt $X = X_{t-k}$ (quá khứ nguồn), $Y = Y_t$ (hiện tại đích), $Z = Y_{t-\ell}$ (quá khứ đích). TENDE đề xuất 3 công thức tương đương:

1. **Dạng 1 (Kỳ vọng KL - Khuyên dùng vì luôn $\ge 0$):**
   $$I(X; Y \mid Z) = \mathbb{E}_{(y,z) \sim p_{Y,Z}} \left[ D_{\mathrm{KL}}(p_{X \mid y,z} \parallel p_{X \mid z}) \right] \simeq \mathbb{E}_{(y,z)} \left[ e(p_{X \mid y,z}, \, p_{X \mid z}) \right]$$
2. **Dạng 2 (Hiệu Entropy có điều kiện qua biến phụ trợ Gauss $\phi_\sigma$):**
   $$I(X; Y \mid Z) = H(X \mid Z) - H(X \mid Y, Z) \simeq \mathbb{E}_{(y,z)}\left[e(p_{X \mid y,z}, \phi_\sigma)\right] - \mathbb{E}_z\left[e(p_{X \mid z}, \phi_\sigma)\right]$$
3. **Dạng 3 (Hiệu Thông tin Tương hỗ):**
   $$I(X; Y \mid Z) = I(X; [Y, Z]) - I(X; Z) \simeq \mathbb{E}_{(y,z)}\left[e(p_{X \mid y,z}, p_X)\right] - \mathbb{E}_z\left[e(p_{X \mid z}, p_X)\right]$$

---

### 2.6. Mạng Điểm Học Gộp (Amortized Score Network)

Thay vì huấn luyện 2–3 mạng riêng biệt cho từng hàm điểm, TENDE huấn luyện duy nhất một mạng $S(Y, X, Z, t^*, e)$ với vector điều kiện $e = [e_1, e_2, e_3] \in \{-1, 0, 1\}^3$:
- $1$: Biến cần tính hàm điểm.
- $0$: Biến làm điều kiện (conditioning signal).
- $-1$: Biến bị loại bỏ / tích phân biên (marginalized out, gán về 0).
*Ví dụ:* $S_{p_{Y_t \mid x, z}} \leftrightarrow [1, 0, 0]$; $S_{p_{Y_t \mid z}} \leftrightarrow [1, -1, 0]$; $S_{p_{Y_t}} \leftrightarrow [1, -1, -1]$.

---

### 2.7. Tính Bất biến và Tính Cộng tính (Theorems B.1 & B.2)

- **Bổ đề B.1 (Bất biến trước chiều nhiễu dư thừa):** Nếu ghép $d$ chiều nhiễu trắng độc lập $\varepsilon_t$ vào $x_t$ và $y_t$, thì $\mathrm{TE}_{\widetilde{X} \to \widetilde{Y}} = \mathrm{TE}_{X \to Y}$ không đổi.
- **Bổ đề B.2 (Tính cộng tính tuyến tính):** Nếu ghép $d$ bản sao độc lập của hệ thống, thì $\mathrm{TE}_{\widetilde{X} \to \widetilde{Y}} = \sum_{j=1}^d \mathrm{TE}_{X_j \to Y_j}$.

---

# PHẦN 3: BẢNG SO SÁNH ĐỐI CHIẾU TRỰC DIỆN
### *(TREET vs. TENDE vs. AMORTIZED MINE CỦA NHÓM)*

| Tiêu chí So sánh | TREET (IEEE Access 2025) | TENDE (AISTATS 2026) | Amortized MINE (Dự án PQRST) |
|---|---|---|---|
| **Trường phái Toán học** | Cận biến phân Donsker-Varadhan (DV) + Causal Attention | Score-based Diffusion SDE + Mối liên hệ KL-Score vi phân | Amortized Neural MINE + Chia sẻ tham số Masked Network |
| **Cơ chế ước lượng** | Hiệu 2 phân kỳ KL đối với phân phối tham chiếu $\widetilde{Y}$ | Tích phân khoảng cách hàm điểm khuếch tán $\int_0^T \|S_p - S_q\|^2 dt$ | Mạng nơ-ron học tỉ số mật độ liên hợp/biên qua Donsker-Varadhan |
| **Bảo đảm Lý thuyết** | Định lý Birkhoff Ergodic + Causal Universal Approximation | Định lý co Logarithmic Sobolev + Triệt tiêu sai số đồng pha | 4 Mệnh đề giải tích: Bất khả tách, Giảm phương sai Cov > 0, Lipschitz AWGN, Ngưỡng $N^*$ |
| **Chế độ dữ liệu mục tiêu** | Chuỗi dài (long-memory context, $l \le 99$) | Chiều cao (high-dimensional, 70 chiều), $T \ge 10.000$ mẫu | **Chuỗi siêu ngắn ($N \le 50 - 100$ mẫu), phi dừng y sinh** |
| **Thời gian chạy / Độ trễ** | $\sim 100 - 300\text{ ms}$ (Transformer CPU/GPU) | **$\sim 20\text{ phút}$ / 1 cặp chuỗi (trên GPU khủng NVIDIA A100!)** | **$7.0\text{ ms}$ (Small MLP) / $11\text{ ms}$ (Deep MLP) trên CPU thường!** |
| **Khả năng triển khai Edge-AI** | Kém (Transformer tốn bộ nhớ attention) | **Hoàn toàn bất khả thi** (chi phí SDE lặp hàng trăm bước khuếch tán) | **Xuất sắc (369 tham số, chiếm vài KB RAM, chạy realtime trên vi điều khiển)** |
| **Độ nhạy cỡ mẫu nhỏ ($N < 50$)** | Bị bias và phương sai lớn nếu $N$ nhỏ | Cần dữ liệu lớn ($T = 10.000 - 50.000$) để học diffusion | **Triệt tiêu phương sai $> 90\%$ nhờ chia sẻ tham số mạng** |
| **Thích nghi Miền (Domain Adaptation)** | Chưa giải quyết (giả định phân phối dừng) | Chưa giải quyết | **Có giao thức UDA tối ưu trực tiếp DV loss trên dữ liệu lâm sàng** |

---

# PHẦN 4: NHỮNG ĐIỂM YẾU CHÍ MẠNG CỦA TREET & TENDE MÀ NHÓM TA VƯỢT TRỘI

Mentor xem toán thấy "xịn" vì hai bài này dùng các công cụ toán học rất hiện đại (Diffusion, SDE, Transformer, Birkhoff Ergodic). Tuy nhiên, **khi đưa vào ứng dụng y sinh thực tế, cả hai bài đều có lỗ hổng rất lớn**:

1. **Lỗ hổng của TENDE: Quá nặng nề, hoàn toàn bất khả thi cho thiết bị y tế đeo:**
   - Trong Mục D (trang 22), TENDE thừa nhận: *"Thời gian chạy trung bình để ước lượng TE cho một cặp chuỗi $T = 10.000$ mẫu là **khoảng 20 phút trên một GPU NVIDIA A100**"*.
   - Với các thiết bị theo dõi tim mạch / hô hấp liên tục (như đo ECG/PPG/PCG trên đồng hồ thông minh hoặc Holter), tín hiệu sinh lý thay đổi theo từng giây và vi điều khiển chỉ có vài chục KB RAM. Không có bệnh viện hay thiết bị đeo nào có thể gắn GPU A100 để tính Transfer Entropy!
   - **Mô hình của nhóm ta:** Small MLP chỉ có **369 tham số, chạy mất 7 mili-giây trên CPU** — nhanh hơn TENDE **170.000 lần**!

2. **Lỗ hổng của TREET: Dễ bị sụp đổ ở chiều cao và cửa sổ ngắn:**
   - Trong chính kết quả thực nghiệm của TENDE (Bảng 1, 2, 3, 4 ở trang 19, 20): Khi số chiều tăng lên hoặc gặp dữ liệu phức tạp, **TREET bị sụp đổ, cho ra giá trị Transfer Entropy âm và phương sai cực lớn** (ví dụ Table 1: TREET ước lượng ra $-3.02 \pm 2.59$ trong khi Ground Truth là $0.00$).
   - Nguyên nhân: Transformer cần lượng dữ liệu khổng lồ để học ma trận Attention. Với các cửa sổ y sinh siêu ngắn ($N \le 50 - 100$), ma trận Attention của TREET bị nhiễu hoàn toàn (noisy attention weights - xem Hình 7b trang 11 của TREET).
   - **Mô hình của nhóm ta:** Thiết kế hàm kích hoạt Lipschitz (ELU) và cơ chế chia sẻ trọng số liên hợp/biên giúp mô hình của ta **không bao giờ bị sụp đổ ở $N$ nhỏ**, vượt trội hơn KSG từ 1.6 đến 2.8 lần.

---

# PHẦN 5: BÀI HỌC ÁP DỤNG ĐỂ NÂNG TẦM TOÁN HỌC CHO BÀI BÁO CỦA NHÓM

Nhóm chúng ta hoàn toàn có thể "mượn" các kỹ thuật lập luận toán học sắc bén của TREET và TENDE để đưa vào bản thảo bài báo IEEE Transactions:

1. **Từ TREET (Cách chứng minh Consistency chặt chẽ):**
   - Áp dụng kỹ thuật của Bổ đề 2 TREET: Đưa biểu diễn phân phối tham chiếu $\widetilde{Y}$ vào phần phụ lục để chứng minh rằng mạng `MaskedStatisticsNetwork` của nhóm ta thực chất đang ước lượng tỉ số mật độ Radon-Nikodym thực sự một cách giải tích.
   - Trích dẫn định lý Birkhoff Ergodic để củng cố tính vững của bước thích nghi miền không giám sát (UDA).

2. **Từ TENDE (Cách phân tích sai số & Né cận McAllester):**
   - TENDE phân tích sai số thành: Sai số xấp xỉ + Sai số mẫu + Sai số đồng pha.
   - Nhóm ta áp dụng lập luận này vào **Mệnh đề 2 (Variance Reduction)**: Chứng minh rằng việc dùng chung một mạng nơ-ron cho cả $I_{\mathrm{full}}$ và $I_{\mathrm{red}}$ tạo ra cơ chế **triệt tiêu sai số đồng pha (Common-mode error cancellation)** y hệt như TENDE, giúp phương sai $\mathrm{Var}(\Delta \mathrm{TE})$ giảm tới 90.87%!

3. **Chiến lược "Chốt hạ" khi trao đổi với Mentor:**
   - *"Toán của TREET và TENDE rất đẹp về mặt lý thuyết nền tảng (Information Theory & SDE Diffusion), nhưng chúng được thiết kế cho bài toán chuỗi dài và cần siêu máy tính GPU A100.*
   - *Đóng góp độc bản của bài báo nhóm mình là giải quyết đúng khoảng trống nghiên cứu (Research Gap) mà cả TREET và TENDE đều bỏ cuộc: **Ước lượng Transfer Entropy trên chuỗi thời gian siêu ngắn y sinh ($N \le 50 - 100$) chạy trực tiếp trên thiết bị nhúng Edge-AI (7 ms, 369 tham số)**, có chứng minh toán học giải tích chặt chẽ về độ bền nhiễu Lipschitz và điểm giao thoa $N^*$."*
