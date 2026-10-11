# TỔNG HỢP VÀ PHÂN TÍCH TOÁN HỌC CHUYÊN SÂU: TREET & TENDE
**Tài liệu phân tích giải tích và đối sánh dành cho Mentor**
*Biên soạn: SPARC Laboratory — ĐHBK Hà Nội*

---

# PHẦN 1: BÀI BÁO 1 — TREET
### *(Transfer Entropy Estimation via Transformers — Luxembourg, Tsur, Permuter 2025)*

---

### 1.1. Khung lý thuyết & Định nghĩa Transfer Entropy mở rộng

Xét hai quá trình ngẫu nhiên dừng $\mathbf{X} = (X_t)_{t \in \mathbb{Z}}$ và $\mathbf{Y} = (Y_t)_{t \in \mathbb{Z}}$ trên không gian xác suất $(\Omega, \mathcal{F}, P)$. Với các bậc nhớ quá khứ $k, l < \infty$, TREET đưa ra định nghĩa Transfer Entropy mở rộng như sau:
$$\mathrm{TE}_{\mathbf{X} \to \mathbf{Y}}(t; k, l) := I(X_{t-l}^t; Y_t \mid Y_{t-k}^{t-1})$$

Khi quá trình đồng thời dừng (jointly stationary), chỉ số thời gian $t$ được loại bỏ và viết gọn là $\mathrm{TE}_{\mathbf{X} \to \mathbf{Y}}(k, l)$:
$$\mathrm{TE}_{\mathbf{X} \to \mathbf{Y}}(k, l) = I(X^l; Y_l \mid Y_{l-k}^{l-1})$$

> **Điểm toán học mới so với Schreiber (2000):** Định nghĩa cổ điển của Schreiber chỉ lấy quá khứ $X_{t-l}^{t-1}$ (tức $I(X^{l-1}; Y_l \mid Y^{l-1})$). TREET chủ động bổ sung mẫu hiện tại $X_t$ vào chuỗi nguồn. Ý nghĩa giải tích: Khi hai quá trình là độc lập cùng phân phối (i.i.d.), công thức của TREET lập tức thu gọn về Mutual Information chuẩn: $\mathrm{TE}_{\mathbf{X} \to \mathbf{Y}}(0, 0) = I(X; Y)$. Khi $k = l$, bài báo ký hiệu vắn tắt là $\mathrm{TE}_{\mathbf{X} \to \mathbf{Y}}(l)$.

---

### 1.2. Mối liên hệ với Directed Information (DI) & Điều kiện xích Markov

Directed Information (Massey 1990, Kramer 1998) đo lường dòng thông tin nhân quả từ chuỗi $X^n$ tới $Y^n$:
$$I(X^n \to Y^n) := \sum_{i=0}^{n-1} I(X^i; Y_i \mid Y^{i-1}) = \sum_{i=0}^{n-1} \mathrm{TE}_{\mathbf{X} \to \mathbf{Y}}(i)$$

Tốc độ thông tin định hướng (DI rate) được định nghĩa qua giới hạn:
$$I(\mathbf{X} \to \mathbf{Y}) := \lim_{n \to \infty} \frac{1}{n} I(X^n \to Y^n) = \lim_{n \to \infty} I(X^n; Y_n \mid Y^{n-1})$$
Giới hạn này luôn tồn tại do $\mathrm{TE}_{\mathbf{X} \to \mathbf{Y}}(l) = h(Y_l \mid Y^{l-1}) - h(Y_l \mid X^l, Y^{l-1})$, và các entropy có điều kiện là không âm, đơn điệu giảm khi độ dài ngữ cảnh $l$ tăng (điều kiện hóa làm giảm entropy).

**Bổ đề 1 (Tính tương đương dưới điều kiện Markov):** Giả sử hai quá trình dừng thỏa mãn cấu trúc xích Markov:
$$Y_l - Y^{l-1} - Y_{-\infty}^{-1} \quad \text{và} \quad Y_l - (X^l, Y^{l-1}) - (X_{-\infty}^{-1}, Y_{-\infty}^{-1})$$
Khi đó, với mọi $m \ge l$, ta có đẳng thức giải tích đóng:
$$\mathrm{TE}_{\mathbf{X} \to \mathbf{Y}}(m) = \mathrm{TE}_{\mathbf{X} \to \mathbf{Y}}(l) = I(\mathbf{X} \to \mathbf{Y})$$
*Ý nghĩa:* Việc ước lượng Transfer Entropy với độ dài bộ nhớ hữu hạn $l$ chính là nghiệm xấp xỉ chính xác của Directed Information Rate trên chuỗi dài vô hạn.

---

### 1.3. Bổ đề 2: Phân rã TE qua Hiệu hai Phân kỳ KL với Phân phối Tham chiếu

Để giải quyết trở ngại tính toán phân phối có điều kiện phức tạp, TREET đưa ra một **phân phối tham chiếu liên tục tùy ý $\widetilde{Y}$** (reference measure) trên không gian quan sát $\mathcal{Y}$ (thường chọn phân phối đều Uniform):

**Bổ đề 2 (Biểu diễn TE qua hiệu 2 phân kỳ KL):**
$$\mathrm{TE}_{\mathbf{X} \to \mathbf{Y}}(l) = \mathcal{D}_{Y^l \mid Y^{l-1} X^l \parallel \widetilde{Y}^l} - \mathcal{D}_{Y^l \mid Y^{l-1} \parallel \widetilde{Y}^l}$$
Trong đó hai số hạng phân kỳ KL có điều kiện được xác định chính xác theo tích phân Lebesgue:
$$\mathcal{D}_{Y^l \mid Y^{l-1} \parallel \widetilde{Y}^l} := D_{\mathrm{KL}}\left( P_{Y_l \mid Y^{l-1}} \,\big\|\, P_{\widetilde{Y}} \;\Big|\; P_{Y^{l-1}} \right) = \int \log \frac{P(y_l \mid y^{l-1})}{P_{\widetilde{Y}}(y_l)} P(x^l, y^l) d(x^l y^l)$$
$$\mathcal{D}_{Y^l \mid Y^{l-1} X^l \parallel \widetilde{Y}^l} := D_{\mathrm{KL}}\left( P_{Y_l \mid Y^{l-1} X^l} \,\big\|\, P_{\widetilde{Y}} \;\Big|\; P_{Y^{l-1} X^l} \right) = \int \log \frac{P(y_l \mid x^l, y^{l-1})}{P_{\widetilde{Y}}(y_l)} P(x^l, y^l) d(x^l y^l)$$

**Chứng minh giải tích (Proof of Lemma 2):** Trừ trực tiếp hai biểu thức tích phân:
$$\mathcal{D}_{Y^l \mid Y^{l-1} X^l \parallel \widetilde{Y}^l} - \mathcal{D}_{Y^l \mid Y^{l-1} \parallel \widetilde{Y}^l} = \int \log \frac{P(y_l \mid x^l, y^{l-1})}{P(y_l \mid y^{l-1})} P(x^l, y^l) d(x^l y^l) = h(Y_l \mid Y^{l-1}) - h(Y_l \mid X^l, Y^{l-1}) = \mathrm{TE}_{\mathbf{X} \to \mathbf{Y}}(l)$$
*Nhận xét:* Mật độ tham chiếu $P_{\widetilde{Y}}(y_l)$ ở mẫu số bị **triệt tiêu hoàn toàn**! Phân phối $\widetilde{Y}$ đóng vai trò như một "điện thế mốc", cho phép chuyển bài toán tính Transfer Entropy có điều kiện phức tạp thành hai bài toán tối ưu KL độc lập.

---

### 1.4. Cận biến phân Donsker-Varadhan (DV) & Nghiệm Mật độ Tối ưu

Mỗi số hạng KL divergence được biểu diễn thông qua công thức biến phân Donsker-Varadhan (1983):
$$D_{\mathrm{KL}}(P \parallel Q) = \sup_{f : \Omega \to \mathbb{R}} \left\{ \mathbb{E}_P[f] - \log \mathbb{E}_Q[e^f] \right\}$$

Áp dụng vào TREET với hai hàm nơ-ron $g_y$ và $g_{xy}$ thuộc họ Causal Transformer:
$$\mathcal{D}_{Y^l \mid Y^{l-1} \parallel \widetilde{Y}^l} = \sup_{g_y} \left\{ \mathbb{E}[g_y(Y^l)] - \log \mathbb{E}\left[\exp(g_y(\widetilde{Y}_l, Y^{l-1}))\right] \right\}$$
$$\mathcal{D}_{Y^l \mid Y^{l-1} X^l \parallel \widetilde{Y}^l} = \sup_{g_{xy}} \left\{ \mathbb{E}[g_{xy}(Y^l, X^l)] - \log \mathbb{E}\left[\exp(g_{xy}(\widetilde{Y}_l, Y^{l-1}, X^l))\right] \right\}$$

Nghiệm tối ưu giải tích $f^*$ đạt được tại tỉ số mật độ Radon-Nikodym thực tế:
$$f_{y,l}^*(Y^l) = \log \frac{d P_{Y^l}}{d(P_{Y^{l-1}} \otimes P_{\widetilde{Y}})} = \log P(y_l \mid y^{l-1}) - \log P_{\widetilde{Y}}(y_l)$$
Trên tập dữ liệu $n$ mẫu $D_n$, kỳ vọng toán học được xấp xỉ bằng trung bình mẫu qua mini-batch gradient ascent.

---

### 1.5. Cơ chế Fixed-Past Causal Attention (FPCA) & Mặt nạ Toeplitz

Để đảm bảo mạng Transformer tuân thủ tính nhân quả (không nhìn trước tương lai) và khóa độ dài ngữ cảnh lịch sử đúng $l$ mẫu, TREET đề xuất mặt nạ băng Toeplitz (Toeplitz-like banded mask $M' \in \mathbb{R}^{L \times L}$):
$$M'[i, j] = \begin{cases} 1 & \text{nếu } 0 \le i - j < l \\ -\infty & \text{ngược lại} \end{cases}$$

Toán tử chú ý FPCA:
$$\mathrm{FPCA}(X) = \mathrm{softmax}\left( \frac{QK^\top}{\sqrt{d_m}} \odot M' \right) V$$

**Modified FPCA cho số hạng tham chiếu:** Để tính kỳ vọng $g_y(\widetilde{Y}_l, Y^{l-1})$, TREET giữ nguyên toàn bộ Key và Value của quá khứ $Y^{l-1}$, chỉ thay thế Query, Key, Value tại thời điểm hiện tại bằng mẫu tham chiếu $\widetilde{Y}_t$. Độ phức tạp tính toán giảm từ $\mathcal{O}(L^2)$ xuống còn $\mathcal{O}(L \cdot l \cdot d_o)$.

---

### 1.6. Định lý 3 (Tính Vững Mạnh - Strong Consistency): Chứng minh Giải tích 3 bước

**Định lý 3:** Cho $\mathbf{X}, \mathbf{Y}$ là các quá trình ngẫu nhiên dừng, ergodic. Khi số mẫu $n \to \infty$, ước lượng $\widehat{\mathrm{TE}}_{\mathbf{X} \to \mathbf{Y}}(D_n; l)$ hội tụ hầu chắc chắn ($P$-a.s.) về giá trị Transfer Entropy lý thuyết: $|\widehat{\mathrm{TE}}(D_n; l) - \mathrm{TE}(l)| \le \varepsilon$.

**Chi tiết chứng minh (Proof of Theorem 3):**
1. **Bước 1 (Biểu diễn biến phân):** TE được phân rã thành hiệu 2 thế năng DV theo Bổ đề 2.
2. **Bước 2 (Công thái học - Generalized Birkhoff Ergodic Theorem):** Theo Định lý Birkhoff (1957), với biến đổi bảo toàn độ đo $T$:
   $$\frac{1}{n} \sum_{i=1}^n g_i(T^i \omega) \xrightarrow{n \to \infty} \mathbb{E}[g] \quad (P\text{-a.s.})$$
   Do quá trình dừng và ergodic, trung bình mẫu của hàm tối ưu $f^*$ hội tụ về kỳ vọng giải tích: $\left| \mathbb{E}[f^*] - \frac{1}{n} \sum f^* \right| < \frac{\varepsilon}{8}$.
3. **Bước 3 (Xấp xỉ phổ quát & Tính Lipschitz của hàm mũ):** Hàm $f^*$ bị chặn đều bởi $M$. Hàm mũ $e^x$ trên khoảng $(-\infty, M]$ có hằng số Lipschitz là $e^M$:
   $$\frac{1}{n} \sum \left| \exp(f^*) - \exp(g) \right| \le e^M \cdot \frac{1}{n} \sum |f^* - g|$$
   Theo Định lý Xấp xỉ Phổ quát của Causal Transformer (Yun et al. 2019), tồn tại mạng $g$ sao cho sai số xấp xỉ $< e^{-M} \cdot \frac{\varepsilon}{4}$. Kết hợp lại ta có sai số toàn phần bị chặn nghiêm ngặt: $|\widehat{\mathcal{D}}(D_n) - \mathcal{D}| \le \frac{\varepsilon}{2}$. Hai số hạng trừ nhau suy ra điều phải chứng minh.

---

### 1.7. Ưu điểm & Điểm yếu chí mạng của TREET

* **Ưu điểm:** Nền tảng lý thuyết ergodic vững chắc, nắm bắt rất tốt ngữ cảnh dài ($l = 99$), song song hóa tốt trên chuỗi dài.
* **LỖ HỔNG CHÍ MẠNG:** Khi áp dụng vào **cửa sổ siêu ngắn y sinh ($N \le 50 - 100$)** hoặc số chiều cao, Transformer không đủ mẫu để học ma trận chú ý, dẫn đến **Attention bị nhiễu loạn hoàn toàn, TE bị ước lượng âm và phương sai bùng nổ** (kết quả đối chứng ở bài TENDE: TREET bị âm tới $-3.02 \pm 2.59$ trong khi ground-truth là $0.00$).

---

# PHẦN 2: BÀI BÁO 2 — TENDE
### *(Transfer Entropy Neural Diffusion Estimation — Galeano Muñoz, Bounoua, Franzese, Michiardi, Filippone — AISTATS 2026)*

---

### 2.1. Đặt vấn đề: Thoát khỏi "Cơn ác mộng" của Cận Biến phân McAllester-Stratos

TENDE mở đầu bằng việc chỉ ra hạn chế nền tảng của các mô hình cận biến phân (như MINE hay TREET) theo định lý McAllester & Stratos (AISTATS 2020):
> *"Mọi bộ ước lượng dựa trên cận dưới biến phân phân kỳ (như Donsker-Varadhan) đều đòi hỏi cỡ mẫu tăng theo cấp số nhân $n = \mathcal{O}(e^I)$ để hội tụ khi giá trị thông tin $I$ lớn."*

Để khắc phục triệt để, TENDE từ bỏ hoàn toàn cận biến phân DV và thiết lập cầu nối giữa **Mô hình Khuếch tán Hàm điểm (Score-based Diffusion)** và **Phân kỳ Kullback-Leibler**.

---

### 2.2. Phương trình Vi phân Ngẫu nhiên (VP-SDE) & Hàm Điểm (Score Function)

TENDE sử dụng phương trình vi phân ngẫu nhiên bảo toàn phương sai (Variance Preserving SDE - Song et al. 2020):
$$dX_t = f_t X_t dt + g_t dW_t, \quad X_0 \sim p_X, \quad t \in [0, T]$$
trong đó $f_t = -\frac{1}{2}\beta(t)$, $g_t = \sqrt{\beta(t)}$, với lịch trình nhiễu tuyến tính $\beta(t) = \beta_{\min} + (\beta_{\max} - \beta_{\min})t$. Nghiệm chuyển tiếp $p_{0t}(x_t \mid x_0)$ có dạng phân phối Gauss giải tích đóng:
$$p_{0t}(x_t \mid x_0) = \mathcal{N}(x_t; \; k_t x_0, \; \sigma_t^2 I), \quad k_t = \exp\left(\int_0^t f_s ds\right), \; \sigma_t^2 = 1 - k_t^2$$

**Định nghĩa Hàm Điểm (Score Function):** $S_p(x) := \nabla_x \log p(x)$. Mạng nơ-ron học hàm điểm qua Denoising Score Matching:
$$\mathcal{L}(\theta) = \int_0^T g_t^2 \mathbb{E}_{x, \widetilde{x}}\left[ \left\| S(\widetilde{x}; \theta) - \nabla_{\widetilde{x}} \log p_{0t}(\widetilde{x} \mid x) \right\|^2 \right] dt$$

---

### 2.3. Cầu nối Toán học Đột phá: Phân kỳ KL qua Tích phân Sai phân Hàm Điểm

**Định lý Cốt lõi (Bounoua 2024, Franzese 2023):** Phân kỳ KL giữa hai phân phối liên tục bất kỳ $p$ và $q$ được biểu diễn chính xác qua sai phân hàm điểm:
$$D_{\mathrm{KL}}(p \parallel q) = \int_0^T \frac{g_t^2}{2} \mathbb{E}_{x \sim p_t}\left[ \left\| S_{p_t}(x) - S_{q_t}(x) \right\|^2 \right] dt + D_{\mathrm{KL}}(p_T \parallel q_T)$$

Khi thời gian khuếch tán $T$ đủ lớn, theo bất đẳng thức Logarithmic Sobolev (Collet & Malrieu 2008), cả hai phân phối đều co về phân phối dừng chuẩn $\mathcal{N}(0, I)$, do đó số hạng đầu cuối $D_{\mathrm{KL}}(p_T \parallel q_T) \to 0$.

Bộ ước lượng TENDE tính phân kỳ KL qua tích phân Monte Carlo:
$$e(p, q) = \int_0^T \frac{g_t^2}{2} \mathbb{E}\left[ \left\| S_{p_t}(x; \theta_1^*) - S_{q_t}(x; \theta_2^*) \right\|^2 \right] dt \simeq D_{\mathrm{KL}}(p \parallel q)$$

---

### 2.4. Phân tích Sai số $d$ & Cơ chế Tự Triệt tiêu Sai số Đồng pha (Common-Mode Cancellation)

TENDE chứng minh công thức phân rã sai số giải tích (Phụ lục A.3):
$$e(p, q) - D_{\mathrm{KL}}(p \parallel q) = d - D_{\mathrm{KL}}(p_T \parallel q_T)$$
trong đó số hạng sai số $d$ có dạng:
$$d = \int_0^T \frac{g_t^2}{2} \mathbb{E}_{x \sim p_t}\left[ \|\epsilon_t^p(x) - \epsilon_t^q(x)\|^2 + 2\langle S_{p_t}(x) - S_{q_t}(x), \; \epsilon_t^p(x) - \epsilon_t^q(x) \rangle \right] dt$$
với $\epsilon_t^p(x) = S(x; \theta^*) - S_{p_t}(x)$ là sai số xấp xỉ của mạng nơ-ron.

> **HAI ƯU THẾ TOÁN HỌC ĐẮT GIÁ:**
> 1. **$d$ không xác định dấu (có thể âm hoặc dương):** $e(p, q)$ **không phải là cận trên cũng không phải cận dưới** của KL divergence. Do đó, TENDE **hoàn toàn thoát khỏi định lý bất khả McAllester-Stratos**!
> 2. **Cơ chế tự triệt tiêu sai số đồng pha:** Nếu mạng nơ-ron mắc cùng một sai số hệ thống trên cả hai phân phối ($\epsilon_t^p(x) = \epsilon_t^q(x)$), thì hiệu sai số triệt tiêu và **$d \equiv 0$ tuyệt đối**, bất kể sai số tuyệt đối của từng hàm điểm lớn đến đâu!

---

### 2.5. Ba Dạng Công thức Tính TE & Mạng Amortized Score Network

Chuyển Transfer Entropy về Thông tin Tương hỗ Có điều kiện CMI: $\mathrm{TE}_{X \to Y}(k, l) = I(Y_t; X_{t-k} \mid Y_{t-l})$ với $X = X_{t-k}$, $Y = Y_t$, $Z = Y_{t-l}$:
* **Dạng 1 (Kỳ vọng KL - Khuyên dùng vì luôn $\ge 0$):**
  $$I(X; Y \mid Z) = \mathbb{E}_{y,z}\left[ D_{\mathrm{KL}}(p_{X \mid y,z} \parallel p_{X \mid z}) \right] \simeq \mathbb{E}_{y,z}\left[ e(p_{X \mid y,z}, p_{X \mid z}) \right]$$
* **Dạng 2 (Hiệu Entropy qua biến Gauss phụ trợ $\phi_\sigma$):**
  $$I(X; Y \mid Z) = H(X \mid Z) - H(X \mid Y, Z) \simeq \mathbb{E}_{y,z}[e(p_{X \mid y,z}, \phi_\sigma)] - \mathbb{E}_z[e(p_{X \mid z}, \phi_\sigma)]$$
* **Dạng 3 (Hiệu Mutual Information):**
  $$I(X; Y \mid Z) = I(X; [Y, Z]) - I(X; Z) \simeq \mathbb{E}_{y,z}[e(p_{X \mid y,z}, p_X)] - \mathbb{E}_z[e(p_{X \mid z}, p_X)]$$

**Mạng Amortized:** Dùng duy nhất một mạng U-Net nhận vector điều kiện $[e_1, e_2, e_3] \in \{-1, 0, 1\}^3$ (1: biến tính score, 0: biến làm điều kiện, -1: biến tích phân biên) để sinh đồng thời cả 3 hàm điểm, tiết kiệm chi phí tham số.

---

### 2.6. Bổ đề B.1 (Bất biến chiều nhiễu) & Bổ đề B.2 (Cộng tính tuyến tính)

* **Bổ đề B.1 (Bất biến trước chiều nhiễu độc lập):** Ghép $d$ chiều nhiễu trắng độc lập $\varepsilon_t$ vào chuỗi: $\widetilde{X}_t = [X_t, \varepsilon_t^x]$, $\widetilde{Y}_t = [Y_t, \varepsilon_t^y]$. Chứng minh giải tích cho thấy $p(\widetilde{Y}_t, \widetilde{X}_{t-k} \mid \widetilde{Y}_{t-l})$ phân rã tích các mật độ nhiễu, các số hạng nhiễu triệt tiêu hoàn toàn trong tích phân logarit, suy ra: **$\mathrm{TE}_{\widetilde{X} \to \widetilde{Y}} = \mathrm{TE}_{X \to Y}$**.
* **Bổ đề B.2 (Tính cộng tính tuyến tính):** Ghép $d$ bản sao độc lập của quá trình: **$\mathrm{TE}_{\widetilde{X} \to \widetilde{Y}} = \sum_{j=1}^d \mathrm{TE}_{X_j \to Y_j}$**.

---

### 2.7. Ưu điểm & Điểm yếu chí mạng của TENDE

* **Ưu điểm:** Độ chính xác cực cao trên số chiều lớn (70 chiều), không bị kẹt cận biến phân, có cơ chế triệt tiêu sai số đồng pha.
* **LỖ HỔNG CHÍ MẠNG:** **Chi phí tính toán khổng lồ!** Tác giả nêu rõ tại Mục D (trang 22): Để ước lượng TE cho một cặp chuỗi $T = 10.000$ mẫu, TENDE mất **khoảng 20 phút trên một GPU khủng NVIDIA A100**. Điều này khiến TENDE **hoàn toàn bất khả thi cho các thiết bị y tế đeo tay (wearables/smartwatch)** cần phân tích trực tuyến thời gian thực.

---

# PHẦN 3: BẢNG SO SÁNH ĐỐI CHIẾU TRỰC DIỆN

| Tiêu chí Đối sánh | TREET (IEEE Access 2025) | TENDE (AISTATS 2026) | Amortized MINE (Dự án PQRST) |
|---|---|---|---|
| **Trường phái Toán học** | Cận biến phân Donsker-Varadhan + Causal Attention | Score-based Diffusion SDE + Mối liên hệ KL-Score vi phân | Amortized Neural MINE + Chia sẻ tham số Masked Network |
| **Cơ chế Ước lượng** | Hiệu 2 phân kỳ KL đối với phân phối tham chiếu độc lập $\widetilde{Y}$ | Tích phân khoảng cách hàm điểm khuếch tán $\int \|S_p - S_q\|^2 dt$ | Tối ưu hóa trực tiếp hàm mất mát Donsker-Varadhan liên hợp/biên |
| **Bảo đảm Lý thuyết Cốt lõi** | Định lý Ergodic Birkhoff + Xấp xỉ phổ quát Causal Transformer | Bất đẳng thức Log-Sobolev + Triệt tiêu sai số đồng pha | 4 Mệnh đề Giải tích: Bất khả tách, Giảm phương sai $\mathrm{Cov} > 0$, Lipschitz AWGN, Ngưỡng $N^*$ |
| **Chế độ Dữ liệu Mục tiêu** | Chuỗi dài (long-memory context, $l = 30 - 99$ mẫu) | Chiều cao (high-dimensional, 70 chiều), $T = 10.000 - 50.000$ | **Chuỗi siêu ngắn y sinh ($N \le 50 - 100$ mẫu), tín hiệu phi dừng** |
| **Thời gian Suy luận / Độ trễ** | $\sim 100 - 300\text{ ms}$ (Transformer trên CPU/GPU) | **$\sim 20\text{ phút}$ / 1 cặp chuỗi (trên GPU khủng NVIDIA A100)** | **$7.0\text{ ms}$ (Small MLP) / $11.2\text{ ms}$ (Deep MLP) trên CPU thường!** |
| **Khả năng Triển khai Edge-AI** | Kém (Transformer ngốn nhiều RAM ma trận Attention) | **Hoàn toàn bất khả thi** (vòng lặp SDE hàng trăm bước) | **Xuất sắc: 369 tham số (~1.5 KB), chạy realtime trên vi điều khiển** |
| **Hành vi ở Cửa sổ Nhỏ ($N \le 50$)** | Dễ sụp đổ, sinh giá trị âm, phương sai nổ tung | Không hội tụ được (cần hàng chục nghìn mẫu huấn luyện) | **Triệt tiêu phương sai $> 90\%$, sai số thấp hơn KSG 1.6 - 2.8 lần** |
| **Thích nghi Miền (Domain Adaptation)** | Chưa có (giả định quá trình dừng nghiêm ngặt) | Chưa có | **Có giao thức UDA tối ưu Donsker-Varadhan không giám sát** |

---

# PHẦN 4: LÀM RÕ BẢN CHẤT MÔ HÌNH VQC (QUANTUM) CỦA DỰ ÁN PQRST

### 4.1. Mã hóa Tín hiệu vào Không gian Hilbert Lượng tử (Bao nhiêu Qubit?)

Đầu vào cổ điển có 4 thành phần: $\mathbf{z} = [y_t, x_{\mathrm{lag}} \cdot \mathrm{mask}, y_{\mathrm{lag}}, \mathrm{mask}] \in \mathbb{R}^4$.
* **Mô hình Hybrid Classical-Quantum (MLP + VQC):** Mã hóa thành **4 qubit**. Dùng Encoder cổ điển nén và chuẩn hóa thành 4 góc quay tiềm ẩn trong $[-\pi, \pi]$, nạp vào 4 qubit qua cổng quay $R_Y$. Không gian trạng thái lượng tử có số chiều là **$2^4 = 16$ chiều biên độ phức**.
* **Mô hình Pure VQC (Thuần lượng tử):** Mã hóa thành **6 qubit** qua kỹ thuật Data Re-uploading. Không gian trạng thái lượng tử có số chiều là **$2^6 = 64$ chiều biên độ phức**.
* **Tại sao gọi là "đưa tín hiệu vào không gian lớn"?** Vì từ không gian số thực 4 chiều cổ điển, hệ thống được ánh xạ phi tuyến vào không gian Hilbert tích tensor $\mathbb{C}^{2^n}$ (16 hoặc 64 chiều phức). Các cổng vướng víu **CNOT dạng vòng tròn** kết nối các qubit, tạo ra sự tương tác chéo phi tuyến cực mạnh mà các hàm cộng tính độc lập cổ điển không thể có được (Mệnh đề 1).

---

### 4.2. Tại sao MSE của Hybrid VQC lại cao hơn MLP? (Phân tích Toán học cho Mentor)

Bóc tách công thức giải tích:
$$\mathrm{MSE} = \mathrm{Bias}^2 + \mathrm{Variance}$$

Nhìn vào Bảng số liệu Ablation của nhóm:
* **Phương sai (Variance) của Hybrid VQC cực kỳ đẹp:** Đạt **$1.77 \times 10^{-5}$** (thậm chí **nhỏ hơn Classical MLP tới 160 lần**, vì MLP là $2.88 \times 10^{-3}$). Mạng lượng tử cho ước lượng cực kỳ ổn định!
* **Độ lệch (Bias) bị âm:** Hybrid VQC ước lượng TE $\approx 0.016$, trong khi Ground Truth là $0.181$ (độ lệch $\mathrm{Bias} = -0.165$). Do đó: $\mathrm{MSE} \approx (-0.165)^2 + 1.77 \times 10^{-5} \approx 0.0274$.
* **NGUYÊN NHÂN:** Do giới hạn tính toán của bộ mô phỏng CPU (`lightning.qubit`) phải tính ma trận trạng thái phức tạp, trong script khảo sát đa kiến trúc Hybrid VQC mới chỉ được train **7 epochs** (bị underfitting độ lệch) thay vì hàng trăm epochs như MLP cổ điển.

---

### 4.3. Định vị Khoa học Chuẩn mực trong Bài báo Q1

* **Ngôi sao thực chiến của bài báo (Primary Champion):** Là **Small MLP (369 tham số, 7.0 ms, MSE = 0.0036)**. Đây là mô hình triển khai chính cho Edge-AI và toàn bộ dữ liệu lâm sàng 40 ca Fantasia.
* **Pure VQC & Hybrid VQC:** Đóng vai trò là **Nghiên cứu thăm dò tính khả thi lượng tử (Future Outlook / Feasibility Exploration)**, chứng minh lý thuyết MINE có thể mở rộng lên không gian Hilbert và kiến trúc lai giúp tăng tốc 2.8 lần so với Pure VQC thuần túy.
