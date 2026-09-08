# Pha P — Hướng dẫn chi tiết (Nhịp 1: Khởi động)

Mục tiêu pha này (nhắc lại từ roadmap): **hạ tầng và dữ liệu sẵn sàng trước khi chạm vào mô hình**. Khi xong pha này, bạn có: môi trường chạy được, 3 bộ sinh dữ liệu tổng hợp cho ground-truth TE, 3 baseline cổ điển chạy đúng và khớp ground-truth.

Repo này dùng **venv + pip**, không dùng conda, vì máy hiện tại không có sẵn conda (đã kiểm tra). Máy có sẵn Python 3.10/3.11/3.12/3.13 và JDK 22 + JDK 25.

---

## 0. Cấu trúc thư mục đã dựng sẵn

```
PQRST/
├── docs/
│   ├── ROADMAP.md              # kế hoạch chi tiết toàn bộ 2 nhịp
│   ├── PHASE_P_GUIDE.md        # file này
│   └── references/             # ghi chú đọc paper, tài liệu API thư viện (bạn tự thêm khi đọc)
├── configs/
│   ├── synthetic/               # tham số sinh dữ liệu (YAML), tách khỏi code
│   │   ├── var_linear_gaussian.yaml
│   │   ├── var_nonlinear.yaml
│   │   └── periodic_coupling.yaml
│   └── baselines/                # tham số cấu hình từng baseline
│       ├── ksg.yaml
│       ├── symbolic_te.yaml
│       └── binning.yaml
├── src/pqrst/                   # package chính, import như `from pqrst.data.synthetic import ...`
│   ├── data/
│   │   ├── synthetic/            # <-- việc chính Pha P
│   │   └── real/                 # để trống, dùng ở Pha S
│   ├── baselines/                # <-- việc chính Pha P
│   ├── estimators/                # để trống, dùng ở Pha Q (MINE) và Pha P' (quantum)
│   ├── evaluation/                # metrics dùng chung, cần 1 phần nhỏ ở Pha P (so sánh vs ground-truth)
│   └── utils/                     # seeding, logging dùng chung toàn dự án
├── scripts/                      # entry point chạy được từ CLI, gọi vào src/pqrst
├── tests/                        # pytest, chạy nhanh, không cần dữ liệu thật
├── notebooks/                    # khám phá nhanh, không phải nơi chứa logic chính
├── data/                         # KHÔNG commit dữ liệu vào git (xem .gitignore)
│   ├── raw/ interim/ processed/ external/
└── results/                      # KHÔNG commit output lớn vào git
    ├── figures/ tables/ logs/
```

**Nguyên tắc:** logic tái sử dụng được luôn nằm trong `src/pqrst/` (import được, test được). `scripts/` chỉ là lớp mỏng gọi vào `src/pqrst/` + parse argument + in/lưu kết quả. `notebooks/` chỉ để thăm dò, không copy-paste logic dài vào notebook.

Tất cả các file `.py` trong `src/pqrst/` hiện là **stub**: có docstring mô tả spec (input/output/tham số/công thức toán), có type hint, nhưng thân hàm là `raise NotImplementedError(...)`. Đó là phần bạn sẽ code. Tôi (Claude) sẽ review lại khi bạn xong.

---

## 1. Cài môi trường

### 1.1. Chọn phiên bản Python

Dùng **Python 3.11** (không dùng 3.13 mặc dù máy có sẵn — IDTxl/JPype thường build chậm hơn để hỗ trợ Python mới nhất, 3.11 là lựa chọn an toàn nhất cho toàn bộ stack PyTorch + IDTxl + PennyLane sau này ở Nhịp 2). Máy đã có Python 3.11 tại:
```
C:\Users\Nguyen Dao Nam Hai\AppData\Local\Programs\Python\Python311\python.exe
```

### 1.2. Tạo virtual environment

Từ thư mục gốc repo (`D:\SPARC Lab\PQRST`), chạy trong PowerShell:

```powershell
& "C:\Users\Nguyen Dao Nam Hai\AppData\Local\Programs\Python\Python311\python.exe" -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Nếu PowerShell chặn script (`execution policy`), chạy 1 lần:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 1.3. Cài package

Đã chuẩn bị sẵn `requirements.txt` ở gốc repo. Cài:
```powershell
pip install -r requirements.txt
pip install -e .
```
Dòng thứ hai cài package `pqrst` (định nghĩa trong `pyproject.toml`, mã nguồn ở `src/pqrst/`) ở chế độ editable — nhờ vậy `import pqrst...` chạy được từ bất kỳ đâu (script, notebook, test) mà không cần chỉnh `sys.path`, và sửa code trong `src/` có hiệu lực ngay không cần cài lại.

Xem nội dung/giải thích từng gói trong file đó. Điểm cần chú ý:

- **PyTorch**: bản CPU là đủ cho Pha P (chưa train mạng gì lớn). Nếu máy có GPU NVIDIA và muốn cài bản CUDA từ bây giờ, dùng lệnh riêng theo [trang chọn bản của PyTorch] thay vì dòng trong requirements.txt.
- **IDTxl**: cần JVM. Máy đã có JDK 22 và 25 — nên set `JAVA_HOME` trỏ tới 1 trong 2 bản đó trước khi cài/dùng `JPype1`:
  ```powershell
  $env:JAVA_HOME = "C:\Program Files\Java\jdk-22"
  ```
  (thêm dòng này vào profile PowerShell hoặc một script `activate_env.ps1` riêng nếu muốn tự động mỗi lần mở terminal).
- **IDTxl không có trên PyPI dưới dạng ổn định lâu dài** — kiểm tra lại tên gói khi cài (`pip install idtxl`) hoặc cài từ GitHub (`pip install git+https://github.com/pwollstadt/IDTxl.git`) nếu bản PyPI lỗi thời. **Đây là việc đầu tiên cần xác minh** vì tài liệu/API của IDTxl có thể đã đổi — đọc README trên GitHub của IDTxl trước khi viết `src/pqrst/baselines/ksg.py`.
- **Symbolic transfer entropy**: chưa chắc IDTxl có sẵn estimator này (cần bạn xác minh khi đọc doc). Nếu không có, 2 lựa chọn: (a) cài thêm `pyinform` hoặc `PyIF`/gói tương đương có symbolic TE, hoặc (b) tự cài đặt nhẹ theo Staniek & Lehnertz (2008): mã hoá chuỗi thành ordinal pattern (thứ tự tương đối của m điểm liên tiếp) rồi tính TE rời rạc trên chuỗi symbol — có thể tái dùng bộ ước lượng binning/discrete đã có cho bước tính TE rời rạc cuối cùng. Ghi quyết định vào `docs/references/symbolic_te_decision.md`.

### 1.4. Kiểm tra cài đặt xong

```powershell
python -c "import torch; print(torch.__version__)"
python -c "import idtxl; print(idtxl.__file__)"
python -c "import jpype; jpype.startJVM(); print('JVM OK'); jpype.shutdownJVM()"
```

Nếu bước JVM lỗi, khả năng cao là `JAVA_HOME` chưa đúng hoặc JPype không tìm thấy `jvm.dll` — kiểm tra `JAVA_HOME\bin\server\jvm.dll` tồn tại.

---

## 2. Ba bộ sinh dữ liệu tổng hợp

File tương ứng: `src/pqrst/data/synthetic/var_linear_gaussian.py`, `var_nonlinear.py`, `periodic_coupling.py`. Config mẫu tương ứng trong `configs/synthetic/*.yaml` (đã tạo sẵn khung, bạn điền/điều chỉnh giá trị).

### 2.1. VAR tuyến tính Gaussian (`var_linear_gaussian.py`)

Mô hình chuẩn (bivariate VAR(1)):
```
X[t] = a * X[t-1] + eps_x[t]
Y[t] = b * Y[t-1] + c * X[t-1] + eps_y[t]
eps_x, eps_y ~ N(0, sigma^2), độc lập
```
`c` là cường độ ghép nối X→Y (transfer entropy thật sự chỉ tồn tại theo chiều này nếu c ≠ 0 — dùng để test cả hướng "TE đúng phải > 0" và hướng ngược lại "TE đúng phải ≈ 0").

**Vì sao chọn dạng này:** với hệ tuyến tính Gaussian, TE(X→Y) có công thức đóng (closed-form) dựa trên phương sai phần dư của hồi quy tuyến tính (tương đương Granger causality theo Geweke 1982: `TE = 0.5 * log(var(residual không có X) / var(residual có X))`). Đây là ground-truth chính xác, không phải ước lượng — lý do bộ dữ liệu này luôn nên là bộ đầu tiên dùng để test bất kỳ estimator mới nào.

**Chữ ký hàm đề xuất:**
```python
def generate_var_linear_gaussian(
    n_samples: int,
    a: float, b: float, c: float,
    noise_std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Trả về (x, y, te_ground_truth) — te_ground_truth tính theo công thức đóng ở trên."""
```

### 2.2. VAR phi tuyến (`var_nonlinear.py`)

Thêm phi tuyến vào ghép nối, ví dụ:
```
X[t] = a * X[t-1] + eps_x[t]
Y[t] = b * Y[t-1] + c * tanh(X[t-1]) + eps_y[t]      # hoặc c * X[t-1]^2, tuỳ chọn
```
**Không có công thức đóng cho TE** ở đây. Ground-truth phải lấy từ **ước lượng hội tụ ở N rất lớn** (ví dụ N = 200.000, dùng KSG hoặc binning độ phân giải cao) — coi giá trị đó là "pseudo ground-truth" và ghi rõ trong docstring/kết quả rằng đây là giá trị tham chiếu, không phải chân lý tuyệt đối. Việc này cần chạy 1 lần, lưu kết quả vào `configs/synthetic/var_nonlinear.yaml` (trường `pseudo_ground_truth_te`) để không phải tính lại mỗi lần.

### 2.3. Ghép nối tuần hoàn (`periodic_coupling.py`)

Mô phỏng ghép nối tim–não kiểu tuần hoàn (gần với coupling sinh lý hơn VAR): 2 dao động pha-耦合, ví dụ dạng Kuramoto đơn giản hoá hoặc sine-coupled:
```
phi_x[t+1] = phi_x[t] + omega_x + eps_x[t]
phi_y[t+1] = phi_y[t] + omega_y + K * sin(phi_x[t] - phi_y[t]) + eps_y[t]
x[t] = sin(phi_x[t]);  y[t] = sin(phi_y[t])
```
`K` là cường độ ghép nối pha. Ground-truth cũng theo kiểu pseudo (N lớn), tương tự mục 2.2.

### 2.4. Test cho cả 3 bộ sinh

`tests/test_synthetic_generators.py` — mỗi bộ sinh cần ít nhất:
- Test shape output đúng `n_samples`.
- Test seed cố định → kết quả tái lập được (chạy 2 lần cùng seed phải ra y hệt).
- Với VAR tuyến tính: test công thức ground-truth closed-form khớp với TE ước lượng bằng KSG ở N lớn (sanity check chéo).
- Test trường hợp `c = 0` (không ghép nối) → TE ground-truth phải ≈ 0.

---

## 3. Ba baseline cổ điển

File: `src/pqrst/baselines/base.py` (interface chung), `ksg.py`, `symbolic_te.py`, `binning.py`.

### 3.1. Interface chung (`base.py`)

```python
from abc import ABC, abstractmethod
import numpy as np

class BaseTEEstimator(ABC):
    """Interface chung cho mọi baseline, để Pha Q/R/S/T gọi thống nhất qua 1 API,
    và để Nhịp 2 cắm T_theta vào cùng vị trí mà không đổi code gọi."""

    @abstractmethod
    def estimate(self, x: np.ndarray, y: np.ndarray, **kwargs) -> float:
        """Ước lượng TE(X -> Y) từ 2 chuỗi thời gian 1D cùng độ dài."""
        raise NotImplementedError
```

`ksg.py`, `symbolic_te.py`, `binning.py` mỗi file implement 1 class kế thừa `BaseTEEstimator`, gọi vào IDTxl/JIDT (hoặc thư viện thay thế đã chọn ở mục 1.3) bên trong `estimate()`. Tham số cấu hình riêng (số neighbor `k` cho KSG, số bin cho binning, độ dài pattern `m` cho symbolic) đọc từ `configs/baselines/*.yaml`, không hard-code trong Python.

### 3.2. Vì sao bọc interface chung ngay từ đầu

Đây là quyết định kiến trúc quan trọng nhất của Pha P: **toàn bộ Pha Q→T′ sẽ gọi estimator qua interface `BaseTEEstimator.estimate(x, y)`**, không quan tâm bên trong là KSG, MLP (`T_phi`), hay mạch lượng tử (`T_theta`) sau này. Nhịp 2 chỉ thêm 1 class mới `QuantumTEEstimator(BaseTEEstimator)` — không sửa bất kỳ code gọi nào ở Pha R/S/T. Đây chính là cơ chế hiện thực hoá "hoán đổi có kiểm soát" mà roadmap mô tả.

---

## 4. Script validate baseline vs ground-truth

`scripts/validate_baselines.py` — chữ ký ý tưởng:
1. Đọc config sinh dữ liệu (mục 2) + config baseline (mục 3).
2. Sinh N_reps thực hiện (ví dụ 50 lần) của mỗi cấu hình, mỗi baseline ước lượng TE, so với ground-truth.
3. Tính bias trung bình, MSE, in bảng ra console + lưu `results/tables/phase_p_baseline_validation.csv`.
4. (tuỳ chọn) vẽ boxplot sai số mỗi baseline → `results/figures/phase_p_baseline_validation.png`.

---

## 5. Checklist thoát Pha P

Đánh dấu khi xong — đây là tiêu chí bàn giao cho tôi review:

- [ ] `pip install -r requirements.txt` chạy xong không lỗi trong `.venv`.
- [ ] `python -c "import idtxl"` và JVM start/shutdown không lỗi.
- [ ] 3 bộ sinh dữ liệu tổng hợp implement xong, `pytest tests/test_synthetic_generators.py` pass.
- [ ] 3 baseline implement xong, kế thừa đúng `BaseTEEstimator`.
- [ ] `python scripts/validate_baselines.py` chạy xong, bảng sai số hợp lý (bias nhỏ, không NaN/Inf).
- [ ] Quyết định về symbolic TE (dùng thư viện nào / tự code) đã ghi vào `docs/references/symbolic_te_decision.md`.
- [ ] Không có file dữ liệu lớn hoặc bí mật bị commit vào git (kiểm tra `git status` trước khi commit).

Khi checklist trên xong, báo tôi — tôi sẽ review code (đúng interface, đúng công thức, test coverage, không có bug logic) trước khi sang Pha Q.
