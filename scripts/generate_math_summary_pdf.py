"""Script generate pristine, publication-grade PDF report summarizing the mathematics of TREET and TENDE.

Ensures:
- 100% supported Arial glyphs (NO tofu/square boxes).
- Removes the intro summary box.
- Removes Section 5 (Bài học kế thừa).
- Extremely detailed, rigorous mathematical exposition of TREET and TENDE.
- Comparison table and comprehensive VQC analysis.
"""

from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

BASE = Path(__file__).resolve().parent.parent
PDF_PATH = BASE / "docs" / "TOM_TAT_TOAN_TREET_TENDE.pdf"

# Register Unicode Vietnamese TrueType Fonts
FONT_REGULAR = "C:/Windows/Fonts/arial.ttf"
FONT_BOLD = "C:/Windows/Fonts/arialbd.ttf"
FONT_ITALIC = "C:/Windows/Fonts/ariali.ttf"
FONT_BOLDITALIC = "C:/Windows/Fonts/arialbi.ttf"

pdfmetrics.registerFont(TTFont("Arial", FONT_REGULAR))
pdfmetrics.registerFont(TTFont("Arial-Bold", FONT_BOLD))
pdfmetrics.registerFont(TTFont("Arial-Italic", FONT_ITALIC))
pdfmetrics.registerFont(TTFont("Arial-BoldItalic", FONT_BOLDITALIC))

# Load font object for character glyph validation
arial_font_obj = pdfmetrics.getFont("Arial")


def sanitize_text(text: str) -> str:
    """Replaces unsupported mathematical Unicode glyphs with 100% supported Arial characters."""
    replacements = {
        # Blackboard bold & script letters
        "𝕏": "<b>X</b>",
        "𝕐": "<b>Y</b>",
        "𝕑": "<b>Z</b>",
        "𝒟": "<i>D</i>",
        "ℒ": "<i>L</i>",
        "𝔼": "<b>E</b>",
        "ℝ": "<b>R</b>",
        "ℂ": "<b>C</b>",
        "𝒩": "<i>N</i>",
        "𝒪": "<i>O</i>",
        "𝒢": "<i>G</i>",
        "ℱ": "<i>F</i>",
        "ℋ": "<i>H</i>",
        "𝒫": "<i>P</i>",
        "𝒬": "<i>Q</i>",
        # Math operators & symbols not in standard Arial
        "∥": " || ",
        "∇": "grad ",
        "∈": " ∈ ",  # check below, replaced with 'thuộc' or in
        "∼": "~",
        "≃": " ≈ ",
        "⋆": "*",
        "⊙": " * ",
        "⊤": "<sup>T</sup>",
        "̃": "~",
        "̂": "^",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    # Specific replacements for characters missing in Arial
    text = text.replace("∈", " thuộc ")
    text = text.replace("∉", " không thuộc ")
    text = text.replace("⊂", " tập con của ")
    text = text.replace("∪", " hợp ")
    text = text.replace("∩", " giao ")
    text = text.replace("⊗", " (x) ")
    text = text.replace("⟨", "&lt;")
    text = text.replace("⟩", "&gt;")

    return text


def validate_text(text: str):
    """Asserts that all characters in the text (outside HTML tags) exist in Arial."""
    plain = re.sub(r"<[^>]+>", "", text)
    missing = [c for c in plain if ord(c) not in arial_font_obj.face.charWidths]
    if missing:
        unique_missing = list(set(missing))
        missing_hex = [f"{c} (U+{ord(c):04X})" for c in unique_missing]
        raise ValueError(f"Found unsupported glyphs in text: {', '.join(missing_hex)}")


def P(text: str, style: ParagraphStyle) -> Paragraph:
    """Helper to sanitize, validate, and return a ReportLab Paragraph."""
    clean_text = sanitize_text(text)
    validate_text(clean_text)
    return Paragraph(clean_text, style)


def make_header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Arial", 8)
    canvas.setFillColor(colors.HexColor("#555555"))

    if doc.page > 1:
        canvas.drawString(
            16 * mm,
            286 * mm,
            "TỔNG HỢP TOÁN HỌC: TREET & TENDE — ĐỐI SÁNH VỚI AMORTIZED MINE (PQRST)",
        )
        canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
        canvas.setLineWidth(0.6)
        canvas.line(16 * mm, 283 * mm, 194 * mm, 283 * mm)

    page_text = f"Trang {doc.page}"
    canvas.drawRightString(194 * mm, 10 * mm, page_text)
    canvas.drawString(
        16 * mm,
        10 * mm,
        "SPARC Laboratory — ĐHBK Hà Nội | Phân tích Lý thuyết Thông tin & Ước lượng Transfer Entropy",
    )
    canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
    canvas.setLineWidth(0.6)
    canvas.line(16 * mm, 14 * mm, 194 * mm, 14 * mm)

    canvas.restoreState()


def build_math_summary_pdf():
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Arial-Bold",
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#0A2540"),
        alignment=1,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Arial-Italic",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#334155"),
        alignment=1,
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Arial-Bold",
        fontSize=12,
        leading=15.5,
        textColor=colors.HexColor("#0F2B5C"),
        spaceBefore=7,
        spaceAfter=2.5,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Normal"],
        fontName="Arial-Bold",
        fontSize=10,
        leading=13.5,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=5,
        spaceAfter=2,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Arial",
        fontSize=8.6,
        leading=12.2,
        textColor=colors.HexColor("#1E293B"),
        alignment=4,
    )

    body_bold = ParagraphStyle(
        "Body_Bold",
        parent=body_style,
        fontName="Arial-Bold",
    )

    formula_style = ParagraphStyle(
        "Formula_Custom",
        parent=styles["Normal"],
        fontName="Arial-Italic",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#0F172A"),
        leftIndent=8,
        rightIndent=8,
        spaceBefore=2,
        spaceAfter=2,
    )

    story = []

    # =========================================================================
    # BANNER TIÊU ĐỀ
    # =========================================================================
    story.append(P("TỔNG HỢP VÀ PHÂN TÍCH TOÁN HỌC CHUYÊN SÂU", title_style))
    story.append(Spacer(1, 1.5 * mm))
    story.append(
        P(
            "Phân tích Chi tiết Giải tích & Lý thuyết Thông tin: <b>TREET</b> (Transformers) vs. <b>TENDE</b> (Score-based Diffusion)<br/>"
            "và Luận cứ Khoa học của Kiến trúc <b>Amortized MINE / Quantum Hybrid (Dự án PQRST)</b>",
            subtitle_style,
        )
    )
    story.append(Spacer(1, 2.5 * mm))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0A2540"), spaceAfter=3.5))

    # =========================================================================
    # PHẦN 1: BÀI 1 — TREET
    # =========================================================================
    story.append(P("1. Bài báo 1 — TREET: Transfer Entropy Estimation via Transformers", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#0F2B5C"), spaceAfter=2.5))
    story.append(P(
        "<i>Tác giả: Omer Luxembourg, Dor Tsur, Haim Permuter — Ben-Gurion University of the Negev, Israel (IEEE Access 2025 / arXiv:2402.06919).</i>",
        body_style,
    ))
    story.append(Spacer(1, 1 * mm))

    story.append(P("1.1. Khung lý thuyết & Định nghĩa Transfer Entropy mở rộng", h2_style))
    story.append(P(
        "Xét hai quá trình ngẫu nhiên dừng <b>X</b> = (X<sub>t</sub>) và <b>Y</b> = (Y<sub>t</sub>) trên không gian xác suất (Ω, ℱ, P). "
        "Với các bậc nhớ quá khứ k, l &lt; ∞, TREET đưa ra định nghĩa Transfer Entropy mở rộng như sau:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>TE<sub>X → Y</sub>(t; k, l) := I(X<sub>t−l</sub><sup>t</sup> ; Y<sub>t</sub> | Y<sub>t−k</sub><sup>t−1</sup>)</b><br/>"
        "Khi quá trình đồng thời dừng (jointly stationary), chỉ số thời gian t được loại bỏ và viết gọn là TE<sub>X → Y</sub>(k, l):<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>TE<sub>X → Y</sub>(k, l) = I(X<sup>l</sup> ; Y<sub>l</sub> | Y<sub>l−k</sub><sup>l−1</sup>)</b><br/>"
        "<b>Điểm toán học mới so với Schreiber (2000):</b> Định nghĩa cổ điển của Schreiber chỉ lấy quá khứ X<sub>t−l</sub><sup>t−1</sup> "
        "(tức I(X<sup>l−1</sup> ; Y<sub>l</sub> | Y<sup>l−1</sup>)). TREET chủ động bổ sung mẫu hiện tại X<sub>t</sub> vào chuỗi nguồn. "
        "Ý nghĩa toán học: Khi hai quá trình là độc lập cùng phân phối (i.i.d.), công thức của TREET lập tức thu gọn về Mutual Information chuẩn: "
        "TE<sub>X → Y</sub>(0, 0) = I(X; Y). Khi k = l, bài báo ký hiệu vắn tắt là TE<sub>X → Y</sub>(l).",
        body_style,
    ))

    story.append(P("1.2. Mối liên hệ với Directed Information (DI) & Điều kiện xích Markov", h2_style))
    story.append(P(
        "Directed Information (Massey 1990, Kramer 1998) đo lường dòng thông tin nhân quả từ chuỗi X<sup>n</sup> tới Y<sup>n</sup>:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;I(X<sup>n</sup> → Y<sup>n</sup>) := ∑<sub>i=0</sub><sup>n−1</sup> I(X<sup>i</sup> ; Y<sub>i</sub> | Y<sup>i−1</sup>) "
        "= ∑<sub>i=0</sub><sup>n−1</sup> TE<sub>X → Y</sub>(i)<br/>"
        "Tốc độ thông tin định hướng (DI rate) được định nghĩa qua giới hạn:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;I(<b>X</b> → <b>Y</b>) := lim<sub>n→∞</sub> (1/n) I(X<sup>n</sup> → Y<sup>n</sup>) = lim<sub>n→∞</sub> I(X<sup>n</sup> ; Y<sub>n</sub> | Y<sup>n−1</sup>)<br/>"
        "Giới hạn này luôn tồn tại do TE<sub>X → Y</sub>(l) = h(Y<sub>l</sub> | Y<sup>l−1</sup>) − h(Y<sub>l</sub> | X<sup>l</sup>, Y<sup>l−1</sup>), "
        "và các entropy có điều kiện là không âm, đơn điệu giảm khi độ dài ngữ cảnh l tăng (điều kiện hóa làm giảm entropy).<br/>"
        "<b>Bổ đề 1 (Tính tương đương dưới điều kiện Markov):</b> Giả sử hai quá trình dừng thỏa mãn cấu trúc Markov:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;Y<sub>l</sub> − Y<sup>l−1</sup> − Y<sub>−∞</sub><sup>−1</sup> &nbsp;&nbsp;&nbsp;&nbsp;và&nbsp;&nbsp;&nbsp;&nbsp; "
        "Y<sub>l</sub> − (X<sup>l</sup>, Y<sup>l−1</sup>) − (X<sub>−∞</sub><sup>−1</sup>, Y<sub>−∞</sub><sup>−1</sup>)<br/>"
        "Khi đó, với mọi m ≥ l, ta có đẳng thức giải tích đóng: <b>TE<sub>X → Y</sub>(m) = TE<sub>X → Y</sub>(l) = I(<b>X</b> → <b>Y</b>)</b>. "
        "Bổ đề này chứng minh rằng việc ước lượng TE với độ dài bộ nhớ hữu hạn l chính là nghiệm chính xác của DI rate vô hạn.",
        body_style,
    ))

    story.append(P("1.3. Bổ đề 2: Phân rã TE qua Hiệu hai Phân kỳ KL với Phân phối Tham chiếu", h2_style))
    story.append(P(
        "Để giải quyết trở ngại tính toán phân phối có điều kiện phức tạp, TREET đưa ra một <b>phân phối tham chiếu liên tục tùy ý Ỹ</b> "
        "(reference measure) trên không gian quan sát <b>Y</b> (thường chọn phân phối đều Uniform):<br/>"
        "<b>Bổ đề 2 (Biểu diễn TE qua hiệu 2 phân kỳ KL):</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>TE<sub>X → Y</sub>(l) = D<sub>Y<sup>l</sup> | Y<sup>l−1</sup> X<sup>l</sup> || Ỹ<sup>l</sup></sub> − D<sub>Y<sup>l</sup> | Y<sup>l−1</sup> || Ỹ<sup>l</sup></sub></b><br/>"
        "Trong đó hai số hạng phân kỳ KL có điều kiện được xác định chính xác theo tích phân Lebesgue:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;• D<sub>Y<sup>l</sup> | Y<sup>l−1</sup> || Ỹ<sup>l</sup></sub> := D<sub>KL</sub>( P<sub>Y<sub>l</sub> | Y<sup>l−1</sup></sub> || P<sub>Ỹ</sub> | P<sub>Y<sup>l−1</sup></sub> ) "
        "= ∫ log [ P(y<sub>l</sub> | y<sup>l−1</sup>) / P<sub>Ỹ</sub>(y<sub>l</sub>) ] P(x<sup>l</sup>, y<sup>l</sup>) d(x<sup>l</sup> y<sup>l</sup>)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;• D<sub>Y<sup>l</sup> | Y<sup>l−1</sup> X<sup>l</sup> || Ỹ<sup>l</sup></sub> := D<sub>KL</sub>( P<sub>Y<sub>l</sub> | Y<sup>l−1</sup> X<sup>l</sup></sub> || P<sub>Ỹ</sub> | P<sub>Y<sup>l−1</sup> X<sup>l</sup></sub> ) "
        "= ∫ log [ P(y<sub>l</sub> | x<sup>l</sup>, y<sup>l−1</sup>) / P<sub>Ỹ</sub>(y<sub>l</sub>) ] P(x<sup>l</sup>, y<sup>l</sup>) d(x<sup>l</sup> y<sup>l</sup>)<br/>"
        "<b>Chứng minh giải tích (Proof of Lemma 2):</b> Trừ trực tiếp hai biểu thức tích phân:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;D<sub>Y<sup>l</sup> | Y<sup>l−1</sup> X<sup>l</sup> || Ỹ<sup>l</sup></sub> − D<sub>Y<sup>l</sup> | Y<sup>l−1</sup> || Ỹ<sup>l</sup></sub> "
        "= ∫ log [ P(y<sub>l</sub> | x<sup>l</sup>, y<sup>l−1</sup>) / P(y<sub>l</sub> | y<sup>l−1</sup>) ] P(x<sup>l</sup>, y<sup>l</sup>) d(x<sup>l</sup> y<sup>l</sup>)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= h(Y<sub>l</sub> | Y<sup>l−1</sup>) − h(Y<sub>l</sub> | X<sup>l</sup>, Y<sup>l−1</sup>) = TE<sub>X → Y</sub>(l).<br/>"
        "<i>Nhận xét:</i> Mật độ tham chiếu P<sub>Ỹ</sub>(y<sub>l</sub>) ở mẫu số bị <b>triệt tiêu hoàn toàn</b>! "
        "Ỹ đóng vai trò như một 'điện thế mốc', cho phép chuyển bài toán tính Transfer Entropy có điều kiện phức tạp thành hai bài toán tối ưu KL độc lập.",
        body_style,
    ))

    story.append(P("1.4. Cận biến phân Donsker-Varadhan (DV) & Nghiệm Mật độ Tối ưu", h2_style))
    story.append(P(
        "Mỗi số hạng KL divergence được biểu diễn thông qua công thức biến phân Donsker-Varadhan (1983):<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;D<sub>KL</sub>(P || Q) = sup<sub>f : Ω → <b>R</b></sub> { <b>E</b><sub>P</sub>[f] − log <b>E</b><sub>Q</sub>[e<sup>f</sup>] }<br/>"
        "Áp dụng vào TREET với hai hàm nơ-ron g<sub>y</sub> và g<sub>xy</sub> thuộc họ Causal Transformer:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;D<sub>Y<sup>l</sup> | Y<sup>l−1</sup> || Ỹ<sup>l</sup></sub> = sup<sub>g<sub>y</sub></sub> { <b>E</b>[g<sub>y</sub>(Y<sup>l</sup>)] − log <b>E</b>[exp(g<sub>y</sub>(Ỹ<sub>l</sub>, Y<sup>l−1</sup>))] }<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;D<sub>Y<sup>l</sup> | Y<sup>l−1</sup> X<sup>l</sup> || Ỹ<sup>l</sup></sub> = sup<sub>g<sub>xy</sub></sub> { <b>E</b>[g<sub>xy</sub>(Y<sup>l</sup>, X<sup>l</sup>)] − log <b>E</b>[exp(g<sub>xy</sub>(Ỹ<sub>l</sub>, Y<sup>l−1</sup>, X<sup>l</sup>))] }<br/>"
        "Nghiệm tối ưu giải tích f<sup>*</sup> đạt được tại tỉ số mật độ Radon-Nikodym thực tế:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;f<sub>y,l</sub><sup>*</sup>(Y<sup>l</sup>) = log [ dP<sub>Y<sup>l</sup></sub> / d(P<sub>Y<sup>l−1</sup></sub> (x) P<sub>Ỹ</sub>) ] = log P(y<sub>l</sub> | y<sup>l−1</sup>) − log P<sub>Ỹ</sub>(y<sub>l</sub>).<br/>"
        "Trên tập dữ liệu n mẫu D<sub>n</sub>, kỳ vọng toán học được xấp xỉ bằng trung bình mẫu qua mini-batch gradient ascent.",
        body_style,
    ))

    story.append(P("1.5. Cơ chế Fixed-Past Causal Attention (FPCA) & Mặt nạ Toeplitz", h2_style))
    story.append(P(
        "Để đảm bảo mạng Transformer tuân thủ tính nhân quả (không nhìn trước tương lai) và khóa độ dài ngữ cảnh lịch sử đúng l mẫu, "
        "TREET đề xuất mặt nạ băng Toeplitz (Toeplitz-like banded mask M′ thuộc <b>R</b><sup>L × L</sup>):<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;M′[i, j] = 1 nếu 0 ≤ i − j &lt; l, và M′[i, j] = −∞ nếu ngược lại.<br/>"
        "Công thức Attention của TREET:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>FPCA(X) = softmax( (QK<sup>T</sup> / √d<sub>m</sub>) * M′ ) V</b> &nbsp;&nbsp;&nbsp;&nbsp;(với * là tích Hadamard phần tử).<br/>"
        "<b>Modified FPCA cho số hạng tham chiếu:</b> Để tính kỳ vọng g<sub>y</sub>(Ỹ<sub>l</sub>, Y<sup>l−1</sup>), TREET giữ nguyên toàn bộ "
        "Key và Value của quá khứ Y<sup>l−1</sup>, chỉ thay thế Query, Key, Value tại thời điểm hiện tại bằng mẫu tham chiếu Ỹ<sub>t</sub>. "
        "Độ phức tạp tính toán giảm từ 𝒪(L<sup>2</sup>) xuống còn <b>𝒪(L · l · d<sub>o</sub>)</b>.",
        body_style,
    ))

    story.append(P("1.6. Định lý 3 (Tính Vững Mạnh - Strong Consistency): Chứng minh Giải tích 3 bước", h2_style))
    story.append(P(
        "<b>Định lý 3:</b> Cho <b>X</b>, <b>Y</b> là các quá trình ngẫu nhiên dừng, ergodic. Khi số mẫu n → ∞, ước lượng TÊ<sub>X → Y</sub>(D<sub>n</sub>; l) "
        "hội tụ hầu chắc chắn (P-a.s.) về giá trị Transfer Entropy lý thuyết: |TÊ(D<sub>n</sub>; l) − TE(l)| ≤ ε.<br/>"
        "<b>Chi tiết chứng minh (Proof of Theorem 3):</b><br/>"
        "• <b>Bước 1 (Biểu diễn biến phân):</b> TE được phân rã thành hiệu 2 thế năng DV theo Bổ đề 2.<br/>"
        "• <b>Bước 2 (Công thái học - Generalized Birkhoff Ergodic Theorem):</b> Theo Định lý Birkhoff (1957), với biến đổi bảo toàn độ đo T:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;(1/n) ∑<sub>i=1</sub><sup>n</sup> g<sub>i</sub>(T<sup>i</sup>ω) → <b>E</b>[g] &nbsp;&nbsp;(P-a.s.). "
        "Do quá trình dừng và ergodic, trung bình mẫu của hàm tối ưu f<sup>*</sup> hội tụ về kỳ vọng giải tích: "
        "|<b>E</b>[f<sup>*</sup>] − (1/n) ∑ f<sup>*</sup>| &lt; ε/8.<br/>"
        "• <b>Bước 3 (Xấp xỉ phổ quát & Tính Lipschitz của hàm mũ):</b> Hàm f<sup>*</sup> bị chặn đều bởi M. "
        "Hàm mũ e<sup>x</sup> trên khoảng (−∞, M] có hằng số Lipschitz là e<sup>M</sup>:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;(1/n) ∑ |exp(f<sup>*</sup>) − exp(g)| ≤ e<sup>M</sup> · (1/n) ∑ |f<sup>*</sup> − g|.<br/>"
        "Theo Định lý Xấp xỉ Phổ quát của Causal Transformer (Yun et al. 2019), tồn tại mạng g sao cho sai số xấp xỉ &lt; e<sup>−M</sup> · (ε/4). "
        "Kết hợp lại ta có sai số toàn phần bị chặn nghiêm ngặt: |D̂(D<sub>n</sub>) − D| ≤ ε/2. Hai số hạng trừ nhau suy ra điều phải chứng minh.",
        body_style,
    ))

    story.append(P("1.7. Ưu điểm & Điểm yếu chí mạng của TREET", h2_style))
    story.append(P(
        "• <b>Ưu điểm:</b> Nền tảng lý thuyết ergodic vững chắc, nắm bắt rất tốt ngữ cảnh dài (l = 99), song song hóa tốt trên chuỗi dài.<br/>"
        "• <b>LỖ HỔNG CHÍ MẠNG:</b> Khi áp dụng vào <b>cửa sổ siêu ngắn y sinh (N ≤ 50–100)</b> hoặc số chiều cao, "
        "Transformer không đủ mẫu để học ma trận chú ý, dẫn đến <b>Attention bị nhiễu loạn hoàn toàn, TE bị ước lượng âm và phương sai bùng nổ</b> "
        "(kết quả đối chứng ở bài TENDE: TREET bị âm tới −3.02 ± 2.59 trong khi ground-truth là 0.00).",
        body_style,
    ))

    story.append(Spacer(1, 3 * mm))
    story.append(PageBreak())

    # =========================================================================
    # PHẦN 2: BÀI 2 — TENDE
    # =========================================================================
    story.append(P("2. Bài báo 2 — TENDE: Transfer Entropy Neural Diffusion Estimation", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#0F2B5C"), spaceAfter=2.5))
    story.append(P(
        "<i>Tác giả: Simon Pedro Galeano Muñoz, Mustapha Bounoua, Giulio Franzese, Pietro Michiardi, Maurizio Filippone — KAUST & EURECOM (AISTATS 2026 / arXiv:2510.14096).</i>",
        body_style,
    ))
    story.append(Spacer(1, 1 * mm))

    story.append(P("2.1. Đặt vấn đề: Thoát khỏi 'Cơn ác mộng' của Cận Biến phân McAllester-Stratos", h2_style))
    story.append(P(
        "TENDE mở đầu bằng việc chỉ ra hạn chế nền tảng của các mô hình cận biến phân (như MINE hay TREET) theo định lý McAllester & Stratos (AISTATS 2020):<br/>"
        "<i>'Mọi bộ ước lượng dựa trên cận dưới biến phân phân kỳ (như Donsker-Varadhan) đều đòi hỏi cỡ mẫu tăng theo cấp số nhân "
        "n = 𝒪(e<sup>I</sup>) để hội tụ khi giá trị thông tin I lớn.'</i><br/>"
        "Để khắc phục triệt để, TENDE từ bỏ hoàn toàn cận biến phân DV và thiết lập cầu nối giữa <b>Mô hình Khuếch tán Hàm điểm (Score-based Diffusion)</b> "
        "và <b>Phân kỳ Kullback-Leibler</b>.",
        body_style,
    ))

    story.append(P("2.2. Phương trình Vi phân Ngẫu nhiên (VP-SDE) & Hàm Điểm (Score Function)", h2_style))
    story.append(P(
        "TENDE sử dụng phương trình vi phân ngẫu nhiên bảo toàn phương sai (Variance Preserving SDE - Song et al. 2020):<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>dX<sub>t</sub> = f<sub>t</sub> X<sub>t</sub> dt + g<sub>t</sub> dW<sub>t</sub>, &nbsp;&nbsp;&nbsp;&nbsp; X<sub>0</sub> ∼ p<sub>X</sub>, &nbsp;&nbsp; t thuộc [0, T]</b><br/>"
        "trong đó f<sub>t</sub> = −(1/2)β(t), g<sub>t</sub> = √β(t), với lịch trình nhiễu tuyến tính β(t) = β<sub>min</sub> + (β<sub>max</sub> − β<sub>min</sub>)t. "
        "Nghiệm chuyển tiếp p<sub>0t</sub>(x<sub>t</sub> | x<sub>0</sub>) có dạng phân phối Gauss giải tích đóng:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;p<sub>0t</sub>(x<sub>t</sub> | x<sub>0</sub>) = <i>N</i>(x<sub>t</sub> ; k<sub>t</sub> x<sub>0</sub>, σ<sub>t</sub><sup>2</sup> I), "
        "&nbsp;&nbsp;&nbsp;&nbsp;với k<sub>t</sub> = exp(∫<sub>0</sub><sup>t</sup> f<sub>s</sub> ds), &nbsp; σ<sub>t</sub><sup>2</sup> = 1 − k<sub>t</sub><sup>2</sup>.<br/>"
        "<b>Định nghĩa Hàm Điểm (Score Function):</b> S<sub>p</sub>(x) := grad<sub>x</sub> log p(x). Mạng nơ-ron học hàm điểm qua Denoising Score Matching:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;ℒ(θ) = ∫<sub>0</sub><sup>T</sup> g<sub>t</sub><sup>2</sup> <b>E</b><sub>x, x~</sub> [ || S(x~; θ) − grad<sub>x~</sub> log p<sub>0t</sub>(x~ | x) ||<sup>2</sup> ] dt.",
        body_style,
    ))

    story.append(P("2.3. Cầu nối Toán học Đột phá: Phân kỳ KL qua Tích phân Sai phân Hàm Điểm", h2_style))
    story.append(P(
        "<b>Định lý Cốt lõi (Bounoua 2024, Franzese 2023):</b> Phân kỳ KL giữa hai phân phối liên tục bất kỳ p và q được biểu diễn chính xác qua sai phân hàm điểm:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>D<sub>KL</sub>(p || q) = ∫<sub>0</sub><sup>T</sup> (g<sub>t</sub><sup>2</sup> / 2) <b>E</b><sub>x ∼ p<sub>t</sub></sub> [ || S<sub>p<sub>t</sub></sub>(x) − S<sub>q<sub>t</sub></sub>(x) ||<sup>2</sup> ] dt + D<sub>KL</sub>(p<sub>T</sub> || q<sub>T</sub>)</b><br/>"
        "Khi thời gian khuếch tán T đủ lớn, theo bất đẳng thức Logarithmic Sobolev (Collet & Malrieu 2008), cả hai phân phối đều co về phân phối dừng chuẩn <i>N</i>(0, I), "
        "do đó số hạng đầu cuối D<sub>KL</sub>(p<sub>T</sub> || q<sub>T</sub>) → 0.<br/>"
        "Bộ ước lượng TENDE tính phân kỳ KL qua tích phân Monte Carlo:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>e(p, q) = ∫<sub>0</sub><sup>T</sup> (g<sub>t</sub><sup>2</sup> / 2) <b>E</b> [ || S<sub>p<sub>t</sub></sub>(x; θ<sub>1</sub><sup>*</sup>) − S<sub>q<sub>t</sub></sub>(x; θ<sub>2</sub><sup>*</sup>) ||<sup>2</sup> ] dt ≈ D<sub>KL</sub>(p || q)</b>.",
        body_style,
    ))

    story.append(P("2.4. Phân tích Sai số d & Cơ chế Tự Triệt tiêu Sai số Đồng pha (Common-Mode Cancellation)", h2_style))
    story.append(P(
        "TENDE chứng minh công thức phân rã sai số giải tích (Phụ lục A.3):<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>e(p, q) − D<sub>KL</sub>(p || q) = d − D<sub>KL</sub>(p<sub>T</sub> || q<sub>T</sub>)</b><br/>"
        "trong đó số hạng sai số d có dạng:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;d = ∫<sub>0</sub><sup>T</sup> (g<sub>t</sub><sup>2</sup> / 2) <b>E</b><sub>x ∼ p<sub>t</sub></sub> [ || ε<sub>t</sub><sup>p</sup>(x) − ε<sub>t</sub><sup>q</sup>(x) ||<sup>2</sup> + 2 &lt; S<sub>p<sub>t</sub></sub>(x) − S<sub>q<sub>t</sub></sub>(x), ε<sub>t</sub><sup>p</sup>(x) − ε<sub>t</sub><sup>q</sup>(x) &gt; ] dt<br/>"
        "với ε<sub>t</sub><sup>p</sup>(x) = S(x; θ<sup>*</sup>) − S<sub>p<sub>t</sub></sub>(x) là sai số xấp xỉ của mạng nơ-ron.<br/>"
        "<b>HAI ƯU THẾ TOÁN HỌC ĐẮT GIÁ:</b><br/>"
        "1. <b>d không xác định dấu (có thể âm hoặc dương):</b> e(p, q) <b>không phải là cận trên cũng không phải cận dưới</b> của KL divergence. "
        "Do đó, TENDE <b>hoàn toàn thoát khỏi định lý bất khả McAllester-Stratos</b>!<br/>"
        "2. <b>Cơ chế tự triệt tiêu sai số đồng pha:</b> Nếu mạng nơ-ron mắc cùng một sai số hệ thống trên cả hai phân phối (ε<sub>t</sub><sup>p</sup>(x) = ε<sub>t</sub><sup>q</sup>(x)), "
        "thì hiệu sai số triệt tiêu và <b>d ≡ 0 tuyệt đối</b>, bất kể sai số tuyệt đối của từng hàm điểm lớn đến đâu!",
        body_style,
    ))

    story.append(P("2.5. Ba Dạng Công thức Tính TE & Mạng Amortized Score Network", h2_style))
    story.append(P(
        "Chuyển Transfer Entropy về Thông tin Tương hỗ Có điều kiện CMI: TE<sub>X → Y</sub>(k, l) = I(Y<sub>t</sub> ; X<sub>t−k</sub> | Y<sub>t−l</sub>) với X = X<sub>t−k</sub>, Y = Y<sub>t</sub>, Z = Y<sub>t−l</sub>:<br/>"
        "• <b>Dạng 1 (Kỳ vọng KL - Khuyên dùng vì luôn ≥ 0):</b> I(X; Y | Z) = <b>E</b><sub>y,z</sub> [ D<sub>KL</sub>(p<sub>X|y,z</sub> || p<sub>X|z</sub>) ] ≈ <b>E</b><sub>y,z</sub> [ e(p<sub>X|y,z</sub>, p<sub>X|z</sub>) ]<br/>"
        "• <b>Dạng 2 (Hiệu Entropy qua biến Gauss phụ trợ φ<sub>σ</sub>):</b> I(X; Y | Z) = H(X | Z) − H(X | Y, Z) ≈ <b>E</b><sub>y,z</sub>[e(p<sub>X|y,z</sub>, φ<sub>σ</sub>)] − <b>E</b><sub>z</sub>[e(p<sub>X|z</sub>, φ<sub>σ</sub>)]<br/>"
        "• <b>Dạng 3 (Hiệu Mutual Information):</b> I(X; Y | Z) = I(X; [Y, Z]) − I(X; Z) ≈ <b>E</b><sub>y,z</sub>[e(p<sub>X|y,z</sub>, p<sub>X</sub>)] − <b>E</b><sub>z</sub>[e(p<sub>X|z</sub>, p<sub>X</sub>)]<br/>"
        "<b>Mạng Amortized:</b> Dùng duy nhất một mạng U-Net nhận vector điều kiện [e<sub>1</sub>, e<sub>2</sub>, e<sub>3</sub>] thuộc {−1, 0, 1}<sup>3</sup> "
        "(1: biến tính score, 0: biến làm điều kiện, −1: biến tích phân biên) để sinh đồng thời cả 3 hàm điểm, tiết kiệm chi phí tham số.",
        body_style,
    ))

    story.append(P("2.6. Bổ đề B.1 (Bất biến chiều nhiễu) & Bổ đề B.2 (Cộng tính tuyến tính)", h2_style))
    story.append(P(
        "• <b>Bổ đề B.1 (Bất biến trước chiều nhiễu độc lập):</b> Ghép d chiều nhiễu trắng độc lập ε<sub>t</sub> vào chuỗi: X̃<sub>t</sub> = [X<sub>t</sub>, ε<sub>t</sub><sup>x</sup>], Ỹ<sub>t</sub> = [Y<sub>t</sub>, ε<sub>t</sub><sup>y</sup>]. "
        "Chứng minh giải tích cho thấy p(Ỹ<sub>t</sub>, X̃<sub>t−k</sub> | Ỹ<sub>t−l</sub>) phân rã tích các mật độ nhiễu, các số hạng nhiễu triệt tiêu hoàn toàn trong tích phân logarit, suy ra: <b>TE<sub>X̃ → Ỹ</sub> = TE<sub>X → Y</sub></b>.<br/>"
        "• <b>Bổ đề B.2 (Tính cộng tính tuyến tính):</b> Ghép d bản sao độc lập của quá trình: <b>TE<sub>X̃ → Ỹ</sub> = ∑<sub>j=1</sub><sup>d</sup> TE<sub>X<sub>j</sub> → Y<sub>j</sub></sub></b>.",
        body_style,
    ))

    story.append(P("2.7. Ưu điểm & Điểm yếu chí mạng của TENDE", h2_style))
    story.append(P(
        "• <b>Ưu điểm:</b> Độ chính xác cực cao trên số chiều lớn (70 chiều), không bị kẹt cận biến phân, có cơ chế triệt tiêu sai số đồng pha.<br/>"
        "• <b>LỖ HỔNG CHÍ MẠNG:</b> <b>Chi phí tính toán khổng lồ!</b> Tác giả nêu rõ tại Mục D (trang 22): "
        "Để ước lượng TE cho một cặp chuỗi T = 10.000 mẫu, TENDE mất <b>khoảng 20 phút trên một GPU khủng NVIDIA A100</b>. "
        "Điều này khiến TENDE <b>hoàn toàn bất khả thi cho các thiết bị y tế đeo tay (wearables/smartwatch)</b> cần phân tích trực tuyến thời gian thực.",
        body_style,
    ))

    story.append(Spacer(1, 3 * mm))
    story.append(PageBreak())

    # =========================================================================
    # PHẦN 3: BẢNG SO SÁNH ĐỐI CHIẾU TRỰC DIỆN
    # =========================================================================
    story.append(P("3. Bảng Đối Sánh Trực Diện: TREET vs. TENDE vs. Amortized MINE (PQRST)", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#0F2B5C"), spaceAfter=3))

    comp_data = [
        [
            P("<b>Tiêu chí Đối sánh</b>", body_bold),
            P("<b>TREET (IEEE Access 2025)</b>", body_bold),
            P("<b>TENDE (AISTATS 2026)</b>", body_bold),
            P("<b>Amortized MINE (Dự án PQRST)</b>", body_bold),
        ],
        [
            P("<b>Trường phái Toán học</b>", body_style),
            P("Cận biến phân Donsker-Varadhan + Causal Attention", body_style),
            P("Score-based Diffusion SDE + Mối liên hệ KL-Score vi phân", body_style),
            P("Amortized Neural MINE + Chia sẻ tham số Masked Network", body_style),
        ],
        [
            P("<b>Cơ chế Ước lượng</b>", body_style),
            P("Hiệu 2 phân kỳ KL đối với phân phối tham chiếu độc lập Ỹ", body_style),
            P("Tích phân khoảng cách hàm điểm khuếch tán ∫ ||S<sub>p</sub> − S<sub>q</sub>||<sup>2</sup> dt", body_style),
            P("Tối ưu hóa trực tiếp hàm mất mát Donsker-Varadhan liên hợp/biên", body_style),
        ],
        [
            P("<b>Bảo đảm Lý thuyết Cốt lõi</b>", body_style),
            P("Định lý Ergodic Birkhoff + Xấp xỉ phổ quát Causal Transformer", body_style),
            P("Bất đẳng thức Log-Sobolev + Triệt tiêu sai số đồng pha", body_style),
            P("4 Mệnh đề Giải tích: Bất khả tách, Giảm phương sai Cov &gt; 0, Lipschitz AWGN, Ngưỡng N*", body_style),
        ],
        [
            P("<b>Chế độ Dữ liệu Mục tiêu</b>", body_style),
            P("Chuỗi dài (long-memory context, l = 30–99 mẫu)", body_style),
            P("Chiều cao (high-dimensional, 70 chiều), T = 10.000–50.000", body_style),
            P("<b>Chuỗi siêu ngắn y sinh (N ≤ 50–100 mẫu), tín hiệu phi dừng</b>", body_style),
        ],
        [
            P("<b>Thời gian Suy luận / Độ trễ</b>", body_style),
            P("~100–300 ms (Transformer trên CPU/GPU)", body_style),
            P("<b>~20 phút / 1 cặp chuỗi (trên GPU khủng NVIDIA A100)</b>", body_style),
            P("<b>7.0 ms (Small MLP) / 11.2 ms (Deep MLP) trên CPU thường!</b>", body_style),
        ],
        [
            P("<b>Khả năng Triển khai Edge-AI</b>", body_style),
            P("Kém (Transformer ngốn nhiều RAM ma trận Attention)", body_style),
            P("<b>Hoàn toàn bất khả thi</b> (vòng lặp SDE hàng trăm bước)", body_style),
            P("<b>Xuất sắc: 369 tham số (~1.5 KB), chạy realtime trên vi điều khiển</b>", body_style),
        ],
        [
            P("<b>Hành vi ở Cửa sổ Nhỏ (N ≤ 50)</b>", body_style),
            P("Dễ sụp đổ, sinh giá trị âm, phương sai nổ tung", body_style),
            P("Không hội tụ được (cần hàng chục nghìn mẫu huấn luyện)", body_style),
            P("<b>Triệt tiêu phương sai &gt; 90%, sai số thấp hơn KSG 1.6–2.8 lần</b>", body_style),
        ],
        [
            P("<b>Thích nghi Miền (Domain Adaptation)</b>", body_style),
            P("Chưa có (giả định quá trình dừng nghiêm ngặt)", body_style),
            P("Chưa có", body_style),
            P("<b>Có giao thức UDA tối ưu Donsker-Varadhan không giám sát</b>", body_style),
        ],
    ]

    comp_table = Table(comp_data, colWidths=[38 * mm, 46 * mm, 48 * mm, 46 * mm])
    comp_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F2B5C")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("TOPPADDING", (0, 0), (-1, -1), 2.2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
            ("LEFTPADDING", (0, 0), (-1, -1), 2.8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2.8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ])
    )
    story.append(comp_table)
    story.append(Spacer(1, 3.5 * mm))

    # =========================================================================
    # PHẦN 4: LÀM RÕ MÔ HÌNH VQC (QUANTUM) CỦA DỰ ÁN
    # =========================================================================
    story.append(P("4. Làm Rõ Bản Chất Mô Hình VQC (Quantum) của Dự Án PQRST", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#0F2B5C"), spaceAfter=2.5))

    story.append(P("4.1. Mã hóa Tín hiệu vào Không gian Hilbert Lượng tử (Bao nhiêu Qubit?)", h2_style))
    story.append(P(
        "Đầu vào cổ điển có 4 thành phần: z = [y<sub>t</sub>, x<sub>lag</sub> · mask, y<sub>lag</sub>, mask] thuộc <b>R</b><sup>4</sup>.<br/>"
        "• <b>Mô hình Hybrid Classical-Quantum (MLP + VQC):</b> Mã hóa thành <b>4 qubit</b>. "
        "Dùng Encoder cổ điển nén và chuẩn hóa thành 4 góc quay tiềm ẩn trong [−π, π], nạp vào 4 qubit qua cổng quay R<sub>Y</sub>. "
        "Không gian trạng thái lượng tử có số chiều là <b>2<sup>4</sup> = 16 chiều biên độ phức</b>.<br/>"
        "• <b>Mô hình Pure VQC (Thuần lượng tử):</b> Mã hóa thành <b>6 qubit</b> qua kỹ thuật Data Re-uploading. "
        "Không gian trạng thái lượng tử có số chiều là <b>2<sup>6</sup> = 64 chiều biên độ phức</b>.<br/>"
        "<b>Tại sao gọi là 'đưa tín hiệu vào không gian lớn'?</b> Vì từ không gian số thực 4 chiều cổ điển, hệ thống được ánh xạ phi tuyến "
        "vào không gian Hilbert tích tensor <b>C</b><sup>2<sup>n</sup></sup> (16 hoặc 64 chiều phức). Các cổng vướng víu <b>CNOT dạng vòng tròn</b> "
        "kết nối các qubit, tạo ra sự tương tác chéo phi tuyến cực mạnh mà các hàm cộng tính độc lập cổ điển không thể có được (Mệnh đề 1).",
        body_style,
    ))

    story.append(P("4.2. Tại sao MSE của Hybrid VQC lại cao hơn MLP? (Phân tích Toán học cho Mentor)", h2_style))
    story.append(P(
        "Bóc tách công thức: <b>MSE = Bias<sup>2</sup> + Variance</b>.<br/>"
        "Nhìn vào Bảng số liệu Ablation của nhóm:<br/>"
        "• <b>Phương sai (Variance) của Hybrid VQC cực kỳ đẹp:</b> Đạt <b>1.77 × 10<sup>−5</sup></b> (thậm chí <b>nhỏ hơn Classical MLP tới 160 lần</b>, vì MLP là 2.88 × 10<sup>−3</sup>). Mạng lượng tử cho ước lượng cực kỳ ổn định!<br/>"
        "• <b>Độ lệch (Bias) bị âm:</b> Hybrid VQC ước lượng TE ≈ 0.016, trong khi Ground Truth là 0.181 (độ lệch Bias = −0.165). Do đó: MSE ≈ (−0.165)<sup>2</sup> + 1.77 × 10<sup>−5</sup> ≈ 0.0274.<br/>"
        "<b>NGUYÊN NHÂN:</b> Do giới hạn tính toán của bộ mô phỏng CPU (`lightning.qubit`) phải tính ma trận trạng thái phức tạp, "
        "trong script khảo sát đa kiến trúc Hybrid VQC mới chỉ được train <b>7 epochs</b> (bị underfitting độ lệch) thay vì hàng trăm epochs như MLP cổ điển.",
        body_style,
    ))

    story.append(P("4.3. Định vị Khoa học Chuẩn mực trong Bài báo Q1", h2_style))
    story.append(P(
        "• <b>Ngôi sao thực chiến của bài báo (Primary Champion):</b> Là <b>Small MLP (369 tham số, 7.0 ms, MSE = 0.0036)</b>. "
        "Đây là mô hình triển khai chính cho Edge-AI và toàn bộ dữ liệu lâm sàng 40 ca Fantasia.<br/>"
        "• <b>Pure VQC & Hybrid VQC:</b> Đóng vai trò là <b>Nghiên cứu thăm dò tính khả thi lượng tử (Future Outlook / Feasibility Exploration)</b>, "
        "chứng minh lý thuyết MINE có thể mở rộng lên không gian Hilbert và kiến trúc lai giúp tăng tốc 2.8 lần so với Pure VQC thuần túy.",
        body_style,
    ))

    doc.build(story, onFirstPage=make_header_footer, onLaterPages=make_header_footer)
    print(f"Successfully generated pristine, error-free PDF at: {PDF_PATH}")


if __name__ == "__main__":
    build_math_summary_pdf()
