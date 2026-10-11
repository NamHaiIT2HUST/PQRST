"""Script generate publication-grade PDF report summarizing the mathematics of TREET and TENDE.

Outputs to docs/TOM_TAT_TOAN_TREET_TENDE.pdf
"""

from __future__ import annotations

import os
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
        fontSize=17,
        leading=21,
        textColor=colors.HexColor("#0A2540"),
        alignment=1,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Arial-Italic",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#334155"),
        alignment=1,
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Arial-Bold",
        fontSize=12.5,
        leading=16,
        textColor=colors.HexColor("#0F2B5C"),
        spaceBefore=7,
        spaceAfter=3,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Normal"],
        fontName="Arial-Bold",
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=5,
        spaceAfter=2,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Arial",
        fontSize=8.8,
        leading=12.5,
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
        fontSize=8.6,
        leading=12.5,
        textColor=colors.HexColor("#0F172A"),
        leftIndent=10,
        rightIndent=10,
        spaceBefore=2.5,
        spaceAfter=2.5,
    )

    box_text = ParagraphStyle(
        "BoxText",
        parent=styles["Normal"],
        fontName="Arial",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#0F172A"),
    )

    story = []

    # =========================================================================
    # BANNER TIÊU ĐỀ
    # =========================================================================
    story.append(Paragraph("TỔNG HỢP VÀ PHÂN TÍCH TOÁN HỌC CHUYÊN SÂU", title_style))
    story.append(Spacer(1, 1.5 * mm))
    story.append(
        Paragraph(
            "Đối sánh Lý thuyết Thông tin & Giải tích: <b>TREET</b> (Transformers) vs. <b>TENDE</b> (Score-based Diffusion)<br/>"
            "và Định vị Khoa học của Kiến trúc <b>Amortized MINE / Quantum Hybrid (PQRST)</b>",
            subtitle_style,
        )
    )
    story.append(Spacer(1, 2.5 * mm))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0A2540"), spaceAfter=3))

    # Hộp tóm tắt tổng quan
    intro_table = Table(
        [
            [
                Paragraph(
                    "<b>TỔNG QUAN HỌC THUẬT:</b> Hai bài báo <b>TREET (IEEE Access 2025)</b> và <b>TENDE (AISTATS 2026)</b> "
                    "đại diện cho hai đỉnh cao toán học mới nhất trong bài toán ước lượng Transfer Entropy (TE). "
                    "TREET khai thác biểu diễn biến phân Donsker-Varadhan (DV) kết hợp Causal Attention với độ trễ thấp và chứng minh Birkhoff Ergodic. "
                    "Trong khi đó, TENDE sử dụng Phương trình Vi phân Ngẫu nhiên (SDE) và Hàm điểm (Score functions) để đo chính xác phân kỳ KL, "
                    "giúp thoát khỏi hoàn toàn định lý bất khả McAllester-Stratos. Dưới đây là phân tích chi tiết từng công thức, "
                    "bản chất giải tích, và luận điểm so sánh trực diện làm nổi bật tính khả thi Edge-AI độc bản của nghiên cứu nhóm ta.",
                    box_text,
                )
            ]
        ],
        colWidths=[178 * mm],
    )
    intro_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#94A3B8")),
            ("TOPPADDING", (0, 0), (-1, -1), 3 * mm),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3 * mm),
            ("LEFTPADDING", (0, 0), (-1, -1), 3.5 * mm),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3.5 * mm),
        ])
    )
    story.append(intro_table)
    story.append(Spacer(1, 3 * mm))

    # =========================================================================
    # PHẦN 1: BÀI 1 — TREET
    # =========================================================================
    story.append(Paragraph("1. Bài báo 1 — TREET: Transfer Entropy Estimation via Transformers", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#0F2B5C"), spaceAfter=2.5))
    story.append(Paragraph(
        "<i>Tác giả: Omer Luxembourg, Dor Tsur, Haim Permuter — Đại học Ben-Gurion của Negev, Israel (IEEE Access 2025 / arXiv:2402.06919).</i>",
        body_style,
    ))
    story.append(Spacer(1, 1 * mm))

    story.append(Paragraph("1.1. Khung lý thuyết & Định nghĩa Transfer Entropy mở rộng", h2_style))
    story.append(Paragraph(
        "TREET đưa ra định nghĩa tổng quát của Transfer Entropy giữa hai quá trình dừng ngẫu nhiên 𝕏 và 𝕐 với các bậc nhớ (k, l):<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>TE<sub>𝕏 → 𝕐</sub>(t; k, l) := I(X<sub>t−l</sub><sup>t</sup>; Y<sub>t</sub> | Y<sub>t−k</sub><sup>t−1</sup>)</b><br/>"
        "Khi quá trình dừng, chỉ số thời gian t được lược bỏ và viết gọn là TE<sub>𝕏 → 𝕐</sub>(k, l). "
        "<b>Điểm toán học mới:</b> TREET đưa cả quan sát hiện tại X<sub>t</sub> vào định nghĩa (thay vì chỉ lấy quá khứ X<sub>t−1</sub> như Schreiber 2000). "
        "Nhờ đó, khi hai quá trình độc lập cùng phân phối (i.i.d.), công thức lập tức thu về Mutual Information chuẩn: TE(0, 0) = I(X; Y).",
        body_style,
    ))

    story.append(Paragraph("1.2. Mối liên hệ với Directed Information (DI) & Điều kiện xích Markov", h2_style))
    story.append(Paragraph(
        "TREET thiết lập mối liên hệ giải tích giữa Transfer Entropy và Directed Information Rate I(𝕏 → 𝕐):<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;I(X<sup>n</sup> → Y<sup>n</sup>) = ∑<sub>i=0</sub><sup>n−1</sup> TE<sub>𝕏 → 𝕐</sub>(i), "
        "&nbsp;&nbsp;&nbsp;&nbsp;và&nbsp;&nbsp;&nbsp;&nbsp; I(𝕏 → 𝕐) = lim<sub>n→∞</sub> (1/n) I(X<sup>n</sup> → Y<sup>n</sup>) = lim<sub>n→∞</sub> I(X<sup>n</sup>; Y<sub>n</sub> | Y<sup>n−1</sup>)<br/>"
        "<b>Bổ đề 1 (Tính tương đương dưới điều kiện Markov):</b> Nếu quá trình thỏa mãn tính chất xích Markov "
        "Y<sub>l</sub> − Y<sup>l−1</sup> − Y<sub>−∞</sub><sup>−1</sup> và Y<sub>l</sub> − (X<sup>l</sup>, Y<sup>l−1</sup>) − (X<sub>−∞</sub><sup>−1</sup>, Y<sub>−∞</sub><sup>−1</sup>), "
        "thì với mọi m ≥ l, ta có đẳng thức tuyệt đối: <b>TE<sub>𝕏 → 𝕐</sub>(m) = TE<sub>𝕏 → 𝕐</sub>(l) = I(𝕏 → 𝕐)</b>. "
        "Điều này chứng minh TE trên độ dài ngữ cảnh hữu hạn l chính là nghiệm chính xác của thông tin định hướng vô hạn.",
        body_style,
    ))

    story.append(Paragraph("1.3. Bổ đề 2: Biểu diễn TE qua Hiệu hai Phân kỳ KL với Phân phối Tham chiếu", h2_style))
    story.append(Paragraph(
        "Điểm đột phá toán học của TREET là đưa vào một <b>phân phối tham chiếu liên tục tùy ý Ỹ</b> (chọn phân phối đều Uniform) để phân rã Transfer Entropy:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>TE<sub>𝕏 → 𝕐</sub>(l) = 𝒟<sub>Y<sup>l</sup> | Y<sup>l−1</sup> X<sup>l</sup> ∥ Ỹ<sup>l</sup></sub> − 𝒟<sub>Y<sup>l</sup> | Y<sup>l−1</sup> ∥ Ỹ<sup>l</sup></sub></b><br/>"
        "Trong đó hai số hạng phân kỳ KL có điều kiện được xác định bởi:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;• 𝒟<sub>Y<sup>l</sup> | Y<sup>l−1</sup> ∥ Ỹ<sup>l</sup></sub> := D<sub>KL</sub>(P<sub>Y<sub>l</sub> | Y<sup>l−1</sup></sub> ∥ P<sub>Ỹ</sub> | P<sub>Y<sup>l−1</sup></sub>) "
        "= ∫ log [ P(y<sub>l</sub> | y<sup>l−1</sup>) / P<sub>Ỹ</sub>(y<sub>l</sub>) ] P(x<sup>l</sup>, y<sup>l</sup>) d(x<sup>l</sup> y<sup>l</sup>)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;• 𝒟<sub>Y<sup>l</sup> | Y<sup>l−1</sup> X<sup>l</sup> ∥ Ỹ<sup>l</sup></sub> := D<sub>KL</sub>(P<sub>Y<sub>l</sub> | Y<sup>l−1</sup> X<sup>l</sup></sub> ∥ P<sub>Ỹ</sub> | P<sub>Y<sup>l−1</sup> X<sup>l</sup></sub>) "
        "= ∫ log [ P(y<sub>l</sub> | x<sup>l</sup>, y<sup>l−1</sup>) / P<sub>Ỹ</sub>(y<sub>l</sub>) ] P(x<sup>l</sup>, y<sup>l</sup>) d(x<sup>l</sup> y<sup>l</sup>)<br/>"
        "<b>Trực giác toán học:</b> Khi lấy hiệu hai số hạng, mật độ tham chiếu P<sub>Ỹ</sub>(y<sub>l</sub>) ở mẫu số <b>triệt tiêu hoàn toàn</b>, "
        "đưa về đúng hiệu hai entropy có điều kiện h(Y<sub>l</sub> | Y<sup>l−1</sup>) − h(Y<sub>l</sub> | X<sup>l</sup>, Y<sup>l−1</sup>) = TE. "
        "Phân phối Ỹ đóng vai trò như một 'điện thế mốc' giúp chuyển bài toán TE thành 2 bài toán ước lượng KL độc lập.",
        body_style,
    ))

    story.append(Paragraph("1.4. Cận biến phân Donsker-Varadhan (DV) & Mạng Fixed-Past Causal Attention (FPCA)", h2_style))
    story.append(Paragraph(
        "Mỗi phân kỳ KL được tối ưu hóa qua cận biến phân Donsker-Varadhan (1983):<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;𝒟(P ∥ Q) = sup<sub>g</sub> { 𝔼<sub>P</sub>[g] − log 𝔼<sub>Q</sub>[e<sup>g</sup>] }<br/>"
        "Nghiệm tối ưu lý thuyết đạt được tại hàm tỉ số mật độ: f<sub>y,l</sub><sup>*</sup> = log P(y<sub>l</sub> | y<sup>l−1</sup>) − log P<sub>Ỹ</sub>(y<sub>l</sub>).<br/>"
        "Để đảm bảo mạng không 'nhìn trộm' tương lai và duy trì bộ nhớ đúng l bước, TREET thiết kế mặt nạ chú ý dạng băng Toeplitz (Toeplitz-like banded mask M′ ∈ ℝ<sup>L×L</sup>):<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;M′[i, j] = 1 nếu 0 ≤ i − j < l, và M′[i, j] = −∞ nếu ngược lại.<br/>"
        "Toán tử chú ý FPCA: <b>FPCA(X) = softmax( (QK<sup>⊤</sup> / √d<sub>m</sub>) ⊙ M′ ) V</b>. "
        "Độ phức tạp chỉ là 𝒪(L · l · d<sub>o</sub>), thấp hơn nhiều so với Transformer thông thường.",
        body_style,
    ))

    story.append(Paragraph("1.5. Chứng minh Tính Vững Mạnh (Strong Consistency — Theorem 3)", h2_style))
    story.append(Paragraph(
        "TREET chứng minh ước lượng TÊ(D<sub>n</sub>; l) hội tụ hầu chắc chắn (P-a.s.) về TE lý thuyết qua 3 bước giải tích chặt chẽ:<br/>"
        "1. <b>Bước đại diện:</b> Phân rã TE thành hiệu 2 thế năng DV (Bổ đề 2).<br/>"
        "2. <b>Bước công thái học:</b> Áp dụng <b>Định lý Birkhoff Ergodic tổng quát (1957)</b>: (1/n) ∑ g<sub>i</sub>(T<sup>i</sup>ω) → 𝔼[g]. "
        "Do tính dừng và ergodic, trung bình mẫu hội tụ về kỳ vọng giải tích với sai số < ε/8.<br/>"
        "3. <b>Bước xấp xỉ hàm:</b> Sử dụng tính chất liên tục Lipschitz của hàm mũ e<sup>x</sup> trên đoạn (−∞, M]: "
        "(1/n) ∑ |e<sup>f<sup>*</sup></sup> − e<sup>g</sup>| ≤ e<sup>M</sup> (1/n) ∑ |f<sup>*</sup> − g|. "
        "Kết hợp với Định lý xấp xỉ phổ quát (Universal Approximation Theorem) cho Causal Transformer, sai số toàn phần bị chặn bởi ε/2.",
        body_style,
    ))

    story.append(Paragraph("1.6. Điểm mạnh và Điểm yếu chí mạng của TREET", h2_style))
    story.append(Paragraph(
        "• <b>Điểm mạnh:</b> Xử lý tốt ngữ cảnh lịch sử dài (l = 99), song song hóa được qua attention, nền tảng lý thuyết ergodic vững chắc.<br/>"
        "• <b>LỖ HỔNG CHÍ MẠNG:</b> Khi áp dụng vào <b>cửa sổ siêu ngắn (N ≤ 50–100)</b> hoặc khi số chiều tăng cao, "
        "ma trận Attention của Transformer bị thiếu dữ liệu nghiêm trọng, dẫn đến <b>nhiễu loạn trọng số attention, TE bị ước lượng âm và phương sai bùng nổ</b> "
        "(trong bài TENDE đối chứng, TREET bị âm tới −3.02 ± 2.59 trong khi ground-truth là 0.00).",
        body_style,
    ))

    story.append(Spacer(1, 3 * mm))
    story.append(PageBreak())

    # =========================================================================
    # PHẦN 2: BÀI 2 — TENDE
    # =========================================================================
    story.append(Paragraph("2. Bài báo 2 — TENDE: Transfer Entropy Neural Diffusion Estimation", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#0F2B5C"), spaceAfter=2.5))
    story.append(Paragraph(
        "<i>Tác giả: Simon Pedro Galeano Muñoz, Mustapha Bounoua, Giulio Franzese, Pietro Michiardi, Maurizio Filippone — KAUST & EURECOM (AISTATS 2026 / arXiv:2510.14096).</i>",
        body_style,
    ))
    story.append(Spacer(1, 1 * mm))

    story.append(Paragraph("2.1. Đặt vấn đề: Thoát khỏi 'Cơn ác mộng' của Cận Biến phân McAllester-Stratos", h2_style))
    story.append(Paragraph(
        "TENDE mở đầu bằng một luận điểm lý thuyết chấn động từ bài báo của McAllester & Stratos (AISTATS 2020):<br/>"
        "<i>'Mọi bộ ước lượng dựa trên cận dưới biến phân phân kỳ (như Donsker-Varadhan / MINE / TREET) đều yêu cầu cỡ mẫu tăng theo cấp số nhân "
        "n = 𝒪(e<sup>I</sup>) để hội tụ khi giá trị thông tin I lớn.'</i><br/>"
        "Để phá vỡ giới hạn này, TENDE từ bỏ hoàn toàn cận biến phân DV và thiết lập cầu nối giữa <b>Mô hình Khuếch tán Hàm điểm (Score-based Diffusion)</b> "
        "và <b>Phân kỳ Kullback-Leibler</b>.",
        body_style,
    ))

    story.append(Paragraph("2.2. Phương trình SDE Bảo toàn Phương sai & Hàm Điểm (Score Function)", h2_style))
    story.append(Paragraph(
        "TENDE xây dựng quá trình khuếch tán tiến dựa trên SDE bảo toàn phương sai (VP-SDE - Song et al. 2020):<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>dX<sub>t</sub> = f<sub>t</sub> X<sub>t</sub> dt + g<sub>t</sub> dW<sub>t</sub>, &nbsp;&nbsp;&nbsp;&nbsp; X<sub>0</sub> ∼ p<sub>X</sub>, &nbsp;&nbsp; t ∈ [0, T]</b><br/>"
        "với f<sub>t</sub> = −(1/2)β(t), g<sub>t</sub> = √β(t). Nghiệm chuyển tiếp phân phối điều kiện p<sub>0t</sub>(x<sub>t</sub> | x<sub>0</sub>) có dạng Gauss giải tích đóng:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;p<sub>0t</sub>(x<sub>t</sub> | x<sub>0</sub>) = 𝒩(x<sub>t</sub>; k<sub>t</sub> x<sub>0</sub>, σ<sub>t</sub><sup>2</sup> I), "
        "&nbsp;&nbsp; k<sub>t</sub> = exp(∫<sub>0</sub><sup>t</sup> f<sub>s</sub> ds), &nbsp; σ<sub>t</sub><sup>2</sup> = 1 − k<sub>t</sub><sup>2</sup>.<br/>"
        "<b>Định nghĩa Hàm Điểm (Score Function):</b> S<sub>p</sub>(x) := ∇<sub>x</sub> log p(x). Mạng nơ-ron học hàm điểm qua Denoising Score Matching:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;ℒ(θ) = ∫<sub>0</sub><sup>T</sup> g<sub>t</sub><sup>2</sup> 𝔼<sub>x, x̃</sub> [ ∥ S(x̃; θ) − ∇<sub>x̃</sub> log p<sub>0t</sub>(x̃ | x) ∥<sup>2</sup> ] dt.",
        body_style,
    ))

    story.append(Paragraph("2.3. Cầu nối Toán học Đột phá: Phân kỳ KL qua Tích phân Sai phân Hàm Điểm", h2_style))
    story.append(Paragraph(
        "<b>Định lý Cốt lõi (Bounoua 2024, Franzese 2023):</b> Phân kỳ KL giữa hai phân phối liên tục bất kỳ p và q được biểu diễn chính xác qua các hàm điểm khuếch tán:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>D<sub>KL</sub>(p ∥ q) = ∫<sub>0</sub><sup>T</sup> (g<sub>t</sub><sup>2</sup> / 2) 𝔼<sub>x ∼ p<sub>t</sub></sub> [ ∥ S<sub>p<sub>t</sub></sub>(x) − S<sub>q<sub>t</sub></sub>(x) ∥<sup>2</sup> ] dt + D<sub>KL</sub>(p<sub>T</sub> ∥ q<sub>T</sub>)</b><br/>"
        "Khi T đủ lớn, theo bất đẳng thức Logarithmic Sobolev (Collet & Malrieu 2008), cả hai phân phối đều co về phân phối dừng chuẩn 𝒩(0, I), "
        "do đó số hạng đầu cuối D<sub>KL</sub>(p<sub>T</sub> ∥ q<sub>T</sub>) → 0. "
        "Bộ ước lượng TENDE tính KL qua: <b>e(p, q) = ∫<sub>0</sub><sup>T</sup> (g<sub>t</sub><sup>2</sup> / 2) 𝔼 [ ∥ S<sub>p<sub>t</sub></sub>(x; θ<sub>1</sub><sup>*</sup>) − S<sub>q<sub>t</sub></sub>(x; θ<sub>2</sub><sup>*</sup>) ∥<sup>2</sup> ] dt</b>.",
        body_style,
    ))

    story.append(Paragraph("2.4. Phân tích Sai số d & Cơ chế Tự Triệt tiêu Sai số Đồng pha (Common-Mode Cancellation)", h2_style))
    story.append(Paragraph(
        "TENDE chứng minh sai số xấp xỉ có cấu trúc cực đẹp (Phụ lục A.3):<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>e(p, q) − D<sub>KL</sub>(p ∥ q) = d − D<sub>KL</sub>(p<sub>T</sub> ∥ q<sub>T</sub>)</b><br/>"
        "với d = ∫<sub>0</sub><sup>T</sup> (g<sub>t</sub><sup>2</sup> / 2) 𝔼 [ ∥ ϵ<sub>t</sub><sup>p</sup> − ϵ<sub>t</sub><sup>q</sup> ∥<sup>2</sup> + 2 ⟨ S<sub>p<sub>t</sub></sub> − S<sub>q<sub>t</sub></sub>, ϵ<sub>t</sub><sup>p</sup> − ϵ<sub>t</sub><sup>q</sup> ⟩ ] dt, "
        "trong đó ϵ<sub>t</sub><sup>p</sup> = S(θ<sup>*</sup>) − S<sub>p</sub> là sai số học hàm điểm.<br/>"
        "<b>Ý NGHĨA TOÁN HỌC ĐẮT GIÁ:</b><br/>"
        "1. <b>d không xác định dấu:</b> e(p, q) không phải là cận trên cũng không phải cận dưới của KL divergence. Vì vậy nó <b>hoàn toàn miễn nhiễm với định lý bất khả McAllester-Stratos</b>!<br/>"
        "2. <b>Triệt tiêu sai số đồng pha:</b> Nếu mạng nơ-ron mắc cùng một sai số hệ thống trên cả 2 phân phối (ϵ<sub>t</sub><sup>p</sup>(x) = ϵ<sub>t</sub><sup>q</sup>(x)), "
        "thì <b>d ≡ 0 tuyệt đối</b>, bất kể sai số riêng lẻ của từng hàm điểm lớn đến đâu!",
        body_style,
    ))

    story.append(Paragraph("2.5. Ba Dạng Công thức Tính TE & Mạng Amortized Score Network", h2_style))
    story.append(Paragraph(
        "Chuyển Transfer Entropy về Thông tin Tương hỗ Có điều kiện CMI: TE<sub>X → Y</sub>(k, ℓ) = I(Y<sub>t</sub>; X<sub>t−k</sub> | Y<sub>t−ℓ</sub>) với X = X<sub>t−k</sub>, Y = Y<sub>t</sub>, Z = Y<sub>t−ℓ</sub>:<br/>"
        "• <b>Dạng 1 (Kỳ vọng KL - Khuyên dùng vì luôn ≥ 0):</b> I(X; Y | Z) = 𝔼<sub>y,z</sub> [ D<sub>KL</sub>(p<sub>X|y,z</sub> ∥ p<sub>X|z</sub>) ] ≃ 𝔼<sub>y,z</sub> [ e(p<sub>X|y,z</sub>, p<sub>X|z</sub>) ]<br/>"
        "• <b>Dạng 2 (Hiệu Entropy qua biến phụ trợ Gauss φ<sub>σ</sub>):</b> I(X; Y | Z) = H(X | Z) − H(X | Y, Z) ≃ 𝔼<sub>y,z</sub>[e(p<sub>X|y,z</sub>, φ<sub>σ</sub>)] − 𝔼<sub>z</sub>[e(p<sub>X|z</sub>, φ<sub>σ</sub>)]<br/>"
        "• <b>Dạng 3 (Hiệu Mutual Information):</b> I(X; Y | Z) = I(X; [Y, Z]) − I(X; Z) ≃ 𝔼<sub>y,z</sub>[e(p<sub>X|y,z</sub>, p<sub>X</sub>)] − 𝔼<sub>z</sub>[e(p<sub>X|z</sub>, p<sub>X</sub>)]<br/>"
        "<b>Mạng Amortized:</b> Dùng duy nhất một mạng U-Net nhận vector [e<sub>1</sub>, e<sub>2</sub>, e<sub>3</sub>] ∈ {−1, 0, 1}<sup>3</sup> "
        "(1: biến tính score, 0: biến điều kiện, −1: biến tích phân biên) để sinh đồng thời cả 3 hàm điểm, tiết kiệm bộ nhớ.",
        body_style,
    ))

    story.append(Paragraph("2.6. Điểm mạnh và Điểm yếu chí mạng của TENDE", h2_style))
    story.append(Paragraph(
        "• <b>Điểm mạnh:</b> Độ chính xác cực cao trên số chiều lớn (70 chiều), bất biến trước chiều nhiễu độc lập (Bổ đề B.1) và có tính cộng tính tuyến tính (Bổ đề B.2).<br/>"
        "• <b>LỖ HỔNG CHÍ MẠNG:</b> <b>Chi phí tính toán khổng lồ!</b> Tác giả nêu rõ tại Mục D (trang 22): "
        "Để ước lượng TE cho một cặp chuỗi T = 10.000 mẫu, TENDE mất <b>khoảng 20 phút trên một GPU khủng NVIDIA A100</b>. "
        "Điều này khiến TENDE <b>hoàn toàn bất khả thi cho thiết bị y tế đeo tay (wearables/smartwatch)</b> hoặc phân tích trực tuyến thời gian thực.",
        body_style,
    ))

    story.append(Spacer(1, 3 * mm))
    story.append(PageBreak())

    # =========================================================================
    # PHẦN 3: BẢNG SO SÁNH ĐỐI CHIẾU TRỰC DIỆN
    # =========================================================================
    story.append(Paragraph("3. Bảng Đối Sánh Trực Diện: TREET vs. TENDE vs. Amortized MINE (PQRST)", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#0F2B5C"), spaceAfter=3))

    comp_data = [
        [
            Paragraph("<b>Tiêu chí Đối sánh</b>", body_bold),
            Paragraph("<b>TREET (IEEE Access 2025)</b>", body_bold),
            Paragraph("<b>TENDE (AISTATS 2026)</b>", body_bold),
            Paragraph("<b>Amortized MINE (Dự án PQRST)</b>", body_bold),
        ],
        [
            Paragraph("<b>Trường phái Toán học</b>", body_style),
            Paragraph("Cận biến phân Donsker-Varadhan + Causal Attention", body_style),
            Paragraph("Score-based Diffusion SDE + Mối liên hệ KL-Score vi phân", body_style),
            Paragraph("Amortized Neural MINE + Chia sẻ tham số Masked Network", body_style),
        ],
        [
            Paragraph("<b>Cơ chế Ước lượng</b>", body_style),
            Paragraph("Hiệu 2 phân kỳ KL đối với phân phối tham chiếu độc lập Ỹ", body_style),
            Paragraph("Tích phân khoảng cách hàm điểm khuếch tán ∫ ∥S<sub>p</sub> − S<sub>q</sub>∥<sup>2</sup> dt", body_style),
            Paragraph("Tối ưu hóa trực tiếp hàm mất mát Donsker-Varadhan liên hợp/biên", body_style),
        ],
        [
            Paragraph("<b>Bảo đảm Lý thuyết Cốt lõi</b>", body_style),
            Paragraph("Định lý Ergodic Birkhoff + Xấp xỉ phổ quát Causal Transformer", body_style),
            Paragraph("Bất đẳng thức Log-Sobolev + Triệt tiêu sai số đồng pha", body_style),
            Paragraph("4 Mệnh đề Giải tích: Bất khả tách, Giảm phương sai Cov > 0, Lipschitz AWGN, Ngưỡng N*", body_style),
        ],
        [
            Paragraph("<b>Chế độ Dữ liệu Mục tiêu</b>", body_style),
            Paragraph("Chuỗi dài (long-memory context, l = 30–99 mẫu)", body_style),
            Paragraph("Chiều cao (high-dimensional, 70 chiều), T = 10.000–50.000", body_style),
            Paragraph("<b>Chuỗi siêu ngắn y sinh (N ≤ 50–100 mẫu), tín hiệu phi dừng</b>", body_style),
        ],
        [
            Paragraph("<b>Thời gian Suy luận / Độ trễ</b>", body_style),
            Paragraph("~100–300 ms (Transformer trên CPU/GPU)", body_style),
            Paragraph("<b>~20 phút / 1 cặp chuỗi (trên GPU khủng NVIDIA A100)</b>", body_style),
            Paragraph("<b>7.0 ms (Small MLP) / 11.2 ms (Deep MLP) trên CPU thường!</b>", body_style),
        ],
        [
            Paragraph("<b>Khả năng Triển khai Edge-AI</b>", body_style),
            Paragraph("Kém (Transformer ngốn nhiều RAM ma trận Attention)", body_style),
            Paragraph("<b>Hoàn toàn bất khả thi</b> (vòng lặp SDE hàng trăm bước)", body_style),
            Paragraph("<b>Xuất sắc: 369 tham số (~1.5 KB), chạy realtime trên vi điều khiển</b>", body_style),
        ],
        [
            Paragraph("<b>Hành vi ở Cửa sổ Nhỏ (N ≤ 50)</b>", body_style),
            Paragraph("Dễ sụp đổ, sinh giá trị âm, phương sai nổ tung", body_style),
            Paragraph("Không hội tụ được (cần hàng chục nghìn mẫu huấn luyện)", body_style),
            Paragraph("<b>Triệt tiêu phương sai > 90%, sai số thấp hơn KSG 1.6–2.8 lần</b>", body_style),
        ],
        [
            Paragraph("<b>Thích nghi Miền (Domain Adaptation)</b>", body_style),
            Paragraph("Chưa có (giả định quá trình dừng nghiêm ngặt)", body_style),
            Paragraph("Chưa có", body_style),
            Paragraph("<b>Có giao thức UDA tối ưu Donsker-Varadhan không giám sát</b>", body_style),
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
    story.append(Paragraph("4. Làm Rõ Bản Chất Mô Hình VQC (Quantum) của Dự Án PQRST", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#0F2B5C"), spaceAfter=2.5))

    story.append(Paragraph("4.1. Mã hóa Tín hiệu vào Không gian Hilbert Lượng tử (Bao nhiêu Qubit?)", h2_style))
    story.append(Paragraph(
        "Đầu vào cổ điển có 4 thành phần: z = [y<sub>t</sub>, x<sub>lag</sub> · mask, y<sub>lag</sub>, mask] ∈ ℝ<sup>4</sup>.<br/>"
        "• <b>Mô hình Hybrid Classical-Quantum (MLP + VQC):</b> Mã hóa thành <b>4 qubit</b>. "
        "Dùng Encoder cổ điển nén và chuẩn hóa thành 4 góc quay tiềm ẩn trong [−π, π], nạp vào 4 qubit qua cổng quay R<sub>Y</sub>. "
        "Không gian trạng thái lượng tử có số chiều là <b>2<sup>4</sup> = 16 chiều biên độ phức</b>.<br/>"
        "• <b>Mô hình Pure VQC (Thuần lượng tử):</b> Mã hóa thành <b>6 qubit</b> qua kỹ thuật Data Re-uploading. "
        "Không gian trạng thái lượng tử có số chiều là <b>2<sup>6</sup> = 64 chiều biên độ phức</b>.<br/>"
        "<b>Tại sao gọi là 'đưa tín hiệu vào không gian lớn'?</b> Vì từ không gian số thực 4 chiều cổ điển, hệ thống được ánh xạ phi tuyến "
        "vào không gian Hilbert tích tensor ℂ<sup>2<sup>n</sup></sup> (16 hoặc 64 chiều phức). Các cổng vướng víu <b>CNOT dạng vòng tròn</b> "
        "kết nối các qubit, tạo ra sự tương tác chéo phi tuyến cực mạnh mà các hàm cộng tính độc lập cổ điển không thể có được (Mệnh đề 1).",
        body_style,
    ))

    story.append(Paragraph("4.2. Tại sao MSE của Hybrid VQC lại cao hơn MLP? (Phân tích Toán học cho Mentor)", h2_style))
    story.append(Paragraph(
        "Bóc tách công thức: <b>MSE = Bias<sup>2</sup> + Variance</b>.<br/>"
        "Nhìn vào Bảng số liệu Ablation của nhóm:<br/>"
        "• <b>Phương sai (Variance) của Hybrid VQC cực kỳ đẹp:</b> Đạt <b>1.77 × 10<sup>−5</sup></b> (thậm chí <b>nhỏ hơn Classical MLP tới 160 lần</b>, vì MLP là 2.88 × 10<sup>−3</sup>). Mạng lượng tử cho ước lượng cực kỳ ổn định!<br/>"
        "• <b>Độ lệch (Bias) bị âm:</b> Hybrid VQC ước lượng TE ≈ 0.016, trong khi Ground Truth là 0.181 (độ lệch Bias = −0.165). Do đó: MSE ≈ (−0.165)<sup>2</sup> + 1.77 × 10<sup>−5</sup> ≈ 0.0274.<br/>"
        "<b>NGUYÊN NHÂN:</b> Do giới hạn tính toán của bộ mô phỏng CPU (`lightning.qubit`) phải tính ma trận trạng thái phức tạp, "
        "trong script khảo sát đa kiến trúc Hybrid VQC mới chỉ được train <b>7 epochs</b> (bị underfitting độ lệch) thay vì hàng trăm epochs như MLP cổ điển.",
        body_style,
    ))

    story.append(Paragraph("4.3. Định vị Khoa học Chuẩn mực trong Bài báo Q1", h2_style))
    story.append(Paragraph(
        "• <b>Ngôi sao thực chiến của bài báo (Primary Champion):</b> Là <b>Small MLP (369 tham số, 7.0 ms, MSE = 0.0036)</b>. "
        "Đây là mô hình triển khai chính cho Edge-AI và toàn bộ dữ liệu lâm sàng 40 ca Fantasia.<br/>"
        "• <b>Pure VQC & Hybrid VQC:</b> Đóng vai trò là <b>Nghiên cứu thăm dò tính khả thi lượng tử (Future Outlook / Feasibility Exploration)</b>, "
        "chứng minh lý thuyết MINE có thể mở rộng lên không gian Hilbert và kiến trúc lai giúp tăng tốc 2.8 lần so với Pure VQC thuần túy.",
        body_style,
    ))

    story.append(Spacer(1, 3 * mm))

    # =========================================================================
    # PHẦN 5: BÀI HỌC ÁP DỤNG & KẾT LUẬN
    # =========================================================================
    story.append(Paragraph("5. Bài Học Kế Thừa Toán Học để Nâng Tầm Bài Báo Q1", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#0F2B5C"), spaceAfter=2.5))
    story.append(Paragraph(
        "1. <b>Kế thừa từ TREET:</b> Mượn cách lập luận của Bổ đề 2 (phân phối tham chiếu Ỹ) để đưa vào phụ lục, "
        "chứng minh rằng việc chia sẻ tham số mạng trong Amortized MINE thực chất là đang ước lượng tỉ số Radon-Nikodym thực sự một cách giải tích.<br/>"
        "2. <b>Kế thừa từ TENDE:</b> Đưa lập luận <i>Triệt tiêu sai số đồng pha (Common-mode error cancellation)</i> vào Mệnh đề 2 của nhóm mình, "
        "làm sáng tỏ vì sao việc dùng chung một mạng `MaskedStatisticsNetwork` kết hợp với cùng một hoán vị biên π giúp triệt tiêu tới 90.87% phương sai.<br/>"
        "3. <b>Thông điệp cốt lõi gửi Reviewer:</b> Trong khi TREET và TENDE tập trung vào chuỗi dài và cần cụm GPU khổng lồ, "
        "công trình của nhóm ta là giải pháp duy nhất giải quyết trọn vẹn bài toán <b>Ước lượng Transfer Entropy trên chuỗi thời gian siêu ngắn y sinh (N ≤ 50–100) "
        "chạy realtime trên thiết bị đeo nhúng (7 ms, 369 tham số)</b>, mở đường cho y tế số và thiết bị chăm sóc sức khỏe cá nhân hóa.",
        body_style,
    ))

    doc.build(story, onFirstPage=make_header_footer, onLaterPages=make_header_footer)
    print(f"Successfully generated pristine PDF at: {PDF_PATH}")


if __name__ == "__main__":
    build_math_summary_pdf()
