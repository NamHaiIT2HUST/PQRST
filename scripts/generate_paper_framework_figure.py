"""
Script generating the perfected, publication-grade Framework Architecture & Pipeline Diagram
for IEEE Transactions (TBME / TNNLS).
Outputs high-resolution 300 DPI figure to docs/paper/Fig/framework_architecture.png
and docs/paper/figures/framework_architecture.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_perfect_architecture():
    # Set high quality styling
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.edgecolor'] = '#CBD5E1'
    plt.rcParams['mathtext.fontset'] = 'cm' # Computer Modern for math

    fig = plt.figure(figsize=(19, 10.2), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 19)
    ax.set_ylim(0, 10.2)
    ax.axis('off')

    # Color Palette (IEEE Standard Aesthetic)
    c_blue = "#1D4ED8"         # Blue
    c_blue_bg = "#F0F7FF"
    c_teal = "#0F766E"         # Teal
    c_teal_bg = "#F0FDFA"
    c_amber = "#B45309"        # Amber
    c_amber_bg = "#FFFBEB"
    c_purple = "#6D28D9"       # Purple
    c_purple_bg = "#FAF5FF"
    c_rose = "#BE123C"         # Rose / Red
    c_rose_bg = "#FFF1F2"
    c_gray = "#475569"
    c_dark = "#0F172A"

    # Background
    fig.patch.set_facecolor('white')

    # -------------------------------------------------------------
    # MASTER TITLE
    # -------------------------------------------------------------
    ax.text(9.5, 9.85, "PROPOSED AMORTIZED NEURAL TRANSFER ENTROPY FRAMEWORK (ANTE-UDA)",
            ha='center', va='center', fontsize=15.5, fontweight='bold', color=c_dark)
    ax.text(9.5, 9.50, "A Unified Paradigm for Ultra-Short Physiological Windows ($N \\leq 100$), Variance Suppression, and Edge Deployment",
            ha='center', va='center', fontsize=11, fontstyle='italic', color=c_gray)

    # Helper function for drawing rounded cards
    def draw_card(x, y, w, h, bg_color, border_color, title, subtitle):
        # Outer Card
        card = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.18",
                                      facecolor=bg_color, edgecolor=border_color, linewidth=1.6, zorder=1)
        ax.add_patch(card)
        # Title Pill
        pill_h = 0.38
        pill = patches.FancyBboxPatch((x + 0.15, y + h - pill_h - 0.1), w - 0.3, pill_h,
                                      boxstyle="round,pad=0.03,rounding_size=0.1",
                                      facecolor=border_color, edgecolor='none', zorder=2)
        ax.add_patch(pill)
        ax.text(x + w / 2, y + h - pill_h / 2 - 0.1, title, ha='center', va='center',
                fontsize=9.2, fontweight='bold', color='white', zorder=3)
        # Subtitle
        if subtitle:
            ax.text(x + w / 2, y + h - 0.62, subtitle, ha='center', va='center',
                    fontsize=8.2, fontstyle='italic', color=c_gray, zorder=3)

    # =============================================================
    # PANEL A: INPUT & EMBEDDING (x=0.5, w=3.9, y=4.7, h=4.5)
    # =============================================================
    draw_card(0.5, 4.7, 3.9, 4.5, c_blue_bg, c_blue,
              "A. NON-STATIONARY BIOSIGNAL INPUT", "Ultra-Short Observation Window ($N \\leq 100$)")
    
    # Inner Signal Box
    sig_box = patches.FancyBboxPatch((0.7, 6.2), 3.5, 2.1, boxstyle="round,pad=0.06",
                                     facecolor='white', edgecolor='#93C5FD', linewidth=1.2, zorder=2)
    ax.add_patch(sig_box)
    ax.text(2.45, 8.05, "Raw Physiological Channels", ha='center', fontsize=8.5, fontweight='bold', color=c_dark, zorder=3)

    # Plot synthetic waves
    t_pts = np.linspace(0, 1, 80)
    sig_resp = 0.28 * np.sin(2 * np.pi * 2.5 * t_pts) + 7.60
    sig_rr = 0.25 * np.cos(2 * np.pi * 2.5 * t_pts + 0.4) + 0.10 * np.sin(2 * np.pi * 6 * t_pts) + 7.00
    ax.plot(np.linspace(1.3, 4.0, 80), sig_resp, color='#0284C7', lw=1.6, zorder=3)
    ax.text(0.85, 7.60, "$X$: Resp", fontsize=8.2, color='#0284C7', fontweight='bold', va='center', zorder=3)
    ax.plot(np.linspace(1.3, 4.0, 80), sig_rr, color='#DC2626', lw=1.6, zorder=3)
    ax.text(0.85, 7.00, "$Y$: RR Int", fontsize=8.2, color='#DC2626', fontweight='bold', va='center', zorder=3)

    ax.text(2.45, 6.30, "Sliding Window: $N \\in [20, 100]$ samples", ha='center', fontsize=7.8, fontweight='bold', color='#0369A1', zorder=3)

    # Embedding Box
    emb_box = patches.FancyBboxPatch((0.7, 4.88), 3.5, 1.15, boxstyle="round,pad=0.06",
                                     facecolor='#F8FAFC', edgecolor='#CBD5E1', linewidth=1.1, zorder=2)
    ax.add_patch(emb_box)
    ax.text(2.45, 5.75, "Lagged Coordinate Embedding & Z-Score", ha='center', fontsize=7.6, fontweight='bold', color=c_gray, zorder=3)
    ax.text(2.45, 5.40, r"$\mathbf{z}_t = (y_t, \, x_{t-1}, \, y_{t-1}) \in \mathbb{R}^3$", ha='center', fontsize=8.5, fontweight='bold', color=c_dark, zorder=3)
    ax.text(2.45, 5.05, "Scale Invariance via Window Standardization", ha='center', fontsize=7.2, fontstyle='italic', color='#64748B', zorder=3)

    # =============================================================
    # PANEL B: MASKED STATISTICS NETWORK (x=4.7, w=6.0, y=4.7, h=4.5)
    # =============================================================
    draw_card(4.7, 4.7, 6.0, 4.5, c_teal_bg, c_teal,
              "B. MASKED STATISTICS NETWORK & VARIANCE REDUCTION", "Single Shared Backbone $T_\\phi$ with Auxiliary Conditioning Mask $m$")
    
    ax.text(7.7, 8.35, "Dual Operational Paths via Binary Conditioning Mask $m \\in \\{0, 1\\}$",
            ha='center', fontsize=8.2, fontweight='bold', color='#0F766E', zorder=3)

    # Upper Path (Full MI)
    path_full = patches.FancyBboxPatch((4.9, 7.30), 5.6, 0.92, boxstyle="round,pad=0.05",
                                       facecolor='white', edgecolor='#0D9488', linewidth=1.2, zorder=2)
    ax.add_patch(path_full)
    ax.text(5.1, 7.95, "Full Joint (m = 1):", fontsize=8.0, fontweight='bold', color='#0F766E', zorder=3)
    ax.text(6.85, 7.95, r"$\mathbf{z}_t^{\mathrm{full}} = (y_t, \, x_{t-1}, \, y_{t-1}, \, 1)$", fontsize=8.0, color=c_dark, zorder=3)
    ax.text(5.1, 7.50, r"$\to$ Shared $T_\phi \Rightarrow \hat{I}_{\mathrm{full}} = I(Y_t; \, X_{t-1}, Y_{t-1})$",
            fontsize=8.0, color='#0F766E', fontweight='bold', zorder=3)

    # Lower Path (Reduced Self MI)
    path_red = patches.FancyBboxPatch((4.9, 6.20), 5.6, 0.92, boxstyle="round,pad=0.05",
                                      facecolor='white', edgecolor='#0D9488', linewidth=1.2, zorder=2)
    ax.add_patch(path_red)
    ax.text(5.1, 6.85, "Reduced Self (m = 0):", fontsize=8.0, fontweight='bold', color='#0F766E', zorder=3)
    ax.text(7.05, 6.85, r"$\mathbf{z}_t^{\mathrm{red}} = (y_t, \, 0, \, y_{t-1}, \, 0)$", fontsize=8.0, color=c_dark, zorder=3)
    ax.text(5.1, 6.40, r"$\to$ Shared $T_\phi \Rightarrow \hat{I}_{\mathrm{red}} = I(Y_t; \, Y_{t-1})$",
            fontsize=8.0, color='#0F766E', fontweight='bold', zorder=3)

    # Proposition 2 Proof Highlight Box
    prop_box = patches.FancyBboxPatch((4.9, 4.88), 5.6, 1.15, boxstyle="round,pad=0.05",
                                      facecolor='#CCFBF1', edgecolor='#14B8A6', linewidth=1.1, zorder=2)
    ax.add_patch(prop_box)
    ax.text(7.7, 5.75, r"Proposition 2 (Error Cancellation Guarantee):",
            ha='center', fontsize=8.0, fontweight='bold', color='#115E59', zorder=3)
    ax.text(7.7, 5.40, r"$\mathrm{Var}(\hat{TE}) = \mathrm{Var}(\hat{I}_{\mathrm{full}}) + \mathrm{Var}(\hat{I}_{\mathrm{red}}) - 2\,\mathrm{Cov}(\hat{I}_{\mathrm{full}}, \hat{I}_{\mathrm{red}})$",
            ha='center', fontsize=8.0, fontweight='bold', color='#042F2E', zorder=3)
    ax.text(7.7, 5.05, r"Shared weights $\phi$ + Unified seed $\pi \Rightarrow \mathrm{Cov} > 0 \Rightarrow >90\%$ Variance Reduction",
            ha='center', fontsize=7.6, color='#0F766E', zorder=3)

    # =============================================================
    # PANEL C: TEST-TIME UDA (x=11.0, w=7.5, y=4.7, h=4.5)
    # =============================================================
    draw_card(11.0, 4.7, 7.5, 4.5, c_amber_bg, c_amber,
              "C. TEST-TIME UNSUPERVISED DOMAIN ADAPTATION (UDA)", "Label-Free Latent Manifold Projection via Variational DV Bound")
    
    # Left Subbox: Marginal Permutation
    shuff_box = patches.FancyBboxPatch((11.2, 6.25), 3.4, 2.05, boxstyle="round,pad=0.05",
                                       facecolor='white', edgecolor='#D97706', linewidth=1.1, zorder=2)
    ax.add_patch(shuff_box)
    ax.text(12.9, 8.00, "Temporal Marginal Shuffling", ha='center', fontsize=8.0, fontweight='bold', color='#B45309', zorder=3)
    ax.text(12.9, 7.60, r"Joint: $\mathbb{P}_{UV} \sim (\mathbf{z}_t)$", ha='center', fontsize=7.8, color=c_dark, zorder=3)
    ax.text(12.9, 7.20, r"Marginal: $\mathbb{P}_U \otimes \mathbb{P}_V \sim (\mathbf{z}_{\pi(t)})$", ha='center', fontsize=7.8, color=c_dark, zorder=3)
    ax.text(12.9, 6.75, r"Synchronized Seed $\pi$ for Both Paths", ha='center', fontsize=7.2, fontstyle='italic', color='#78350F', zorder=3)
    ax.text(12.9, 6.42, r"★ Zero Ground-Truth Labels Needed!", ha='center', fontsize=7.4, fontweight='bold', color='#B45309', zorder=3)

    # Right Subbox: DV Objective
    opt_box = patches.FancyBboxPatch((14.8, 6.25), 3.5, 2.05, boxstyle="round,pad=0.05",
                                     facecolor='white', edgecolor='#D97706', linewidth=1.1, zorder=2)
    ax.add_patch(opt_box)
    ax.text(16.55, 8.00, "Empirical Donsker-Varadhan Loss", ha='center', fontsize=8.0, fontweight='bold', color='#B45309', zorder=3)
    ax.text(16.55, 7.45, r"$\mathcal{L}_{\mathrm{DV}} = - \left[ \frac{1}{N}\sum T_\phi(z_i) - \log \frac{1}{N}\sum e^{T_\phi(\tilde{z}_i)} \right]$",
            ha='center', fontsize=7.6, color='#92400E', fontweight='bold', zorder=3)
    ax.text(16.55, 6.80, r"• Stratified 5-Fold Cross-Validation", ha='center', fontsize=7.4, color='#475569', zorder=3)
    ax.text(16.55, 6.45, r"• Conservative $\eta = 10^{-4}$, 5 Epochs", ha='center', fontsize=7.4, color='#475569', zorder=3)

    # Bottom Subbox: Manifold Projection
    align_box = patches.FancyBboxPatch((11.2, 4.88), 7.1, 1.20, boxstyle="round,pad=0.05",
                                       facecolor='#FEF3C7', edgecolor='#F59E0B', linewidth=1.0, zorder=2)
    ax.add_patch(align_box)
    ax.text(14.75, 5.80, "Latent Manifold Alignment (Bypassing Reality Gap):", ha='center', fontsize=8.0, fontweight='bold', color='#92400E', zorder=3)
    ax.text(14.75, 5.45, r"Synthetic Prior $\leftarrow$ [Unsupervised Gradient Update] $\leftarrow$ Patient Biosignals", ha='center', fontsize=7.8, color='#78350F', zorder=3)
    ax.text(14.75, 5.08, r"Eliminates Covariate Shift $\Rightarrow$ Restores Correct Directionality ($\Delta TE > 0$) without Semantic Drift",
            ha='center', fontsize=7.6, fontweight='bold', color='#B45309', zorder=3)

    # =============================================================
    # PANEL D: MULTI-ARCHITECTURE BENCHMARK (x=0.5, w=10.2, y=0.4, h=4.0)
    # =============================================================
    draw_card(0.5, 0.4, 10.2, 4.0, c_purple_bg, c_purple,
              "D. MULTI-MODEL ARCHITECTURAL BENCHMARKING SPACE", "Hardware-Viable Classical Edge MLPs vs. Hilbert Space Quantum Circuits")

    # Sub-card 1: Small Edge MLP (Champion)
    edge_card = patches.FancyBboxPatch((0.7, 0.6), 4.7, 3.0, boxstyle="round,pad=0.06",
                                       facecolor='white', edgecolor='#10B981', linewidth=1.5, zorder=2)
    ax.add_patch(edge_card)
    ax.text(3.05, 3.32, "★ PRIMARY EDGE-AI CHAMPION", ha='center', fontsize=8.5, fontweight='bold', color='#059669', zorder=3)
    ax.text(3.05, 3.05, "Compact 2-Layer MLP (16-16 ELU)", ha='center', fontsize=7.8, color='#334155', zorder=3)

    ax.text(0.9, 2.65, "• Parameters:", fontsize=8.0, fontweight='bold', color=c_dark, zorder=3)
    ax.text(2.6, 2.65, "369 params (~1.5 KB SRAM)", fontsize=8.0, fontweight='bold', color='#059669', zorder=3)

    ax.text(0.9, 2.30, "• Inference Latency:", fontsize=8.0, fontweight='bold', color=c_dark, zorder=3)
    ax.text(2.6, 2.30, "7.0 ms per window (CPU)", fontsize=8.0, fontweight='bold', color='#059669', zorder=3)

    ax.text(0.9, 1.95, "• Estimation MSE:", fontsize=8.0, fontweight='bold', color=c_dark, zorder=3)
    ax.text(2.6, 1.95, "0.0036 (matches 25k Deep MLP)", fontsize=8.0, color='#334155', zorder=3)

    ax.text(0.9, 1.60, "• Noise Resilience:", fontsize=8.0, fontweight='bold', color=c_dark, zorder=3)
    ax.text(2.6, 1.60, "2.8x lower MSE vs KSG @ 20dB", fontsize=8.0, color='#047857', zorder=3)

    # Edge Deployment Banner
    arm_box = patches.FancyBboxPatch((0.85, 0.75), 4.4, 0.55, boxstyle="round,pad=0.04",
                                     facecolor='#ECFDF5', edgecolor='#A7F3D0', linewidth=0.8, zorder=3)
    ax.add_patch(arm_box)
    ax.text(3.05, 1.02, "Target: Ultra-Low-Power Wearables (ARM Cortex-M)", ha='center', fontsize=7.5, fontweight='bold', color='#065F46', zorder=4)

    # Sub-card 2: Hybrid Classical-Quantum
    q_card = patches.FancyBboxPatch((5.7, 0.6), 4.8, 3.0, boxstyle="round,pad=0.06",
                                    facecolor='white', edgecolor='#7C3AED', linewidth=1.5, zorder=2)
    ax.add_patch(q_card)
    ax.text(8.1, 3.32, "◆ HYBRID CLASSICAL-QUANTUM (MLP + 4Q VQC)", ha='center', fontsize=8.3, fontweight='bold', color='#6D28D9', zorder=3)
    ax.text(8.1, 3.05, "4-Qubit Circular CNOT Entangling Ansatz", ha='center', fontsize=7.8, color='#334155', zorder=3)

    ax.text(5.9, 2.65, "• State Space:", fontsize=8.0, fontweight='bold', color=c_dark, zorder=3)
    ax.text(7.9, 2.65, r"$\mathbb{C}^{16}$ complex amplitudes", fontsize=8.0, color='#334155', zorder=3)

    ax.text(5.9, 2.30, "• Model Variance:", fontsize=8.0, fontweight='bold', color=c_dark, zorder=3)
    ax.text(7.9, 2.30, r"$1.77 \times 10^{-5}$ (160x lower than MLP!)", fontsize=8.0, fontweight='bold', color='#7C3AED', zorder=3)

    ax.text(5.9, 1.95, "• VQC Speedup:", fontsize=8.0, fontweight='bold', color=c_dark, zorder=3)
    ax.text(7.9, 1.95, "2.8x faster (4.8 s vs 13.2 s)", fontsize=8.0, color='#6D28D9', zorder=3)

    ax.text(5.9, 1.60, "• Investigation Role:", fontsize=8.0, fontweight='bold', color=c_dark, zorder=3)
    ax.text(7.9, 1.60, "Quantum Feasibility Outlook", fontsize=8.0, fontstyle='italic', color='#334155', zorder=3)

    # Mini Quantum Circuit Schematic
    q_box = patches.FancyBboxPatch((5.85, 0.75), 4.5, 0.55, boxstyle="round,pad=0.04",
                                   facecolor='#F5F3FF', edgecolor='#DDD6FE', linewidth=0.8, zorder=3)
    ax.add_patch(q_box)
    ax.text(8.1, 1.02, r"$|0\rangle^{\otimes 4} \to R_Y(\theta) \to \mathrm{CNOT\ Ring} \to \mathrm{Var}(R_X, R_Y, R_Z) \to \langle Z_i \rangle$",
            ha='center', fontsize=7.2, color='#4C1D95', fontweight='bold', zorder=4)

    # =============================================================
    # PANEL E: CLINICAL VALIDATION (x=11.0, w=7.5, y=0.4, h=4.0)
    # =============================================================
    draw_card(11.0, 0.4, 7.5, 4.0, c_rose_bg, c_rose,
              "E. CLINICAL AGING VALIDATION (FANTASIA 40)", "Full Cohort Autonomic Cardiorespiratory Decoupling")

    res_card = patches.FancyBboxPatch((11.2, 0.6), 7.1, 3.0, boxstyle="round,pad=0.06",
                                      facecolor='white', edgecolor='#F43F5E', linewidth=1.3, zorder=2)
    ax.add_patch(res_card)

    ax.text(14.75, 3.32, r"PhysioNet Fantasia Database ($N_{\mathrm{total}} = 40$ records)", ha='center', fontsize=8.5, fontweight='bold', color='#BE123C', zorder=3)
    ax.text(14.75, 3.05, "20 Young (ages 21--34) vs. 20 Old (ages 68--85) — 9,443 analysis windows", ha='center', fontsize=7.8, color='#475569', zorder=3)

    ax.text(11.4, 2.65, "• Directionality in Young:", fontsize=8.0, fontweight='bold', color=c_dark, zorder=3)
    ax.text(14.2, 2.65, r"90.0% (18/20) Resp $\to$ RR ($p = 6.68 \times 10^{-5}$)", fontsize=8.0, color='#BE123C', fontweight='bold', zorder=3)

    ax.text(11.4, 2.30, "• Age Decoupling (RSA):", fontsize=8.0, fontweight='bold', color=c_dark, zorder=3)
    ax.text(14.2, 2.30, r"$p = 0.0045$, Cohen's $d = 0.67$ (Robust Blunting)", fontsize=8.0, color='#BE123C', fontweight='bold', zorder=3)

    ax.text(11.4, 1.95, "• Empirical Variance Reduction:", fontsize=8.0, fontweight='bold', color=c_dark, zorder=3)
    ax.text(14.2, 1.95, "2.2x lower std dev over KSG (0.0184 vs 0.0406)", fontsize=8.0, color='#059669', fontweight='bold', zorder=3)

    ax.text(11.4, 1.60, "• Cross-Method Agreement:", fontsize=8.0, fontweight='bold', color=c_dark, zorder=3)
    ax.text(14.2, 1.60, r"High window-by-window correlation $r = 0.89$", fontsize=8.0, color='#1E293B', zorder=3)

    # Resolution of Censoring Bias Banner
    bias_box = patches.FancyBboxPatch((11.4, 0.75), 6.7, 0.55, boxstyle="round,pad=0.04",
                                      facecolor='#FFE4E6', edgecolor='#FDA4AF', linewidth=0.8, zorder=3)
    ax.add_patch(bias_box)
    ax.text(14.75, 1.02, "Resolves Prior Algorithmic Censoring Bias: Rescues 85% of Elderly Records (17/20)",
            ha='center', fontsize=7.4, color='#9F1239', fontweight='bold', zorder=4)

    # =============================================================
    # FLOW ARROWS
    # =============================================================
    # A -> B
    ax.annotate("", xy=(4.7, 6.95), xytext=(4.4, 6.95),
                arrowprops=dict(arrowstyle="simple,tail_width=0.20,head_width=0.65,head_length=0.65",
                                color=c_blue, lw=0.5), zorder=5)

    # B -> C
    ax.annotate("", xy=(11.0, 6.95), xytext=(10.7, 6.95),
                arrowprops=dict(arrowstyle="simple,tail_width=0.20,head_width=0.65,head_length=0.65",
                                color=c_teal, lw=0.5), zorder=5)

    # B -> D (downwards)
    ax.annotate("", xy=(5.2, 4.4), xytext=(5.2, 4.7),
                arrowprops=dict(arrowstyle="simple,tail_width=0.20,head_width=0.65,head_length=0.65",
                                color=c_teal, lw=0.5), zorder=5)

    # C -> E (downwards)
    ax.annotate("", xy=(14.75, 4.4), xytext=(14.75, 4.7),
                arrowprops=dict(arrowstyle="simple,tail_width=0.20,head_width=0.65,head_length=0.65",
                                color=c_amber, lw=0.5), zorder=5)

    # D -> E (horizontal)
    ax.annotate("", xy=(11.0, 2.4), xytext=(10.7, 2.4),
                arrowprops=dict(arrowstyle="simple,tail_width=0.20,head_width=0.65,head_length=0.65",
                                color=c_purple, lw=0.5), zorder=5)

    # Output paths
    os.makedirs("docs/paper/Fig", exist_ok=True)
    os.makedirs("docs/paper/figures", exist_ok=True)
    
    out_1 = "docs/paper/Fig/framework_architecture.png"
    out_2 = "docs/paper/figures/framework_architecture.png"

    plt.savefig(out_1, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(out_2, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Exported perfectly aligned publication diagram to:\n", out_1, "\n", out_2)

if __name__ == "__main__":
    generate_perfect_architecture()
