"""
Script generating the comprehensive Framework Architecture & Pipeline Diagram
for IEEE Transactions submission (Q1 Publication-Grade Quality).
Outputs high-resolution 300 DPI figure to docs/paper/Fig/framework_architecture.png
and docs/paper/figures/framework_architecture.png
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

def draw_system_architecture():
    fig = plt.figure(figsize=(18, 9.5), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 9.5)
    ax.axis('off')

    # Color Palette (IEEE Journal Standard)
    c_bg = "#F8FAFC"
    c_blue = "#1E40AF"      # Primary Blue
    c_blue_light = "#EFF6FF"
    c_teal = "#0D9488"      # Teal for Masking/Features
    c_teal_light = "#F0FDFA"
    c_purple = "#7C3AED"    # Quantum/Theoretical
    c_purple_light = "#FAF5FF"
    c_amber = "#D97706"     # UDA / Adaptation
    c_amber_light = "#FFFBEB"
    c_rose = "#E11D48"      # Clinical / Outcome
    c_rose_light = "#FFF1F2"
    c_gray = "#475569"
    c_border = "#94A3B8"

    # Set background
    fig.patch.set_facecolor('white')

    # Header Title Banner
    ax.text(9.0, 9.15, "PROPOSED AMORTIZED NEURAL CAUSALITY ESTIMATION FRAMEWORK (ANTE-UDA)",
            ha='center', va='center', fontsize=15, fontweight='bold', color='#0F172A', fontfamily='sans-serif')
    ax.text(9.0, 8.85, "A Unified Paradigm for Ultra-Short Physiological Time Series ($N \\leq 100$), Variance Suppression, and Edge Deployment",
            ha='center', va='center', fontsize=11, fontstyle='italic', color=c_gray, fontfamily='sans-serif')

    # Helper function for fancy rounded panels
    def add_card(x, y, w, h, bg_color, border_color, title="", subtitle=""):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12,rounding_size=0.2",
                                      facecolor=bg_color, edgecolor=border_color, linewidth=1.5, zorder=1)
        ax.add_patch(rect)
        if title:
            # Title pill
            pill = patches.FancyBboxPatch((x + 0.2, y + h - 0.45), w - 0.4, 0.35,
                                          boxstyle="round,pad=0.04,rounding_size=0.1",
                                          facecolor=border_color, edgecolor='none', zorder=2)
            ax.add_patch(pill)
            ax.text(x + w/2, y + h - 0.28, title, ha='center', va='center',
                    fontsize=9.5, fontweight='bold', color='white', zorder=3)
        if subtitle:
            ax.text(x + w/2, y + h - 0.65, subtitle, ha='center', va='center',
                    fontsize=8.5, color=c_gray, zorder=3)

    # -------------------------------------------------------------
    # PANEL A: INPUT & EMBEDDING (Left Top)
    # -------------------------------------------------------------
    add_card(0.5, 4.4, 3.8, 4.1, c_blue_light, c_blue,
             "A. NON-STATIONARY BIOSIGNAL INPUT", "Ultra-Short Observation Window ($N \\leq 100$)")
    
    # Sub-elements in Panel A
    # Signal representation box
    sig_box = patches.FancyBboxPatch((0.7, 5.8), 3.4, 1.8, boxstyle="round,pad=0.08",
                                     facecolor='white', edgecolor='#CBD5E1', linewidth=1.2, zorder=2)
    ax.add_patch(sig_box)
    ax.text(2.4, 7.35, "Raw Physiological Channels", ha='center', fontsize=8.5, fontweight='bold', color='#1E293B', zorder=3)
    
    # Draw mini signal waves
    import numpy as np
    t = np.linspace(0, 1, 60)
    sig_resp = 0.4 * np.sin(2 * np.pi * 2.5 * t) + 6.8
    sig_rr = 0.35 * np.cos(2 * np.pi * 2.5 * t + 0.4) + 0.15 * np.sin(2 * np.pi * 6 * t) + 6.05
    ax.plot(np.linspace(0.9, 3.9, 60), sig_resp, color='#0284C7', lw=1.5, zorder=3)
    ax.text(0.85, 6.8, "$X$: Resp", fontsize=8, color='#0284C7', fontweight='bold', va='center', zorder=3)
    
    ax.plot(np.linspace(0.9, 3.9, 60), sig_rr, color='#DC2626', lw=1.5, zorder=3)
    ax.text(0.85, 6.05, "$Y$: RR Int", fontsize=8, color='#DC2626', fontweight='bold', va='center', zorder=3)
    
    # Window dimension indicator
    ax.text(2.4, 5.5, r"Sliding Window: $N \in [20, 100]$ samples", ha='center', fontsize=8, fontweight='bold', color='#0369A1', zorder=3)
    
    # Coordinate embedding box
    embed_box = patches.FancyBboxPatch((0.7, 4.55), 3.4, 0.75, boxstyle="round,pad=0.08",
                                       facecolor='#F1F5F9', edgecolor='#94A3B8', linewidth=1.0, zorder=2)
    ax.add_patch(embed_box)
    ax.text(2.4, 4.95, "Lagged Coordinate Embedding & Z-Score:", ha='center', fontsize=7.5, color='#334155', zorder=3)
    ax.text(2.4, 4.7, r"$\mathbf{z}_t = (y_t, \, x_{t-1}, \, y_{t-1}) \in \mathbb{R}^3$", ha='center', fontsize=8.5, fontweight='bold', color='#0F172A', zorder=3)

    # -------------------------------------------------------------
    # PANEL B: UNIFIED MASKED STATISTICS NETWORK (Center Top)
    # -------------------------------------------------------------
    add_card(4.6, 4.4, 5.3, 4.1, c_teal_light, c_teal,
             "B. MASKED STATISTICS NETWORK & VARIANCE REDUCTION", "Single Shared Backbone $T_\phi$ with Auxiliary Mask $m$")
    
    # Masking logic
    ax.text(7.25, 7.6, r"Dual Operational Paths via Binary Mask $m \in \{0, 1\}$", ha='center', fontsize=8.5, fontweight='bold', color='#0F766E', zorder=3)
    
    # Upper Path (Full MI)
    path_full = patches.FancyBboxPatch((4.8, 6.55), 4.9, 0.85, boxstyle="round,pad=0.06",
                                       facecolor='white', edgecolor='#0D9488', linewidth=1.2, zorder=2)
    ax.add_patch(path_full)
    ax.text(5.0, 7.15, "Full Joint (m = 1):", fontsize=8, fontweight='bold', color='#0F766E', zorder=3)
    ax.text(7.25, 7.15, r"$\mathbf{z}_t^{\mathrm{full}} = (y_t, \, x_{t-1}, \, y_{t-1}, \, 1)$", fontsize=8, color='#1E293B', zorder=3)
    ax.text(5.0, 6.75, r"$\to$ Shared Backbone $T_\phi \rightarrow \hat{I}_{\mathrm{full}} = I(Y_t; \, X_{t-1}, Y_{t-1})$", fontsize=8, color='#0F766E', fontweight='bold', zorder=3)

    # Lower Path (Reduced MI)
    path_red = patches.FancyBboxPatch((4.8, 5.5), 4.9, 0.85, boxstyle="round,pad=0.06",
                                      facecolor='white', edgecolor='#0D9488', linewidth=1.2, zorder=2)
    ax.add_patch(path_red)
    ax.text(5.0, 6.1, "Reduced Self (m = 0):", fontsize=8, fontweight='bold', color='#0F766E', zorder=3)
    ax.text(7.25, 6.1, r"$\mathbf{z}_t^{\mathrm{red}} = (y_t, \, 0, \, y_{t-1}, \, 0)$", fontsize=8, color='#1E293B', zorder=3)
    ax.text(5.0, 5.7, r"$\to$ Shared Backbone $T_\phi \rightarrow \hat{I}_{\mathrm{red}} = I(Y_t; \, Y_{t-1})$", fontsize=8, color='#0F766E', fontweight='bold', zorder=3)

    # Proposition 2 Proof highlight box
    prop_box = patches.FancyBboxPatch((4.8, 4.55), 4.9, 0.78, boxstyle="round,pad=0.06",
                                      facecolor='#CCFBF1', edgecolor='#14B8A6', linewidth=1.0, zorder=2)
    ax.add_patch(prop_box)
    ax.text(7.25, 5.05, r"Proposition 2 (Error Cancellation): $\mathrm{Var}(\hat{TE}) = \mathrm{Var}(\hat{I}_{\mathrm{full}}) + \mathrm{Var}(\hat{I}_{\mathrm{red}}) - 2\,\mathrm{Cov}$",
            ha='center', fontsize=8, fontweight='bold', color='#115E59', zorder=3)
    ax.text(7.25, 4.75, r"Shared weights $\phi$ + Unified permutation $\pi \rightarrow \mathrm{Cov} > 0 \rightarrow >90\%$ Variance Reduction",
            ha='center', fontsize=7.8, color='#0F766E', zorder=3)

    # -------------------------------------------------------------
    # PANEL C: TEST-TIME UDA PROTOCOL (Right Top)
    # -------------------------------------------------------------
    add_card(10.2, 4.4, 7.3, 4.1, c_amber_light, c_amber,
             "C. TEST-TIME UNSUPERVISED DOMAIN ADAPTATION (UDA)", "Label-Free Latent Manifold Projection via Variational DV Bound")
    
    # Joint vs Marginal Shuffling Box
    shuff_box = patches.FancyBboxPatch((10.4, 6.15), 3.3, 1.9, boxstyle="round,pad=0.06",
                                       facecolor='white', edgecolor='#D97706', linewidth=1.1, zorder=2)
    ax.add_patch(shuff_box)
    ax.text(12.05, 7.75, "Temporal Marginal Product", ha='center', fontsize=8, fontweight='bold', color='#B45309', zorder=3)
    ax.text(12.05, 7.4, r"Joint: $\mathbb{P}_{UV} \sim (z_t^{\text{joint}})$", ha='center', fontsize=7.8, color='#1E293B', zorder=3)
    ax.text(12.05, 7.05, r"Marginal: $\mathbb{P}_U \otimes \mathbb{P}_V \sim (z_{\pi(t)}^{\text{marg}})$", ha='center', fontsize=7.8, color='#1E293B', zorder=3)
    ax.text(12.05, 6.6, r"Fixed Permutation Seed $\pi$", ha='center', fontsize=7.5, fontstyle='italic', color='#78350F', zorder=3)
    ax.text(12.05, 6.3, r"No Ground-Truth Labels Needed!", ha='center', fontsize=7.5, fontweight='bold', color='#B45309', zorder=3)

    # DV Objective Optimization Box
    opt_box = patches.FancyBboxPatch((13.9, 6.15), 3.4, 1.9, boxstyle="round,pad=0.06",
                                     facecolor='white', edgecolor='#D97706', linewidth=1.1, zorder=2)
    ax.add_patch(opt_box)
    ax.text(15.6, 7.75, "Empirical Donsker-Varadhan Loss", ha='center', fontsize=8, fontweight='bold', color='#B45309', zorder=3)
    ax.text(15.6, 7.25, r"$\mathcal{L}_{\mathrm{DV}} = - \left[ \frac{1}{N}\sum T_\phi(z_i) - \log \frac{1}{N}\sum e^{T_\phi(\tilde{z}_i)} \right]$",
            ha='center', fontsize=7.8, color='#92400E', fontweight='bold', zorder=3)
    ax.text(15.6, 6.65, r"• Stratified 5-Fold Cross-Validation", ha='center', fontsize=7.5, color='#475569', zorder=3)
    ax.text(15.6, 6.35, r"• Conservative $\eta = 10^{-4}$, 5 Epochs", ha='center', fontsize=7.5, color='#475569', zorder=3)

    # Manifold Alignment Visual Box
    align_box = patches.FancyBboxPatch((10.4, 4.55), 6.9, 1.4, boxstyle="round,pad=0.06",
                                       facecolor='#FEF3C7', edgecolor='#F59E0B', linewidth=1.0, zorder=2)
    ax.add_patch(align_box)
    ax.text(13.85, 5.65, "Latent Manifold Alignment (Bypassing Reality Gap):", ha='center', fontsize=8, fontweight='bold', color='#92400E', zorder=3)
    ax.text(13.85, 5.35, r"Synthetic Manifold $\leftarrow$ [UDA Backpropagation] $\leftarrow$ Patient-Specific Biosignals", ha='center', fontsize=8, color='#78350F', zorder=3)
    ax.text(13.85, 4.85, r"Eliminates Covariate Shift $\rightarrow$ Restores Correct Directionality ($\Delta TE > 0$) without Semantic Drift",
            ha='center', fontsize=7.8, fontweight='bold', color='#B45309', zorder=3)

    # -------------------------------------------------------------
    # PANEL D: MULTI-ARCHITECTURE SPACE (Bottom Left & Center)
    # -------------------------------------------------------------
    add_card(0.5, 0.4, 10.5, 3.7, c_purple_light, c_purple,
             "D. MULTI-MODEL ARCHITECTURAL BENCHMARKING SPACE", "Hardware-Viable Classical Edge MLPs vs. Hilbert Space Quantum Circuits")
    
    # Sub-card 1: Small Edge MLP (Champion)
    edge_card = patches.FancyBboxPatch((0.7, 0.6), 4.8, 2.7, boxstyle="round,pad=0.08",
                                       facecolor='white', edgecolor='#10B981', linewidth=1.5, zorder=2)
    ax.add_patch(edge_card)
    # Champion ribbon
    ax.text(3.1, 3.05, "★ PRIMARY EDGE-AI CHAMPION", ha='center', fontsize=8.5, fontweight='bold', color='#059669', zorder=3)
    ax.text(3.1, 2.75, "Compact 2-Layer MLP (16-16 ELU)", ha='center', fontsize=8, color='#334155', zorder=3)
    
    ax.text(1.0, 2.35, r"• Parameters:", fontsize=8, fontweight='bold', color='#1E293B', zorder=3)
    ax.text(2.6, 2.35, "369 parameters (~1.5 KB)", fontsize=8, fontweight='bold', color='#059669', zorder=3)
    
    ax.text(1.0, 2.0, r"• CPU Latency:", fontsize=8, fontweight='bold', color='#1E293B', zorder=3)
    ax.text(2.6, 2.0, "7.0 ms per window", fontsize=8, fontweight='bold', color='#059669', zorder=3)
    
    ax.text(1.0, 1.65, r"• Estimation MSE:", fontsize=8, fontweight='bold', color='#1E293B', zorder=3)
    ax.text(2.6, 1.65, "0.0036 (matches 25k-param MLP)", fontsize=8, color='#334155', zorder=3)
    
    ax.text(1.0, 1.3, r"• Noise Tolerance:", fontsize=8, fontweight='bold', color='#1E293B', zorder=3)
    ax.text(2.6, 1.3, "2.8x lower MSE vs KSG @ 20dB", fontsize=8, color='#047857', zorder=3)

    ax.text(3.1, 0.85, "[Ideal for ARM Cortex-M / Wearable Continuous Monitoring]", ha='center', fontsize=7.5, fontstyle='italic', color='#047857', zorder=3)

    # Sub-card 2: Hybrid Classical-Quantum (MLP + VQC)
    q_card = patches.FancyBboxPatch((5.8, 0.6), 5.0, 2.7, boxstyle="round,pad=0.08",
                                    facecolor='white', edgecolor='#7C3AED', linewidth=1.5, zorder=2)
    ax.add_patch(q_card)
    ax.text(8.3, 3.05, "◆ HYBRID CLASSICAL-QUANTUM (MLP + 4Q VQC)", ha='center', fontsize=8.5, fontweight='bold', color='#6D28D9', zorder=3)
    ax.text(8.3, 2.75, "4-Qubit Circular CNOT Entangling Ansatz", ha='center', fontsize=8, color='#334155', zorder=3)

    ax.text(6.1, 2.35, r"• Hilbert Space:", fontsize=8, fontweight='bold', color='#1E293B', zorder=3)
    ax.text(7.7, 2.35, r"$\mathbb{C}^{16}$ complex amplitudes", fontsize=8, color='#334155', zorder=3)

    ax.text(6.1, 2.0, r"• Ultra-Low Variance:", fontsize=8, fontweight='bold', color='#1E293B', zorder=3)
    ax.text(7.7, 2.0, r"$1.77 \times 10^{-5}$ (160x lower than MLP!)", fontsize=8, fontweight='bold', color='#7C3AED', zorder=3)

    ax.text(6.1, 1.65, r"• Speedup over Pure VQC:", fontsize=8, fontweight='bold', color='#1E293B', zorder=3)
    ax.text(7.7, 1.65, "2.8x faster (4.8 s vs 13.2 s)", fontsize=8, color='#6D28D9', zorder=3)

    ax.text(6.1, 1.3, r"• Role in Paper:", fontsize=8, fontweight='bold', color='#1E293B', zorder=3)
    ax.text(7.7, 1.3, "Quantum Feasibility Outlook", fontsize=8, fontstyle='italic', color='#334155', zorder=3)

    # Mini Circuit Scheme Diagram inside Quantum Card
    circ_box = patches.FancyBboxPatch((6.0, 0.72), 4.6, 0.45, boxstyle="round,pad=0.04",
                                      facecolor='#EDE9FE', edgecolor='#C4B5FD', linewidth=0.8, zorder=3)
    ax.add_patch(circ_box)
    ax.text(8.3, 0.95, r"$|0\rangle^{\otimes 4} \rightarrow R_Y(\theta)\ \mathrm{Enc} \rightarrow \mathrm{CNOT\ Ring} \rightarrow \mathrm{Var}(R_X, R_Y, R_Z) \rightarrow \langle Z_i \rangle \rightarrow \mathrm{Readout}$",
            ha='center', fontsize=7.0, color='#4C1D95', fontweight='bold', zorder=4)

    # -------------------------------------------------------------
    # PANEL E: CLINICAL VALIDATION & OUTCOMES (Bottom Right)
    # -------------------------------------------------------------
    add_card(11.3, 0.4, 6.2, 3.7, c_rose_light, c_rose,
             "E. CLINICAL AGING VALIDATION (FANTASIA 40)", "Full Cohort Autonomic Cardiorespiratory Decoupling")
    
    res_card = patches.FancyBboxPatch((11.5, 0.6), 5.8, 2.7, boxstyle="round,pad=0.08",
                                      facecolor='white', edgecolor='#F43F5E', linewidth=1.2, zorder=2)
    ax.add_patch(res_card)
    
    ax.text(14.4, 3.05, "PhysioNet Fantasia Database ($N_{\mathrm{total}} = 40$)", ha='center', fontsize=8.5, fontweight='bold', color='#BE123C', zorder=3)
    ax.text(14.4, 2.75, "20 Young (ages 21--34) vs. 20 Old (ages 68--85)", ha='center', fontsize=8, color='#475569', zorder=3)

    ax.text(11.7, 2.35, r"• Directionality in Young:", fontsize=8, fontweight='bold', color='#1E293B', zorder=3)
    ax.text(14.1, 2.35, r"90.0% (18/20) Resp $\rightarrow$ RR ($p = 6.68 \times 10^{-5}$)", fontsize=8, color='#BE123C', fontweight='bold', zorder=3)

    ax.text(11.7, 2.0, r"• Age Decoupling (RSA):", fontsize=8, fontweight='bold', color='#1E293B', zorder=3)
    ax.text(14.1, 2.0, r"$p = 0.0045$, Cohen's $d = 0.67$", fontsize=8, color='#BE123C', fontweight='bold', zorder=3)

    ax.text(11.7, 1.65, r"• Empirical Variance:", fontsize=8, fontweight='bold', color='#1E293B', zorder=3)
    ax.text(14.1, 1.65, "2.2x lower variance over KSG", fontsize=8, color='#059669', fontweight='bold', zorder=3)

    ax.text(11.7, 1.3, r"• Age Decay Correlation:", fontsize=8, fontweight='bold', color='#1E293B', zorder=3)
    ax.text(14.1, 1.3, r"$\rho = -0.391\ (p = 0.013)$", fontsize=8, color='#334155', zorder=3)

    # Censoring bias resolution pill
    bias_box = patches.FancyBboxPatch((11.7, 0.72), 5.4, 0.45, boxstyle="round,pad=0.04",
                                      facecolor='#FFE4E6', edgecolor='#FDA4AF', linewidth=0.8, zorder=3)
    ax.add_patch(bias_box)
    ax.text(14.4, 0.95, "Resolves Prior Algorithmic Censoring Bias (rescues 85% elderly records)",
            ha='center', fontsize=7.2, color='#9F1239', fontweight='bold', zorder=4)

    # -------------------------------------------------------------
    # FLOW CONNECTING ARROWS
    # -------------------------------------------------------------
    # Arrow A -> B
    ax.annotate("", xy=(4.6, 6.45), xytext=(4.3, 6.45),
                arrowprops=dict(arrowstyle="simple,tail_width=0.18,head_width=0.6,head_length=0.6",
                                color=c_blue, lw=0.5), zorder=5)
    
    # Arrow B -> C
    ax.annotate("", xy=(10.2, 6.45), xytext=(9.9, 6.45),
                arrowprops=dict(arrowstyle="simple,tail_width=0.18,head_width=0.6,head_length=0.6",
                                color=c_teal, lw=0.5), zorder=5)

    # Arrow B -> D (downwards)
    ax.annotate("", xy=(5.2, 4.1), xytext=(5.2, 4.4),
                arrowprops=dict(arrowstyle="simple,tail_width=0.18,head_width=0.6,head_length=0.6",
                                color=c_teal, lw=0.5), zorder=5)

    # Arrow C -> E (downwards)
    ax.annotate("", xy=(14.4, 4.1), xytext=(14.4, 4.4),
                arrowprops=dict(arrowstyle="simple,tail_width=0.18,head_width=0.6,head_length=0.6",
                                color=c_amber, lw=0.5), zorder=5)

    # Arrow D -> E (across bottom)
    ax.annotate("", xy=(11.3, 2.25), xytext=(11.0, 2.25),
                arrowprops=dict(arrowstyle="simple,tail_width=0.18,head_width=0.6,head_length=0.6",
                                color=c_purple, lw=0.5), zorder=5)

    # Save destinations
    os.makedirs("docs/paper/Fig", exist_ok=True)
    os.makedirs("docs/paper/figures", exist_ok=True)
    
    out_path_1 = "docs/paper/Fig/framework_architecture.png"
    out_path_2 = "docs/paper/figures/framework_architecture.png"
    
    plt.savefig(out_path_1, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(out_path_2, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Successfully generated high-res publication diagram at:\n  - {out_path_1}\n  - {out_path_2}")

if __name__ == "__main__":
    draw_system_architecture()
