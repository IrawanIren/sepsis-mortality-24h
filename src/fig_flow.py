"""Figure 1: research workflow.

Rendered with the same typography and palette as the data figures so
that all figures in the manuscript are visually consistent. Colour
distinguishes the four stages; greyscale legibility is preserved by
keeping fills light and relying on position and text rather than hue.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from figures import OKABE, apply_style, save

# Light fills keep text readable and survive greyscale conversion.
FILL = {
    'data':   '#DCEEF7',   # light sky
    'design': '#FBEBCF',   # light orange
    'exp':    '#D9F0E6',   # light green
    'final':  '#F5DCD0',   # light vermillion
}
EDGE = {
    'data':   OKABE['sky'],
    'design': OKABE['orange'],
    'exp':    OKABE['green'],
    'final':  OKABE['red'],
}


def box(ax, x, y, w, h, text, kind, fontsize=8.5, bold=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.012,rounding_size=0.02',
        facecolor=FILL[kind], edgecolor=EDGE[kind], linewidth=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold' if bold else 'normal',
            linespacing=1.35, zorder=3)
    return (x + w / 2, y, x + w / 2, y + h, x, y + h / 2, x + w, y + h / 2)


def arrow(ax, xy_from, xy_to, style='-|>', dashed=False):
    ax.add_patch(FancyArrowPatch(
        xy_from, xy_to, arrowstyle=style, mutation_scale=11,
        linewidth=0.9, color='0.25', shrinkA=1, shrinkB=1,
        linestyle='--' if dashed else '-',
        connectionstyle='arc3,rad=0'))


def stage_label(ax, x, y, text, kind):
    ax.text(x, y, text, ha='left', va='center', fontsize=8,
            style='italic', color=EDGE[kind])


def build(outdir):
    apply_style()
    fig, ax = plt.subplots(figsize=(6.77, 9.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.01, 1.0)
    ax.axis('off')

    L, W = 0.19, 0.68          # main column
    HW = 0.325                 # half-width box
    XL, XR = 0.19, 0.535       # left and right half-column positions

    # ---- Stage 1: data preparation -----------------------------------
    stage_label(ax, 0.005, 0.960, 'Data\npreparation', 'data')

    b1 = box(ax, L, 0.945, W, 0.040,
             'MIMIC-IV v2.1  (credentialed access)', 'data')
    b2 = box(ax, L, 0.845, W, 0.070,
             'Cohort selection\n'
             'Sepsis-3  ·  age $\\geq$ 18  ·  first ICU stay\n'
             'LOS $\\geq$ 24 h  ·  laboratory data available\n'
             'n = 19,902   (3,036 deaths, 15.25%)', 'data')
    b3 = box(ax, L, 0.705, W, 0.108,
             'Preprocessing\n'
             'Hourly resampling to 24 time steps  ·  hourly differences\n'
             'Physiological bound clipping\n'
             'Race collapsed from 33 to 6 levels\n'
             'Three constant variables moved to the static pathway\n'
             '23 dynamic (46 channels)  ·  16 static  ·  8 one-hot', 'data')

    # ---- Stage 2: frozen partition -----------------------------------
    stage_label(ax, 0.005, 0.665, 'Frozen\npartition', 'design')

    b4 = box(ax, L, 0.645, W, 0.040,
             'Stratified split, generated once, shared by every model',
             'design', bold=True)
    b5 = box(ax, XL, 0.535, HW, 0.070,
             'Development set  (80%)\nn = 15,921\nFive stratified folds',
             'design')
    b6 = box(ax, XR, 0.535, HW, 0.070,
             'Test set  (20%)\nn = 3,981\nSealed until Stage 4', 'design')

    # ---- Stage 3: development experiments -----------------------------
    stage_label(ax, 0.005, 0.470, 'Development\nexperiments', 'exp')

    b7 = box(ax, XL, 0.440, HW, 0.058,
             'E0  Hyperparameter search\n15 trials\nLocked thereafter', 'exp')
    b8 = box(ax, XR, 0.440, HW, 0.058,
             'Baselines\nXGBoost  ·  CatBoost\nFirst-day SOFA', 'exp')

    b9 = box(ax, XL, 0.330, HW, 0.070,
             'E1  Window grid\n12 combinations of W and S\n'
             'E1b  five seeds at the best', 'exp')
    b10 = box(ax, XR, 0.330, HW, 0.070,
              'E2  Architecture ablation\nfour variants, five seeds\n'
              'E3  Observation length', 'exp')

    b11 = box(ax, L, 0.245, W, 0.048,
              'Configuration locked:  W = 6, S = 1, full architecture\n'
              'One score per patient from the right-aligned window',
              'exp', bold=True)

    # ---- Stage 4: final evaluation ------------------------------------
    stage_label(ax, 0.005, 0.190, 'Final\nevaluation', 'final')

    b12 = box(ax, L, 0.170, W, 0.045,
              'E5  Held-out test set, evaluated once\n'
              'Proposed model and all baselines, five seeds', 'final')
    b13 = box(ax, L, 0.062, W, 0.078,
              'Analysis\n'
              'AUROC and AUPRC  ·  Brier score and calibration slope\n'
              'Youden index and matched sensitivity 0.80\n'
              'DeLong  ·  bootstrap  ·  Benjamini-Hochberg', 'final')
    b14 = box(ax, L, 0.012, W, 0.034,
              'Twelve tables  ·  twelve figures  ·  predictions retained',
              'final')

    # ---- Connections --------------------------------------------------
    def down(a, b):
        arrow(ax, (a[0], a[1]), (b[0], b[3]))

    down(b1, b2); down(b2, b3); down(b3, b4)
    arrow(ax, (b4[0], b4[1]), (b5[0], b5[3]))
    arrow(ax, (b4[0], b4[1]), (b6[0], b6[3]))
    arrow(ax, (b5[0], b5[1]), (b7[0], b7[3]))
    arrow(ax, (b5[0], b5[1]), (b8[0], b8[3]))
    down(b7, b9); down(b8, b10)
    arrow(ax, (b9[0], b9[1]), (b11[0], b11[3]))
    arrow(ax, (b10[0], b10[1]), (b11[0], b11[3]))
    down(b11, b12); down(b12, b13); down(b13, b14)

    # The test set is opened only at Stage 4. Drawn dashed and routed
    # around the development column to show that it takes no part in any
    # development decision.
    xr = 0.885
    arrow(ax, (b6[6], b6[7]), (xr, b6[7]), style='-', dashed=True)
    arrow(ax, (xr, b6[7]), (xr, b12[7]), style='-', dashed=True)
    arrow(ax, (xr, b12[7]), (b12[6], b12[7]), dashed=True)
    ax.text(xr + 0.018, 0.36, 'sealed', rotation=90, fontsize=7.5,
            style='italic', color='0.35', ha='center', va='center')

    save(fig, outdir, 'fig01_research_flow')


if __name__ == '__main__':
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else '.')
