"""Figure 4: model architecture.

Two parallel pathways drawn left to right, meeting at the concatenation.
Tensor shapes are annotated beneath each block so that the effect of
temporal pooling is visible, and equation numbers are shown in the
corner of each block that the text formalises.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from figures import OKABE, apply_style, save

DYN_FILL, DYN_EDGE = '#DCEEF7', OKABE['sky']
STA_FILL, STA_EDGE = '#FBEBCF', OKABE['orange']
MRG_FILL, MRG_EDGE = '#EDEDED', '#707070'
OUT_FILL, OUT_EDGE = '#D9F0E6', OKABE['green']


def block(ax, x, y, w, h, label, shape, fill, edge, eq=None, bold=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.006,rounding_size=0.012',
        facecolor=fill, edgecolor=edge, linewidth=1.0))
    ax.text(x + w / 2, y + h / 2, label, ha='center', va='center',
            fontsize=8, linespacing=1.35,
            fontweight='bold' if bold else 'normal', zorder=3)
    if shape:
        ax.text(x + w / 2, y - 0.028, shape, ha='center', va='center',
                fontsize=7.2, color='0.35', style='italic')
    if eq:
        ax.text(x + w / 2, y + h + 0.022, eq, ha='center', va='center',
                fontsize=6.8, color=edge)
    return {'cx': x + w / 2, 'cy': y + h / 2,
            'left': x, 'right': x + w, 'top': y + h, 'bottom': y}


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(
        a, b, arrowstyle='-|>', mutation_scale=9, linewidth=0.85,
        color='0.3', shrinkA=1, shrinkB=1))


def build(outdir, W=6, n_dyn=46, n_static=24):
    apply_style()
    fig, ax = plt.subplots(figsize=(6.77, 3.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    H = 0.150                       # block height
    YD, YS = 0.62, 0.20             # dynamic and static rows

    # ---- Dynamic pathway ---------------------------------------------
    xs, w = [], 0.096
    x = 0.012
    for _ in range(5):
        xs.append(x)
        x += w + 0.028

    d0 = block(ax, xs[0], YD, w, H, 'Hourly\nsequence',
               f'({W}, {n_dyn})', DYN_FILL, DYN_EDGE)
    d1 = block(ax, xs[1], YD, w, H, 'Conv1D 64\nk = 3\nBatchNorm',
               f'({W}, 64)', DYN_FILL, DYN_EDGE, eq='(7)')
    d2 = block(ax, xs[2], YD, w, H, 'MaxPool 2\nDropout',
               f'({W // 2}, 64)', DYN_FILL, DYN_EDGE)
    d3 = block(ax, xs[3], YD, w, H, 'Bi-GRU 32\nsequence\noutput',
               f'({W // 2}, 64)', DYN_FILL, DYN_EDGE, eq='(8)-(12)')
    d4 = block(ax, xs[4], YD, w, H, 'Bi-GRU 32\nfinal state',
               '(64)', DYN_FILL, DYN_EDGE)

    for a, b in zip([d0, d1, d2, d3], [d1, d2, d3, d4]):
        arrow(ax, (a['right'], a['cy']), (b['left'], b['cy']))

    # ---- Static pathway ----------------------------------------------
    s0 = block(ax, xs[0], YS, w, H, 'Static\nfeatures',
               f'({n_static})', STA_FILL, STA_EDGE)
    s1 = block(ax, xs[1], YS, w, H, 'Dense 32\nBatchNorm\nDropout',
               '(32)', STA_FILL, STA_EDGE, eq='(13)')
    s2 = block(ax, xs[2], YS, w, H, 'Dense 16',
               '(16)', STA_FILL, STA_EDGE)

    arrow(ax, (s0['right'], s0['cy']), (s1['left'], s1['cy']))
    arrow(ax, (s1['right'], s1['cy']), (s2['left'], s2['cy']))

    # ---- Merge and output --------------------------------------------
    YM = (YD + YS) / 2
    m = block(ax, 0.640, YM, 0.100, H, 'Concatenate',
              '(80)', MRG_FILL, MRG_EDGE)
    f1 = block(ax, 0.768, YM, 0.100, H, 'Dense 32\nDropout',
               '(32)', MRG_FILL, MRG_EDGE)
    f2 = block(ax, 0.896, YM, 0.100, H, 'Dense 1\nsigmoid',
               'risk score', OUT_FILL, OUT_EDGE, eq='(14)', bold=True)

    # Both pathways feed the concatenation.
    arrow(ax, (d4['right'], d4['cy']), (m['left'], m['cy'] + 0.030))
    arrow(ax, (s2['right'], s2['cy']), (m['left'], m['cy'] - 0.030))
    arrow(ax, (m['right'], m['cy']), (f1['left'], f1['cy']))
    arrow(ax, (f1['right'], f1['cy']), (f2['left'], f2['cy']))

    # ---- Pathway labels ----------------------------------------------
    ax.text(0.012, YD + H + 0.090, 'Dynamic pathway',
            fontsize=8.5, style='italic', color=DYN_EDGE, ha='left')
    ax.text(0.012, YS + H + 0.090, 'Static pathway',
            fontsize=8.5, style='italic', color=STA_EDGE, ha='left')

    ax.text(0.99, 0.030,
            f'Shapes shown for W = {W}; temporal pooling halves the '
            f'sequence length before the recurrent layers',
            fontsize=7, color='0.4', ha='right', style='italic')

    save(fig, outdir, 'fig04_architecture')


if __name__ == '__main__':
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else '.')
