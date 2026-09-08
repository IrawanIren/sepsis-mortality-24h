"""Figure 2: patient selection flow.

Drawn in the CONSORT style, with exclusions shown to the right of the
main column so that the count remaining after each criterion is read
down a single line. Typography and palette match the other figures.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from figures import OKABE, apply_style, save

MAIN_FILL, MAIN_EDGE = '#DCEEF7', OKABE['sky']
EXC_FILL, EXC_EDGE = '#F5F5F5', '#808080'
OUT_FILL, OUT_EDGE = '#FBEBCF', OKABE['orange']


def box(ax, x, y, w, h, text, fill, edge, fontsize=8.5, bold=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.010,rounding_size=0.015',
        facecolor=fill, edgecolor=edge, linewidth=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold' if bold else 'normal',
            linespacing=1.4, zorder=3)
    return {'cx': x + w / 2, 'bottom': y, 'top': y + h,
            'left': x, 'right': x + w, 'mid': y + h / 2}


def arrow(ax, a, b, dashed=False):
    ax.add_patch(FancyArrowPatch(
        a, b, arrowstyle='-|>', mutation_scale=11, linewidth=0.9,
        color='0.25', shrinkA=1, shrinkB=1,
        linestyle='--' if dashed else '-'))


def build(outdir):
    apply_style()
    fig, ax = plt.subplots(figsize=(6.77, 5.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.0)
    ax.axis('off')

    ML, MW = 0.07, 0.46        # main column
    EL, EW = 0.60, 0.38        # exclusion column
    cx = ML + MW / 2

    b1 = box(ax, ML, 0.880, MW, 0.075,
             'Patients with sepsis in MIMIC-IV v2.1\n'
             'identified by the Sepsis-3 criteria\n'
             'n = 32,532', MAIN_FILL, MAIN_EDGE)

    e1 = box(ax, EL, 0.700, EW, 0.075,
             'Excluded  (n = 7,186)\n'
             'Age below 18 years\n'
             'Repeat ICU admission', EXC_FILL, EXC_EDGE)

    b2 = box(ax, ML, 0.590, MW, 0.060,
             'Adults on first ICU admission\nn = 25,346',
             MAIN_FILL, MAIN_EDGE)

    e2 = box(ax, EL, 0.420, EW, 0.075,
             'Excluded  (n = 5,444)\n'
             'ICU stay shorter than 24 h\n'
             'No laboratory measurement in 24 h',
             EXC_FILL, EXC_EDGE)

    b3 = box(ax, ML, 0.300, MW, 0.070,
             'Final study cohort\nn = 19,902',
             MAIN_FILL, MAIN_EDGE, bold=True)

    o1 = box(ax, 0.055, 0.100, 0.42, 0.070,
             'Survivors at 30 days\nn = 16,866  (84.75%)',
             OUT_FILL, OUT_EDGE)
    o2 = box(ax, 0.525, 0.100, 0.42, 0.070,
             'Died within 30 days\nn = 3,036  (15.25%)',
             OUT_FILL, OUT_EDGE)

    # Main column, top to bottom.
    arrow(ax, (cx, b1['bottom']), (cx, b2['top']))
    arrow(ax, (cx, b2['bottom']), (cx, b3['top']))

    # Exclusions branch to the right at the midpoint of each transition.
    for e, upper, lower in [(e1, b1, b2), (e2, b2, b3)]:
        y = (upper['bottom'] + lower['top']) / 2
        arrow(ax, (cx, y), (e['left'], e['mid']))

    # Outcome split.
    arrow(ax, (cx, b3['bottom']), (o1['cx'], o1['top']))
    arrow(ax, (cx, b3['bottom']), (o2['cx'], o2['top']))

    save(fig, outdir, 'fig02_patient_flow')


if __name__ == '__main__':
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else '.')
