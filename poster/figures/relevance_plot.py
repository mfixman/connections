from pathlib import Path

from matplotlib import pyplot
import seaborn

from results_plots import deep, set_style

here = Path(__file__).parent
core_fill = '#ccd8e3'

# Kusrow's scores for SYN054+1, and whether each clause is in the SAT core of all three proofs.
clauses = {
    r'$\neg P(x) \vee Q(x) \vee R(x)$': (0.7333, True),
    r'$\neg S(x) \vee \neg Q(x)$': (0.6454, True),
    r'$\neg Q(x) \vee S(x)$': (0.5432, True),
    r'$P(a) \vee Q(b)$': (0.4238, True),
    r'$\neg R(x) \vee S(x)$': (0.0628, False),
}

def main():
    set_style()
    pyplot.rcParams['mathtext.fontset'] = 'cm'

    scores = [score for score, _ in clauses.values()]
    figure, axis = pyplot.subplots(figsize = (10.3, 4), layout = 'constrained')
    seaborn.barplot(x = scores, y = list(clauses), color = deep[3], saturation = 1, ax = axis)

    for bars in axis.containers:
        axis.bar_label(bars, fmt = '{:.2f}', padding = 6)

    for label, (_, core) in zip(axis.get_yticklabels(), clauses.values()):
        fill = core_fill if core else 'white'
        label.set_bbox(dict(boxstyle = 'round,pad=0.3', edgecolor = 'black', facecolor = fill))

    axis.tick_params(axis = 'y', pad = 12)
    axis.set(xlim = (0, 1), xticks = [0, 0.5, 1])

    figure.get_layout_engine().set(w_pad = 0.2)
    figure.savefig(here / 'relevance.pdf')
    pyplot.close(figure)

if __name__ == '__main__':
    main()
