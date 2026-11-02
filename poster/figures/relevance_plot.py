from pathlib import Path

from matplotlib import pyplot
import seaborn

from results_plots import deep, set_style

here = Path(__file__).parent

# Kusrow's scores for SYN054+1; green clauses are in the SAT core of all three proofs, red in none.
clauses = {
    r'$\neg P(x) \vee Q(x) \vee R(x)$': (0.7333, deep[2]),
    r'$\neg S(x) \vee \neg Q(x)$': (0.6454, deep[2]),
    r'$\neg Q(x) \vee S(x)$': (0.5432, deep[2]),
    r'$P(a) \vee Q(b)$': (0.4238, deep[2]),
    r'$\neg R(x) \vee S(x)$': (0.0628, deep[3]),
}

def main():
    set_style()
    pyplot.rcParams['mathtext.fontset'] = 'cm'

    labels = list(clauses)
    scores = [score for score, _ in clauses.values()]
    palette = {label: colour for label, (_, colour) in clauses.items()}

    figure, axis = pyplot.subplots(figsize = (10.3, 4), layout = 'constrained')
    seaborn.barplot(x = scores, y = labels, hue = labels, palette = palette, saturation = 1, legend = False, ax = axis)

    for bars in axis.containers:
        axis.bar_label(bars, fmt = '{:.2f}', padding = 6)

    axis.set(xlim = (0, 1), xticks = [0, 0.5, 1])
    figure.savefig(here / 'relevance.pdf')
    pyplot.close(figure)

if __name__ == '__main__':
    main()
