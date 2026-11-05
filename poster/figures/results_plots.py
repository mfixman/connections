from pathlib import Path

from matplotlib import font_manager, pyplot
from matplotlib.ticker import PercentFormatter
import seaborn

here = Path(__file__).parent
fonts = Path('/usr/share/texmf-dist/fonts/truetype/typoland/lato')
cam_blue = '#003E72'
deep = seaborn.color_palette('deep')
torok = '1 seed'
kusrow = '5 seeds'

colours = {
    'No guidance': deep[0],
    'Ordinary': deep[0],
    torok: deep[2],
    kusrow: deep[3],
}

charts = {
    'satresetcop': ('SATResetCoP', 722, {'No guidance': 322, torok: 369, kusrow: 461}),
    'prover9': ('Prover9', 59, {'Ordinary': 30, torok: 31, kusrow: 33}),
    'vampire': ('Vampire', 59, {'Ordinary': 38, torok: 38, kusrow: 36}),
}

notes = {'vampire': 'Still some way to go!'}

def set_style():
    for path in fonts.glob('Lato-*.ttf'):
        font_manager.fontManager.addfont(path)

    sizes = ['font.size', 'axes.labelsize', 'xtick.labelsize', 'ytick.labelsize']
    rc = {'font.weight': 300, 'pdf.fonttype': 42, **dict.fromkeys(sizes, 26)}
    seaborn.set_theme(style = 'darkgrid', font = 'Lato', rc = rc)

def bar_text(model, percentage, starred):
    suffix = '*' if starred and model in (torok, kusrow) else ''
    return f'{percentage:.1f}%{suffix}'

def draw_bars(axis, proved, total, starred):
    names = list(proved)
    percentages = [100 * count / total for count in proved.values()]
    seaborn.barplot(
        x = names,
        y = percentages,
        hue = names,
        palette = colours,
        saturation = 1,
        legend = False,
        width = 0.6,
        ax = axis,
    )

    for bars, model, percentage in zip(axis.containers, names, percentages):
        axis.bar_label(bars, labels = [bar_text(model, percentage, starred)], padding = 4)

def draw(name, title, total, proved):
    figure, axis = pyplot.subplots(figsize = (13.9, 3.5), layout = 'constrained')
    draw_bars(axis, proved, total, name in notes)

    axis.set(ylim = (0, 90), yticks = [0, 25, 50, 75], ylabel = 'Proved')
    axis.yaxis.set_major_formatter(PercentFormatter())
    axis.set_title(title, fontsize = 36, color = cam_blue, fontweight = 'normal')
    axis.set_title(notes.get(name, ''), loc = 'right', fontsize = 24)

    figure.savefig(here / f'results-{name}.pdf')
    pyplot.close(figure)

def main():
    set_style()
    for name, chart in charts.items():
        draw(name, *chart)

if __name__ == '__main__':
    main()
