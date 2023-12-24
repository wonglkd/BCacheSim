# Similar to seaborn contexts (https://seaborn.pydata.org/tutorial/aesthetics.html)
# See https://github.com/mwaskom/seaborn/blob/master/seaborn/rcmod.py
import matplotlib as mpl


def single():
    """Default: for 1:1 meetings, presenting just one graph"""
    return {
        'font.size': 14,
        'axes.titlesize': 22,
        'axes.labelsize': 18,
        'figure.labelsize': 16,
        'lines.linewidth': 3,
        'lines.markersize': 8,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 14,

        # 'font.size': 14,
        # 'axes.titlesize': 22,
        # 'axes.labelsize': 23,
        # 'figure.labelsize': 23,
        # 'lines.linewidth': 3,
        # 'lines.markersize': 8,
        # 'xtick.labelsize': 20,
        # 'ytick.labelsize': 20,
        # 'legend.fontsize': 16,

        'axes.labelweight': 'bold',
        'figure.labelweight': 'bold',
        'font.weight': 'bold',
        'axes.grid': True,
        'figure.constrained_layout.use': True,

        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',

        # Default
        # 'figure.figsize': [6.4, 4.8],

        # Avoid Type 3 fonts and use TrueType instead
        # See http://phyletica.org/matplotlib-fonts/
        'pdf.fonttype' : 42,
        'ps.fonttype': 42,

        # For export
        # 'figure.dpi': 300,

        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Bitstream Vera Sans',
        'mathtext.it': 'Bitstream Vera Sans:italic',
        'mathtext.bf': 'Bitstream Vera Sans:bold',

        'legend.columnspacing': .8,
        'legend.frameon': False,
    }


def wide():
    dct = single()
    dct['figure.figsize'] = [6.4 + 1, 4.8]
    return dct


def double():
    """When you have to squeeze in 2 graphs into one slide"""
    return rescale(single(), factor=1.8)


def triple():
    """When you have to squeeze in 3 graphs into one slide"""
    return rescale(single(), factor=1.9)


def poster():
    raise NotImplementedError
    return rescale(single(), factor=2.)


def notebook():
    return rescale(single(), factor=1.)


def paper():
    return rescale(single(), factor=1.)


def rescale(params, factor=1.):
    return {k: v * factor if k.endswith('size') and k != 'figure.figsize' else v
            for k, v in params.items()}


def get(prop):
    return mpl.rcParams[prop]


def use(name):
    if isinstance(name, str):
        name = globals()[name]
    mpl.rcParams.update(name())


# use('wide')
use('paper')
