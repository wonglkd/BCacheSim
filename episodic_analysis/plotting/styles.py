import matplotlib as mpl
from matplotlib import colors as mpl_colors


# http://colorbrewer2.org/#type=diverging&scheme=Spectral&n=4
COLORLISTS = {
    2: [
        mpl_colors.colorConverter.to_rgb("#a6cee3"),
        mpl_colors.colorConverter.to_rgb("#7fbf7b"),
    ],
    3: [
        mpl_colors.colorConverter.to_rgb("#a6cee3"),
        mpl_colors.colorConverter.to_rgb("#1f78b4"),
        mpl_colors.colorConverter.to_rgb("#b2df8a"),
    ],
    4: [
        mpl_colors.colorConverter.to_rgb("#d53e4f"),
        mpl_colors.colorConverter.to_rgb("#fdae61"),
        mpl_colors.colorConverter.to_rgb("#abdda4"),
        mpl_colors.colorConverter.to_rgb("#2b83ba")
    ],
    5: [
        mpl_colors.colorConverter.to_rgb("#a6cee3"),
        mpl_colors.colorConverter.to_rgb("#1f78b4"),
        mpl_colors.colorConverter.to_rgb("#b2df8a"),
        mpl_colors.colorConverter.to_rgb("#33a02c"),
        mpl_colors.colorConverter.to_rgb("#fb9a99")
],
    6: [
        mpl_colors.colorConverter.to_rgb("#d53e4f"),
        mpl_colors.colorConverter.to_rgb("#fdae61"),
        mpl_colors.colorConverter.to_rgb("#abdda4"),
        mpl_colors.colorConverter.to_rgb("#2b83ba"),
        "turquoise",
        "pink",
    ],
    8: [
        mpl_colors.colorConverter.to_rgb("#d53e4f"),
        mpl_colors.colorConverter.to_rgb("#f46d43"),
        mpl_colors.colorConverter.to_rgb("#fdae61"),
        mpl_colors.colorConverter.to_rgb("#fee08b"),
        mpl_colors.colorConverter.to_rgb("#e6f598"),
        mpl_colors.colorConverter.to_rgb("#abdda4"),
        mpl_colors.colorConverter.to_rgb("#66c2a5"),
        mpl_colors.colorConverter.to_rgb("#3288bd")
    ],
    12: [
        mpl_colors.colorConverter.to_rgb("#a6cee3"),
        mpl_colors.colorConverter.to_rgb("#1f78b4"),
        mpl_colors.colorConverter.to_rgb("#b2df8a"),
        mpl_colors.colorConverter.to_rgb("#33a02c"),
        mpl_colors.colorConverter.to_rgb("#fb9a99"),
        mpl_colors.colorConverter.to_rgb("#e31a1c"),
        mpl_colors.colorConverter.to_rgb("#fdbf6f"),
        mpl_colors.colorConverter.to_rgb("#ff7f00"),
        mpl_colors.colorConverter.to_rgb("#cab2d6"),
        mpl_colors.colorConverter.to_rgb("#6a3d9a"),
        mpl_colors.colorConverter.to_rgb("#f229ef"),
        #         mpl_colors.colorConverter.to_rgb("#ffff99"),
        mpl_colors.colorConverter.to_rgb("#b15928")
    ]
}


COLORS = {
    "grey": mpl_colors.colorConverter.to_rgb("#4D4D4D"),
    "blue": mpl_colors.colorConverter.to_rgb("#2B83BA"),
    "orange": mpl_colors.colorConverter.to_rgb("#FDAE61"),
    "green": mpl_colors.colorConverter.to_rgb("#ABDDA4"),
    "seafoam": mpl_colors.colorConverter.to_rgb("#9DD192"),
    "pink": mpl_colors.colorConverter.to_rgb("#F2686D"),
    "brown": mpl_colors.colorConverter.to_rgb("#B2912F"),
    "purple": mpl_colors.colorConverter.to_rgb("#C988BB"),
    "yellow": mpl_colors.colorConverter.to_rgb("#DECF3F"),
    "red": mpl_colors.colorConverter.to_rgb("#d7191c"),
    "grey1": mpl_colors.colorConverter.to_rgb("#ffffff"),
    "grey2": mpl_colors.colorConverter.to_rgb("#f2f2f2"),
    "grey3": mpl_colors.colorConverter.to_rgb("#e6e6e6"),
    "grey4": mpl_colors.colorConverter.to_rgb("#d9d9d9"),
    "grey5": mpl_colors.colorConverter.to_rgb("#bfbfbf"),
}


MARKERS = ["o", "v", "D", "*", "p", "8", "h", "x", "x", "d", "|", "2", "3", "4"]
LINESTYLES = ["solid", "dashed", "dotted", "dashdot"]
MARKERSIZES = [1] * len(MARKERS)
# * is small compared to the others.
MARKERSIZES[3] = 1.5

CAPSIZES = [10, 6, 3, 8, 4]

LINEGROUPS = {
    'fg': dict(lw=2),
    'fg-e': dict(lw=2, markeredgewidth=1),
    'bg': dict(lw=1, alpha=0.7, linestyle='--'),
    'bg-e': dict(lw=1, alpha=0.7, linestyle='--', markeredgewidth=1),
}


def default(n=None, labels=None):
    if n is None:
        n = len(labels)
    idx = min(k for k in COLORLISTS.keys() if k >= n)
    colors = COLORLISTS[idx]
    if labels:
        return [{'color': color, 'marker': marker, 'markersize': ms * mpl.rcParams['lines.markersize'], 'label': label}
                for color, marker, ms, label in zip(colors, MARKERS, MARKERSIZES, labels)]
    else:
        return [{'color': color, 'marker': marker, 'markersize': ms * mpl.rcParams['lines.markersize']}
                for color, marker, ms in zip(colors, MARKERS, MARKERSIZES)]
