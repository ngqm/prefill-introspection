"""Shared figure style: Paul Tol's colorblind-safe 'muted' colors, Arial type, embeddable PDF fonts."""
import matplotlib.pyplot as plt

PALETTE = {
    "indigo": "#332288",
    "cyan":   "#88CCEE",
    "rose":   "#CC6677",
}


def apply():
    """Apply the shared rcParams. Call once before drawing."""
    plt.rcParams.update({
        "font.family":     "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":       9,
        "axes.titlesize":  9,
        "axes.labelsize":  9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype":  42,
    })
