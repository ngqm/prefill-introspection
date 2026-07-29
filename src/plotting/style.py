"""Shared figure style for the project's plots.

Colorblind-safe palette, serif typography, vector-friendly PDF embedding.
Call apply() at the top of any plot script for consistent styling.
"""
import matplotlib.pyplot as plt


# Colorblind-safe palette: Paul Tol's 'muted' qualitative scheme. Dict keys are
# legacy role names kept so downstream scripts need no change; primary two-way
# contrast is indigo (baseline) vs rose (accent).
PALETTE = {
    "terracotta":      "#CC6677",   # Tol rose (accent / second series)
    "terracotta_dark": "#882255",   # Tol wine
    "terracotta_lite": "#E39AA5",
    "slate":      "#332288",   # Tol indigo (primary / baseline series)
    "slate_lite": "#88CCEE",   # Tol cyan
    "sage":       "#44AA99",   # Tol teal
    "sage_lite":  "#99DDCC",
    "grey":       "#BBBBBB",   # neutral grey
}

# Stable per-family color assignment for model-comparison plots.
MODEL_FAMILY_COLORS = {
    "qwen":   PALETTE["terracotta"],
    "llama":  PALETTE["slate"],
    "gemma":  PALETTE["sage"],
}


def family_of(model_name: str) -> str:
    """Return the model family ("qwen", "llama", "gemma", or "other")."""
    m = model_name.lower()
    if "qwen" in m:  return "qwen"
    if "llama" in m: return "llama"
    if "gemma" in m: return "gemma"
    return "other"


def color_for(model_name: str) -> str:
    """Return the palette color for the model's family, or grey if unknown."""
    return MODEL_FAMILY_COLORS.get(family_of(model_name), PALETTE["grey"])


def apply():
    """Apply the shared rcParams. Call once at the top of a plot script."""
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
