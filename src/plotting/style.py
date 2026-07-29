"""Shared figure style for the project's plots.

Warm terracotta, slate, and sage palette; serif typography; vector-friendly PDF embedding.
Import + call apply() at the top of any plot script for consistent styling
across the project.
"""
import matplotlib.pyplot as plt


# Warm terracotta, slate, and sage palette.
PALETTE = {
    "terracotta":      "#CC785C",   # primary terracotta
    "terracotta_dark": "#A8533B",
    "terracotta_lite": "#E5A584",
    "slate":      "#3E4A52",   # dark slate/charcoal
    "slate_lite": "#7A8A93",
    "sage":       "#4A6B5C",   # dark muted sage
    "sage_lite":  "#8FA985",
    "grey":       "#6B655E",   # warm neutral grey
}

# Stable per-family color assignment for model-comparison plots.
MODEL_FAMILY_COLORS = {
    "qwen":   PALETTE["terracotta"],
    "llama":  PALETTE["slate"],
    "gemma":  PALETTE["sage"],
}


def family_of(model_name: str) -> str:
    m = model_name.lower()
    if "qwen" in m:  return "qwen"
    if "llama" in m: return "llama"
    if "gemma" in m: return "gemma"
    return "other"


def color_for(model_name: str) -> str:
    return MODEL_FAMILY_COLORS.get(family_of(model_name), PALETTE["grey"])


def apply():
    """Apply the shared rcParams. Call once at the top of a plot script."""
    plt.rcParams.update({
        "font.family":     "serif",
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
