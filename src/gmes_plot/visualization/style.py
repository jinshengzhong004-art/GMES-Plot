from __future__ import annotations


def configure_matplotlib() -> None:
    """Configure publication-friendly defaults with Chinese font fallback."""
    import matplotlib

    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial", "DejaVu Sans"
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["savefig.dpi"] = 300
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["svg.fonttype"] = "none"

