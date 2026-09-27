"""Redraw the figures from the saved tables, without refitting anything.

    uv run --extra cpu python perceptual/replot.py outputs/perceptual

``scale_converted.py`` measures audio, fits probits and runs a 2000-draw
bootstrap; none of that changes when a marker size does.  Everything the figures
need is already on disk, so adjusting them is a read and a redraw.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from perceptual.scale_converted import make_figure


def line_from(table: pd.DataFrame, slope_prefix: str, offset_prefix: str
              ) -> tuple[float, float]:
    """The (offset, slope) pair a bootstrap table stores as two named rows."""
    def value(prefix):
        rows = table[table["quantity"].str.startswith(prefix)]
        if rows.empty:
            raise KeyError(
                f"no row starting {prefix!r} in the table; columns are "
                f"{list(table['quantity'])}")
        return float(rows["estimate"].iloc[0])
    return value(offset_prefix), value(slope_prefix)


def main(output_dir: Path) -> None:
    per_target = pd.read_csv(output_dir / "perceived_by_target.csv")
    per_stimulus = pd.read_csv(output_dir / "perceived_by_stimulus.csv")
    converted_line = line_from(
        pd.read_csv(output_dir / "perceived_linear.csv"), "slope", "offset")
    real_line = line_from(pd.read_csv(output_dir / "ruler.csv"), "slope", "offset")

    figure_dir = output_dir / "figures"
    make_figure(per_target, per_stimulus, converted_line, real_line,
                figure_dir / "perceived_vs_requested.png",
                figure_dir / "perceived_vs_requested_two_panel.png")
    # Untitled copies for the paper, where the caption carries the title.
    make_figure(per_target, per_stimulus, converted_line, real_line,
                figure_dir / "perceived_vs_requested_paper.pdf",
                figure_dir / "perceived_vs_requested_two_panel_paper.pdf",
                titled=False, small=True)
    print(f"conversions  offset {converted_line[0]:+.2f} dB, "
          f"slope {converted_line[1]:.3f}")
    print(f"real         offset {real_line[0]:+.2f} dB, slope {real_line[1]:.3f}")
    print(f"redrew both figures in {figure_dir}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "."))
