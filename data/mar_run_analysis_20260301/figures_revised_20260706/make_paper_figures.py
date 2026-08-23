"""Regenerate the revised paper figures (Figs 1-9 as numbered in the July-16
draft) from data/mar_run_analysis_20260301, using the plot functions in
src/adam_impact_study/analysis/plots.py (with the y-label / NaN-binning fixes).

APPLY_PROGRADE=True  -> the *_PROGRADE.png set (adopted cohort, i < 90 deg)
APPLY_PROGRADE=False -> the *_REVISED.png set (full cohort, retrogrades included)

Run with the repo venv:
    MPLCONFIGDIR=.mplconfig .venv/bin/python \
        data/mar_run_analysis_20260301/figures_revised_20260706/make_paper_figures.py
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/Users/kathleenkiker/impacts_paper/adam_impacts_study"
DATA = f"{REPO}/data/mar_run_analysis_20260301"
OUT = f"{DATA}/figures_revised_20260706"
sys.path.insert(0, f"{REPO}/src")

from adam_core.time import Timestamp
from adam_impact_study.types import ImpactorResultSummary
from adam_impact_study.analysis.plots import (
    plot_discovered_by_diameter_impact_period,
    plot_observed_vs_unobserved_elements,
    plot_max_impact_probability_by_diameter_decade,
    plot_iawn_threshold_not_reached_by_diameter_decade,
    plot_arc_length_by_diameter_decade,
    plot_warning_time_by_diameter_year,
    plot_warning_time_histogram,
)

APPLY_PROGRADE = True
TAG = "PROGRADE" if APPLY_PROGRADE else "REVISED"

summary = ImpactorResultSummary.from_parquet(f"{DATA}/summary_results.parquet")
if APPLY_PROGRADE:
    summary = summary.apply_mask(summary.prograde_orbits())
print(f"cohort: {len(summary)} rows ({TAG})")


def save(fig, name):
    fig.savefig(f"{OUT}/{name}_{TAG}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"{name}_{TAG}.png done")


# Fig 1 - discovery by 5-year impact window (paper uses the 2070 cut, 1-yr limit)
fig, _ = plot_discovered_by_diameter_impact_period(
    summary, period="5year",
    max_impact_time=Timestamp.from_iso8601(["2070-01-01"]),
    time_before_impact_limit_days=365)
save(fig, "fig1_discovered_by_diameter_5year_to2070")

fig, _ = plot_discovered_by_diameter_impact_period(
    summary, period="5year", time_before_impact_limit_days=365)
save(fig, "fig1_variant_full_range_to2125")

# Fig 2/3 - elements, 3 detection categories, 140 m (caption's blue/orange/red)
fig, _ = plot_observed_vs_unobserved_elements(summary, diameter=0.14)
save(fig, "fig2_elements_observed_vs_unobserved_140m")

# Fig 4 (old 3) - mean maximum IP by diameter and impact decade
fig, _ = plot_max_impact_probability_by_diameter_decade(summary)
save(fig, "fig3_max_impact_probability_by_diameter_decade")

# Fig 5 (old 4) - % of discovered not reaching 1% IAWN threshold
# (y-axis capped at 60% per review feedback; max value in data is ~55%)
fig, ax = plot_iawn_threshold_not_reached_by_diameter_decade(summary)
ax.set_ylim(0, 60)
ax.set_title(ax.get_title(), pad=14)          # keep title clear of the y-axis
if ax.get_legend() is not None:
    ax.get_legend().set_frame_on(True)        # boxed legend, consistent with fig 2
save(fig, "fig4_iawn_not_reached_by_diameter_decade")

# Fig 6 (old 5) - mean arc length by diameter and impact decade
fig, _ = plot_arc_length_by_diameter_decade(summary)
save(fig, "fig6_arc_length_by_diameter_decade")

# Fig 8 (old 7) - mean warning time by impact year and diameter
fig, _ = plot_warning_time_by_diameter_year(summary)
save(fig, "fig8_warning_time_by_diameter_year")

# Fig 9 (old 8) - warning-time histogram ("Count" label; crossed-1% cohort)
fig, _ = plot_warning_time_histogram(summary)
save(fig, "fig9_warning_time_histogram")
