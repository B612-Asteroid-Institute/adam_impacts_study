"""Vertically stacked shared-x line panels (prograde cohort, decade bins):

  fig_stacked3_disc_obs_iawn_PROGRADE.png
      1. % discovered            (of all objects)
      2. % observed but not discovered (of all objects)
      3. % of discovered not reaching the 1% IAWN threshold

  fig_stacked2_iawn_obs_PROGRADE.png
      1. % of discovered not reaching the 1% IAWN threshold
      2. % observed but not discovered

Run with the repo venv:
    MPLCONFIGDIR=.mplconfig .venv/bin/python \
        data/mar_run_analysis_20260301/figures_revised_20260706/make_stacked_panels.py
"""
import sys
import numpy as np
import pyarrow.compute as pc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/Users/kathleenkiker/impacts_paper/adam_impacts_study"
DATA = f"{REPO}/data/mar_run_analysis_20260301"
OUT = f"{DATA}/figures_revised_20260706"
sys.path.insert(0, f"{REPO}/src")
from adam_impact_study.types import ImpactorResultSummary

s = ImpactorResultSummary.from_parquet(f"{DATA}/summary_results.parquet")
s = s.apply_mask(s.prograde_orbits())
s = s.apply_mask(s.complete())

diam = s.orbit.diameter.to_numpy(zero_copy_only=False)
iy = np.array([t.datetime.year for t in s.orbit.impact_time.to_astropy()])
disc = ~pc.is_null(s.discovery_time.mjd()).to_numpy(zero_copy_only=False)
reached1 = ~pc.is_null(s.ip_threshold_1_percent.mjd()).to_numpy(zero_copy_only=False)
obs = s.observations.to_numpy(zero_copy_only=False)

dec = (iy // 10) * 10
uniq = sorted(set(dec))
labels = [str(d) for d in uniq]
period = np.array([uniq.index(d) for d in dec])
NP = len(labels)
xs = np.arange(NP)
diams = sorted(set(diam))
colors = plt.cm.viridis(np.linspace(0, 1, len(diams)))

# per-diameter series
P_DISC, P_OBS, P_IAWN = {}, {}, {}
for d in diams:
    m = diam == d
    tot = np.array([(m & (period == p)).sum() for p in range(NP)], float)
    P_DISC[d] = np.array([(m & (period == p) & disc).sum() for p in range(NP)]) / tot * 100
    P_OBS[d] = np.array([(m & (period == p) & ~disc & (obs > 0)).sum()
                         for p in range(NP)]) / tot * 100
    P_IAWN[d] = np.array([(~reached1[m & (period == p) & disc]).mean() * 100
                          for p in range(NP)])

PANELS = {
    "disc": ("Percentage of Objects Discovered", P_DISC, (0, 100)),
    "obs": ("Percentage of Objects Observed But Not Discovered", P_OBS, (0, 40)),
    "iawn": ("Percentage of Discovered Objects Not Reaching IAWN Threshold (1%)",
             P_IAWN, (0, 60)),
}


def stacked(panel_keys, fname, height_per_panel=3.2):
    n = len(panel_keys)
    fig, axes = plt.subplots(n, 1, figsize=(11, height_per_panel * n), dpi=200,
                             sharex=True)
    axes = np.atleast_1d(axes)
    for ax, key in zip(axes, panel_keys):
        title, series, ylim = PANELS[key]
        for d, c in zip(diams, colors):
            ax.plot(xs, series[d], "-o", color=c, ms=4, lw=1.8, label=f"{d:.3f} km")
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Percentage")
        ax.set_ylim(*ylim)
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    axes[-1].set_xticks(xs)
    axes[-1].set_xticklabels(labels)
    axes[-1].set_xlabel("Impact Decade")
    handles, hlabels = axes[0].get_legend_handles_labels()
    fig.legend(handles, hlabels, title="Diameter [km]", loc="center left",
               bbox_to_anchor=(0.92, 0.5), frameon=True)
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    fig.savefig(f"{OUT}/{fname}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"{fname} done")


stacked(["disc", "obs", "iawn"], "fig_stacked3_disc_obs_iawn_PROGRADE.png")
stacked(["iawn", "obs"], "fig_stacked2_iawn_obs_PROGRADE.png")

# each panel as its own standalone chart
def single(key, fname):
    title, series, ylim = PANELS[key]
    fig, ax = plt.subplots(1, 1, figsize=(11, 6), dpi=200)
    for d, c in zip(diams, colors):
        ax.plot(xs, series[d], "-o", color=c, ms=4, lw=1.8, label=f"{d:.3f} km")
    ax.set_title(title, pad=12)
    ax.set_ylabel("Percentage")
    ax.set_ylim(*ylim)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Impact Decade")
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.legend(title="Diameter [km]", frameon=True, bbox_to_anchor=(1.01, 1),
              loc="upper left")
    fig.tight_layout(rect=[0, 0, 0.87, 1])
    fig.savefig(f"{OUT}/{fname}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"{fname} done")

single("disc", "fig_line_discovered_PROGRADE.png")
single("obs", "fig_line_observed_not_discovered_PROGRADE.png")
single("iawn", "fig_line_iawn_not_reached_PROGRADE.png")
