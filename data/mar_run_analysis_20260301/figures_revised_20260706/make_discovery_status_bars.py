"""Discovery-rate figure (per Braxton's 2026-08 simplification): per impact
decade and diameter, stacked percentage of objects
  - solid  : discovered
  - hollow : observed but not discovered
The IAWN-threshold color-coding lives in the separate IAWN figure.
Prograde + complete cohort.

Run with the repo venv:
    MPLCONFIGDIR=.mplconfig .venv/bin/python \
        data/mar_run_analysis_20260301/figures_revised_20260706/make_discovery_status_bars.py
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
diams = sorted(set(diam))
colors = plt.cm.viridis(np.linspace(0, 1, len(diams)))

GROUP_W, W_SCALE, GROUP_SPACING = 0.8, 0.9, 0.2
n_d = len(diams)
bar_w = GROUP_W / n_d * W_SCALE
xg = np.arange(NP) * (1 + GROUP_SPACING)

fig, ax = plt.subplots(1, 1, dpi=200, figsize=(12, 6))
for i, (d, c) in enumerate(zip(diams, colors)):
    off = (i - n_d / 2 + 0.5) * (bar_w * 1.1)
    md_ = diam == d
    tot = np.array([(md_ & (period == p)).sum() for p in range(NP)], float)
    solid = np.array([(md_ & (period == p) & disc).sum()
                      for p in range(NP)]) / tot * 100
    hollow = np.array([(md_ & (period == p) & ~disc & (obs > 0)).sum()
                       for p in range(NP)]) / tot * 100
    ax.bar(xg + off, solid, width=bar_w, color=c, alpha=1, label=f"{d:.3f} km")
    ax.bar(xg + off, hollow, bottom=solid, width=bar_w,
           edgecolor=c, facecolor="none", alpha=0.6)

ax.set_xticks(xg)
ax.set_xticklabels(labels)
ax.set_xlabel("Impact Decade")
ax.set_ylabel("Percentage")
ax.set_ylim(0, 100)
ax.set_xlim(min(xg) - 0.7, max(xg) + 0.7)
ax.yaxis.grid(True, linestyle="--", alpha=0.7)
ax.set_title("Percentage of Objects Discovered and Observed")

diameter_legend = ax.legend(title="Diameter [km]", frameon=True,
                            bbox_to_anchor=(1.01, 1), loc="upper left")
ax.add_artist(diameter_legend)
pattern_handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor="gray", alpha=1, label="Discovered"),
    plt.Rectangle((0, 0), 1, 1, edgecolor="gray", facecolor="none",
                  label="Observed But Not Discovered"),
]
ax.legend(handles=pattern_handles, frameon=True, bbox_to_anchor=(1.01, 0.45),
          loc="center left", fontsize=9)
plt.tight_layout(rect=[0, 0, 0.82, 1])
fig.savefig(f"{OUT}/fig_discovery_status_by_diameter_decade_PROGRADE.png",
            bbox_inches="tight", dpi=300)
plt.close(fig)
print("fig_discovery_status_by_diameter_decade_PROGRADE.png done")
