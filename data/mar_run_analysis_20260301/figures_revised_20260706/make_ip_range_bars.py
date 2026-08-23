"""Standalone maximum-IP range figure: for each (diameter, impact decade) a
floating bar spans the 25th-75th percentile of maximum impact probability over
discovered objects (prograde, complete runs). No mean/median line - the bar
top and bottom ARE the range.

Degenerate ranges (Q1 = Q3, e.g. everything pinned at IP = 1) are drawn with a
minimum visible thickness.

Run with the repo venv:
    MPLCONFIGDIR=.mplconfig .venv/bin/python \
        data/mar_run_analysis_20260301/figures_revised_20260706/make_ip_range_bars.py
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
max_ip = s.maximum_impact_probability.to_numpy(zero_copy_only=False)

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
MIN_H = 0.012  # minimum visible bar thickness for degenerate ranges

fig, ax = plt.subplots(1, 1, dpi=200, figsize=(12, 6))
for i, (d, c) in enumerate(zip(diams, colors)):
    off = (i - n_d / 2 + 0.5) * (bar_w * 1.1)
    md_ = diam == d
    q25 = np.array([np.nanpercentile(max_ip[md_ & (period == p) & disc], 25)
                    for p in range(NP)])
    q75 = np.array([np.nanpercentile(max_ip[md_ & (period == p) & disc], 75)
                    for p in range(NP)])
    height = np.maximum(q75 - q25, MIN_H)
    bottom = np.minimum(q25, 1.0 - height)   # keep sliver bars inside [0, 1]
    ax.bar(xg + off, height, bottom=bottom, width=bar_w, color=c,
           label=f"{d:.3f} km")

ax.set_xticks(xg)
ax.set_xticklabels(labels)
ax.set_xlabel("Impact Decade")
ax.set_ylabel("Maximum Impact Probability")
ax.set_title("Range of Maximum Impact Probability by Diameter and Impact Decade\n"
             "(bars span the 25th-75th percentile of discovered objects)")
ax.set_ylim(0, 1.05)
ax.set_xlim(min(xg) - 0.7, max(xg) + 0.7)
ax.yaxis.grid(True, linestyle="--", alpha=0.7)
ax.legend(title="Diameter [km]", frameon=True, bbox_to_anchor=(1.01, 1),
          loc="upper left")
plt.tight_layout(rect=[0, 0, 0.87, 1])
fig.savefig(f"{OUT}/fig_ip_range_by_diameter_decade_PROGRADE.png",
            bbox_inches="tight", dpi=300)
plt.close(fig)
print("fig_ip_range_by_diameter_decade_PROGRADE.png done")
