"""Percentage of objects observed but not discovered, in three survey-relative
impact-date bins: first 5 survey years, second 5 survey years, after the survey.
Prograde + complete cohort. Survey start taken as 2025-11-01 (first impacts in
the population are 2025-11-03, matching the baseline_v4.3.1 start).

Run with the repo venv:
    MPLCONFIGDIR=.mplconfig .venv/bin/python \
        data/mar_run_analysis_20260301/figures_revised_20260706/make_obs_not_disc_survey_bins.py
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
from adam_core.time import Timestamp
from adam_impact_study.types import ImpactorResultSummary

STYLE = "bars"          # "bars" | "line"

s = ImpactorResultSummary.from_parquet(f"{DATA}/summary_results.parquet")
s = s.apply_mask(s.prograde_orbits())
s = s.apply_mask(s.complete())

diam = s.orbit.diameter.to_numpy(zero_copy_only=False)
disc = ~pc.is_null(s.discovery_time.mjd()).to_numpy(zero_copy_only=False)
obs = s.observations.to_numpy(zero_copy_only=False)
imp_mjd = s.orbit.impact_time.mjd().to_numpy(zero_copy_only=False)

start_mjd = Timestamp.from_iso8601(["2025-11-01"]).mjd()[0].as_py()
y5, y10 = start_mjd + 5 * 365.25, start_mjd + 10 * 365.25
period = np.digitize(imp_mjd, [y5, y10])   # 0: yrs 1-5, 1: yrs 6-10, 2: after survey
labels = ["Survey Years 1-5\n(2025-2030)", "Survey Years 6-10\n(2030-2035)",
          "After Survey\n(2035-2125)"]
NP = len(labels)

diams = sorted(set(diam))
colors = plt.cm.viridis(np.linspace(0, 1, len(diams)))
bar_w = 0.8 / len(diams) * 0.9
xg = np.arange(NP)

fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=200)
for i, (d, c) in enumerate(zip(diams, colors)):
    m = diam == d
    tot = np.array([(m & (period == p)).sum() for p in range(NP)], float)
    y = np.array([(m & (period == p) & ~disc & (obs > 0)).sum()
                  for p in range(NP)]) / tot * 100
    n_str = "/".join(f"{int(t):,}" for t in tot)
    if STYLE == "bars":
        off = (i - len(diams) / 2 + 0.5) * (bar_w * 1.1)
        ax.bar(xg + off, y, width=bar_w, color=c, label=f"{d:.3f} km")
    else:
        ax.plot(xg, y, "-o", color=c, ms=5, lw=1.8, label=f"{d:.3f} km")

ax.set_xticks(xg)
ax.set_xticklabels(labels)
ax.set_xlabel("Impact Date")
ax.set_ylabel("Percentage")
ax.set_ylim(0, 40)
ax.yaxis.grid(True, linestyle="--", alpha=0.7)
ax.set_title("Percentage of Objects Observed But Not Discovered", pad=12)
ax.legend(title="Diameter [km]", frameon=True, bbox_to_anchor=(1.01, 1),
          loc="upper left")
fig.tight_layout(rect=[0, 0, 0.86, 1])
fig.savefig(f"{OUT}/fig_obs_not_disc_survey_bins_PROGRADE.png",
            bbox_inches="tight", dpi=300)
plt.close(fig)
print("fig_obs_not_disc_survey_bins_PROGRADE.png done")

# bin populations for the caption
for p, lbl in enumerate(labels):
    print(f"{lbl.splitlines()[0]}: {(period == p).sum():,} objects")
