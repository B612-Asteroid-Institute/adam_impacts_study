"""Distribution of impact probabilities 5.0 years before impact (paper figure
remake, prograde + complete cohort). For each object with at least one IP
window at or before (impact_time - 5 yr), the latest IP estimate at that epoch
is used. Zeros (no impacting variants yet) are included in the first bin.

Needs window_results.parquet (72 MB, not committed to git).

Run with the repo venv:
    MPLCONFIGDIR=.mplconfig .venv/bin/python \
        data/mar_run_analysis_20260301/figures_revised_20260706/make_ip_hist_5yr.py
"""
import sys
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/Users/kathleenkiker/impacts_paper/adam_impacts_study"
DATA = f"{REPO}/data/mar_run_analysis_20260301"
OUT = f"{DATA}/figures_revised_20260706"
sys.path.insert(0, f"{REPO}/src")
from adam_impact_study.types import ImpactorResultSummary

LEAD_YR = 5.0
BINS = np.arange(0, 1.0001, 0.1)

s = ImpactorResultSummary.from_parquet(f"{DATA}/summary_results.parquet")
s = s.apply_mask(s.prograde_orbits())
s = s.apply_mask(s.complete())
oid = np.array(s.orbit.orbit_id.to_pylist())
imp_mjd = s.orbit.impact_time.mjd().to_numpy(zero_copy_only=False)

wr = pq.read_table(f"{DATA}/window_results.parquet",
                   columns=["orbit_id", "observation_end", "impact_probability"])
wdf = pd.DataFrame({
    "orbit_id": wr["orbit_id"].to_pandas(),
    "end_days": pc.struct_field(wr["observation_end"], "days").to_pandas().astype(float),
    "ip": wr["impact_probability"].to_pandas(),
}).dropna(subset=["end_days"])
imp_map = pd.Series(imp_mjd, index=oid)
wdf = wdf[wdf.orbit_id.isin(imp_map.index)]
wdf["lead_yr"] = (wdf.orbit_id.map(imp_map) - wdf.end_days) / 365.25
wdf = wdf.sort_values(["orbit_id", "end_days"])

ip5 = wdf[wdf.lead_yr >= LEAD_YR].groupby("orbit_id").last().ip.dropna()
print(f"objects with an IP estimate >= {LEAD_YR} yr before impact: {len(ip5):,} "
      f"(of which IP = 0: {(ip5 == 0).sum():,})")

fig, ax = plt.subplots(1, 1, figsize=(11, 6), dpi=200)
ax.hist(ip5, bins=BINS, color=plt.cm.viridis(0.35), edgecolor="black", lw=0.5)
ax.set_xlabel("Impact Probability")
ax.set_ylabel("Number of Objects")
ax.set_title(f"Distribution of Impact Probabilities ({LEAD_YR} Years Before Impact)",
             pad=12)
# ax.set_yscale("log")            # log-y makes the middle of the distribution visible
ax.set_xlim(-0.02, 1.02)
ax.yaxis.grid(True, linestyle="--", alpha=0.7)
ax.annotate(f"n = {len(ip5):,} objects", xy=(0.4, 0.92), xycoords="axes fraction",
            fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_ip_hist_5yr_PROGRADE.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("fig_ip_hist_5yr_PROGRADE.png done")
