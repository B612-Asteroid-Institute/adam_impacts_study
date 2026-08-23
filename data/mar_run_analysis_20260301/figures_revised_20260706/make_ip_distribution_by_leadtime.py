"""IP distribution of synthetic impactors N years before impact, one image per
style (filled histogram / step histogram / line), log-y.

For each object (prograde + complete runs), takes the latest IP window at or
before (impact_time - N years). Objects with IP = 0 at that epoch are excluded
from the log bins (counts printed at the end).

Run with the repo venv:
    MPLCONFIGDIR=.mplconfig .venv/bin/python \
        data/mar_run_analysis_20260301/figures_revised_20260706/make_ip_distribution_by_leadtime.py
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

s = ImpactorResultSummary.from_parquet(f"{DATA}/summary_results.parquet")
prog = s.prograde_orbits().to_numpy(zero_copy_only=False)
comp = s.complete().to_numpy(zero_copy_only=False)
oid = np.array(s.orbit.orbit_id.to_pylist())
imp_mjd = s.orbit.impact_time.mjd().to_numpy(zero_copy_only=False)

wr = pq.read_table(f"{DATA}/window_results.parquet",
                   columns=["orbit_id", "observation_end", "impact_probability"])
wdf = pd.DataFrame({
    "orbit_id": wr["orbit_id"].to_pandas(),
    "end_days": pc.struct_field(wr["observation_end"], "days").to_pandas().astype(float),
    "ip": wr["impact_probability"].to_pandas(),
}).dropna(subset=["end_days"])

keep = prog & comp
imp_map = pd.Series(imp_mjd[keep], index=oid[keep])
wdf = wdf[wdf.orbit_id.isin(imp_map.index)]
wdf["impact_mjd"] = wdf.orbit_id.map(imp_map)
wdf["lead_yr"] = (wdf.impact_mjd - wdf.end_days) / 365.25
wdf = wdf.sort_values(["orbit_id", "end_days"])

lead_times = [1, 5, 10, 20]
lead_colors = plt.cm.plasma(np.linspace(0.05, 0.8, len(lead_times)))
bins = np.arange(-4.0, 0.01, 0.25)  # log10(IP)
centers = (bins[:-1] + bins[1:]) / 2

hists, ns, nz = {}, {}, {}
for xyr in lead_times:
    sel = wdf[wdf.lead_yr >= xyr].groupby("orbit_id").last()
    ip = sel.ip.dropna()
    pos = ip[ip > 0]
    h, _ = np.histogram(np.log10(np.clip(pos, 1e-4, 1.0)), bins=bins)
    hists[xyr], ns[xyr], nz[xyr] = h, len(pos), (ip == 0).sum()

styles = {"filled": "filled histogram", "step": "step histogram", "line": "line"}
for key, style in styles.items():
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.2), dpi=200)
    for xyr, c in zip(lead_times, lead_colors):
        lbl = f"{xyr} yr before impact (n={ns[xyr]:,})"
        if key == "filled":
            ax.hist(centers, bins=bins, weights=hists[xyr], alpha=0.35, color=c, label=lbl)
        elif key == "step":
            ax.hist(centers, bins=bins, weights=hists[xyr], histtype="step",
                    lw=1.8, color=c, label=lbl)
        else:
            ax.plot(centers, hists[xyr], "-", lw=1.8, color=c, label=lbl)
    ax.set_xlabel("log10(Impact Probability)")
    ax.set_ylabel("Number of Impactors")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_yscale("log")
    ax.set_ylim(0.7, None)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_title("IP Distribution of Synthetic Impactors N Years Before Impact", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_ip_distribution_by_leadtime_{key}_PROGRADE.png",
                bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"{style} image done")

for xyr in lead_times:
    print(f"lead {xyr:>2} yr: n={ns[xyr]:,} positive-IP objects, IP=0 at that epoch: {nz[xyr]:,}")
