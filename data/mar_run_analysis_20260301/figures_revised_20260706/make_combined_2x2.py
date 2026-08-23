"""Paper Figs 2/4/5/6 combined into ONE 2x2 figure (prograde cohort, viridis
diameter colors). Configurable:
  BINNING  : "30yr" (2025-2054/2055-2084/2085-2114/2115-2125) or "decade"
  STYLES   : any of "bars", "step", "line"
  Panel B  : percentage observed but not discovered (IP panel moved to
             make_ip_range_bars.py per review feedback)

Run with the repo venv:
    MPLCONFIGDIR=.mplconfig .venv/bin/python \
        data/mar_run_analysis_20260301/figures_revised_20260706/make_combined_2x2.py
"""
import sys
import numpy as np
import pyarrow.compute as pc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BINNING = "decade"
STYLES = ["bars", "step", "line"]
PANEL_B_MODES = ["obs_not_disc"]

REPO = "/Users/kathleenkiker/impacts_paper/adam_impacts_study"
DATA = f"{REPO}/data/mar_run_analysis_20260301"
OUT = f"{DATA}/figures_revised_20260706"
sys.path.insert(0, f"{REPO}/src")
from adam_impact_study.types import ImpactorResultSummary

s = ImpactorResultSummary.from_parquet(f"{DATA}/summary_results.parquet")
s = s.apply_mask(s.prograde_orbits())
comp = s.apply_mask(s.complete())   # all four originals filter to complete internally

diam = comp.orbit.diameter.to_numpy(zero_copy_only=False)
iy = np.array([t.datetime.year for t in comp.orbit.impact_time.to_astropy()])
disc = ~pc.is_null(comp.discovery_time.mjd()).to_numpy(zero_copy_only=False)
reached1 = ~pc.is_null(comp.ip_threshold_1_percent.mjd()).to_numpy(zero_copy_only=False)
obs = comp.observations.to_numpy(zero_copy_only=False)
max_ip = comp.maximum_impact_probability.to_numpy(zero_copy_only=False)
arc0 = pc.fill_null(comp.arc_length(), 0).to_numpy(zero_copy_only=False)
lead_days = pc.subtract(comp.orbit.impact_time.mjd(),
                        comp.discovery_time.mjd()).to_numpy(zero_copy_only=False)

if BINNING == "30yr":
    edges_yr = [2025, 2055, 2085, 2115, 2126]
    labels = ["2025-2054", "2055-2084", "2085-2114", "2115-2125"]
    period = np.digitize(iy, edges_yr) - 1
else:  # decade
    dec = (iy // 10) * 10
    uniq = sorted(set(dec))
    labels = [str(d) for d in uniq]
    period = np.array([uniq.index(d) for d in dec])
NP = len(labels)
diams = sorted(set(diam))
colors = plt.cm.viridis(np.linspace(0, 1, len(diams)))
T_LIM = 365  # 1 yr before impact, as in the paper's Fig 2

# ---- precompute panel values per diameter ----
V = {}
for d in diams:
    md_ = diam == d
    tot = np.array([(md_ & (period == p)).sum() for p in range(NP)], float)
    early = disc & (lead_days >= T_LIM)
    a1 = np.array([(md_ & (period == p) & early & reached1).sum() for p in range(NP)]) / tot * 100
    a2 = np.array([(md_ & (period == p) & early & ~reached1).sum() for p in range(NP)]) / tot * 100
    a3 = np.array([(md_ & (period == p) & disc & (lead_days < T_LIM)).sum() for p in range(NP)]) / tot * 100
    a4 = np.array([(md_ & (period == p) & ~disc & (obs > 0)).sum() for p in range(NP)]) / tot * 100
    sel = [max_ip[md_ & (period == p) & disc] for p in range(NP)]
    V[d] = dict(
        a1=a1, a2=a2, a3=a3, a4=a4, disc_total=a1 + a2 + a3,
        B_mean=np.array([np.nanmean(x) for x in sel]),
        B_med=np.array([np.nanmedian(x) for x in sel]),
        B_q25=np.array([np.nanpercentile(x, 25) for x in sel]),
        B_q75=np.array([np.nanpercentile(x, 75) for x in sel]),
        C=np.array([(~reached1[md_ & (period == p) & disc]).mean() * 100 for p in range(NP)]),
        D=np.array([arc0[md_ & (period == p)].mean() for p in range(NP)]),
    )

# bar layout constants (as in plots.py)
GROUP_W, W_SCALE, GROUP_SPACING = 0.8, 0.9, 0.2
n_d = len(diams)
bar_w = GROUP_W / n_d * W_SCALE
xg = np.arange(NP) * (1 + GROUP_SPACING)          # bar group positions
step_edges = np.arange(NP + 1, dtype=float)       # contiguous bins for step/line
centers = step_edges[:-1] + 0.5

def render(style, panel_b):
    fig, axes = plt.subplots(2, 2, dpi=200, figsize=(16, 10))
    axA, axB, axC, axD = axes.flat

    for i, (d, c) in enumerate(zip(diams, colors)):
        v = V[d]
        if style == "bars":
            off = (i - n_d / 2 + 0.5) * (bar_w * 1.1)
            axA.bar(xg + off, v["disc_total"], width=bar_w, color=c, label=f"{d:.3f} km")
            for ax, key in ((axB, "a4"), (axC, "C"), (axD, "D")):
                ax.bar(xg + off, v[key], width=bar_w, color=c, label=f"{d:.3f} km")
        elif style == "step":
            for ax, key in ((axA, "disc_total"), (axB, "a4"), (axC, "C"), (axD, "D")):
                ax.stairs(v[key], step_edges, color=c, lw=1.8, label=f"{d:.3f} km")
        else:  # line
            for ax, key in ((axA, "disc_total"), (axB, "a4"), (axC, "C"), (axD, "D")):
                ax.plot(centers, v[key], "-o", color=c, ms=4, lw=1.8, label=f"{d:.3f} km")

    axA.set_title("Percentage of Discovered Objects")
    axA.set_ylabel("Percentage")
    axA.set_ylim(0, 100)
    axB.set_title("Percentage of Objects Observed But Not Discovered")
    axB.set_ylabel("Percentage")
    axB.set_ylim(0, 100)
    axC.set_title("Percentage of Discovered Objects Not Reaching IAWN Threshold (1%)")
    axC.set_ylabel("Percentage Not Reaching IAWN Threshold")
    axC.set_ylim(0, 100)
    axD.set_title("Mean Arc Length by Diameter and Impact Period")
    axD.set_ylabel("Mean Arc Length [days]")

    ticks = xg if style == "bars" else centers
    lim = (min(ticks) - 0.7, max(ticks) + 0.7) if style == "bars" else (0, NP)
    for ax in axes.flat:
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Impact Period")
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)
        ax.set_xlim(*lim)

    handles, hlabels = axB.get_legend_handles_labels()
    fig.legend(handles, hlabels, title="Diameter [km]", loc="center left",
               bbox_to_anchor=(0.92, 0.5), frameon=True)
    fig.tight_layout(rect=[0, 0, 0.91, 1])
    fname = f"{OUT}/fig_combined_2x2_{BINNING}_{style}_PROGRADE.png"
    fig.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"{fname.rsplit('/', 1)[-1]} done")

for style in STYLES:
    for panel_b in PANEL_B_MODES:
        render(style, panel_b)
