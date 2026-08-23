"""Four candidate designs for the maximum-IP figure (prograde, complete,
discovered objects). The distribution is bimodal (certify at ~1 or stall
near 0), so each design shows the OUTCOME MIX rather than a single summary:

  A. ipalt_A_outcome_stack : small multiples per diameter; stacked fraction of
     discovered objects per decade in three bands (certified >=90%, elevated
     1-90%, never reaches 1%)
  B. ipalt_B_certified_lines : one panel; %% of discovered reaching >=90%
     (solid) and >=1% (dashed) vs decade, per diameter
  C. ipalt_C_violins : raw max-IP distributions as violins, three
     representative diameters per decade
  D. ipalt_D_heatmap : diameter x decade grid, cell = %% certified (>=90%)

Run with the repo venv:
    MPLCONFIGDIR=.mplconfig .venv/bin/python \
        data/mar_run_analysis_20260301/figures_revised_20260706/make_ip_alternatives.py
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
max_ip = np.nan_to_num(s.maximum_impact_probability.to_numpy(zero_copy_only=False), nan=0.0)

dec = (iy // 10) * 10
uniq = sorted(set(dec))
labels = [str(d) for d in uniq]
period = np.array([uniq.index(d) for d in dec])
NP = len(labels)
diams = sorted(set(diam))
vir = plt.cm.viridis(np.linspace(0, 1, len(diams)))

CERT, ELEV = 0.90, 0.01
frac_cert, frac_elev, frac_low = {}, {}, {}
for d in diams:
    md_ = (diam == d) & disc
    fc, fe, fl = [], [], []
    for p in range(NP):
        x = max_ip[md_ & (period == p)]
        fc.append((x >= CERT).mean() * 100)
        fe.append(((x >= ELEV) & (x < CERT)).mean() * 100)
        fl.append((x < ELEV).mean() * 100)
    frac_cert[d], frac_elev[d], frac_low[d] = map(np.array, (fc, fe, fl))

xg = np.arange(NP)

# ---------------- A: outcome-mix stacked bars, small multiples ----------------
c_cert, c_elev, c_low = "#21918c", "#f6b93b", "#d3d3d3"
fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), dpi=200, sharex=True, sharey=True)
for ax, d in zip(axes.flat, diams):
    ax.bar(xg, frac_cert[d], color=c_cert, width=0.8, label="certified (IP ≥ 90%)")
    ax.bar(xg, frac_elev[d], bottom=frac_cert[d], color=c_elev, width=0.8,
           label="elevated (1% ≤ IP < 90%)")
    ax.bar(xg, frac_low[d], bottom=frac_cert[d] + frac_elev[d], color=c_low,
           width=0.8, label="never reaches 1%")
    ax.set_title(f"{d*1000:.0f} m", fontsize=11)
    ax.set_xticks(xg[::2])
    ax.set_xticklabels(labels[::2])
    ax.set_ylim(0, 100)
for ax in axes[:, 0]:
    ax.set_ylabel("% of discovered objects")
for ax in axes[1]:
    ax.set_xlabel("Impact Decade")
handles, hl = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, hl, loc="lower center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Maximum Impact Probability Outcomes of Discovered Impactors", fontsize=13)
fig.tight_layout(rect=[0, 0.03, 1, 0.96])
fig.savefig(f"{OUT}/ipalt_A_outcome_stack_PROGRADE.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("A done")

# ---------------- B: certified / actionable lines ----------------
fig, ax = plt.subplots(1, 1, figsize=(11, 6), dpi=200)
for d, c in zip(diams, vir):
    ax.plot(xg, frac_cert[d], "-o", color=c, ms=4.5, lw=1.9, label=f"{d:.3f} km")
    ax.plot(xg, frac_cert[d] + frac_elev[d], "--", color=c, lw=1.3)
ax.set_xticks(xg)
ax.set_xticklabels(labels)
ax.set_xlabel("Impact Decade")
ax.set_ylabel("% of discovered objects")
ax.set_ylim(0, 102)
ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.set_title("Discovered Impactors Reaching IP ≥ 90% (solid) and IP ≥ 1% (dashed)")
ax.legend(title="Diameter [km]", frameon=True, bbox_to_anchor=(1.01, 1), loc="upper left")
fig.tight_layout(rect=[0, 0, 0.86, 1])
fig.savefig(f"{OUT}/ipalt_B_certified_lines_PROGRADE.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("B done")

# ---------------- C: violins, three representative diameters ----------------
sel_d = [0.04, 0.14, 1.0]
sel_c = [vir[0], vir[2], vir[5]]
off = [-0.28, 0.0, 0.28]
fig, ax = plt.subplots(1, 1, figsize=(13, 6), dpi=200)
for d, c, o in zip(sel_d, sel_c, off):
    md_ = (diam == d) & disc
    data = [max_ip[md_ & (period == p)] for p in range(NP)]
    vp = ax.violinplot(data, positions=xg + o, widths=0.26, showextrema=False)
    for b in vp["bodies"]:
        b.set_facecolor(c)
        b.set_alpha(0.75)
ax.set_xticks(xg)
ax.set_xticklabels(labels)
ax.set_xlabel("Impact Decade")
ax.set_ylabel("Maximum Impact Probability")
ax.set_ylim(-0.03, 1.05)
ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.set_title("Distribution of Maximum Impact Probability (discovered objects)")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=c, alpha=0.75,
                                 label=f"{d*1000:.0f} m") for d, c in zip(sel_d, sel_c)],
          frameon=True, loc="center left", bbox_to_anchor=(1.01, 0.5))
fig.tight_layout(rect=[0, 0, 0.9, 1])
fig.savefig(f"{OUT}/ipalt_C_violins_PROGRADE.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("C done")

# ---------------- D: heatmap of % certified ----------------
grid = np.array([frac_cert[d] for d in diams])
fig, ax = plt.subplots(1, 1, figsize=(11, 5), dpi=200)
im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0, vmax=100, origin="lower")
ax.set_xticks(xg)
ax.set_xticklabels(labels)
ax.set_yticks(range(len(diams)))
ax.set_yticklabels([f"{d*1000:.0f} m" for d in diams])
ax.set_xlabel("Impact Decade")
ax.set_ylabel("Diameter")
for r in range(len(diams)):
    for cix in range(NP):
        val = grid[r, cix]
        ax.text(cix, r, f"{val:.0f}", ha="center", va="center", fontsize=8.5,
                color="white" if val < 55 else "black")
ax.set_title("Percentage of Discovered Impactors Reaching IP ≥ 90%")
fig.colorbar(im, ax=ax, label="% certified")
fig.tight_layout()
fig.savefig(f"{OUT}/ipalt_D_heatmap_PROGRADE.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("D done")
