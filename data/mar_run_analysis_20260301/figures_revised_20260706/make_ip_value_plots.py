"""Designs that put actual maximum-IP VALUES on the y-axis (log scale),
prograde + complete + discovered cohort:

  E. ipval_E_boxplot  : grouped box-and-whisker per decade x diameter
                        (box = IQR, line = median, whiskers = 5th-95th pct)
  F. ipval_F_strip    : jittered scatter of every object's max IP
                        (three representative diameters)
  G. ipval_G_envelope : small multiples per diameter; median line with
                        25-75 and 10-90 percentile bands

Objects whose IP never resolves above the Monte Carlo floor (max IP < 1e-4,
i.e. zero impacting variants) are drawn AT the floor; the dashed line marks
the 1/10,000-variant resolution limit.

Run with the repo venv:
    MPLCONFIGDIR=.mplconfig .venv/bin/python \
        data/mar_run_analysis_20260301/figures_revised_20260706/make_ip_value_plots.py
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

FLOOR = 1e-4          # MC resolution: 1 impacting variant of 10,000
DRAW_FLOOR = 6e-5     # where sub-floor objects are drawn
ip_c = np.where(max_ip < FLOOR, DRAW_FLOOR, max_ip)

dec = (iy // 10) * 10
uniq = sorted(set(dec))
labels = [str(d) for d in uniq]
period = np.array([uniq.index(d) for d in dec])
NP = len(labels)
diams = sorted(set(diam))
vir = plt.cm.viridis(np.linspace(0, 1, len(diams)))

GROUP_SPACING = 0.2
xg = np.arange(NP) * (1 + GROUP_SPACING)
n_d = len(diams)
box_w = 0.8 / n_d * 0.85


def style_log_axis(ax):
    ax.set_yscale("log")
    ax.set_ylim(4e-5, 1.6)
    ax.axhline(FLOOR, color="grey", lw=0.9, ls="--", zorder=0)
    ax.set_yticks([DRAW_FLOOR, 1e-4, 1e-3, 1e-2, 1e-1, 1])
    ax.set_yticklabels(["<10$^{-4}$", "10$^{-4}$", "10$^{-3}$", "10$^{-2}$",
                        "10$^{-1}$", "1"])
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)


# ---------------- E: grouped box plots ----------------
fig, ax = plt.subplots(1, 1, figsize=(13, 6.5), dpi=200)
for i, (d, c) in enumerate(zip(diams, vir)):
    off = (i - n_d / 2 + 0.5) * (box_w * 1.15)
    data = [ip_c[(diam == d) & disc & (period == p)] for p in range(NP)]
    bp = ax.boxplot(data, positions=xg + off, widths=box_w, whis=(5, 95),
                    showfliers=False, patch_artist=True,
                    medianprops=dict(color="black", lw=1.1),
                    boxprops=dict(facecolor=c, edgecolor="black", lw=0.5),
                    whiskerprops=dict(color=c, lw=1.1),
                    capprops=dict(color=c, lw=1.1))
ax.set_xticks(xg)
ax.set_xticklabels(labels)
ax.set_xlabel("Impact Decade")
ax.set_ylabel("Maximum Impact Probability")
style_log_axis(ax)
ax.set_xlim(min(xg) - 0.7, max(xg) + 0.7)
ax.set_title("Maximum Impact Probability of Discovered Impactors\n"
             "(box: 25th-75th percentile; line: median; whiskers: 5th-95th; "
             "dashed line: Monte Carlo resolution floor)")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="black",
                                 lw=0.5, label=f"{d:.3f} km")
                   for d, c in zip(diams, vir)],
          title="Diameter [km]", frameon=True, bbox_to_anchor=(1.01, 1),
          loc="upper left")
fig.tight_layout(rect=[0, 0, 0.88, 1])
fig.savefig(f"{OUT}/ipval_E_boxplot_PROGRADE.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("E done")

# ---------------- F: jittered strip plot ----------------
rng = np.random.default_rng(612)
sel_d = [0.04, 0.14, 1.0]
sel_c = [vir[0], vir[2], vir[5]]
off3 = [-0.32, 0.0, 0.32]
fig, ax = plt.subplots(1, 1, figsize=(13, 6.5), dpi=200)
for d, c, o in zip(sel_d, sel_c, off3):
    for p in range(NP):
        y = ip_c[(diam == d) & disc & (period == p)]
        x = xg[p] + o + rng.uniform(-0.10, 0.10, len(y))
        ax.scatter(x, y, s=5, color=c, alpha=0.25, linewidths=0,
                   label=f"{d*1000:.0f} m" if p == 0 else None)
ax.set_xticks(xg)
ax.set_xticklabels(labels)
ax.set_xlabel("Impact Decade")
ax.set_ylabel("Maximum Impact Probability")
style_log_axis(ax)
ax.set_xlim(min(xg) - 0.7, max(xg) + 0.7)
ax.set_title("Maximum Impact Probability of Every Discovered Impactor")
leg = ax.legend(title="Diameter", frameon=True, bbox_to_anchor=(1.01, 1),
                loc="upper left", markerscale=3)
for h in leg.legend_handles:
    h.set_alpha(1)
fig.tight_layout(rect=[0, 0, 0.9, 1])
fig.savefig(f"{OUT}/ipval_F_strip_PROGRADE.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("F done")

# ---------------- G: percentile-envelope small multiples ----------------
fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), dpi=200, sharex=True, sharey=True)
xs = np.arange(NP)
for ax, d, c in zip(axes.flat, diams, vir):
    md_ = (diam == d) & disc
    q = {k: np.array([np.nanpercentile(ip_c[md_ & (period == p)], k)
                      for p in range(NP)]) for k in (10, 25, 50, 75, 90)}
    ax.fill_between(xs, q[10], q[90], color=c, alpha=0.18, lw=0, label="10th-90th pct")
    ax.fill_between(xs, q[25], q[75], color=c, alpha=0.40, lw=0, label="25th-75th pct")
    ax.plot(xs, q[50], "-o", color="black", ms=3.5, lw=1.4, label="median")
    ax.set_title(f"{d*1000:.0f} m", fontsize=11)
    style_log_axis(ax)
    ax.set_xticks(xs[::2])
    ax.set_xticklabels(labels[::2])
for ax in axes[:, 0]:
    ax.set_ylabel("Maximum IP")
for ax in axes[1]:
    ax.set_xlabel("Impact Decade")
handles, hl = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, hl, loc="lower center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Maximum Impact Probability of Discovered Impactors "
             "(median and percentile bands)", fontsize=13)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig(f"{OUT}/ipval_G_envelope_PROGRADE.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("G done")
