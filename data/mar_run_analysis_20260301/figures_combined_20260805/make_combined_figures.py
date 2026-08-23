"""
Prototype revised figures for B12_asteroid paper (2026-08-05):
  A1: 2x2 combining Figs 2/4/5/6 as grouped BARS, 30-yr impact-epoch bins
  A2: same 2x2 as LINES per diameter, decade resolution
  B:  IP distribution at X years before impact, 3 render styles
      (filled hist / step outline / frequency polygon)

Filters everywhere: status == complete AND prograde (Lz > 0) — retrograde
population from the impact-enforcement method excluded.
"""
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

D = "/Users/kathleenkiker/impacts_paper/adam_impacts_study/data/mar_run_analysis_20260301"
OUT = f"{D}/figures_combined_20260805"
DIAMS = [0.04, 0.08, 0.14, 0.25, 0.5, 1.0]
DLABELS = ["40 m", "80 m", "140 m", "250 m", "500 m", "1 km"]
COLORS = [cm.viridis(x) for x in np.linspace(0, 1, 6)]
MARKERS = ["o", "s", "D", "^", "v", "*"]
X_YEARS = 5.0  # "IP at X years before impact"

plt.rcParams.update({
    "figure.dpi": 200, "font.size": 9, "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.axisbelow": True,
})


def mjd_to_year(mjd):
    return 2000.0 + (mjd - 51544.5) / 365.25


# ---------------- load + flatten summary ----------------
t = pq.read_table(f"{D}/impactor_results_summary.parquet")
orbit = t.column("orbit")
coords = pc.struct_field(orbit, "coordinates")
x = pc.struct_field(coords, "x").to_numpy(zero_copy_only=False)
y = pc.struct_field(coords, "y").to_numpy(zero_copy_only=False)
vx = pc.struct_field(coords, "vx").to_numpy(zero_copy_only=False)
vy = pc.struct_field(coords, "vy").to_numpy(zero_copy_only=False)
lz = x * vy - y * vx
prograde = lz > 0

status = t.column("status").to_numpy(zero_copy_only=False)
complete = status == "complete"

orbit_id = pc.struct_field(orbit, "orbit_id").to_numpy(zero_copy_only=False)
diam = pc.struct_field(orbit, "diameter").to_numpy(zero_copy_only=False)
# true (enforced) impact time from the orbit definition — populated for all
# rows, unlike mean_impact_time which exists only where the IP pipeline ran
impact_mjd = pc.struct_field(pc.struct_field(orbit, "impact_time"), "days").to_numpy(zero_copy_only=False).astype(float)
disc_mjd = pc.struct_field(t.column("discovery_time"), "days").to_numpy(zero_copy_only=False)
ip1_mjd = pc.struct_field(t.column("ip_threshold_1_percent"), "days").to_numpy(zero_copy_only=False)
first_mjd = pc.struct_field(t.column("first_observation"), "days").to_numpy(zero_copy_only=False)
last_mjd = pc.struct_field(t.column("last_observation"), "days").to_numpy(zero_copy_only=False)
max_ip = t.column("maximum_impact_probability").to_numpy(zero_copy_only=False)

keep = complete & prograde & np.isfinite(impact_mjd)
print(f"rows: {len(t):,} | complete: {complete.sum():,} | prograde: {prograde.sum():,} | kept: {keep.sum():,}")

impact_year = mjd_to_year(impact_mjd)
discovered = np.isfinite(disc_mjd)
never_1pct = ~np.isfinite(ip1_mjd)
arc_days = last_mjd - first_mjd

# ---------------- panel statistics ----------------
BIN30 = [2025, 2055, 2085, 2115, 2125.01]
BIN30_LABELS = ["2025–2055", "2055–2085", "2085–2115", "2115–2125"]
DECADES = np.arange(2025, 2126, 10)
DEC_MID = (DECADES[:-1] + DECADES[1:]) / 2


def stats_for(bins):
    """per (diameter, epoch-bin): pct_discovered, mean max IP, pct never>=1%, mean arc"""
    nb = len(bins) - 1
    out = {k: np.full((6, nb), np.nan) for k in ("pct_disc", "mean_maxip", "pct_no1pct", "mean_arc")}
    for di, d in enumerate(DIAMS):
        md = keep & np.isclose(diam, d)
        for bi in range(nb):
            mb = md & (impact_year >= bins[bi]) & (impact_year < bins[bi + 1])
            n = mb.sum()
            if n == 0:
                continue
            mdisc = mb & discovered
            out["pct_disc"][di, bi] = 100.0 * mdisc.sum() / n
            if mdisc.sum():
                out["mean_maxip"][di, bi] = np.nanmean(max_ip[mdisc])
                out["pct_no1pct"][di, bi] = 100.0 * (mdisc & never_1pct).sum() / mdisc.sum()
                out["mean_arc"][di, bi] = np.nanmean(arc_days[mdisc])
    return out


PANELS = [
    ("pct_disc", "Discovered [%]", "(a) Percentage discovered"),
    ("mean_maxip", "Mean maximum IP", "(b) Mean maximum impact probability (discovered)"),
    ("pct_no1pct", "Never reach 1% IP [%]", "(c) Discovered objects never reaching IAWN 1% [%]"),
    ("mean_arc", "Mean arc length [days]", "(d) Mean arc length (discovered)"),
]


def fig_2x2_bars():
    s = stats_for(BIN30)
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4), constrained_layout=True)
    nb = len(BIN30_LABELS)
    xg = np.arange(nb)
    w = 0.8 / 6
    for ax, (key, ylab, title) in zip(axes.flat, PANELS):
        for di in range(6):
            ax.bar(xg + (di - 2.5) * w, s[key][di], width=w * 0.92,
                   color=COLORS[di], edgecolor="black", linewidth=0.4,
                   label=DLABELS[di])
        ax.set_xticks(xg, BIN30_LABELS)
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left")
        if key in ("pct_disc",):
            ax.set_ylim(0, 100)
        if key == "mean_maxip":
            ax.set_ylim(0, 1.0)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, ncols=6, loc="outside upper center", fontsize=8, frameon=False)
    fig.supxlabel("Impact epoch (30-year bins; last bin is the 2115–2125 decade)", fontsize=9)
    fig.savefig(f"{OUT}/figA1_combined_2x2_bars_30yr.png")
    plt.close(fig)


def fig_2x2_lines():
    s = stats_for(DECADES)
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4), constrained_layout=True)
    for ax, (key, ylab, title) in zip(axes.flat, PANELS):
        for di in range(6):
            ax.plot(DEC_MID, s[key][di], color=COLORS[di], lw=1.8,
                    marker=MARKERS[di], ms=4.5, markeredgecolor="0.15",
                    markeredgewidth=0.5, label=DLABELS[di])
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left")
        if key == "pct_disc":
            ax.set_ylim(0, 100)
        if key == "mean_maxip":
            ax.set_ylim(0, 1.02)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, ncols=6, loc="outside upper center", fontsize=8, frameon=False)
    fig.supxlabel("Impact decade", fontsize=9)
    fig.savefig(f"{OUT}/figA2_combined_2x2_lines_decade.png")
    plt.close(fig)


# ---------------- Fig B: IP distribution at X years before impact ----------------
def ip_at_years_before():
    w = pq.read_table(
        f"{D}/window_results.parquet",
        columns=["orbit_id", "observation_end", "impact_probability"],
    )
    wid = w.column("orbit_id").to_numpy(zero_copy_only=False)
    wend = pc.struct_field(w.column("observation_end"), "days").to_numpy(zero_copy_only=False)
    wip = w.column("impact_probability").to_numpy(zero_copy_only=False)

    sel = keep & discovered
    target = dict(zip(orbit_id[sel], impact_mjd[sel] - X_YEARS * 365.25))
    dmap = dict(zip(orbit_id[sel], diam[sel]))

    # latest window ending on/before target, per orbit
    best_end, best_ip = {}, {}
    for oid, e, p in zip(wid, wend, wip):
        tgt = target.get(oid)
        if tgt is None or not np.isfinite(e) or e > tgt or not np.isfinite(p):
            continue
        if oid not in best_end or e > best_end[oid]:
            best_end[oid] = e
            best_ip[oid] = p
    print(f"objects with an IP estimate {X_YEARS:.0f} yr before impact: {len(best_ip):,} "
          f"of {sel.sum():,} discovered")
    ips_by_d = {d: [] for d in DIAMS}
    for oid, p in best_ip.items():
        ips_by_d[dmap[oid]].append(p)
    return {d: np.array(v) for d, v in ips_by_d.items()}


def fig_ip_styles(ips_by_d):
    bins = np.linspace(0, 1, 21)
    centers = (bins[:-1] + bins[1:]) / 2
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharey=True, constrained_layout=True)

    for di, d in enumerate(DIAMS):
        v = ips_by_d[d]
        cnt, _ = np.histogram(v, bins=bins)
        # (1) filled overlapping histogram
        axes[0].hist(v, bins=bins, color=COLORS[di], alpha=0.45, label=DLABELS[di])
        # (2) step outline
        axes[1].hist(v, bins=bins, histtype="step", lw=1.8, color=COLORS[di], label=DLABELS[di])
        # (3) frequency polygon
        axes[2].plot(centers, cnt, color=COLORS[di], lw=1.8, marker=MARKERS[di],
                     ms=4, markeredgecolor="0.15", markeredgewidth=0.5, label=DLABELS[di])

    for ax, title in zip(axes, ["(a) filled histogram", "(b) step outline", "(c) frequency polygon"]):
        ax.set_title(title, loc="left")
        ax.set_xlabel(f"Impact probability {X_YEARS:.0f} years before impact")
    axes[0].set_ylabel("Number of impactors")
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, ncols=6, loc="outside upper center", fontsize=8, frameon=False)
    fig.savefig(f"{OUT}/figB_ip_dist_{int(X_YEARS)}yr_styles_linear.png")
    for ax in axes:
        ax.set_yscale("log")
    fig.savefig(f"{OUT}/figB_ip_dist_{int(X_YEARS)}yr_styles_logy.png")
    plt.close(fig)


fig_2x2_bars()
fig_2x2_lines()
fig_ip_styles(ip_at_years_before())
print("done ->", OUT)
