import logging
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from adam_impact_study.types import ImpactorResultSummary

logger = logging.getLogger(__name__)


def plot_warning_time_histogram(
    summary: ImpactorResultSummary,
) -> Tuple[plt.Figure, plt.Axes]:

    fig, ax = plt.subplots(1, 1, dpi=200)

    warning_time_max = pc.ceil(pc.max(summary.warning_time)).as_py() / 365.25
    bins = np.arange(0, warning_time_max, 1)

    unique_diameters = summary.orbit.diameter.unique().sort().to_pylist()
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(unique_diameters)))
    for diameter, color in zip(unique_diameters, colors):

        orbits_at_diameter = summary.select("orbit.diameter", diameter)

        warning_time = orbits_at_diameter.warning_time.to_numpy(zero_copy_only=False)

        ax.hist(
            np.where(np.isnan(warning_time), 0, warning_time) / 365.25,
            histtype="step",
            label=f"{diameter:.3f} km",
            color=color,
            bins=bins,
            density=True,
        )

    ax.set_xlim(0, warning_time_max)
    ax.set_xticks(np.arange(0, warning_time_max + 20, 20))
    ax.legend(frameon=False, bbox_to_anchor=(1.01, 0.75))
    ax.set_xlabel("Warning Time for Discoveries [years]")
    ax.set_ylabel("PDF")

    return fig, ax


def plot_realization_time_histogram(
    summary: ImpactorResultSummary,
) -> Tuple[plt.Figure, plt.Axes]:

    fig, ax = plt.subplots(1, 1, dpi=200)

    realization_time_max = pc.ceil(pc.max(summary.realization_time)).as_py()
    if realization_time_max > 100:
        realization_time_max = 100
    bins = np.linspace(0, realization_time_max, 100)

    unique_diameters = summary.orbit.diameter.unique().sort().to_pylist()
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(unique_diameters)))
    for diameter, color in zip(unique_diameters, colors):

        orbits_at_diameter = summary.select("orbit.diameter", diameter)

        realization_time = orbits_at_diameter.realization_time.to_numpy(
            zero_copy_only=False
        )

        ax.hist(
            realization_time[~np.isnan(realization_time)],
            histtype="step",
            label=f"{diameter:.3f} km",
            color=color,
            bins=bins,
            density=True,
        )

    # Identify number of objects beyond 100 days
    realization_time = summary.realization_time.to_numpy(zero_copy_only=False)
    n_objects_beyond_100_days = np.sum(realization_time > 100)

    ax.text(
        99,
        0.01,
        rf"$N_{{objects}}$(>100 d)={n_objects_beyond_100_days}",
        ha="right",
        rotation=90,
    )

    ax.set_xlim(0, realization_time_max)
    ax.set_xlabel("Realization Time for Discoveries [days]")
    ax.set_ylabel("PDF")
    ax.legend(frameon=False, bbox_to_anchor=(1.01, 0.75))
    return fig, ax


def plot_discoveries_by_diameter(
    summary: ImpactorResultSummary,
) -> Tuple[plt.Figure, plt.Axes]:

    # Calculate the discovery summary
    discovery_summary = summary.summarize_discoveries()

    fig, ax = plt.subplots(1, 1, dpi=200)

    unique_diameters = discovery_summary.diameter.unique().sort().to_pylist()
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(unique_diameters)))
    for i, (diameter, color) in enumerate(zip(unique_diameters, colors)):

        # Filter to the results for this diameter
        discoveries_at_diameter = discovery_summary.select("diameter", diameter)
        percent_discovered = pc.multiply(
            pc.divide(
                pc.cast(pc.sum(discoveries_at_diameter.discovered), pa.float64()),
                pc.cast(pc.sum(discoveries_at_diameter.total), pa.float64()),
            ),
            100,
        ).as_py()

        ax.bar(i, height=percent_discovered, color=color)
        ax.text(
            i,
            percent_discovered + 1,
            f"{percent_discovered:.2f}%",
            ha="center",
            fontsize=10,
        )

    x_ticks = np.arange(0, len(unique_diameters), 1)
    x_tick_labels = [f"{diameter:.3f}" for diameter in unique_diameters]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Diameter [km]")
    ax.set_ylabel("Discovered [%]")

    return fig, ax


def make_analysis_plots(
    summary: ImpactorResultSummary,
    out_dir: str,
) -> None:

    fig, ax = plot_warning_time_histogram(summary)
    fig.savefig(
        os.path.join(out_dir, "warning_time_histogram.jpg"),
        bbox_inches="tight",
        dpi=200,
    )
    logger.info("Generated warning time histogram")
    plt.close(fig)

    fig, ax = plot_realization_time_histogram(summary)
    fig.savefig(
        os.path.join(out_dir, "realization_time_histogram.jpg"),
        bbox_inches="tight",
        dpi=200,
    )
    logger.info("Generated realization time histogram")
    plt.close(fig)

    fig, ax = plot_discoveries_by_diameter(summary)
    fig.savefig(
        os.path.join(out_dir, "discoveries_by_diameter.jpg"),
        bbox_inches="tight",
        dpi=200,
    )
    logger.info("Generated discoveries by diameter plot")
    plt.close(fig)

    return
