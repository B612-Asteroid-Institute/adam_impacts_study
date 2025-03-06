import logging
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from adam_core.dynamics.impacts import CollisionConditions
from adam_core.time import Timestamp

from adam_impact_study.analysis.utils import collect_all_window_results
from adam_impact_study.types import ImpactorOrbits, ImpactorResultSummary, WindowResult
from adam_impact_study.utils import get_study_paths

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


def plot_runtime_by_diameter(
    summary: ImpactorResultSummary,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the average window runtime for each diameter.

    Parameters
    ----------
    summary : ImpactorResultSummary
        The summary of impact study results.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        The figure and axes objects for the plot.
    """
    # we want to filter out the orbits with no results_timing.orbit_id
    summary = summary.apply_mask(pc.invert(pc.is_null(summary.results_timing.orbit_id)))

    fig, ax = plt.subplots(1, 1, dpi=200)

    unique_diameters = summary.orbit.diameter.unique().sort().to_pylist()
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(unique_diameters)))

    for i, (diameter, color) in enumerate(zip(unique_diameters, colors)):
        # Filter to the results for this diameter
        orbits_at_diameter = summary.select("orbit.diameter", diameter)

        # Calculate mean runtime in minutes
        mean_runtime = pc.divide(
            pc.mean(orbits_at_diameter.results_timing.total_window_runtime),
            60,  # Convert seconds to minutes
        ).as_py()

        ax.bar(i, height=mean_runtime, color=color)
        ax.text(
            i,
            mean_runtime + (ax.get_ylim()[1] * 0.02),  # Position label 2% above bar
            f"{mean_runtime:.1f}m",
            ha="center",
            fontsize=10,
        )

    x_ticks = np.arange(0, len(unique_diameters), 1)
    x_tick_labels = [f"{diameter:.3f}" for diameter in unique_diameters]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)
    ax.set_xlabel("Diameter [km]")
    ax.set_ylabel("Average Window Runtime [minutes]")

    # Add a title
    ax.set_title("Window Runtime by Object Diameter")

    return fig, ax


def plot_ip_over_time(
    impacting_orbits: ImpactorOrbits,
    impact_study_results: WindowResult,
    run_dir: str,
    out_dir: str | None = None,
    survey_start: Timestamp | None = None,
) -> None:
    """
    Plot the impact probability (IP) over time for each object in the provided orbits.

    Parameters
    ----------
    impact_study_results : `~quiver.Table`
        Table containing the impact study results with columns 'object_id', 'day', and 'impact_probability
    run_dir : str
        Directory for this study run
    impacting_orbits : `~adam_core.orbits.Orbits`
        Table containing the impacting orbits. The impact time is the coordinates.time + 30 days
    survey_start : `~adam_core.time.Timestamp`, optional
        The start time of the survey. If provided, will add an x-axis showing days since survey start.

    Returns
    -------
    None
        This function does not return any value. It generates and displays plots for each object.
    """

    # Filter out objects with errors
    impact_study_results = impact_study_results.apply_mask(
        pc.is_null(impact_study_results.error)
    )
    orbit_ids = impact_study_results.orbit_id.unique().to_pylist()

    for orbit_id in orbit_ids:
        paths = get_study_paths(run_dir, orbit_id)
        orbit_dir = paths["orbit_base_dir"]
        logger.info(f"Orbit ID Plotting: {orbit_id}")

        # Create figure with multiple x-axes
        fig, ax1 = plt.subplots()

        # Get data for this object
        ips = impact_study_results.apply_mask(
            pc.equal(impact_study_results.orbit_id, orbit_id)
        )

        # Sort by observation end time
        mjd_times = ips.observation_end.mjd().to_numpy(zero_copy_only=False)
        probabilities = ips.impact_probability.to_numpy(zero_copy_only=False)
        sort_indices = mjd_times.argsort()
        mjd_times = mjd_times[sort_indices]
        probabilities = probabilities[sort_indices]

        # Plot sorted data on primary axis (MJD)
        ax1.set_xlabel("MJD")
        ax1.set_ylabel("Impact Probability")
        for condition in ips.condition_id.unique():
            results_at_condition = ips.select("condition_id", condition)
            ax1.plot(
                results_at_condition.observation_end.mjd(),
                results_at_condition.impact_probability,
                label=condition,
                lw=1,
            )
            ax1.scatter(
                results_at_condition.observation_end.mjd(),
                results_at_condition.impact_probability,
                label=condition,
            )

        # Create x-axis labels, 10 in total over the range of mjd_times (to the nearest integers)
        # make the number of labels dynamic for when we have less than 10 days of data
        num_x_ticks = min(10, len(mjd_times))
        x_ticks = np.linspace(mjd_times[0], mjd_times[-1], num_x_ticks)
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels([f"{tick:.0f}" for tick in x_ticks])

        # Set the y-axis range to be from 0 to 1.1
        ax1.set_ylim(0, 1.05)

        # Set the y-axis tick labels to stop at 1.0
        y_ticks = np.arange(0, 1.1, 0.1)
        ax1.set_yticks(y_ticks)
        ax1.set_yticklabels([f"{y_tick:.1f}" for y_tick in y_ticks])
        # Get impact time for this object
        impact_orbit = impacting_orbits.apply_mask(
            pc.equal(impacting_orbits.orbit_id, orbit_id)
        )
        if len(impact_orbit) > 0:
            impact_time = impact_orbit.impact_time.mjd()[0].as_py()

            # Add days until impact axis
            ax2 = ax1.twiny()
            days_until_impact = impact_time - mjd_times
            new_tick_locations = np.array([0, 0.25, 0.5, 0.75, 1])
            ax2.set_xlim(ax1.get_xlim())
            ax2.set_xticks(
                ax1.get_xlim()[0]
                + new_tick_locations * (ax1.get_xlim()[1] - ax1.get_xlim()[0])
            )
            ax2.set_xticklabels(
                [
                    f"{days:.1f}"
                    for days in (
                        days_until_impact[0]
                        + new_tick_locations
                        * (days_until_impact[-1] - days_until_impact[0])
                    )
                ]
            )
            ax2.set_xlabel("Days Until Impact")

        # Add days since survey start axis if survey_start is provided
        if survey_start is not None:
            ax3 = ax1.twiny()
            days_since_start = mjd_times - survey_start.mjd()

            # Adjust the offset for the top spine
            if len(impact_orbit) > 0:
                ax3.spines["top"].set_position(("axes", 1.15))

            new_tick_locations = np.array([0, 0.25, 0.5, 0.75, 1])
            ax3.set_xlim(ax1.get_xlim())
            ax3.set_xticks(
                ax1.get_xlim()[0]
                + new_tick_locations * (ax1.get_xlim()[1] - ax1.get_xlim()[0])
            )
            ax3.set_xticklabels(
                [
                    f"{days:.1f}"
                    for days in (
                        days_since_start[0]
                        + new_tick_locations
                        * (days_since_start[-1] - days_since_start[0])
                    )
                ]
            )
            ax3.set_xlabel("Days Since Survey Start")

        fig.suptitle(orbit_id)
        if out_dir is not None:
            fig.savefig(
                os.path.join(out_dir, f"IP_{orbit_id}.png"),
            )
        else:
            fig.savefig(
                os.path.join(orbit_dir, f"IP_{orbit_id}.png"),
            )
        plt.close()


def plot_all_ip_over_time(
    impacting_orbits: ImpactorOrbits,
    window_results: WindowResult,
    run_dir: str,
    out_dir: str | None = None,
) -> None:
    """Plot impact probability over time for all orbits in a run.

    Parameters
    ----------
    impacting_orbits : ImpactorOrbits
        The impactor orbits to plot
    window_results : WindowResult
        The window results to plot
    run_dir : str
        Directory containing the run results
    out_dir : str | None, optional
        Directory to save plots to. If None, plots will be saved in run_dir/plots/ip_over_time
    """
    try:
        plot_ip_over_time(
            impacting_orbits,
            window_results,
            run_dir,
            out_dir=out_dir,
        )
    except Exception as e:
        logger.error(f"Failed to plot results: {e}")


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

    fig, ax = plot_runtime_by_diameter(summary)
    fig.savefig(
        os.path.join(out_dir, "runtime_by_diameter.jpg"),
        bbox_inches="tight",
        dpi=200,
    )
    logger.info("Generated runtime by diameter plot")
    plt.close(fig)

    return
