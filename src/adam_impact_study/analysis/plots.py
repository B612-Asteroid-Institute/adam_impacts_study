import logging
import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
from adam_core.time import Timestamp

from adam_impact_study.types import (
    DiscoveryDates,
    ImpactorOrbits,
    ImpactorResultSummary,
    WindowResult,
)

logger = logging.getLogger(__name__)


def plot_warning_time_histogram(
    summary: ImpactorResultSummary,
) -> Tuple[plt.Figure, plt.Axes]:

    # Filter to only include complete results
    summary = summary.apply_mask(summary.complete())

    fig, ax = plt.subplots(1, 1, dpi=200)

    warning_time_max = (
        pc.ceil(
            pc.max(
                pc.subtract(
                    summary.orbit.impact_time.mjd(),
                    summary.ip_threshold_1_percent.mjd(),
                )
            )
        ).as_py()
        / 365.25
    )
    bins = np.arange(0, warning_time_max, 1)

    unique_diameters = summary.orbit.diameter.unique().sort().to_pylist()
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(unique_diameters)))
    for diameter, color in zip(unique_diameters, colors):

        orbits_at_diameter = summary.select("orbit.diameter", diameter)

        warning_time = pc.subtract(
            orbits_at_diameter.orbit.impact_time.mjd(),
            orbits_at_diameter.ip_threshold_1_percent.mjd(),
        ).to_numpy(zero_copy_only=False)

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

    # Filter to only include complete results
    summary = summary.apply_mask(summary.complete())

    fig, ax = plt.subplots(1, 1, dpi=200)

    realization_time_max = (
        pc.ceil(
            pc.max(
                pc.subtract(
                    summary.ip_threshold_0_dot_01_percent.mjd(),
                    summary.discovery_time.mjd(),
                )
            )
        ).as_py()
        / 365.25
    )
    if realization_time_max > 100:
        realization_time_max = 100

    # For the case where all values are 0, use a small range
    if realization_time_max == 0:
        bins = np.array([0, 0.1])  # Just two bins to show the spike at 0
    else:
        bins = np.linspace(0, realization_time_max, 100)

    unique_diameters = summary.orbit.diameter.unique().sort().to_pylist()
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(unique_diameters)))
    for diameter, color in zip(unique_diameters, colors):
        orbits_at_diameter = summary.select("orbit.diameter", diameter)
        diameter_realization_time = pc.subtract(
            orbits_at_diameter.ip_threshold_0_dot_01_percent.mjd(),
            orbits_at_diameter.discovery_time.mjd(),
        ).to_numpy(zero_copy_only=False)
        ax.hist(
            diameter_realization_time[~np.isnan(diameter_realization_time)],
            histtype="step",
            label=f"{diameter:.3f} km",
            color=color,
            bins=bins,
            density=True,
        )

    # Identify number of objects beyond 100 days
    realization_time = pc.subtract(
        summary.ip_threshold_0_dot_01_percent.mjd(), summary.discovery_time.mjd()
    ).to_numpy(zero_copy_only=False)
    n_objects_beyond_100_days = np.sum(realization_time > 100)

    if n_objects_beyond_100_days > 0:
        ax.text(
            99,
            0.01,
            rf"$N_{{objects}}$(>100 d)={n_objects_beyond_100_days}",
            ha="right",
            rotation=90,
        )

    ax.set_xlim(0, max(0.1, realization_time_max))  # Ensure x-axis shows some range
    ax.set_xlabel("Realization Time for Discoveries [days]")
    ax.set_ylabel("PDF")
    ax.legend(frameon=False, bbox_to_anchor=(1.01, 0.75))
    return fig, ax


def plot_discoveries_by_diameter(
    summary: ImpactorResultSummary,
) -> Tuple[plt.Figure, plt.Axes]:

    # Filter to only include complete results
    summary = summary.apply_mask(summary.complete())

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
    # Filter to only include complete results
    summary = summary.apply_mask(summary.complete())

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
            3600,  # Convert seconds to hours
        ).as_py()

        ax.bar(i, height=mean_runtime, color=color)
        ax.text(
            i,
            mean_runtime + (ax.get_ylim()[1] * 0.02),  # Position label 2% above bar
            f"{mean_runtime:.1f}h",
            ha="center",
            fontsize=10,
        )

    x_ticks = np.arange(0, len(unique_diameters), 1)
    x_tick_labels = [f"{diameter:.3f}" for diameter in unique_diameters]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)
    ax.set_xlabel("Diameter [km]")
    ax.set_ylabel("Average IP Runtime [hours]")

    # Add a title
    ax.set_title("IP Runtime by Object Diameter")

    return fig, ax


def plot_incomplete_by_diameter(
    summary: ImpactorResultSummary,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the number of incomplete results by diameter, with percentage labels.
    """
    fig, ax = plt.subplots(1, 1, dpi=200)

    incomplete_summary = summary.apply_mask(summary.incomplete())

    unique_diameters = summary.orbit.diameter.unique().sort().to_pylist()
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(unique_diameters)))

    for i, (diameter, color) in enumerate(zip(unique_diameters, colors)):
        orbits_at_diameter = summary.select("orbit.diameter", diameter)
        incomplete_orbits_at_diameter = incomplete_summary.select(
            "orbit.diameter", diameter
        )

        # Calculate raw count and percentage
        incomplete_count = len(incomplete_orbits_at_diameter)
        total_count = len(orbits_at_diameter)
        percentage = (incomplete_count / total_count * 100) if total_count > 0 else 0

        # Plot bar with raw count height
        ax.bar(i, height=incomplete_count, color=color)

        # Add percentage label above bar
        ax.text(
            i,
            incomplete_count + 0.5,  # Adjust the 0.5 offset as needed
            f"{incomplete_count}\n({percentage:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    x_ticks = np.arange(0, len(unique_diameters), 1)
    x_tick_labels = [f"{diameter:.3f}" for diameter in unique_diameters]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)
    ax.set_xlabel("Diameter [km]")
    ax.set_ylabel("Number of Incomplete Results")
    ax.set_title("Number of Incomplete Results by Object Diameter")

    return fig, ax


def plot_collective_ip_over_time(
    window_results: WindowResult,
) -> None:
    """
    Plot the impact probability (IP) over time for all objects in the provided orbits.
    """
    # We want to align it so the first window results is at the same location
    # on the x-axis
    fig, ax = plt.subplots(1, 1, dpi=200, figsize=(10, 6))

    completed_window_results = window_results.apply_mask(
        pc.equal(window_results.status, "complete")
    )

    # get the unique orbit_ids
    orbit_ids = completed_window_results.orbit_id.unique().to_pylist()

    # Calculate dynamic alpha value based on the number of orbits
    # Formula: alpha = min(0.3, 10/n) where n is the number of orbits
    # This ensures alpha decreases as number of orbits increases
    n_orbits = len(orbit_ids)
    alpha = min(0.1, 100 / max(1, n_orbits))

    # Use a single color with dynamic alpha for all plots
    plot_color = "steelblue"

    # Plot the IP for each orbit
    for orbit_id in orbit_ids:
        orbit_ips = completed_window_results.apply_mask(
            pc.equal(completed_window_results.orbit_id, orbit_id)
        )

        # Sort by observation end time
        mjd_times = orbit_ips.observation_end.mjd().to_numpy(zero_copy_only=False)
        probabilities = orbit_ips.impact_probability.to_numpy(zero_copy_only=False)
        sort_indices = mjd_times.argsort()
        mjd_times = mjd_times[sort_indices]
        probabilities = probabilities[sort_indices]

        # Adjust all times to be relative to the orbit's earliest observation_end
        earliest_observation_end = orbit_ips.observation_end.min().mjd()
        offset_times = mjd_times - earliest_observation_end

        # Plot the IP with shaded area using the same dynamic alpha for both line and fill
        ax.plot(
            offset_times, probabilities, color=plot_color, alpha=alpha, linewidth=0.8
        )
        # ax.fill_between(offset_times, 0, probabilities, color=plot_color, alpha=alpha)

    # Improve grid for better readability with many overlapping lines
    ax.grid(True, alpha=0.3, linestyle="--")

    # Add statistics including number of orbits and alpha value used
    ax.text(
        0.98,
        0.02,
        f"Total orbits: {n_orbits}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel("Days Since First Observation")
    ax.set_ylabel("Impact Probability")
    ax.set_title("Impact Probability Over Time for All Orbits")

    return fig, ax


def plot_individual_orbit_ip_over_time(
    impacting_orbits: ImpactorOrbits,
    window_results: WindowResult,
    out_dir: str,
    summary_results: Optional[ImpactorResultSummary] = None,
    survey_start: Optional[Timestamp] = None,
) -> None:
    """
    Plot the impact probability (IP) over time for each object in the provided orbits.

    Parameters
    ----------
    impacting_orbits : `~adam_impact_study.types.ImpactorOrbits`
        Table containing the impacting orbits. The impact time is the coordinates.time + 30 days
    window_results : `~adam_impact_study.types.WindowResult`
        Table containing the window results.
    out_dir : str
        Directory for this study run
    summary_results : `~adam_impact_study.types.ImpactorResultSummary`, optional
        Table containing summary results including discovery times
    survey_start : `~adam_core.time.Timestamp`, optional
        The start time of the survey. If provided, will add an x-axis showing days since survey start.

    Returns
    -------
    None
        This function does not return any value. It generates and displays plots for each object.
    """

    # Filter out objects with errors
    window_results = window_results.apply_mask(pc.is_null(window_results.error))

    # Filter out objects with incomplete status
    window_results = window_results.apply_mask(
        pc.equal(window_results.status, "complete")
    )

    orbit_ids = window_results.orbit_id.unique().to_pylist()

    # Store discovery times for quick lookup
    discovery_times = {}
    if summary_results is not None:
        summary_orbit_ids = summary_results.orbit.orbit_id.to_pylist()
        for i, orbit_id in enumerate(summary_orbit_ids):
            orbit_summary = summary_results.take([i])
            if orbit_summary.discovery_time is not None:
                discovery_times[orbit_id] = orbit_summary.discovery_time.mjd()[
                    0
                ].as_py()

    for orbit_id in orbit_ids:
        logger.info(f"Orbit ID Plotting: {orbit_id}")

        # Create figure with multiple x-axes
        fig, ax1 = plt.subplots()

        # Get data for this object
        ips = window_results.apply_mask(pc.equal(window_results.orbit_id, orbit_id))
        if len(ips) == 0:
            logger.warning(f"No complete results found for orbit {orbit_id}")
            continue

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

        # Now add discovery time marker IF AVAILABLE - MOVED TO END
        if orbit_id in discovery_times:
            discovery_time = discovery_times[orbit_id]
            if discovery_time is None:
                continue
            # Check if discovery time is within the plot range
            x_min, x_max = ax1.get_xlim()
            if discovery_time < x_min or discovery_time > x_max:
                # If not, expand the range slightly
                buffer = (x_max - x_min) * 0.05  # 5% buffer
                ax1.set_xlim(
                    min(x_min, discovery_time - buffer),
                    max(x_max, discovery_time + buffer),
                )

            # Draw a VERY visible vertical line
            ax1.axvline(
                x=discovery_time,
                color="#FF0000",  # Pure red
                linestyle="-",  # Solid line
                linewidth=1,  # Thick line
                zorder=100,  # Very high z-order
                label="Discovery",  # Add to legend
            )

            # Add visible text
            y_range = ax1.get_ylim()[1] - ax1.get_ylim()[0]
            ax1.text(
                discovery_time + ((x_max - x_min) * 0.02),  # Slight offset
                0.5,  # Middle of y-axis
                "DISCOVERY",
                color="red",
                fontsize=6,
                fontweight="bold",
                rotation=90,
                zorder=100,
            )

        fig.suptitle(orbit_id)
        plt.tight_layout()
        fig.savefig(
            os.path.join(out_dir, f"IP_{orbit_id}.png"),
            bbox_inches="tight",
        )

        plt.close()


class DiscoveryByDiameterDecade(qv.Table):
    diameter = qv.Float64Column()
    decade = qv.LargeStringColumn()
    percentage_discovered = qv.Float64Column()


def plot_discoveries_by_diameter_decade(
    summary: ImpactorResultSummary,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the percentage discovered broken down by diameter and impact decade.
    """
    # Filter to only include complete results
    summary = summary.apply_mask(summary.complete())

    impact_decades, unique_decades, unique_diameters = summary.get_diameter_decade_data()


    discovery_by_diameter_decade = DiscoveryByDiameterDecade.empty()

    # Plot each decade as a set of bars
    for decade in unique_decades:
        # Filter by impact decade
        orbits_at_decade = summary.apply_mask(impact_decades == decade)
        for diameter in unique_diameters:
            # Filter by diameter
            orbits_at_diameter_and_decade = orbits_at_decade.select(
                "orbit.diameter", diameter
            )

            discovery_mask = pc.invert(
                pc.is_null(orbits_at_diameter_and_decade.discovery_time.days)
            )
            discovered = pc.sum(discovery_mask).as_py()
            total = len(orbits_at_diameter_and_decade)

            percentage = (discovered / total * 100) if total > 0 else 0

            discovery_by_diameter_decade = qv.concatenate(
                [
                    discovery_by_diameter_decade,
                    DiscoveryByDiameterDecade.from_kwargs(
                        decade=[f"{decade}"],
                        diameter=[diameter],
                        percentage_discovered=[percentage],
                    ),
                ]
            )

    # Plot the data using a bar plot. We want the x-axis to be decade and y-axis to be percentage discovered
    # We want to plot each diameter as a separate bar
    width = 0.2
    x = np.arange(len(unique_decades))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_diameters)))
    fig, ax = plt.subplots(1, 1, dpi=200, figsize=(10, 6))
    for i, diameter in enumerate(unique_diameters):
        diameter_data = discovery_by_diameter_decade.select("diameter", diameter)
        ax.bar(
            x + i * width,
            diameter_data.percentage_discovered.to_numpy(zero_copy_only=False),
            width=width,
            color=colors[i],
        )

    ax.set_xticks(x + width * (len(unique_diameters) - 1) / 2)
    ax.set_xticklabels(unique_decades)
    ax.set_xlabel("Impact Decade")
    ax.set_ylabel("Percentage Discovered")
    ax.set_title("Percentage of Objects Discovered by Diameter and Impact Decade")
    ax.legend(
        unique_diameters,
        title="Diameter [km]",
        frameon=False,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    return fig, ax


class RealizationByDiameterDecade(qv.Table):
    diameter = qv.Float64Column()
    decade = qv.LargeStringColumn()
    percentage_realized = qv.Float64Column()


def plot_realizations_by_diameter_decade(
    summary: ImpactorResultSummary,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the percentage of objects that have been realized (reached a non-zero impact probability)
    broken down by diameter and impact decade.

    Parameters
    ----------
    summary : ImpactorResultSummary
        The summary of impact study results.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        The figure and axes objects for the plot.
    """
    # Filter to only include complete results (needed for the rest of the function)
    summary = summary.apply_mask(summary.complete())
    # Get common data
    impact_decades, unique_decades, unique_diameters = (
        summary.get_diameter_decade_data()
    )

    realization_by_diameter_decade = RealizationByDiameterDecade.empty()

    # Process each decade
    for decade in unique_decades:
        # Filter by impact decade
        orbits_at_decade = summary.apply_mask(impact_decades == decade)
        for diameter in unique_diameters:
            # Filter by diameter
            orbits_at_diameter_and_decade = orbits_at_decade.select(
                "orbit.diameter", diameter
            )

            # Count objects with non-null realization time (meaning they were realized)
            realization_mask = pc.invert(
                pc.is_null(orbits_at_diameter_and_decade.ip_threshold_0_dot_01_percent)
            )
            realized = pc.sum(realization_mask).as_py()
            total = len(orbits_at_diameter_and_decade)

            percentage = (realized / total * 100) if total > 0 else 0

            realization_by_diameter_decade = qv.concatenate(
                [
                    realization_by_diameter_decade,
                    RealizationByDiameterDecade.from_kwargs(
                        decade=[f"{decade}"],
                        diameter=[diameter],
                        percentage_realized=[percentage],
                    ),
                ]
            )

    # Create the plot
    width = 0.2
    x = np.arange(len(unique_decades))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_diameters)))
    fig, ax = plt.subplots(1, 1, dpi=200, figsize=(10, 6))

    for i, diameter in enumerate(unique_diameters):
        diameter_data = realization_by_diameter_decade.select("diameter", diameter)
        ax.bar(
            x + i * width,
            diameter_data.percentage_realized.to_numpy(zero_copy_only=False),
            width=width,
            color=colors[i],
        )

    ax.set_xticks(x + width * (len(unique_diameters) - 1) / 2)
    ax.set_xticklabels(unique_decades)
    ax.set_xlabel("Impact Decade")
    ax.set_ylabel("Percentage Realized")
    ax.set_title("Percentage of Objects Realized by Diameter and Impact Decade")
    ax.legend(
        unique_diameters,
        title="Diameter [km]",
        frameon=False,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)

    return fig, ax


class MaxImpactProbabilityByDiameterDecade(qv.Table):
    diameter = qv.Float64Column()
    decade = qv.LargeStringColumn()
    mean_max_impact_probability = qv.Float64Column()


def plot_max_impact_probability_by_diameter_decade(
    summary: ImpactorResultSummary,
    window_results: WindowResult,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the mean maximum impact probability reached during the survey,
    broken down by diameter and impact decade.

    Parameters
    ----------
    summary : ImpactorResultSummary
        The summary of impact study results.
    window_results : WindowResult
        The window results containing impact probabilities.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        The figure and axes objects for the plot.
    """
    # Get common data
    impact_decades, unique_decades, unique_diameters = (
        summary.get_diameter_decade_data()
    )

    # Filter to only include complete results (needed for the rest of the function)
    summary = summary.apply_mask(summary.complete())

    max_ip_by_diameter_decade = MaxImpactProbabilityByDiameterDecade.empty()

    # Process each decade
    for decade in unique_decades:
        # Filter by impact decade
        orbits_at_decade = summary.apply_mask(impact_decades == decade)
        for diameter in unique_diameters:
            # Filter by diameter
            orbits_at_diameter_and_decade = orbits_at_decade.select(
                "orbit.diameter", diameter
            )

            # Get orbit IDs for this diameter and decade
            orbit_ids = orbits_at_diameter_and_decade.orbit.orbit_id.to_pylist()

            # Calculate max impact probability for each orbit
            max_impact_probs = []
            for orbit_id in orbit_ids:
                # Filter window results for this orbit
                orbit_results = window_results.apply_mask(
                    pc.equal(window_results.orbit_id, orbit_id)
                )

                if len(orbit_results) > 0:
                    # Get maximum impact probability for this orbit
                    max_ip = pc.max(
                        pc.fill_null(orbit_results.impact_probability, 0)
                    ).as_py()
                    max_impact_probs.append(max_ip)

            # Calculate mean of maximum impact probabilities
            mean_max_ip = np.mean(max_impact_probs) if max_impact_probs else 0

            max_ip_by_diameter_decade = qv.concatenate(
                [
                    max_ip_by_diameter_decade,
                    MaxImpactProbabilityByDiameterDecade.from_kwargs(
                        decade=[f"{decade}"],
                        diameter=[diameter],
                        mean_max_impact_probability=[mean_max_ip],
                    ),
                ]
            )

    # Create the plot
    width = 0.2
    x = np.arange(len(unique_decades))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_diameters)))
    fig, ax = plt.subplots(1, 1, dpi=200, figsize=(10, 6))

    for i, diameter in enumerate(unique_diameters):
        diameter_data = max_ip_by_diameter_decade.select("diameter", diameter)
        ax.bar(
            x + i * width,
            diameter_data.mean_max_impact_probability.to_numpy(zero_copy_only=False),
            width=width,
            color=colors[i],
        )

    ax.set_xticks(x + width * (len(unique_diameters) - 1) / 2)
    ax.set_xticklabels(unique_decades)
    ax.set_xlabel("Impact Decade")
    ax.set_ylabel("Mean Maximum Impact Probability")
    ax.set_title("Mean Maximum Impact Probability by Diameter and Impact Decade")
    ax.legend(
        unique_diameters,
        title="Diameter [km]",
        frameon=False,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)

    return fig, ax


class IAWNThresholdByDiameterDecade(qv.Table):
    diameter = qv.Float64Column()
    decade = qv.LargeStringColumn()
    percentage_not_reaching_threshold = qv.Float64Column()


def plot_iawn_threshold_by_diameter_decade(
    summary: ImpactorResultSummary,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the percentage of objects that do not reach the IAWN threshold (1% impact probability)
    broken down by diameter and impact decade.

    Parameters
    ----------
    summary : ImpactorResultSummary
        The summary of impact study results.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        The figure and axes objects for the plot.
    """
    # Get common data
    impact_decades, unique_decades, unique_diameters = (
        summary.get_diameter_decade_data()
    )

    # Filter to only include complete results (needed for the rest of the function)
    summary = summary.apply_mask(summary.complete())

    iawn_by_diameter_decade = IAWNThresholdByDiameterDecade.empty()

    # Process each decade
    for decade in unique_decades:
        # Filter by impact decade
        orbits_at_decade = summary.apply_mask(impact_decades == decade)
        for diameter in unique_diameters:
            # Filter by diameter
            orbits_at_diameter_and_decade = orbits_at_decade.select(
                "orbit.diameter", diameter
            )

            # Count objects with null IAWN_time (meaning they never reached the threshold)
            iawn_null_mask = pc.is_null(orbits_at_diameter_and_decade.ip_threshold_1_percent)
            not_reaching_threshold = pc.sum(iawn_null_mask).as_py()
            total = len(orbits_at_diameter_and_decade)

            percentage = (not_reaching_threshold / total * 100) if total > 0 else 0

            iawn_by_diameter_decade = qv.concatenate(
                [
                    iawn_by_diameter_decade,
                    IAWNThresholdByDiameterDecade.from_kwargs(
                        decade=[f"{decade}"],
                        diameter=[diameter],
                        percentage_not_reaching_threshold=[percentage],
                    ),
                ]
            )

    # Create the plot
    width = 0.2
    x = np.arange(len(unique_decades))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_diameters)))
    fig, ax = plt.subplots(1, 1, dpi=200, figsize=(10, 6))

    for i, diameter in enumerate(unique_diameters):
        diameter_data = iawn_by_diameter_decade.select("diameter", diameter)
        ax.bar(
            x + i * width,
            diameter_data.percentage_not_reaching_threshold.to_numpy(
                zero_copy_only=False
            ),
            width=width,
            color=colors[i],
        )

    ax.set_xticks(x + width * (len(unique_diameters) - 1) / 2)
    ax.set_xticklabels(unique_decades)
    ax.set_xlabel("Impact Decade")
    ax.set_ylabel("Percentage Not Reaching IAWN Threshold")
    ax.set_title(
        "Percentage of Objects Not Reaching IAWN Threshold (1%) by Diameter and Impact Decade"
    )
    ax.legend(
        unique_diameters,
        title="Diameter [km]",
        frameon=False,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)

    return fig, ax



def make_analysis_plots(
    summary: ImpactorResultSummary,
    window_results: WindowResult,
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

    fig, ax = plot_incomplete_by_diameter(summary)
    fig.savefig(
        os.path.join(out_dir, "incomplete_by_diameter.jpg"),
        bbox_inches="tight",
        dpi=200,
    )
    logger.info("Generated incomplete by diameter plot")
    plt.close(fig)

    fig, ax = plot_discoveries_by_diameter_decade(summary)
    fig.savefig(
        os.path.join(out_dir, "percentage_discovered.jpg"),
        bbox_inches="tight",
        dpi=200,
    )
    logger.info("Generated percentage discovered plot")
    plt.close(fig)

    fig, ax = plot_realizations_by_diameter_decade(summary)
    fig.savefig(
        os.path.join(out_dir, "percentage_realized.jpg"),
        bbox_inches="tight",
        dpi=200,
    )
    logger.info("Generated percentage realized plot")
    plt.close(fig)

    fig, ax = plot_max_impact_probability_by_diameter_decade(summary, window_results)
    fig.savefig(
        os.path.join(out_dir, "max_impact_probability.jpg"),
        bbox_inches="tight",
        dpi=200,
    )
    logger.info("Generated max impact probability plot")
    plt.close(fig)

    fig, ax = plot_iawn_threshold_by_diameter_decade(summary)
    fig.savefig(
        os.path.join(out_dir, "iawn_threshold.jpg"),
        bbox_inches="tight",
        dpi=200,
    )
    logger.info("Generated iawn threshold plot")
    plt.close(fig)

    fig, ax = plot_collective_ip_over_time(window_results)
    fig.savefig(
        os.path.join(out_dir, "collective_ip_over_time.jpg"),
        bbox_inches="tight",
        dpi=200,
    )
    logger.info("Generated collective IP over time plot")
    plt.close(fig)
