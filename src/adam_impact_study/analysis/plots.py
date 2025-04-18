import logging
import os
from typing import Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
import scipy.ndimage
from adam_core.time import Timestamp

from adam_impact_study.types import ImpactorOrbits, ImpactorResultSummary, WindowResult

logger = logging.getLogger(__name__)

BAR_GROUP_WIDTH = 0.9
BAR_WIDTH_SCALE = 0.9
BAR_GROUP_SPACING = 0.05


class WarningTimeByDiameterYear(qv.Table):
    diameter = qv.Float64Column()
    year = qv.Int64Column()
    mean_warning_time = qv.Float64Column()


def plot_warning_time_by_diameter_year(
    summary: ImpactorResultSummary,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Show the average warning time (days from 1% IP threshold to impact) for the first decade, grouped by year and diameter
    """
    summary = summary.apply_mask(summary.complete())
    summary = summary.apply_mask(summary.discovered())

    # Calculate the impact years for each row
    impact_year = np.array(
        [
            impact_time.datetime.year
            for impact_time in summary.orbit.impact_time.to_astropy()
        ]
    )

    # Take the earliest impact year and then filter
    # the include everything starting with that year for 10 years
    earliest_impact_year = np.min(impact_year)
    unique_diameters = summary.orbit.diameter.unique().sort().to_pylist()
    warning_time_by_diameter_year = WarningTimeByDiameterYear.empty()

    # Iterate through each year and diameter and calculate the average warning time
    for year in range(earliest_impact_year + 11, earliest_impact_year + 21):
        for diameter in unique_diameters:
            summary_at_year_and_diameter = summary.apply_mask(impact_year == year)
            summary_at_year_and_diameter = summary_at_year_and_diameter.select(
                "orbit.diameter", diameter
            )
            warning_time = summary_at_year_and_diameter.warning_time()
            mean_warning_time = pc.mean(warning_time).as_py()
            warning_time_by_diameter_year = qv.concatenate(
                [
                    warning_time_by_diameter_year,
                    WarningTimeByDiameterYear.from_kwargs(
                        diameter=[diameter],
                        year=[year],
                        mean_warning_time=[mean_warning_time],
                    ),
                ]
            )

    unique_years = warning_time_by_diameter_year.year.unique().sort().to_pylist()

    # Create figure with improved spacing
    fig, ax = plt.subplots(1, 1, dpi=200, figsize=(12, 6))

    # Calculate bar width and spacing based on number of diameters
    num_diameters = len(unique_diameters)
    group_width = BAR_GROUP_WIDTH  # Width allocated for each year group
    bar_width = (
        group_width / num_diameters * BAR_WIDTH_SCALE
    )  # Slightly narrower bars for spacing between them

    # Create evenly spaced x positions for year groups
    x = np.arange(len(unique_years)) * (
        1 + BAR_GROUP_SPACING
    )  # Add spacing between year groups

    # Define a colormap
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_diameters)))

    # Plot bars for each diameter within each year
    for i, diameter in enumerate(unique_diameters):
        # Calculate x position for this diameter's bars within each group
        offset = (i - num_diameters / 2 + 0.5) * (
            bar_width * 1.1
        )  # Add 10% spacing between bars

        y_values = []
        bar_positions = []

        for j, year in enumerate(unique_years):
            # Get data for this diameter and year
            data = warning_time_by_diameter_year.apply_mask(
                pc.and_(
                    pc.equal(warning_time_by_diameter_year.diameter, diameter),
                    pc.equal(warning_time_by_diameter_year.year, year),
                )
            )

            if len(data) > 0:
                y_values.append(data.mean_warning_time[0].as_py() / 365.25)
                bar_positions.append(x[j] + offset)
            else:
                y_values.append(0)
                bar_positions.append(x[j] + offset)

        # Plot the bars for this diameter across all years
        bars = ax.bar(
            bar_positions,
            y_values,
            width=bar_width,
            color=colors[i],
            label=f"{diameter:.3f} km",
            alpha=0.8,
        )

        # Add value labels above bars
        for bar, y_val in zip(bars, y_values):
            if y_val > 0:  # Only add labels for bars with data
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.1,  # 1 day offset
                    f"{y_val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                )

    # Position x-ticks at the center of each year group
    ax.set_xticks(x)
    ax.set_xticklabels(unique_years)

    # Add some padding to x-axis limits
    ax.set_xlim(min(x) - 0.5, max(x) + 0.5)

    # Add grid for better readability
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)

    # Set axis labels and title
    ax.set_xlabel("Impact Year")
    ax.set_ylabel("Mean Warning Time (years)")
    ax.set_title("Mean Warning Time by Impact Year and Diameter")

    # Move legend outside to save space
    ax.legend(
        title="Diameter [km]",
        frameon=False,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
    )

    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    return fig, ax


def plot_warning_time_histogram(
    summary: ImpactorResultSummary,
) -> Tuple[plt.Figure, plt.Axes]:

    # Filter to only include complete results
    summary = summary.apply_mask(summary.complete())
    summary = summary.apply_mask(pc.invert(pc.is_null(summary.discovery_time.days)))

    fig, ax = plt.subplots(1, 1, dpi=200)
    warning_time_max = pc.ceil(
        pc.max(
            pc.subtract(
                summary.orbit.impact_time.mjd(),
                summary.ip_threshold_1_percent.mjd(),
            )
        )
    )
    warning_time_max = warning_time_max.as_py() / 365.25
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


def plot_arclength_by_diameter(
    summary: ImpactorResultSummary,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the arc length of the orbits by diameter.
    """
    summary = summary.apply_mask(summary.complete())

    fig, ax = plt.subplots(1, 1, dpi=200)

    unique_diameters = summary.orbit.diameter.unique().sort().to_pylist()
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(unique_diameters)))

    # Get the max arc length
    max_arc_length = pc.max(summary.arc_length()).as_py()
    bins = np.linspace(0, max_arc_length, 100)

    for i, (diameter, color) in enumerate(zip(unique_diameters, colors)):
        orbits_at_diameter = summary.select("orbit.diameter", diameter)
        arc_length = orbits_at_diameter.arc_length().to_numpy(zero_copy_only=False)
        # Default nan to 0
        arc_length[np.isnan(arc_length)] = 0
        ax.hist(
            arc_length,
            bins=bins,
            color=color,
            alpha=0.7,
            label=f"{diameter:.3f} km",
        )

    ax.set_xlabel("Arc Length [days]")
    ax.set_ylabel("Count")
    ax.legend(frameon=False, bbox_to_anchor=(1.01, 0.75))

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

    # Add vertical lines for every year (365.25 days)
    max_days = ax.get_xlim()[1]
    year_lines = np.arange(0, max_days, 365.25)
    for year_line in year_lines:
        ax.axvline(x=year_line, color="gray", linestyle=":", alpha=0.3)

    # Add legend for the vertical year lines
    ax.plot([], [], color="gray", linestyle=":", alpha=0.3, label="Year")
    ax.plot([], [], color=plot_color, alpha=0.8, label="Individual Orbit IP")
    ax.legend(loc="upper right", framealpha=0.8)

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
    plot_discovery_time_optimistic: Optional[bool] = False,
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
    discovery_times_optimistic = {}
    if summary_results is not None:
        summary_orbit_ids = summary_results.orbit.orbit_id.to_pylist()
        for i, orbit_id in enumerate(summary_orbit_ids):
            orbit_summary = summary_results.take([i])
            if orbit_summary.discovery_time is not None:
                discovery_times[orbit_id] = orbit_summary.discovery_time.mjd()[
                    0
                ].as_py()
            if orbit_summary.discovery_time_optimistic is not None:
                discovery_times_optimistic[orbit_id] = (
                    orbit_summary.discovery_time_optimistic.mjd()[0].as_py()
                )

    for orbit_id in orbit_ids:
        logger.info(f"Orbit ID Plotting: {orbit_id}")

        # Create figure with multiple x-axes
        fig, ax1 = plt.subplots()

        # Get data for this object
        ips = window_results.apply_mask(pc.equal(window_results.orbit_id, orbit_id))
        if len(ips) == 0:
            logger.warning(f"No complete results found for orbit {orbit_id}")
            continue

        # Plot sorted data on primary axis (MJD)
        ax1.set_xlabel("MJD")
        ax1.set_ylabel("Impact Probability")
        for condition in ips.condition_id.unique():
            results_at_condition = ips.select("condition_id", condition)

            # Sort by observation end time
            mjd_times = results_at_condition.observation_end.mjd().to_numpy(
                zero_copy_only=False
            )
            probabilities = results_at_condition.impact_probability.to_numpy(
                zero_copy_only=False
            )
            sort_indices = mjd_times.argsort()
            mjd_times = mjd_times[sort_indices]
            probabilities = probabilities[sort_indices]

            ax1.plot(
                mjd_times,
                probabilities,
                label=condition,
                lw=1,
            )
            ax1.scatter(
                mjd_times,
                probabilities,
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

        if orbit_id in discovery_times_optimistic and plot_discovery_time_optimistic:
            discovery_time_optimistic = discovery_times_optimistic[orbit_id]
            if discovery_time_optimistic is None:
                continue
            # Check if discovery time is within the plot range
            x_min, x_max = ax1.get_xlim()
            if discovery_time_optimistic < x_min or discovery_time_optimistic > x_max:
                # If not, expand the range slightly
                buffer = (x_max - x_min) * 0.05  # 5% buffer
                ax1.set_xlim(
                    min(x_min, discovery_time_optimistic - buffer),
                    max(x_max, discovery_time_optimistic + buffer),
                )

            # Draw a VERY visible vertical line
            ax1.axvline(
                x=discovery_time_optimistic,
                color="#FF0000",  # Pure red
                linestyle="-",  # Solid line
                linewidth=1,  # Thick line
                zorder=100,  # Very high z-order
                label="Discovery (Optimistic)",  # Add to legend
            )

            # Add visible text
            ax1.text(
                discovery_time_optimistic + ((x_max - x_min) * 0.02),  # Slight offset
                0.5,  # Middle of y-axis
                "DISCOVERY (OPTIMISTIC)",
                color="green",
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
    num_discovered = qv.Int64Column()
    num_observed_not_discovered = qv.Int64Column()
    num_unobserved = qv.Int64Column()


def plot_discovered_by_diameter_decade(
    summary: ImpactorResultSummary,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the percentage discovered broken down by diameter and impact decade.
    """
    # Filter to only include complete results
    summary = summary.apply_mask(summary.complete())

    impact_decades, unique_decades, unique_diameters = (
        summary.get_diameter_impact_period_data(period_breakdown="decade")
    )

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

            discovered_mask = orbits_at_diameter_and_decade.discovered()
            num_discovered = pc.sum(discovered_mask).as_py()
            observed_not_discovered_mask = (
                orbits_at_diameter_and_decade.observed_but_not_discovered()
            )
            num_observed_not_discovered = pc.sum(observed_not_discovered_mask).as_py()
            unobserved_mask = pc.equal(orbits_at_diameter_and_decade.observations, 0)
            num_unobserved = pc.sum(unobserved_mask).as_py()

            assert num_discovered + num_observed_not_discovered + num_unobserved == len(
                orbits_at_diameter_and_decade
            )

            discovery_by_diameter_decade = qv.concatenate(
                [
                    discovery_by_diameter_decade,
                    DiscoveryByDiameterDecade.from_kwargs(
                        decade=[f"{decade}"],
                        diameter=[diameter],
                        num_discovered=[num_discovered],
                        num_observed_not_discovered=[num_observed_not_discovered],
                        num_unobserved=[num_unobserved],
                    ),
                ]
            )

    # Create the plot with improved spacing
    fig, ax = plt.subplots(
        1, 1, dpi=200, figsize=(12, 6)
    )  # Wider figure for better spacing

    # Calculate bar width and spacing based on number of diameters
    num_diameters = len(unique_diameters)
    group_width = BAR_GROUP_WIDTH  # Width allocated for each decade group (out of 1.0)
    bar_width = (
        group_width / num_diameters * BAR_WIDTH_SCALE
    )  # Slightly narrower bars for spacing between them

    # Create evenly spaced x positions for decade groups
    x = np.arange(len(unique_decades)) * (
        1 + BAR_GROUP_SPACING
    )  # Add 20% extra space between decade groups

    # Define a colormap
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_diameters)))

    # Plot bars for each diameter
    for i, diameter in enumerate(unique_diameters):
        diameter_data = discovery_by_diameter_decade.select("diameter", diameter)

        # Calculate x position for this diameter's bars within each group
        offset = (i - num_diameters / 2 + 0.5) * (
            bar_width * 1.1
        )  # Add 10% spacing between bars
        bar_positions = x + offset

        num_discovered = diameter_data.num_discovered.to_numpy(zero_copy_only=False)
        num_observed_not_discovered = (
            diameter_data.num_observed_not_discovered.to_numpy(zero_copy_only=False)
        )
        num_unobserved = diameter_data.num_unobserved.to_numpy(zero_copy_only=False)

        # Calculate percentages
        total = num_discovered + num_observed_not_discovered + num_unobserved

        pct_discovered = num_discovered / total * 100
        pct_observed_not_discovered = num_observed_not_discovered / total * 100
        pct_unobserved = num_unobserved / total * 100

        # Plot stacked bars
        ax.bar(
            bar_positions,
            pct_discovered,
            width=bar_width,
            color=colors[i],
            label=f"{diameter:.3f} km",
        )

        ax.bar(
            bar_positions,
            pct_observed_not_discovered,
            bottom=pct_discovered,
            width=bar_width,
            color=colors[i],
            alpha=0.5,
            hatch="///",
            label="_nolegend_",
        )

        ax.bar(
            bar_positions,
            pct_unobserved,
            bottom=pct_discovered + pct_observed_not_discovered,
            width=bar_width,
            color="none",
            edgecolor=colors[i],
            label="_nolegend_",
        )

    # Position x-ticks at the center of each decade group
    ax.set_xticks(x)
    ax.set_xticklabels(unique_decades)

    # Add some padding to x-axis limits
    ax.set_xlim(min(x) - 0.5, max(x) + 0.5)

    ax.set_xlabel("Impact Decade")
    ax.set_ylabel("Percentage Discovered")
    ax.set_title("Percentage of Objects Discovered by Diameter and Impact Decade")
    # Add legend entries for the hatch and empty bar patterns
    # Create a separate legend for the pattern types
    pattern_legend_elements = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor="gray",
            alpha=0.5,
            hatch="///",
            label="Observed but Not Discovered",
        ),
        plt.Rectangle(
            (0, 0), 1, 1, facecolor="white", edgecolor="gray", label="Unobserved"
        ),
    ]

    # Create two legends - one for diameters and one for patterns
    # Save the first legend as a variable so it doesn't get overwritten
    diameter_legend = ax.legend(
        title="Diameter [km]", frameon=True, bbox_to_anchor=(1.01, 1), loc="upper left"
    )

    # Add the first legend explicitly as an artist
    ax.add_artist(diameter_legend)

    # Create second legend
    pattern_legend = ax.legend(
        handles=pattern_legend_elements,
        frameon=True,
        bbox_to_anchor=(1.01, 0.5),
        loc="center left",
    )

    ax.yaxis.grid(True, linestyle="--", alpha=0.7)

    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    return fig, ax


class RealizationByDiameterDecade(qv.Table):
    diameter = qv.Float64Column()
    decade = qv.LargeStringColumn()
    percentage_not_realized = qv.Float64Column()


def plot_not_realized_by_diameter_decade(
    summary: ImpactorResultSummary,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the percentage of discovered objects that have NOT been realized
    (have not reached the 0.01% impact probability threshold) broken down by diameter and impact decade.

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
    summary = summary.apply_mask(summary.discovered())
    # Get common data
    impact_decades, unique_decades, unique_diameters = (
        summary.get_diameter_impact_period_data(period_breakdown="decade")
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

            # Count objects with null realization time or null discovery time
            # (meaning they were NOT realized despite being discovered)
            not_realized_mask = pc.is_null(
                orbits_at_diameter_and_decade.ip_threshold_0_dot_01_percent.mjd()
            )
            not_realized = pc.sum(not_realized_mask).as_py()

            total_at_diameter_and_decade = len(orbits_at_diameter_and_decade)

            percentage = (
                (not_realized / total_at_diameter_and_decade * 100)
                if total_at_diameter_and_decade > 0
                else 0
            )

            realization_by_diameter_decade = qv.concatenate(
                [
                    realization_by_diameter_decade,
                    RealizationByDiameterDecade.from_kwargs(
                        decade=[f"{decade}"],
                        diameter=[diameter],
                        percentage_not_realized=[percentage],
                    ),
                ]
            )

    # Create the plot with improved spacing
    fig, ax = plt.subplots(
        1, 1, dpi=200, figsize=(12, 6)
    )  # Wider figure for better spacing

    # Calculate bar width and spacing based on number of diameters
    num_diameters = len(unique_diameters)
    group_width = BAR_GROUP_WIDTH  # Width allocated for each decade group (out of 1.0)
    bar_width = (
        group_width / num_diameters * BAR_WIDTH_SCALE
    )  # Slightly narrower bars for spacing between them

    # Create evenly spaced x positions for decade groups
    x = np.arange(len(unique_decades)) * (
        1 + BAR_GROUP_SPACING
    )  # Add 20% extra space between decade groups

    # Define a colormap
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_diameters)))

    # Plot bars for each diameter
    for i, diameter in enumerate(unique_diameters):
        diameter_data = realization_by_diameter_decade.select("diameter", diameter)

        # Calculate x position for this diameter's bars within each group
        offset = (i - num_diameters / 2 + 0.5) * (
            bar_width * 1.1
        )  # Add 10% spacing between bars
        bar_positions = x + offset

        ax.bar(
            bar_positions,
            diameter_data.percentage_not_realized.to_numpy(zero_copy_only=False),
            width=bar_width,
            color=colors[i],
            label=f"{diameter:.3f} km",
        )

    # Position x-ticks at the center of each decade group
    ax.set_xticks(x)
    ax.set_xticklabels(unique_decades)

    # Add some padding to x-axis limits
    ax.set_xlim(min(x) - 0.5, max(x) + 0.5)
    ax.set_ylim(0, 100)

    ax.set_xlabel("Impact Decade")
    ax.set_ylabel("Percentage of Discovered Objects Not Realized")
    ax.set_title(
        "Percentage of Discovered Objects Not Reaching 0.01% Impact Probability by Diameter and Impact Decade"
    )

    # Move legend outside to save space
    ax.legend(
        title="Diameter [km]",
        frameon=False,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
    )

    ax.yaxis.grid(True, linestyle="--", alpha=0.7)

    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    return fig, ax


class MaxImpactProbabilityByDiameterDecade(qv.Table):
    diameter = qv.Float64Column()
    decade = qv.LargeStringColumn()
    mean_max_impact_probability = qv.Float64Column()


def plot_max_impact_probability_by_diameter_decade(
    summary: ImpactorResultSummary,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the mean maximum impact probability reached during the survey,
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
    summary = summary.apply_mask(summary.discovered())

    # Get common data
    impact_decades, unique_decades, unique_diameters = (
        summary.get_diameter_impact_period_data(period_breakdown="decade")
    )

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
            max_impact_probabilities = pc.fill_null(
                orbits_at_diameter_and_decade.maximum_impact_probability, 0
            )

            # Calculate mean of maximum impact probabilities
            mean_max_ip = pc.mean(max_impact_probabilities).as_py()
            if mean_max_ip is None:
                mean_max_ip = 0

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

    # Create the plot with improved spacing
    fig, ax = plt.subplots(
        1, 1, dpi=200, figsize=(12, 6)
    )  # Wider figure for better spacing

    # Calculate bar width and spacing based on number of diameters
    num_diameters = len(unique_diameters)
    group_width = BAR_GROUP_WIDTH  # Width allocated for each decade group (out of 1.0)
    bar_width = (
        group_width / num_diameters * BAR_WIDTH_SCALE
    )  # Slightly narrower bars for spacing between them

    # Create evenly spaced x positions for decade groups
    x = np.arange(len(unique_decades)) * (
        1 + BAR_GROUP_SPACING
    )  # Add 20% extra space between decade groups

    # Define a colormap
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_diameters)))

    # Plot bars for each diameter
    for i, diameter in enumerate(unique_diameters):
        diameter_data = max_ip_by_diameter_decade.select("diameter", diameter)

        # Calculate x position for this diameter's bars within each group
        offset = (i - num_diameters / 2 + 0.5) * (
            bar_width * 1.1
        )  # Add 10% spacing between bars
        bar_positions = x + offset

        ax.bar(
            bar_positions,
            diameter_data.mean_max_impact_probability.to_numpy(zero_copy_only=False),
            width=bar_width,
            color=colors[i],
            label=f"{diameter:.3f} km",
        )

    # Position x-ticks at the center of each decade group
    ax.set_xticks(x)
    ax.set_xticklabels(unique_decades)

    # Add some padding to x-axis limits
    ax.set_xlim(min(x) - 0.5, max(x) + 0.5)

    ax.set_xlabel("Impact Decade")
    ax.set_ylabel("Mean Maximum Impact Probability")
    ax.set_title("Mean Maximum Impact Probability by Diameter and Impact Decade")

    # Move legend outside to save space
    ax.legend(
        title="Diameter [km]",
        frameon=False,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
    )

    ax.yaxis.grid(True, linestyle="--", alpha=0.7)

    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    return fig, ax


class ArcLengthByDiameterDecade(qv.Table):
    diameter = qv.Float64Column()
    decade = qv.LargeStringColumn()
    mean_arc_length = qv.Float64Column()


def plot_arc_length_by_diameter_decade(
    summary: ImpactorResultSummary,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the mean arc length of the orbits by diameter and impact decade.
    """
    summary = summary.apply_mask(summary.complete())

    impact_decades, unique_decades, unique_diameters = (
        summary.get_diameter_impact_period_data(period_breakdown="decade")
    )

    arc_length_by_diameter_decade = ArcLengthByDiameterDecade.empty()

    for decade in unique_decades:
        orbits_at_decade = summary.apply_mask(impact_decades == decade)
        for diameter in unique_diameters:
            orbits_at_diameter_and_decade = orbits_at_decade.select(
                "orbit.diameter", diameter
            )
            arc_length = pc.fill_null(
                orbits_at_diameter_and_decade.arc_length(), 0
            ).to_numpy(zero_copy_only=False)

            mean_arc_length = pc.mean(arc_length).as_py() if len(arc_length) > 0 else 0
            arc_length_by_diameter_decade = qv.concatenate(
                [
                    arc_length_by_diameter_decade,
                    ArcLengthByDiameterDecade.from_kwargs(
                        decade=[f"{decade}"],
                        diameter=[diameter],
                        mean_arc_length=[mean_arc_length],
                    ),
                ]
            )

    # Create the plot with improved spacing
    fig, ax = plt.subplots(
        1, 1, dpi=200, figsize=(12, 6)
    )  # Wider figure for better spacing

    # Calculate bar width and spacing based on number of diameters
    num_diameters = len(unique_diameters)
    group_width = BAR_GROUP_WIDTH  # Width allocated for each decade group (out of 1.0)
    bar_width = (
        group_width / num_diameters * BAR_WIDTH_SCALE
    )  # Slightly narrower bars for spacing between them

    # Create evenly spaced x positions for decade groups
    x = np.arange(len(unique_decades)) * (
        1 + BAR_GROUP_SPACING
    )  # Add 20% extra space between decade groups

    # Define a colormap
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_diameters)))

    # Plot bars for each diameter
    for i, diameter in enumerate(unique_diameters):
        diameter_data = arc_length_by_diameter_decade.select("diameter", diameter)

        # Calculate x position for this diameter's bars within each group
        offset = (i - num_diameters / 2 + 0.5) * (
            bar_width * 1.1
        )  # Add 10% spacing between bars
        bar_positions = x + offset

        ax.bar(
            bar_positions,
            diameter_data.mean_arc_length.to_numpy(zero_copy_only=False),
            width=bar_width,
            color=colors[i],
            label=f"{diameter:.3f} km",
        )

    # Position x-ticks at the center of each decade group
    ax.set_xticks(x)
    ax.set_xticklabels(unique_decades)

    # Add some padding to x-axis limits
    ax.set_xlim(min(x) - 0.5, max(x) + 0.5)

    ax.set_xlabel("Impact Decade")
    ax.set_ylabel("Mean Arc Length [days]")
    ax.set_title("Mean Arc Length by Diameter and Impact Decade")

    # Move legend outside to save space
    ax.legend(
        title="Diameter [km]",
        frameon=False,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
    )

    ax.yaxis.grid(True, linestyle="--", alpha=0.7)

    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    return fig, ax


class IAWNThresholdReachedByDiameterImpactTime(qv.Table):
    diameter = qv.Float64Column()
    impact_period = qv.Int64Column()
    percentage_reaching_threshold = qv.Float64Column()


def plot_iawn_threshold_reached_by_diameter_impact_time(
    summary: ImpactorResultSummary,
    period: Literal["year", "5year", "decade"] = "decade",
    max_impact_time: Optional[Timestamp] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the percentage of objects that reach the IAWN threshold (1% impact probability)
    broken down by diameter and impact period.
    """
    summary = summary.apply_mask(summary.complete())

    # Only include discovered objects
    summary = summary.apply_mask(summary.discovered())

    if max_impact_time is not None:
        logger.info(
            f"Filtering {len(summary)} objects that impact before {max_impact_time.to_astropy().iso}"
        )
        summary = summary.apply_mask(
            pc.less_equal(summary.orbit.impact_time.mjd(), max_impact_time.mjd()[0])
        )
        logger.info(f"After filtering, {len(summary)} objects remain")

    # Get common data
    impact_periods, unique_periods, unique_diameters = (
        summary.get_diameter_impact_period_data(period_breakdown=period)
    )

    iawn_by_diameter_impact_period = IAWNThresholdReachedByDiameterImpactTime.empty()

    for impact_period in unique_periods:
        orbits_at_impact_period = summary.apply_mask(impact_periods == impact_period)
        for diameter in unique_diameters:
            orbits_at_diameter_and_impact_period = orbits_at_impact_period.select(
                "orbit.diameter", diameter
            )

            # Count objects with non-null IAWN_time (meaning they reached the threshold)
            reaching_threshold_mask = pc.invert(
                pc.is_null(
                    orbits_at_diameter_and_impact_period.ip_threshold_1_percent.mjd()
                )
            )

            reaching_threshold = pc.sum(reaching_threshold_mask).as_py()

            total_at_diameter_and_impact_period = len(
                orbits_at_diameter_and_impact_period
            )

            percentage = (
                (reaching_threshold / total_at_diameter_and_impact_period * 100)
                if total_at_diameter_and_impact_period > 0
                else 0
            )

            iawn_by_diameter_impact_period = qv.concatenate(
                [
                    iawn_by_diameter_impact_period,
                    IAWNThresholdReachedByDiameterImpactTime.from_kwargs(
                        impact_period=[impact_period],
                        diameter=[diameter],
                        percentage_reaching_threshold=[percentage],
                    ),
                ]
            )

    # Create the plot with improved spacing
    fig, ax = plt.subplots(
        1, 1, dpi=200, figsize=(12, 6)
    )  # Wider figure for better spacing

    # Calculate bar width and spacing based on number of diameters
    num_diameters = len(unique_diameters)
    group_width = (
        BAR_GROUP_WIDTH  # Width allocated for each five year group (out of 1.0)
    )
    bar_width = (
        group_width / num_diameters * BAR_WIDTH_SCALE
    )  # Slightly narrower bars for spacing between them

    # Create evenly spaced x positions for five year groups
    x = np.arange(len(unique_periods)) * (
        1 + BAR_GROUP_SPACING
    )  # Add 20% extra space between five year groups

    # Define a colormap
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_diameters)))

    # Plot bars for each diameter
    for i, diameter in enumerate(unique_diameters):
        diameter_data = iawn_by_diameter_impact_period.select("diameter", diameter)

        # Calculate x position for this diameter's bars within each group
        offset = (i - num_diameters / 2 + 0.5) * (
            bar_width * 1.1
        )  # Add 10% spacing between bars
        bar_positions = x + offset

        ax.bar(
            bar_positions,
            diameter_data.percentage_reaching_threshold.to_numpy(zero_copy_only=False),
            width=bar_width,
            color=colors[i],
            label=f"{diameter:.3f} km",
        )

    # Position x-ticks at the center of each five year group
    ax.set_xticks(x)
    ax.set_xticklabels(unique_periods)

    # Add some padding to x-axis limits
    ax.set_xlim(min(x) - 0.5, max(x) + 0.5)
    ax.set_ylim(0, 100)

    if period == "year":
        ax.set_xlabel("Impact Year")
    elif period == "5year":
        ax.set_xlabel("Impact 5-Year Period")
    elif period == "decade":
        ax.set_xlabel("Impact Decade")

    ax.set_ylabel("Percentage of Discovered Objects Reaching IAWN Threshold (1%)")
    ax.set_title(
        "Percentage of Discovered Objects Reaching IAWN Threshold (1%) by Diameter"
    )

    # Move legend outside to save space
    ax.legend(
        title="Diameter [km]",
        frameon=False,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
    )

    ax.yaxis.grid(True, linestyle="--", alpha=0.7)

    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    return fig, ax


class IAWNThresholdNotReachedByDiameterDecade(qv.Table):
    diameter = qv.Float64Column()
    decade = qv.LargeStringColumn()
    percentage_not_reaching_threshold = qv.Float64Column()


def plot_iawn_threshold_not_reached_by_diameter_decade(
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

    # Filter to only include complete results (needed for the rest of the function)
    summary = summary.apply_mask(summary.complete())

    # Only include discovered objects
    summary = summary.apply_mask(summary.discovered())

    # Get common data
    impact_decades, unique_decades, unique_diameters = (
        summary.get_diameter_impact_period_data(period_breakdown="decade")
    )

    iawn_by_diameter_decade = IAWNThresholdNotReachedByDiameterDecade.empty()

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
            # Only consider objects that have been discovered (to be consistent with realization plot)
            # Use negated AND logic instead of OR for consistency with realization plot
            not_reaching_threshold_mask = pc.is_null(
                orbits_at_diameter_and_decade.ip_threshold_1_percent.mjd()
            )

            not_reaching_threshold = pc.sum(not_reaching_threshold_mask).as_py()

            total_at_diameter_and_decade = len(orbits_at_diameter_and_decade)

            percentage = (
                (not_reaching_threshold / total_at_diameter_and_decade * 100)
                if total_at_diameter_and_decade > 0
                else 0
            )

            iawn_by_diameter_decade = qv.concatenate(
                [
                    iawn_by_diameter_decade,
                    IAWNThresholdNotReachedByDiameterDecade.from_kwargs(
                        decade=[f"{decade}"],
                        diameter=[diameter],
                        percentage_not_reaching_threshold=[percentage],
                    ),
                ]
            )

    # Create the plot with improved spacing
    fig, ax = plt.subplots(
        1, 1, dpi=200, figsize=(12, 6)
    )  # Wider figure for better spacing

    # Calculate bar width and spacing based on number of diameters
    num_diameters = len(unique_diameters)
    group_width = BAR_GROUP_WIDTH  # Width allocated for each decade group (out of 1.0)
    bar_width = (
        group_width / num_diameters * BAR_WIDTH_SCALE
    )  # Slightly narrower bars for spacing between them

    # Create evenly spaced x positions for decade groups
    x = np.arange(len(unique_decades)) * (
        1 + BAR_GROUP_SPACING
    )  # Add 20% extra space between decade groups

    # Define a colormap
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_diameters)))

    # Plot bars for each diameter
    for i, diameter in enumerate(unique_diameters):
        diameter_data = iawn_by_diameter_decade.select("diameter", diameter)

        # Calculate x position for this diameter's bars within each group
        offset = (i - num_diameters / 2 + 0.5) * (
            bar_width * 1.1
        )  # Add 10% spacing between bars
        bar_positions = x + offset

        ax.bar(
            bar_positions,
            diameter_data.percentage_not_reaching_threshold.to_numpy(
                zero_copy_only=False
            ),
            width=bar_width,
            color=colors[i],
            label=f"{diameter:.3f} km",
        )

    # Position x-ticks at the center of each decade group
    ax.set_xticks(x)
    ax.set_xticklabels(unique_decades)

    # Add some padding to x-axis limits
    ax.set_xlim(min(x) - 0.5, max(x) + 0.5)
    ax.set_ylim(0, 100)

    ax.set_xlabel("Impact Decade")
    ax.set_ylabel("Percentage of Discovered Objects Not Reaching IAWN Threshold")
    ax.set_title(
        "Percentage of Discovered Objects Not Reaching IAWN Threshold (1%) by Diameter and Impact Decade"
    )

    # Move legend outside to save space
    ax.legend(
        title="Diameter [km]",
        frameon=False,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
    )

    ax.yaxis.grid(True, linestyle="--", alpha=0.7)

    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    return fig, ax


def plot_elements(
    summary: ImpactorResultSummary,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the elements of the orbits
    """
    # summary = summary.apply_mask(summary.complete())

    kep_coordinates = summary.orbit.coordinates.to_keplerian()

    # Convert pyarrow arrays to numpy arrays for plotting
    a_au = kep_coordinates.a.to_numpy(zero_copy_only=False)
    i_deg = kep_coordinates.i.to_numpy(zero_copy_only=False)
    e = kep_coordinates.e.to_numpy(zero_copy_only=False)

    # make two hexbin plots, a vs i and a vs e
    fig, axes = plt.subplots(1, 2, dpi=200, figsize=(12, 5))  # Adjusted figsize

    # Plot a vs i
    hb1 = axes[0].hexbin(a_au, i_deg, gridsize=50, cmap="viridis", mincnt=1)
    axes[0].set_xlabel("Semimajor Axis (a) [AU]")
    axes[0].set_ylabel("Inclination (i) [deg]")
    axes[0].set_title("Density Plot: a vs i")
    fig.colorbar(hb1, ax=axes[0], label="Count in bin")

    # Plot a vs e
    hb2 = axes[1].hexbin(a_au, e, gridsize=50, cmap="viridis", mincnt=1)
    axes[1].set_xlabel("Semimajor Axis (a) [AU]")
    axes[1].set_ylabel("Eccentricity (e)")
    axes[1].set_title("Density Plot: a vs e")
    fig.colorbar(hb2, ax=axes[1], label="Count in bin")

    # add a title to the entire figure
    fig.suptitle("Density Plot of Elements")

    plt.tight_layout()  # Adjust layout to prevent overlap
    return fig, axes


def plot_1_percent_ip_threshold_percentage_vs_elements(
    summary: ImpactorResultSummary,
    diameter: float = 1,
    contour_levels: int = 4,
    contour_color: str = "black",
    contour_alpha: float = 0.7,
    hist_bins: int = 40,
    contour_fontsize: int = 7,
    min_contour_level: float = 5.0,
    contour_smooth_factor: int = 3,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the percentage of orbits reaching the 1% IP threshold
    as a function of orbital elements using a hexbin plot, with overlaid,
    visually smoothed contours representing the total population density
    (excluding levels below min_contour_level). Contour lines are labeled
    with rounded integer values.

    Parameters
    ----------
    summary : ImpactorResultSummary
        The summary of impact study results.
    diameter : float, optional
        The diameter [km] to filter the results by, by default 1.
    contour_levels : int, optional
        Approximate number of contour levels to draw for the total density. Default is 4.
    contour_color : str, optional
        Color of the density contour lines. Default is black.
    contour_alpha : float, optional
        Transparency of the density contour lines.
    hist_bins : int, optional
        Number of bins to use for the 2D histogram underlying the contours.
    contour_fontsize : int, optional
        Font size for the contour labels. Default is 7.
    min_contour_level : float, optional
        The minimum density value for a contour line to be drawn. Default is 5.0.
    contour_smooth_factor : int, optional
        Factor by which to upscale the histogram grid for smoother contours (e.g., 3).
        Higher values mean smoother but potentially slower plotting. Default is 3.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        The figure and axes objects for the plot.
    """
    # summary = summary.apply_mask(summary.complete()) # Optional: uncomment if only complete needed

    # filter to only include the given diameter
    summary = summary.apply_mask(pc.equal(summary.orbit.diameter, diameter))

    if len(summary) == 0:
        logger.warning(f"No data found for diameter {diameter} km.")
        # Return empty figure/axes or raise error
        fig, axes = plt.subplots(1, 2, dpi=200, figsize=(18, 7))
        fig.suptitle(f"No data for Diameter: {diameter} km")
        return fig, axes

    # Get Keplerian coordinates for the entire filtered population
    kep_coordinates = summary.orbit.coordinates.to_keplerian()
    a_au = kep_coordinates.a.to_numpy(zero_copy_only=False)
    i_deg = kep_coordinates.i.to_numpy(zero_copy_only=False)
    e = kep_coordinates.e.to_numpy(zero_copy_only=False)

    # Create a mask indicating which orbits reached the 1% IP threshold (non-null time)
    # Convert boolean mask to float (1.0 for True, 0.0 for False) for averaging
    reached_1_percent_ip_mask = (
        pc.invert(pc.is_null(summary.ip_threshold_1_percent.mjd()))
        .to_numpy(zero_copy_only=False)
        .astype(float)
    )

    # Create the plots
    fig, axes = plt.subplots(1, 2, dpi=200, figsize=(18, 7))

    # --- Plot a vs i ---
    hb1 = axes[0].hexbin(
        a_au,
        i_deg,
        C=reached_1_percent_ip_mask,
        reduce_C_function=np.mean,
        gridsize=hist_bins,
        cmap="plasma",
        mincnt=2,
        vmin=0,
        vmax=1,
    )
    axes[0].set_xlabel("Semimajor Axis (a) [AU]")
    axes[0].set_ylabel("Inclination (i) [deg]")
    axes[0].set_title("a vs i")
    cb1 = fig.colorbar(hb1, ax=axes[0])
    cb1.set_label("Fraction Reaching 1% IP Threshold")

    # Calculate and plot density contours for a vs i
    H_ai, xedges_ai, yedges_ai = np.histogram2d(
        a_au,
        i_deg,
        bins=hist_bins,
        range=[[np.min(a_au), np.max(a_au)], [np.min(i_deg), np.max(i_deg)]],
    )
    xcenters_ai = (xedges_ai[:-1] + xedges_ai[1:]) / 2
    ycenters_ai = (yedges_ai[:-1] + yedges_ai[1:]) / 2
    X_ai, Y_ai = np.meshgrid(xcenters_ai, ycenters_ai)

    # --- Determine contour levels excluding zero and below threshold ---
    max_val_ai = H_ai.max()
    levels_ai = None  # Initialize levels_ai
    if max_val_ai >= min_contour_level:  # Only proceed if max is above threshold
        # Find min positive value that is also >= min_contour_level
        positive_above_thresh_ai = H_ai[H_ai >= min_contour_level]
        if len(positive_above_thresh_ai) > 0:
            min_level_ai = positive_above_thresh_ai.min()

            if max_val_ai <= min_level_ai:
                # Only one level possible above threshold
                levels_ai = np.round(np.array([max_val_ai])).astype(int)
            else:
                # Calculate levels between the effective min and max
                levels_ai = np.linspace(min_level_ai, max_val_ai, contour_levels)
                # Round levels to nearest integer
                levels_ai = np.round(levels_ai).astype(int)
                # Ensure levels are unique and sorted after rounding
                levels_ai = np.unique(levels_ai)
                # Filter again to ensure they are still >= min_contour_level after rounding
                levels_ai = levels_ai[
                    levels_ai >= np.ceil(min_contour_level)
                ]  # Use ceil for comparison

            # Ensure we have at least one level after filtering/rounding
            if len(levels_ai) == 0:
                levels_ai = None

    if levels_ai is not None and len(levels_ai) > 0:
        # --- Smooth the histogram data for visual appearance ---
        H_ai_smooth = scipy.ndimage.zoom(H_ai.T, contour_smooth_factor, order=3)
        # Create finer grid coordinates corresponding to the smoothed data
        xcenters_ai_smooth = np.linspace(
            xcenters_ai.min(), xcenters_ai.max(), H_ai_smooth.shape[1]
        )
        ycenters_ai_smooth = np.linspace(
            ycenters_ai.min(), ycenters_ai.max(), H_ai_smooth.shape[0]
        )
        X_ai_smooth, Y_ai_smooth = np.meshgrid(xcenters_ai_smooth, ycenters_ai_smooth)

        # Store the contour set plotted on the smoothed grid
        CS_ai = axes[0].contour(
            X_ai_smooth,  # Use smoothed grid
            Y_ai_smooth,  # Use smoothed grid
            H_ai_smooth,  # Use smoothed data
            levels=levels_ai,  # Use original calculated levels
            colors=contour_color,
            alpha=contour_alpha,
            linewidths=0.8,
        )
        # Add labels to the contours, using the original calculated levels
        axes[0].clabel(
            CS_ai, levels=levels_ai, inline=True, fontsize=contour_fontsize, fmt="%1.0f"
        )
    # --- End contour calculation ---

    # --- Plot a vs e ---
    hb2 = axes[1].hexbin(
        a_au,
        e,
        C=reached_1_percent_ip_mask,
        reduce_C_function=np.mean,
        gridsize=hist_bins,
        cmap="plasma",
        mincnt=2,
        vmin=0,
        vmax=1,
    )
    axes[1].set_xlabel("Semimajor Axis (a) [AU]")
    axes[1].set_ylabel("Eccentricity (e)")
    axes[1].set_title("a vs e")
    cb2 = fig.colorbar(hb2, ax=axes[1])
    cb2.set_label("Fraction Reaching 1% IP Threshold")

    # Calculate and plot density contours for a vs e
    H_ae, xedges_ae, yedges_ae = np.histogram2d(
        a_au,
        e,
        bins=hist_bins,
        range=[[np.min(a_au), np.max(a_au)], [np.min(e), np.max(e)]],
    )
    xcenters_ae = (xedges_ae[:-1] + xedges_ae[1:]) / 2
    ycenters_ae = (yedges_ae[:-1] + yedges_ae[1:]) / 2
    X_ae, Y_ae = np.meshgrid(xcenters_ae, ycenters_ae)

    # --- Determine contour levels excluding zero and below threshold ---
    max_val_ae = H_ae.max()
    levels_ae = None  # Initialize levels_ae
    if max_val_ae >= min_contour_level:  # Only proceed if max is above threshold
        # Find min positive value that is also >= min_contour_level
        positive_above_thresh_ae = H_ae[H_ae >= min_contour_level]
        if len(positive_above_thresh_ae) > 0:
            min_level_ae = positive_above_thresh_ae.min()

            if max_val_ae <= min_level_ae:
                # Only one level possible above threshold
                levels_ae = np.round(np.array([max_val_ae])).astype(int)
            else:
                # Calculate levels between the effective min and max
                levels_ae = np.linspace(min_level_ae, max_val_ae, contour_levels)
                # Round levels to nearest integer
                levels_ae = np.round(levels_ae).astype(int)
                # Ensure levels are unique and sorted after rounding
                levels_ae = np.unique(levels_ae)
                # Filter again to ensure they are still >= min_contour_level after rounding
                levels_ae = levels_ae[
                    levels_ae >= np.ceil(min_contour_level)
                ]  # Use ceil for comparison

            # Ensure we have at least one level after filtering/rounding
            if len(levels_ae) == 0:
                levels_ae = None

    if levels_ae is not None and len(levels_ae) > 0:
        # --- Smooth the histogram data for visual appearance ---
        H_ae_smooth = scipy.ndimage.zoom(H_ae.T, contour_smooth_factor, order=3)
        # Create finer grid coordinates corresponding to the smoothed data
        xcenters_ae_smooth = np.linspace(
            xcenters_ae.min(), xcenters_ae.max(), H_ae_smooth.shape[1]
        )
        ycenters_ae_smooth = np.linspace(
            ycenters_ae.min(), ycenters_ae.max(), H_ae_smooth.shape[0]
        )
        X_ae_smooth, Y_ae_smooth = np.meshgrid(xcenters_ae_smooth, ycenters_ae_smooth)

        # Store the contour set plotted on the smoothed grid
        CS_ae = axes[1].contour(
            X_ae_smooth,  # Use smoothed grid
            Y_ae_smooth,  # Use smoothed grid
            H_ae_smooth,  # Use smoothed data
            levels=levels_ae,  # Use original calculated levels
            colors=contour_color,
            alpha=contour_alpha,
            linewidths=0.8,
        )
        # Add labels to the contours, using the original calculated levels
        axes[1].clabel(
            CS_ae, levels=levels_ae, inline=True, fontsize=contour_fontsize, fmt="%1.0f"
        )
    # --- End contour calculation ---

    # Add overall title
    fig.suptitle(
        f"Fraction Reaching 1% IP Threshold (Color) & Total Density (Contours) by Orbital Elements (Diameter: {diameter} km)"
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig, axes


def plot_observation_density_vs_impact_probability(
    summary: ImpactorResultSummary,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the relationship between observation density (observations per month)
    and the maximum impact probability achieved.

    Parameters
    ----------
    summary : ImpactorResultSummary
        Summary of the impact study results

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        The figure and axes objects for the plot
    """
    # Filter to only include complete results
    summary = summary.apply_mask(summary.complete())

    # Initialize arrays to store our data
    diameters = []

    observation_densities_days = pc.divide(summary.observations, summary.arc_length())
    observation_densities_months = pc.divide(observation_densities_days, 30.4375)
    observation_densities_months = pc.fill_null(
        observation_densities_months, 0
    ).to_numpy(zero_copy_only=False)

    max_impact_probabilities = pc.fill_null(
        summary.maximum_impact_probability, 0
    ).to_numpy(zero_copy_only=False)
    diameters = summary.orbit.diameter.to_numpy(zero_copy_only=False)

    # Create figure
    fig, ax = plt.subplots(1, 1, dpi=200, figsize=(10, 6))

    # Create scatter plot with color based on diameter
    unique_diameters = np.unique(diameters)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_diameters)))

    for i, diam in enumerate(unique_diameters):
        mask = diameters == diam
        scatter = ax.scatter(
            observation_densities_months[mask],
            max_impact_probabilities[mask],
            color=colors[i],
            alpha=0.7,
            label=f"{diam:.3f} km",
            s=30,
        )

    # Add trend line
    if len(observation_densities_months) > 1:
        # Use numpy's polyfit to calculate trend line
        z = np.polyfit(observation_densities_months, max_impact_probabilities, 1)
        p = np.poly1d(z)

        # Get sorted x values for smooth line
        x_sorted = np.sort(observation_densities_months)

        # Plot the trend line
        ax.plot(x_sorted, p(x_sorted), "r--", alpha=0.8, label="Trend")

        # Calculate correlation coefficient
        correlation = np.corrcoef(
            observation_densities_months, max_impact_probabilities
        )[0, 1]
        ax.text(
            0.05,
            0.95,
            f"Correlation: {correlation:.3f}",
            transform=ax.transAxes,
            bbox=dict(facecolor="white", alpha=0.8),
        )

    # Set axis labels and title
    ax.set_xlabel("Observation Density (observations per month)")
    ax.set_ylabel("Maximum Impact Probability")
    ax.set_title("Relationship Between Observation Density and Impact Probability")

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle="--")

    # Add legend outside plot area
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)

    # Set y-axis limits
    ax.set_ylim(0, 1.05)

    # If there are very high density values, use log scale for x-axis
    if max(observation_densities_months) > 50:
        ax.set_xscale("log")
        ax.set_xlim(left=0.1)  # Minimum x value

    # Adjust layout to make room for legend
    plt.tight_layout()

    return fig, ax


def plot_total_arc_length_vs_max_impact_probability(
    summary: ImpactorResultSummary,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the relationship between total observation arc length (days) and
    the maximum impact probability achieved, color-coded by impact decade.

    Parameters
    ----------
    summary : ImpactorResultSummary
        Summary of the impact study results

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        The figure and axes objects for the plot
    """
    # Filter to only include complete results
    summary = summary.apply_mask(summary.complete())

    # Extract impact decades
    impact_decades, unique_decades, _ = summary.get_diameter_impact_period_data(
        period_breakdown="decade"
    )

    # Get arc lengths and max impact probabilities
    arc_lengths = summary.arc_length().to_numpy(zero_copy_only=False)
    max_impact_probs = pc.fill_null(summary.maximum_impact_probability, 0).to_numpy(
        zero_copy_only=False
    )

    # Create figure
    fig, ax = plt.subplots(1, 1, dpi=200, figsize=(10, 6))

    # Create scatter plot with color based on impact decade
    unique_decades_sorted = np.sort(unique_decades)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_decades_sorted)))

    # Create a legend handle list
    legend_elements = []

    # Plot each decade with a different color
    for i, decade in enumerate(unique_decades_sorted):
        mask = impact_decades == decade

        # Skip empty decades
        if not np.any(mask):
            continue

        scatter = ax.scatter(
            arc_lengths[mask],
            max_impact_probs[mask],
            color=colors[i],
            alpha=0.7,
            label=f"{decade}s",
            s=30,
        )
        legend_elements.append(scatter)

    # Set axis labels and title
    ax.set_xlabel("Total Observation Arc Length (days)")
    ax.set_ylabel("Maximum Impact Probability")
    ax.set_title("Relationship Between Arc Length and Impact Probability by Decade")

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle="--")

    # Add legend
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)

    # Set y-axis limits
    ax.set_ylim(0, 1.05)

    # Adjust layout to make room for legend
    plt.tight_layout()

    return fig, ax


def plot_window_arc_length_vs_impact_probability(
    summary: ImpactorResultSummary,
    window_results: WindowResult,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the relationship between window arc length (days) and impact probability
    Color-coded by impact decade.
    """
    # Get the impact decades for each orbit
    impact_decades, unique_decades, _ = summary.get_diameter_impact_period_data(
        period_breakdown="decade"
    )

    # Get the window arc length and impact probability
    window_arc_lengths = window_results.arc_length().to_numpy(zero_copy_only=False)
    window_impact_probabilities = pc.fill_null(
        window_results.impact_probability, 0
    ).to_numpy(zero_copy_only=False)

    # Create figure
    fig, ax = plt.subplots(1, 1, dpi=200, figsize=(10, 6))

    # Create scatter plot with color based on impact decade
    unique_decades_sorted = np.sort(unique_decades)[::-1]  # Reverse the sorted decades
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_decades_sorted)))

    # Create a legend handle list
    legend_elements = []

    # Plot each decade with a different color
    for i, decade in enumerate(unique_decades_sorted):
        decade_summary_mask = impact_decades == decade
        orbit_ids_in_decade = pa.array(
            summary.orbit.orbit_id.to_numpy(zero_copy_only=False)[decade_summary_mask],
            pa.large_string(),
        )
        window_results_mask = pc.is_in(window_results.orbit_id, orbit_ids_in_decade)

        scatter = ax.scatter(
            window_arc_lengths[window_results_mask],
            window_impact_probabilities[window_results_mask],
            color=colors[i],
            alpha=0.7,
            label=f"{decade}s",
            s=30,
        )
        legend_elements.append(scatter)

    # Add legend
    ax.legend(
        legend_elements,
        [f"{decade}s" for decade in unique_decades_sorted],
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        frameon=False,
    )

    # Set axis labels and title
    ax.set_xlabel("Window Arc Length (days)")
    ax.set_ylabel("Impact Probability")
    ax.set_title(
        "Relationship Between Window Arc Length and Impact Probability by Decade"
    )

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle="--")

    # Adjust layout to make room for legend
    plt.tight_layout()

    return fig, ax


def plot_window_arc_length_vs_impact_probability_density(
    window_results: WindowResult,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the relationship between window arc length (days) and impact probability
    as a density map, without distinguishing between impact decades.

    Parameters
    ----------
    window_results : WindowResult
        The window results containing impact probabilities and observation times

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        The figure and axes objects for the plot
    """
    arc_lengths = window_results.arc_length().to_numpy(zero_copy_only=False)
    impact_probs = pc.fill_null(window_results.impact_probability, 0).to_numpy(
        zero_copy_only=False
    )

    # Create figure
    fig, ax = plt.subplots(1, 1, dpi=200, figsize=(10, 6))

    # Create density plot using hexbin
    hb = ax.hexbin(
        arc_lengths,
        impact_probs,
        gridsize=50,  # Number of hexagons in each direction
        cmap="viridis",  # Color map
        mincnt=1,  # Minimum count to color the hexagon
        alpha=0.8,
        bins="log",  # Use logarithmic binning to better represent the density
    )

    # Add colorbar
    cb = fig.colorbar(hb, ax=ax, label="Log Count")

    # Set axis labels and title
    ax.set_xlabel("Window Arc Length (days)")
    ax.set_ylabel("Impact Probability")
    ax.set_title("Density of Window Arc Length vs Impact Probability")

    # Add grid for better readability (behind the hexbin plot)
    ax.grid(True, alpha=0.3, linestyle="--", zorder=0)

    # Set y-axis limits
    ax.set_ylim(0, 1.05)

    # If there are very long arcs, use log scale for x-axis
    if np.max(arc_lengths[~np.isnan(arc_lengths)]) > 365:
        ax.set_xscale("log")
        ax.set_xlim(left=0.1)  # Minimum x value

    # Adjust layout
    plt.tight_layout()

    return fig, ax


class MeanRealizationTimeByDiameterDecade(qv.Table):
    diameter = qv.Float64Column()
    decade = qv.Int64Column()
    mean_realization_time = qv.Float64Column()
    count = qv.Int64Column()


def plot_mean_realization_time_by_diameter_decade(
    summary: ImpactorResultSummary,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the mean realization time (number of days since discovery until reaching 0.01% impact probability)
    grouped by impact decade and asteroid diameter.

    Parameters
    ----------
    summary : ImpactorResultSummary
        Summary of the impact study results

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        The figure and axes objects for the plot
    """
    # Filter to only include complete results
    summary = summary.apply_mask(summary.complete())

    # Filter to only include discovered objects
    summary = summary.apply_mask(summary.discovered())

    # Extract impact decades and diameters
    impact_decades, unique_decades, unique_diameters = (
        summary.get_diameter_impact_period_data(period_breakdown="decade")
    )

    # Initialize results table
    mean_realization_time_by_diameter_decade = (
        MeanRealizationTimeByDiameterDecade.empty()
    )

    # Calculate realization time for each orbit
    realization_times = pc.fill_null(
        summary.days_discovery_to_0_dot_01_percent(), 0
    ).to_numpy(zero_copy_only=False)

    # Replace any negative values with 0, as we are only
    # considering default linking discovery times
    realization_times = np.maximum(realization_times, 0)

    # For each combination of diameter and decade, calculate mean realization time
    for decade in unique_decades:
        for diameter in unique_diameters:
            # Create mask for this diameter and decade
            diam_mask = (
                summary.orbit.diameter.to_numpy(zero_copy_only=False) == diameter
            )
            decade_mask = impact_decades == decade
            combined_mask = diam_mask & decade_mask

            # Skip if no data for this combination
            if not np.any(combined_mask):
                continue

            # Get realization times for this group
            group_realization_times = realization_times[combined_mask]

            # Skip if no valid realization times
            valid_times = group_realization_times[~np.isnan(group_realization_times)]
            if len(valid_times) == 0:
                continue

            # Calculate mean realization time
            mean_time = np.mean(valid_times)

            # Add to results table
            mean_realization_time_by_diameter_decade = qv.concatenate(
                [
                    mean_realization_time_by_diameter_decade,
                    MeanRealizationTimeByDiameterDecade.from_kwargs(
                        diameter=[diameter],
                        decade=[decade],
                        mean_realization_time=[mean_time],
                        count=[len(valid_times)],
                    ),
                ]
            )

    # Create the plot with improved spacing
    fig, ax = plt.subplots(
        1, 1, dpi=200, figsize=(12, 6)
    )  # Wider figure for better spacing

    # Calculate bar width and spacing based on number of diameters
    num_diameters = len(unique_diameters)
    group_width = 0.8  # Width allocated for each decade group (out of 1.0)
    bar_width = (
        group_width / num_diameters * 0.8
    )  # Slightly narrower bars for spacing between them

    # Create evenly spaced x positions for decade groups
    x = np.arange(len(unique_decades)) * (
        1 + 0.2
    )  # Add 20% extra space between decade groups

    # Define a colormap
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_diameters)))

    # Plot bars for each diameter
    for i, diameter in enumerate(unique_diameters):
        # Get data for this diameter across all decades
        y_values = []
        count_values = []
        x_positions = []

        for j, decade in enumerate(unique_decades):
            # Get data for this diameter and decade
            data = mean_realization_time_by_diameter_decade.apply_mask(
                pc.and_(
                    pc.equal(
                        mean_realization_time_by_diameter_decade.diameter, diameter
                    ),
                    pc.equal(mean_realization_time_by_diameter_decade.decade, decade),
                )
            )

            if len(data) > 0:
                y_values.append(data.mean_realization_time[0].as_py())
                count_values.append(data.count[0].as_py())
                # Calculate position with proper spacing between bars
                offset = (i - num_diameters / 2 + 0.5) * (
                    bar_width * 1.1
                )  # Add 10% spacing between bars
                x_positions.append(x[j] + offset)
            else:
                # Add 0 or placeholder for missing data to keep bar positions consistent
                y_values.append(0)
                count_values.append(0)
                # Calculate position with proper spacing between bars
                offset = (i - num_diameters / 2 + 0.5) * (
                    bar_width * 1.1
                )  # Add 10% spacing between bars
                x_positions.append(x[j] + offset)

        # Plot bars for this diameter
        bars = ax.bar(
            x_positions,
            y_values,
            width=bar_width,
            color=colors[i],
            label=f"{diameter:.3f} km",
            alpha=0.8,
        )

        # Add count labels above bars with values
        for k, (bar, count) in enumerate(zip(bars, count_values)):
            if count > 0:  # Only add labels for bars with data
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.5,
                    f"n={count}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                )

    # Position x-ticks at the center of each decade group
    ax.set_xticks(x)
    ax.set_xticklabels(unique_decades)

    # Add some padding to x-axis limits
    ax.set_xlim(min(x) - 0.5, max(x) + 0.5)

    # Set axis labels and title
    ax.set_xlabel("Impact Decade")
    ax.set_ylabel("Mean Days from Discovery to 0.01% Impact Probability")
    ax.set_title("Mean Realization Time by Impact Decade and Diameter")

    # Add grid for better readability
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)

    # Add legend
    ax.legend(
        title="Diameter [km]", frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left"
    )

    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    return fig, ax


def make_analysis_plots(
    summary: ImpactorResultSummary,
    window_results: WindowResult,
    out_dir: str,
) -> None:

    for diameter in summary.orbit.diameter.unique().to_pylist():
        fig, ax = plot_observed_vs_unobserved_elements(summary, diameter=diameter)
        fig.savefig(
            os.path.join(out_dir, f"observed_vs_unobserved_elements_{diameter}km.jpg"),
            bbox_inches="tight",
            dpi=200,
        )
        plt.close(fig)

    fig, ax = plot_discovered_by_diameter_impact_period(
        summary,
        period="5year",
    )
    fig.savefig(
        os.path.join(out_dir, "discovered_by_diameter_5year.jpg"),
        bbox_inches="tight",
        dpi=200,
    )
    plt.close(fig)

    fig, ax = plot_discovered_by_diameter_impact_period(
        summary, period="5year", max_impact_time=Timestamp.from_iso8601(["2070-01-01"])
    )
    fig.savefig(
        os.path.join(out_dir, "discovered_by_diameter_5year_2070.jpg"),
        bbox_inches="tight",
        dpi=200,
    )
    plt.close(fig)

    fig, ax = plot_iawn_threshold_reached_by_diameter_impact_time(
        summary, period="5year", max_impact_time=Timestamp.from_iso8601(["2140-01-01"])
    )
    fig.savefig(
        os.path.join(out_dir, "iawn_threshold_reached_by_diameter_5year.jpg"),
        bbox_inches="tight",
        dpi=200,
    )
    plt.close(fig)

    fig, ax = plot_max_impact_probability_histograms_by_diameter_decade(
        summary, include_undiscovered=True,
        x_log_scale=True
    )
    fig.savefig(
        os.path.join(
            out_dir, "max_impact_probability_histograms_by_diameter_decade_all.jpg"
        ),
        bbox_inches="tight",
        dpi=200,
    )
    logger.info(
        "Generated max impact probability histograms by diameter decade plot (all)"
    )

    fig, ax = plot_max_impact_probability_histograms_by_diameter_decade(
        summary, include_undiscovered=False,
        x_log_scale=True
    )
    fig.savefig(
        os.path.join(
            out_dir,
            "max_impact_probability_histograms_by_diameter_decade_discovered.jpg",
        ),
        bbox_inches="tight",
        dpi=200,
    )
    logger.info(
        "Generated max impact probability histograms by diameter decade plot (discovered)"
    )
    plt.close(fig)

    fig, ax = plot_warning_time_histogram(summary)
    fig.savefig(
        os.path.join(out_dir, "warning_time_histogram.jpg"),
        bbox_inches="tight",
        dpi=200,
    )
    logger.info("Generated warning time histogram")
    plt.close(fig)

    fig, ax = plot_warning_time_by_diameter_year(summary)
    fig.savefig(
        os.path.join(out_dir, "warning_time_by_diameter_year.jpg"),
        bbox_inches="tight",
        dpi=200,
    )
    logger.info("Generated warning time by diameter year plot")
    plt.close(fig)

    fig, ax = plot_realization_time_histogram(summary)
    fig.savefig(
        os.path.join(out_dir, "realization_time_histogram.jpg"),
        bbox_inches="tight",
        dpi=200,
    )
    logger.info("Generated realization time histogram")
    plt.close(fig)

    fig, ax = plot_mean_realization_time_by_diameter_decade(summary)
    fig.savefig(
        os.path.join(out_dir, "mean_realization_time_by_diameter_decade.jpg"),
        bbox_inches="tight",
        dpi=200,
    )
    logger.info("Generated realization time by diameter decade plot")
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

    fig, ax = plot_arclength_by_diameter(summary)
    fig.savefig(
        os.path.join(out_dir, "arclength_by_diameter.jpg"),
        bbox_inches="tight",
        dpi=200,
    )
    logger.info("Generated arclength by diameter plot")
    plt.close(fig)

    fig, ax = plot_discovered_by_diameter_decade(summary)
    fig.savefig(
        os.path.join(out_dir, "discovered_by_diameter_decade.jpg"),
        bbox_inches="tight",
        dpi=200,
    )
    logger.info("Generated discovered by diameter decade plot")
    plt.close(fig)

    fig, ax = plot_not_realized_by_diameter_decade(summary)
    fig.savefig(
        os.path.join(out_dir, "not_realized_by_diameter_decade.jpg"),
        bbox_inches="tight",
        dpi=200,
    )
    logger.info("Generated percentage realized plot")
    plt.close(fig)

    fig, ax = plot_max_impact_probability_by_diameter_decade(summary)
    fig.savefig(
        os.path.join(out_dir, "max_impact_probability_by_diameter_decade.jpg"),
        bbox_inches="tight",
        dpi=200,
    )
    logger.info("Generated max impact probability plot")
    plt.close(fig)

    fig, ax = plot_iawn_threshold_not_reached_by_diameter_decade(summary)
    fig.savefig(
        os.path.join(out_dir, "iawn_threshold_by_diameter_decade.jpg"),
        bbox_inches="tight",
        dpi=200,
    )
    logger.info("Generated iawn threshold plot")
    plt.close(fig)

    fig, ax = plot_arc_length_by_diameter_decade(summary)
    fig.savefig(
        os.path.join(out_dir, "arc_length_by_diameter_decade.jpg"),
        bbox_inches="tight",
        dpi=200,
    )
    logger.info("Generated arc length by diameter decade plot")
    plt.close(fig)

    fig, ax = plot_1_percent_ip_threshold_percentage_vs_elements(summary)
    fig.savefig(
        os.path.join(out_dir, "1_percent_ip_threshold_percentage_vs_elements.jpg"),
        bbox_inches="tight",
        dpi=200,
    )
    logger.info("Generated 1% IP threshold percentage vs elements plot")
    plt.close(fig)

    fig, ax = plot_elements(summary)
    fig.savefig(
        os.path.join(out_dir, "elements.jpg"),
        bbox_inches="tight",
        dpi=200,
    )
    logger.info("Generated elements plot")
    plt.close(fig)

    fig, ax = plot_collective_ip_over_time(window_results)
    fig.savefig(
        os.path.join(out_dir, "collective_ip_over_time.jpg"),
        bbox_inches="tight",
        dpi=200,
    )
    logger.info("Generated collective IP over time plot")
    plt.close(fig)


def plot_max_impact_probability_histograms_by_diameter_decade(
    summary: ImpactorResultSummary,
    include_undiscovered: bool = False,
    y_log_scale: bool = False,
    x_log_scale: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a grid of histograms showing the maximum impact probability distribution
    for each diameter and decade combination. Diameters are arranged in rows and
    decades in columns.

    Parameters
    ----------
    summary : ImpactorResultSummary
        The summary of impact study results.
    include_undiscovered : bool, optional
        Whether to include undiscovered objects in the histograms. If False,
        only discovered objects are included. Default is False.
    y_log_scale : bool, optional
        Whether to use a log scale for the y-axis. Default is False.
    x_log_scale : bool, optional
        Whether to use a log scale for the x-axis. Default is False.


    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        The figure and axes objects for the plot.
    """
    # Filter to only include complete results
    summary = summary.apply_mask(summary.complete())
    logger.info(f"Total complete results: {len(summary)}")

    if not include_undiscovered:
        summary = summary.apply_mask(summary.discovered())
        logger.info(f"Total discovered results: {len(summary)}")

    # Get common data
    impact_decades, unique_decades, unique_diameters = (
        summary.get_diameter_impact_period_data(period_breakdown="decade")
    )

    # Sort decades and diameters for consistent ordering
    unique_decades = np.sort(unique_decades)
    unique_diameters = np.sort(unique_diameters)

    # Create figure with subplots
    n_rows = len(unique_diameters)
    n_cols = len(unique_decades)
    fig, axes = plt.subplots(
        n_rows, n_cols, dpi=200, figsize=(3 * n_cols, 2.5 * n_rows)
    )

    # Define a colormap for consistency with other plots
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_diameters)))

    max_y = 0

    # Second pass: plot the histograms
    for i, diameter in enumerate(unique_diameters):
        for j, decade in enumerate(unique_decades):
            # Get the current axis
            ax = axes[i, j] if n_rows > 1 else axes[j]

            # Filter data for this diameter and decade
            diameter_mask = (
                summary.orbit.diameter.to_numpy(zero_copy_only=False) == diameter
            )
            decade_mask = impact_decades == decade
            combined_mask = diameter_mask & decade_mask

            # Get maximum impact probabilities for this group
            max_impact_probs = summary.maximum_impact_probability.to_numpy(
                zero_copy_only=False
            )
            max_impact_probs = np.where(np.isnan(max_impact_probs), 0, max_impact_probs)
            max_impact_probs = max_impact_probs[combined_mask]

            # Skip if no data
            if len(max_impact_probs) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                continue

            # Create histogram
            if x_log_scale:
                bins = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
                ax.hist(
                    np.where(max_impact_probs == 0, 0.00001, max_impact_probs),
                    bins=bins,
                    color=colors[i],
                    alpha=0.7,
                )
                ax.set_xscale('log')
                # Set the x-axis ticks to match our bin edges
                ax.set_xticks(bins)
                # Format the tick labels to be more readable
                ax.set_xticklabels([f"{x:.0e}" for x in bins])
                # Rotate the tick labels
                ax.tick_params(axis='x', rotation=45)
            else:
                ax.hist(
                    max_impact_probs,
                    range=(0, 1),
                    bins=20,
                    color=colors[i],
                    alpha=0.7,
                )

            # Get the y-limit
            y_limit = ax.get_ylim()[1]
            max_y = max(max_y, y_limit)

            # Add grid
            ax.grid(True, alpha=0.3, linestyle="--")

            # Only add x-axis label to bottom row
            if i == n_rows - 1:
                ax.set_xlabel(f"{decade}s")

            # Only add y-axis label to leftmost column
            if j == 0:
                if y_log_scale:
                    ax.set_ylabel("Count (log scale)")
                else:
                    ax.set_ylabel("Count")

    # Collect all the y-limits and set them all to be the same max value
    axes_flat = axes.flatten()
    for ax in axes_flat:
        ax.set_ylim(0.1, max_y * 1.1)  # Start at 0.1 to avoid log(0) issues

    if y_log_scale:
        # Make the y axis log scale with proper tick formatting
        for ax in axes_flat:
            ax.set_yscale("log")
            # Set major ticks at powers of 10
            ax.yaxis.set_major_locator(plt.LogLocator(base=10, numticks=5))
            # Set minor ticks between major ticks
            ax.yaxis.set_minor_locator(
                plt.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=5)
            )
            # Format the tick labels to be more readable
            ax.yaxis.set_major_formatter(plt.ScalarFormatter())
            ax.grid(True, which="major", alpha=0.3, linestyle="--")

    # Add a single legend for all plots
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color=colors[i], label=f"{d:.3f} km")
        for i, d in enumerate(unique_diameters)
    ]
    fig.legend(
        handles=legend_elements,
        title="Diameter",
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
        frameon=False,
    )

    # Add overall title
    if include_undiscovered:
        title = "Maximum Impact Probability Distribution (Including Undiscovered)"
    else:
        title = "Maximum Impact Probability Distribution (Only Discovered)"
    fig.suptitle(title, y=0.99)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])

    return fig, axes


class DiscoveredByDiameterImpactPeriod(qv.Table):
    diameter = qv.Float64Column()
    impact_period = qv.Int64Column()
    num_discovered_above_50_percent = qv.Int64Column()
    num_discovered_1_percent_to_50_percent = qv.Int64Column()
    num_discovered_below_1_percent = qv.Int64Column()
    total_including_undiscovered = qv.Int64Column()


def plot_discovered_by_diameter_impact_period(
    summary: ImpactorResultSummary,
    period: Literal["year", "5year", "decade"] = "5year",
    max_impact_time: Optional[Timestamp] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the percentage of discovered objects broken down by diameter and impact period.
    Each bar is further subdivided into objects that reached the 1% IAWN threshold (solid)
    and those that did not (patterned).

    Parameters
    ----------
    summary : ImpactorResultSummary
        The summary of impact study results.
    period : Literal["year", "5year", "decade"], optional
        The time period to group impacts by. Default is "5year".
    max_impact_time : Optional[Timestamp], optional
        The maximum impact time to consider. If None, all impacts are included.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        The figure and axes objects for the plot.
    """
    # Filter to only include complete results
    summary = summary.apply_mask(summary.complete())

    if max_impact_time is not None:
        logger.info(
            f"Filtering {len(summary)} objects that impact before {max_impact_time.to_astropy().iso}"
        )
        summary = summary.apply_mask(
            pc.less_equal(summary.orbit.impact_time.mjd(), max_impact_time.mjd()[0])
        )
        logger.info(f"After filtering, {len(summary)} objects remain")

    # Get common data
    impact_periods, unique_periods, unique_diameters = (
        summary.get_diameter_impact_period_data(period_breakdown=period)
    )

    discovered_by_diameter_period = DiscoveredByDiameterImpactPeriod.empty()

    for impact_period in unique_periods:
        orbits_at_impact_period = summary.apply_mask(impact_periods == impact_period)
        for diameter in unique_diameters:
            orbits_at_diameter_and_period = orbits_at_impact_period.select(
                "orbit.diameter", diameter
            )

            discovered_orbits_at_diameter_and_period = (
                orbits_at_diameter_and_period.apply_mask(
                    orbits_at_diameter_and_period.discovered()
                )
            )

            # Count discovered objects that reach 50% threshold
            reaching_50_percent_mask = pc.invert(
                pc.is_null(
                    discovered_orbits_at_diameter_and_period.ip_threshold_50_percent.mjd()
                )
            )
            num_above_50_percent = pc.sum(reaching_50_percent_mask).as_py()

            # Count discovered objects that reach 1% threshold but do not reach 50% threshold
            reaching_1_percent_mask = pc.and_(
                pc.invert(
                    pc.is_null(
                        discovered_orbits_at_diameter_and_period.ip_threshold_1_percent.mjd()
                    )
                ),
                pc.is_null(
                    discovered_orbits_at_diameter_and_period.ip_threshold_50_percent.mjd()
                ),
            )
            num_above_1_percent_below_50_percent = pc.sum(
                reaching_1_percent_mask
            ).as_py()

            # Count discovered objects that do not reach the 1% threshold
            num_below_1_percent = pc.sum(
                pc.is_null(
                    discovered_orbits_at_diameter_and_period.ip_threshold_1_percent.mjd()
                )
            ).as_py()

            discovered_by_diameter_period = qv.concatenate(
                [
                    discovered_by_diameter_period,
                    DiscoveredByDiameterImpactPeriod.from_kwargs(
                        impact_period=[impact_period],
                        diameter=[diameter],
                        num_discovered_above_50_percent=[num_above_50_percent],
                        num_discovered_1_percent_to_50_percent=[
                            num_above_1_percent_below_50_percent
                        ],
                        num_discovered_below_1_percent=[num_below_1_percent],
                        total_including_undiscovered=[
                            len(orbits_at_diameter_and_period)
                        ],
                    ),
                ]
            )

    # Create the plot with improved spacing
    fig, ax = plt.subplots(
        1, 1, dpi=200, figsize=(12, 6)
    )  # Wider figure for better spacing

    # Calculate bar width and spacing based on number of diameters
    num_diameters = len(unique_diameters)
    group_width = BAR_GROUP_WIDTH  # Width allocated for each period group
    bar_width = (
        group_width / num_diameters * BAR_WIDTH_SCALE
    )  # Slightly narrower bars for spacing between them

    # Create evenly spaced x positions for period groups
    x = np.arange(len(unique_periods)) * (
        1 + BAR_GROUP_SPACING
    )  # Add spacing between period groups

    # Define a colormap
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_diameters)))

    # Plot bars for each diameter
    for i, diameter in enumerate(unique_diameters):
        diameter_data = discovered_by_diameter_period.select("diameter", diameter)

        # Calculate x position for this diameter's bars within each group
        offset = (i - num_diameters / 2 + 0.5) * (
            bar_width * 1.1
        )  # Add 10% spacing between bars
        bar_positions = x + offset

        # Get the data for this diameter
        above_50_percent = diameter_data.num_discovered_above_50_percent.to_numpy(
            zero_copy_only=False
        )
        above_1_percent = diameter_data.num_discovered_1_percent_to_50_percent.to_numpy(
            zero_copy_only=False
        )
        below_1_percent = diameter_data.num_discovered_below_1_percent.to_numpy(
            zero_copy_only=False
        )
        total = diameter_data.total_including_undiscovered.to_numpy(
            zero_copy_only=False
        )

        # Calculate percentages
        pct_above_50 = above_50_percent / total * 100
        pct_above_1 = above_1_percent / total * 100
        pct_below_1 = below_1_percent / total * 100

        # Plot stacked bars
        ax.bar(
            bar_positions,
            pct_above_50,
            width=bar_width,
            color=colors[i],
            label=f"{diameter:.3f} km",
            alpha=0.8,
        )

        ax.bar(
            bar_positions,
            pct_above_1,
            bottom=pct_above_50,
            width=bar_width,
            color=colors[i],
            alpha=0.6,
            label="_nolegend_",
        )

        ax.bar(
            bar_positions,
            pct_below_1,
            bottom=pct_above_50 + pct_above_1,
            width=bar_width,
            color=colors[i],
            alpha=0.4,
            hatch="///",
            label="_nolegend_",
        )

    # Position x-ticks at the center of each period group
    ax.set_xticks(x)

    # Format tick labels to show year ranges for multi-year periods
    tick_labels = []
    for period_start in unique_periods:
        if isinstance(period_start, str):
            tick_labels.append(period_start)
        else:
            if period == "5year":
                tick_labels.append(f"{period_start}-{period_start+4}")
            elif period == "decade":
                tick_labels.append(f"{period_start}-{period_start+9}")
            else:
                tick_labels.append(str(period_start))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    # Add some padding to x-axis limits
    ax.set_xlim(min(x) - 0.5, max(x) + 0.5)
    ax.set_ylim(0, 100)

    if period == "year":
        ax.set_xlabel("Impact Year")
    elif period == "5year":
        ax.set_xlabel("Impact 5-Year Period")
    elif period == "decade":
        ax.set_xlabel("Impact Decade")

    ax.set_ylabel("Percentage of Discovered Objects")
    ax.set_title("Percentage of Discovered Objects by Diameter and Impact Period")

    # Add legend entries for the hatch pattern
    pattern_legend_elements = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor="gray",
            alpha=0.5,
            hatch="///",
            label="Below 1% IP Threshold",
        ),
        plt.Rectangle(
            (0, 0), 1, 1, facecolor="gray", alpha=0.6, label="1% - 50% IP Threshold"
        ),
        plt.Rectangle(
            (0, 0), 1, 1, facecolor="gray", alpha=0.8, label="Above 50% IP Threshold"
        ),
    ]

    # Create two legends - one for diameters and one for patterns
    diameter_legend = ax.legend(
        title="Diameter [km]", frameon=True, bbox_to_anchor=(1.01, 1), loc="upper left"
    )
    ax.add_artist(diameter_legend)

    # Create second legend for patterns
    pattern_legend = ax.legend(
        handles=pattern_legend_elements,
        frameon=True,
        bbox_to_anchor=(1.01, 0.5),
        loc="center left",
    )

    ax.yaxis.grid(True, linestyle="--", alpha=0.7)

    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    return fig, ax


def plot_observed_vs_unobserved_elements(
    summary: ImpactorResultSummary,
    diameter: float = 1,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the distribution of objects in orbital element space with three categories:
    - Discovered (blue)
    - Observed but not discovered (yellow)
    - Unobserved (red)

    Parameters
    ----------
    summary : ImpactorResultSummary
        The summary of impact study results.
    diameter : float, optional
        The diameter [km] to filter the results by, by default 1.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        The figure and axes objects for the plot.
    """
    summary = summary.apply_mask(summary.complete())
    # Filter to only include the given diameter

    summary = summary.apply_mask(pc.equal(summary.orbit.diameter, diameter))

    orbits_at_diameter = summary.apply_mask(pc.equal(summary.orbit.diameter, diameter))

    # Get Keplerian coordinates
    kep_coordinates = summary.orbit.coordinates.to_keplerian()
    a_au = kep_coordinates.a.to_numpy(zero_copy_only=False)
    i_deg = kep_coordinates.i.to_numpy(zero_copy_only=False)
    e = kep_coordinates.e.to_numpy(zero_copy_only=False)

    # print discovered objedts by those with a non null discovery time
    discovered_objects_mask = pc.invert(
        pc.is_null(summary.discovery_time.days)
    ).to_numpy(zero_copy_only=False)
    # do the same for undiscovered by the null discovery time
    observed_not_discovered_objects_mask = pc.and_(
        pc.is_null(summary.discovery_time.days),
        pc.greater(summary.observations, 0),
    ).to_numpy(zero_copy_only=False)
    unobserved_objects_mask = pc.and_(
        pc.is_null(summary.discovery_time.days), pc.equal(summary.observations, 0)
    ).to_numpy(zero_copy_only=False)

    assert np.sum(discovered_objects_mask) + np.sum(
        observed_not_discovered_objects_mask
    ) + np.sum(unobserved_objects_mask) == len(orbits_at_diameter)
    # Create the plots
    fig, axes = plt.subplots(1, 2, dpi=200, figsize=(18, 7))

    scatter_dot_size = 10
    scatter_dot_alpha = 0.3
    # Uses contrasting colors that don't include yellow
    # colors = plt.cm.coolwarm(np.linspace(0, 1, 3))
    # colors = plt.cm.viridis(np.linspace(0, 1, 3))
    colors = ["blue", "green", "red"]

    # --- Plot a vs i ---
    # Plot discovered (blue) first
    axes[0].scatter(
        a_au[discovered_objects_mask],
        i_deg[discovered_objects_mask],
        c=colors[0],
        alpha=scatter_dot_alpha,
        linewidths=0,
        label="Discovered",
        s=scatter_dot_size,
    )
    # Plot observed but not discovered (yellow) second
    axes[0].scatter(
        a_au[observed_not_discovered_objects_mask],
        i_deg[observed_not_discovered_objects_mask],
        c=colors[1],
        alpha=scatter_dot_alpha,
        linewidths=0,
        label="Observed (Not Discovered)",
        s=scatter_dot_size,
    )
    # Plot unobserved (red) last
    axes[0].scatter(
        a_au[unobserved_objects_mask],
        i_deg[unobserved_objects_mask],
        c=colors[2],
        alpha=scatter_dot_alpha,
        linewidths=0,
        label="Unobserved",
        s=scatter_dot_size,
    )

    axes[0].set_xlabel("Semimajor Axis (a) [AU]")
    axes[0].set_ylabel("Inclination (i) [deg]")
    axes[0].set_title("a vs i")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Plot a vs e ---
    # Plot discovered (blue) first
    axes[1].scatter(
        a_au[discovered_objects_mask],
        e[discovered_objects_mask],
        c=colors[0],
        alpha=scatter_dot_alpha,
        linewidths=0,
        label="Discovered",
        s=scatter_dot_size,
    )
    # Plot observed but not discovered (yellow) second
    axes[1].scatter(
        a_au[observed_not_discovered_objects_mask],
        e[observed_not_discovered_objects_mask],
        c=colors[1],
        alpha=scatter_dot_alpha,
        linewidths=0,
        label="Observed (Not Discovered)",
        s=scatter_dot_size,
    )
    # Plot unobserved (red) last
    axes[1].scatter(
        a_au[unobserved_objects_mask],
        e[unobserved_objects_mask],
        c=colors[2],
        alpha=scatter_dot_alpha,
        linewidths=0,
        label="Unobserved",
        s=scatter_dot_size,
    )

    axes[1].set_xlabel("Semimajor Axis (a) [AU]")
    axes[1].set_ylabel("Eccentricity (e)")
    axes[1].set_title("a vs e")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Add overall title with statistics
    n_total = len(orbits_at_diameter)
    n_discovered = np.sum(discovered_objects_mask)
    n_observed_not_discovered = np.sum(observed_not_discovered_objects_mask)
    n_unobserved = np.sum(unobserved_objects_mask)
    assert n_total == n_discovered + n_observed_not_discovered + n_unobserved

    percent_discovered = (n_discovered / n_total) * 100
    percent_observed_not_discovered = (n_observed_not_discovered / n_total) * 100
    percent_unobserved = (n_unobserved / n_total) * 100

    fig.suptitle(
        f"Distribution of Objects (Diameter: {diameter} km)\n"
        f"Total Objects: {n_total}, "
        f"Discovered: {n_discovered} ({percent_discovered:.1f}%), "
        f"Observed Not Discovered: {n_observed_not_discovered} ({percent_observed_not_discovered:.1f}%), "
        f"Unobserved: {n_unobserved} ({percent_unobserved:.1f}%)"
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig, axes
