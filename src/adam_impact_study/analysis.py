import glob
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
from adam_core.time import Timestamp

from adam_impact_study.types import (
    ImpactorOrbits,
    ImpactorResultSummary,
    Observations,
    WindowResult,
)
from adam_impact_study.utils import get_study_paths

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WarningTimes(qv.Table):
    orbit_id = qv.LargeStringColumn()
    warning_time = qv.Float64Column(nullable=True)


def compute_warning_time(
    impactor_orbits: ImpactorOrbits,
    results: WindowResult,
    threshold: float = 1e-4,
) -> WarningTimes:
    """
    Compute the warning time for each object using their impact study results and their impact time.

    The warning time is the time between the first window where the object's impact probability is above the threshold and the impact time.
    The time difference is calculated as the difference between the last observation in the window and the impact time.

    If the object's impact probability is below the threshold for all windows, the warning time is set to null.

    Parameters
    ----------
    impactor_orbits: ImpactorOrbits
        The impactor orbits to compute the warning time for.
    results: ImpactStudyResults
        The impact study results to compute the warning time for.
    threshold: float, optional
        The threshold for the impact probability. Default is 1e-4.

    Returns
    -------
    WarningTimes
        The warning times for each object.
    """
    # Sort results by object_id and observation_end
    results_sorted = results.sort_by(
        ["orbit_id", "observation_end.days", "observation_end.nanos"]
    )

    # Filter results to cases where impact probability is above threshold
    filtered_results = results_sorted.apply_mask(
        pc.greater_equal(pc.fill_null(results_sorted.impact_probability, 0), threshold)
    )

    # Drop duplicates and keep the first instance
    filtered_results = filtered_results.drop_duplicates(subset=["orbit_id"], keep="first")

    # import pdb; pdsb.set_trace()
    # Convert last observation time to an MJD
    filtered_results = (
        filtered_results.flattened_table()
        .append_column("observation_end_mjd", filtered_results.observation_end.mjd())
        .select(["orbit_id", "observation_end_mjd"])
    )

    # Join with impactor orbits to get impact time
    impactors_table_time = (
        impactor_orbits.flattened_table()
        .append_column("impact_time_mjd", impactor_orbits.impact_time.mjd())
        .select(["orbit_id", "impact_time_mjd"])
    )
    impactors_table_time = impactors_table_time.join(
        filtered_results, "orbit_id", "orbit_id"
    )
    impactors_table_time = impactors_table_time.append_column(
        "warning_time",
        pc.subtract(
            impactors_table_time["impact_time_mjd"],
            impactors_table_time["observation_end_mjd"],
        ),
    )

    return WarningTimes.from_pyarrow(
        impactors_table_time.select(["orbit_id", "warning_time"]).combine_chunks()
    ).sort_by([("orbit_id", "ascending")])


class DiscoveryDates(qv.Table):
    orbit_id = qv.LargeStringColumn()
    discovery_date = Timestamp.as_column(nullable=True)


def compute_discovery_dates(
    impactor_orbits: ImpactorOrbits,
    results: WindowResult,
) -> DiscoveryDates:
    """
    Return when each object is considered to be discoverable.

    TODO: Define what "discoverable" means in more detail.

    Parameters
    ----------
    impactor_orbits: ImpactorOrbits
        The impactor orbits to compute the discovery dates for.
    results: ImpactStudyResults
        The impact study results to compute the discovery dates for.

    Returns
    -------
    DiscoveryDates
        The discovery dates for each object.
    """
    # For now, we will consider an object discoverable if it has 3 unique nights of data
    results_sorted = results.sort_by(
        ["orbit_id", "observation_end.days", "observation_end.nanos"]
    )

    # only consider the first 3 nights of data
    results_sorted = results_sorted.apply_mask(
        pc.greater_equal(results_sorted.observation_nights, 3)
    )

    results_sorted = results_sorted.drop_duplicates(subset=["orbit_id"], keep="first")

    discovery_dates = DiscoveryDates.from_kwargs(
        orbit_id=results_sorted.orbit_id,
        discovery_date=results_sorted.observation_end,
    )

    # get the orbit_ids which did not have 3 unique nights of data
    orbit_ids_without_3_nights = set(results.orbit_id.unique().to_pylist()) - set(
        results_sorted.orbit_id.unique().to_pylist()
    )

    non_discovery_dates = DiscoveryDates.from_kwargs(
        orbit_id=list(orbit_ids_without_3_nights),
        discovery_date=Timestamp.nulls(
            len(orbit_ids_without_3_nights),
            scale="utc",
        ),
    )

    discovery_dates = qv.concatenate([discovery_dates, non_discovery_dates])

    return discovery_dates


class RealizationTimes(qv.Table):
    orbit_id = qv.LargeStringColumn()
    realization_time = qv.Float64Column(nullable=True)


def compute_realization_time(
    impactor_orbits: ImpactorOrbits,
    results: WindowResult,
    discovery_dates: DiscoveryDates,
    threshold: float = 1e-9,
) -> RealizationTimes:
    """
    Compute the realization time for each object using their impact study results and their impact time.

    Realization time is defined as the time between discovery and the first window where the orbit's
    impact probability is above the threshold.

    If the object is not discoverable, the realization time is set to null.

    Parameters
    ----------
    impactor_orbits: ImpactorOrbits
        The impactor orbits to compute the realization time for.
    results: ImpactStudyResults
        The impact study results to compute the realization time for.
    discovery_dates: DiscoveryDates
        The discovery dates for each object.
    threshold: float, optional
        The threshold for the impact probability. Default is 1e-9.

    Returns
    -------
    RealizationTimes
        The realization times for each object.
    """
    results_sorted = results.sort_by(
        ["orbit_id", "observation_end.days", "observation_end.nanos"]
    )

    # Filter results to cases where impact probability is above threshold
    filtered_results = results_sorted.apply_mask(
        pc.greater_equal(pc.fill_null(results_sorted.impact_probability, 0), threshold)
    )

    # Drop duplicates and keep the first instance
    filtered_results = filtered_results.drop_duplicates(subset=["orbit_id"])

    # Convert last observation time to an MJD
    filtered_results_table = filtered_results.flattened_table().append_column(
        "observation_end_mjd", filtered_results.observation_end.mjd()
    )

    # Convert discovery date to an MJD
    discovery_dates_table = discovery_dates.flattened_table().append_column(
        "discovery_date_mjd", discovery_dates.discovery_date.mjd()
    )

    # Join with discovery dates to get discovery_time
    realization_table = (
        impactor_orbits.flattened_table()
        .select(["orbit_id"])
        .join(
            filtered_results_table.select(["orbit_id", "observation_end_mjd"]),
            "orbit_id",
            "orbit_id",
        )
    )
    realization_table = realization_table.join(
        discovery_dates_table, "orbit_id", "orbit_id"
    )
    realization_table = realization_table.append_column(
        "realization_time",
        pc.subtract(
            realization_table["observation_end_mjd"],
            realization_table["discovery_date_mjd"],
        ),
    ).sort_by([("orbit_id", "ascending")])

    return RealizationTimes.from_pyarrow(
        realization_table.select(["orbit_id", "realization_time"]).combine_chunks()
    ).sort_by([("orbit_id", "ascending")])


class ObservationCadence(qv.Table):
    orbit_id = qv.LargeStringColumn()
    tracklets = qv.UInt64Column()
    singletons = qv.UInt64Column()


def compute_observation_cadence(
    observations: Observations,
) -> ObservationCadence:
    """
    Compute the observation cadence for each object (the number of tracklets and singletons) observed overall.

    This is a placeholder function until difi is quivr-ized.

    Parameters
    ----------
    observations: Observations
        The observations to compute the observation cadence for.

    Returns
    -------
    ObservationCadence
        The observation cadence for each object.
    """
    observations_table = observations.flattened_table().select(
        ["orbit_id", "observing_night"]
    )
    observations_grouped = observations_table.group_by(
        ["orbit_id", "observing_night"]
    ).aggregate([("observing_night", "count")])

    # Filter out tracklets and singletons
    tracklets = observations_grouped.filter(
        pc.greater_equal(observations_grouped.column("observing_night_count"), 2)
    )
    singletons = observations_grouped.filter(
        pc.equal(observations_grouped.column("observing_night_count"), 1)
    )

    tracklet_counts = (
        tracklets.group_by("orbit_id")
        .aggregate([("observing_night", "count")])
        .rename_columns(["orbit_id", "tracklets"])
    )
    singleton_counts = (
        singletons.group_by("orbit_id")
        .aggregate([("observing_night", "count")])
        .rename_columns(["orbit_id", "singletons"])
    )

    # Create a table off of the unique orbit ids so we
    # can populate empty singleton and tracklet counts
    # with 0s
    orbit_id_table = pa.Table.from_arrays(
        [observations.orbit_id.unique()], ["orbit_id"]
    )

    tracklet_counts = tracklet_counts.join(orbit_id_table, "orbit_id", "orbit_id")
    singleton_counts = singleton_counts.join(orbit_id_table, "orbit_id", "orbit_id")

    observation_cadence = orbit_id_table.join(
        tracklet_counts, "orbit_id", "orbit_id"
    ).join(singleton_counts, "orbit_id", "orbit_id")

    return ObservationCadence.from_kwargs(
        orbit_id=observation_cadence.column("orbit_id"),
        tracklets=observation_cadence.column("tracklets").fill_null(0),
        singletons=observation_cadence.column("singletons").fill_null(0),
    )


def plot_ip_over_time(
    impacting_orbits: ImpactorOrbits,
    impact_study_results: WindowResult,
    run_dir: str,
    survey_start: Timestamp | None = None,
) -> None:
    """
    Plot the impact probability (IP) over time for each object in the provided observations.

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
        ax1.scatter(mjd_times, probabilities)
        ax1.set_xlabel("MJD")
        ax1.set_ylabel("Impact Probability")
        ax1.plot(mjd_times, probabilities)

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
        # Get impact time for this object (30 days after coordinates.time)
        impact_orbit = impacting_orbits.apply_mask(
            pc.equal(impacting_orbits.orbit_id, orbit_id)
        )
        if len(impact_orbit) > 0:
            impact_time = impact_orbit.coordinates.time.add_days(30).mjd()[0].as_py()

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

        plt.title(orbit_id)
        plt.tight_layout()
        plt.savefig(os.path.join(orbit_dir, f"IP_{orbit_id}.png"))
        plt.close()


def collect_orbit_window_results(
    run_dir: str, orbit_id: str
) -> WindowResult:
    paths = get_study_paths(run_dir, orbit_id)
    orbit_dir = paths["orbit_base_dir"]
    window_result_files = glob.glob(f"{orbit_dir}/windows/*/impact_results_{orbit_id}.parquet")
    window_results = WindowResult.empty()
    for f in window_result_files:
        window_results = qv.concatenate([window_results, WindowResult.from_parquet(f)])
    return window_results


def summarize_impact_study_object_results(
    run_dir: str, orbit_id: str
) -> ImpactorResultSummary:
    """
    Summarize the impact study results for a single object.
    """
    paths = get_study_paths(run_dir, orbit_id)
    orbit_dir = paths["orbit_base_dir"]
    impact_results = collect_orbit_window_results(run_dir, orbit_id)
    impactor_orbits = ImpactorOrbits.from_parquet(
        f"{orbit_dir}/impact_orbits_{orbit_id}.parquet"
    )
    discovery_dates = compute_discovery_dates(impactor_orbits, impact_results)
    warning_times = compute_warning_time(impactor_orbits, impact_results)
    realization_times = compute_realization_time(
        impactor_orbits, impact_results, discovery_dates
    )

    import pdb; pdb.set_trace()
    mean_impact_time = Timestamp.from_mjd(
        [pc.mean(impact_results.mean_impact_time.mjd())],
        impact_results.mean_impact_time.scale,
    )

    # Load sorcha observations
    observations = Observations.from_parquet(
        f"{paths['sorcha_dir']}/observations_{orbit_id}.parquet"
    )

    # Compute the number of singletons and tracklets in each window
    observation_cadence = compute_observation_cadence(observations)

    return ImpactorResultSummary.from_kwargs(
        orbit_id=[orbit_id],
        object_id=impact_results[0].object_id,
        mean_impact_time=mean_impact_time,
        windows=[len(impact_results)],
        nights=[pc.max(impact_results.observation_nights)],
        observations=[pc.max(impact_results.observation_count)],
        singletons=[pc.sum(observation_cadence.singletons)],
        tracklets=[pc.sum(observation_cadence.tracklets)],
        observed=[len(observations) > 0],
        discovery_time=discovery_dates.discovery_date,
        warning_time=warning_times.warning_time,
        realization_time=realization_times.realization_time,
        maximum_impact_probability=[pc.max(impact_results.impact_probability)],
    )


def summarize_impact_study_results(run_dir: str) -> ImpactorResultSummary:
    """
    Summarize the impact study results.
    """
    orbit_ids = [os.path.basename(dir) for dir in glob.glob(f"{run_dir}/*")]
    results = ImpactorResultSummary.empty()
    for orbit_id in orbit_ids:
        results = qv.concatenate(
            [results, summarize_impact_study_object_results(run_dir, orbit_id)]
        )
    return results
