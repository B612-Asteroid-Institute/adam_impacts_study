import argparse
import logging
import multiprocessing as mp
import pathlib
from typing import Optional, Tuple, Union

import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
import ray
from adam_core.ray_cluster import initialize_use_ray
from adam_core.time import Timestamp

from adam_impact_study.analysis.collect import collect_all_results
from adam_impact_study.analysis.plots import (
    make_analysis_plots,
    plot_individual_orbit_ip_over_time,
)
from adam_impact_study.types import (
    DiscoveryDates,
    ImpactorOrbits,
    ImpactorResultSummary,
    Observations,
    ResultsTiming,
    WarningTimes,
    WindowResult,
)
from adam_impact_study.utils import get_study_paths

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_discovery_dates(
    observations: Observations,
    min_tracklets: int = 3,
    max_nights: float = 15,
) -> DiscoveryDates:
    """
    Return when each object is considered to be discoverable. Any object is considered
    discoverable if it has at least 3 tracklets within a 15 day window.

    Parameters
    ----------
    observations: Observations
        The observations for a single object.
    min_tracklets: int, optional
        The minimum number of tracklets to consider an object discoverable. Default is 3.
    max_nights: float, optional
        The maximum number of nights over which to find the number of required tracklets. Default is 15.

    Returns
    -------
    DiscoveryDates
        The discovery dates for each object.
    """
    results = DiscoveryDates.empty()
    # Sort observations by observation_end
    orbit_ids = observations.orbit_id.unique().to_pylist()
    for orbit_id in orbit_ids:
        orbit_observations = observations.select("orbit_id", orbit_id)

        observing_nights = orbit_observations.observing_night.unique().sort()
        discovery_time = Timestamp.nulls(
            1, scale=orbit_observations.coordinates.time.scale
        )
        if len(observing_nights) < min_tracklets:
            results = qv.concatenate(
                [
                    results,
                    DiscoveryDates.from_kwargs(
                        orbit_id=[orbit_id],
                        discovery_date=discovery_time,
                    ),
                ]
            )
            continue

        for observing_night in observing_nights[min_tracklets - 1 :].to_pylist():
            observations_window = orbit_observations.apply_mask(
                pc.and_(
                    pc.less_equal(orbit_observations.observing_night, observing_night),
                    pc.greater_equal(
                        orbit_observations.observing_night, observing_night - max_nights
                    ),
                )
            )

            observing_cadence = compute_observation_cadence(observations_window)
            if observing_cadence.tracklets[0].as_py() >= min_tracklets:
                discovery_time = observations_window.coordinates.time.max()
                break

        results = qv.concatenate(
            [
                results,
                DiscoveryDates.from_kwargs(
                    orbit_id=[orbit_id],
                    discovery_date=discovery_time,
                ),
            ]
        )

    return results


def compute_warning_time(
    impactor_orbits: ImpactorOrbits,
    results: WindowResult,
    discovery_dates: DiscoveryDates,
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
    orbit_ids = impactor_orbits.orbit_id.unique().to_pylist()
    results_sorted = results.sort_by(
        ["orbit_id", "observation_end.days", "observation_end.nanos"]
    )

    # Filter results to cases where impact probability is above threshold
    filtered_results = results_sorted.apply_mask(
        pc.greater_equal(pc.fill_null(results_sorted.impact_probability, 0), threshold)
    )

    # Filter results to only include observation end equal to or after discovery date
    filtered_by_discovery_date = WindowResult.empty()
    for orbit_id in orbit_ids:
        orbit_results = filtered_results.select("orbit_id", orbit_id)
        orbit_discovery_date = discovery_dates.select("orbit_id", orbit_id)
        if (
            len(orbit_results) == 0
            or len(orbit_discovery_date) == 0
            or pc.all(pc.is_null(orbit_discovery_date.discovery_date.days)).as_py()
        ):
            # Undiscovered objects have no warning times.
            continue

        if pc.any(pc.is_null(orbit_results.observation_end.days)).as_py():
            print(f"Orbit {orbit_id} has null observation_end")
        if pc.any(pc.is_null(orbit_discovery_date.discovery_date.days)).as_py():
            print(f"Orbit {orbit_id} has null discovery_date")
        orbit_results = orbit_results.apply_mask(
            pc.greater_equal(
                orbit_results.observation_end.mjd(),
                orbit_discovery_date.discovery_date.mjd()[0],
            )
        )
        # There should never be a situation where there is a discovery date,
        # but not windows equal to or after the discovery date.
        assert (
            len(orbit_results) > 0
        ), f"No windows found for orbit {orbit_id} after discovery date"
        filtered_by_discovery_date = qv.concatenate(
            [filtered_by_discovery_date, orbit_results]
        )

    filtered_results = filtered_by_discovery_date

    # Drop duplicates and keep the first instance
    filtered_results = filtered_results.drop_duplicates(
        subset=["orbit_id"], keep="first"
    )

    filtered_results_table = (
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
    filtered_results_table = filtered_results_table.join(
        impactors_table_time, "orbit_id", "orbit_id"
    )
    filtered_results_table = filtered_results_table.append_column(
        "warning_time",
        pc.subtract(
            filtered_results_table["impact_time_mjd"],
            filtered_results_table["observation_end_mjd"],
        ),
    )

    warning_times = WarningTimes.from_pyarrow(
        filtered_results_table.select(["orbit_id", "warning_time"]).combine_chunks()
    )

    # For orbit_ids that do not have a warning time, return nulls
    missing_orbit_ids = set(orbit_ids) - set(warning_times.orbit_id.to_pylist())
    warning_times = qv.concatenate(
        [
            warning_times,
            WarningTimes.from_kwargs(
                orbit_id=list(missing_orbit_ids),
                warning_time=[None] * len(missing_orbit_ids),
            ),
        ]
    )

    warning_times = warning_times.sort_by([("orbit_id", "ascending")])
    return warning_times


class IPThresholdDate(qv.Table):
    orbit_id = qv.LargeStringColumn()
    ip_threshold = qv.Float64Column()
    date = Timestamp.as_column(nullable=True)


def compute_ip_threshold_date(
    impactor_orbits: ImpactorOrbits,
    results: WindowResult,
    threshold: float,
) -> IPThresholdDate:
    """
    Compute the date when the impact probability first reaches a given threshold.
    """

    results_sorted = results.sort_by(
        ["orbit_id", "observation_end.days", "observation_end.nanos"]
    )

    # Filter results to cases where impact probability is above threshold
    filtered_results = results_sorted.apply_mask(
        pc.greater_equal(pc.fill_null(results_sorted.impact_probability, 0), threshold)
    )

    filtered_results_table = filtered_results.drop_duplicates(
        subset=["orbit_id"], keep="first"
    )

    ip_threshold_dates = IPThresholdDate.from_kwargs(
        orbit_id=filtered_results_table.orbit_id,
        ip_threshold=pa.repeat(threshold, len(filtered_results_table)),
        date=filtered_results_table.observation_end,
    )

    null_orbit_ids = set(impactor_orbits.orbit_id.to_pylist()) - set(
        filtered_results_table.orbit_id.to_pylist()
    )

    null_threshold_dates = IPThresholdDate.from_kwargs(
        orbit_id=list(null_orbit_ids),
        ip_threshold=pa.repeat(threshold, len(null_orbit_ids)),
        date=Timestamp.nulls(len(null_orbit_ids), scale="utc"),
    )

    ip_threshold_dates = qv.concatenate([ip_threshold_dates, null_threshold_dates])

    return ip_threshold_dates


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

    Realization time is defined as the time (in days) between discovery and the first window where the orbit's
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

    # Filter results to only include observation end equal to or after discovery date
    filtered_by_discovery_date = WindowResult.empty()
    for orbit_id in filtered_results.orbit_id.unique().to_pylist():
        orbit_results = filtered_results.select("orbit_id", orbit_id)
        orbit_discovery_date = discovery_dates.select("orbit_id", orbit_id)
        if (
            len(orbit_results) == 0
            or len(orbit_discovery_date) == 0
            or pc.all(pc.is_null(orbit_discovery_date.discovery_date.days)).as_py()
        ):
            continue
        filtered_by_discovery_date = qv.concatenate(
            [
                filtered_by_discovery_date,
                orbit_results.apply_mask(
                    pc.greater_equal(
                        orbit_results.observation_end.mjd(),
                        orbit_discovery_date.discovery_date.mjd()[0],
                    )
                ),
            ]
        )

    filtered_results = filtered_by_discovery_date

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


def _select_ip_at_discovery_time(
    orbit_id: str,
    all_window_results: WindowResult,  # Note! Pass in all windows
    discovery_dates: DiscoveryDates,
) -> float:
    """
    Select the impact probability at discovery time for a given orbit.
    """
    orbit_results = all_window_results.select("orbit_id", orbit_id)
    orbit_discovery_date = discovery_dates.select("orbit_id", orbit_id)
    if len(orbit_results) == 0 or len(orbit_discovery_date) == 0:
        return None

    assert (
        len(orbit_discovery_date) == 1
    ), f"{orbit_id} had {len(orbit_discovery_date)} discovery dates, expected 1"

    if pc.all(pc.is_null(orbit_discovery_date.discovery_date.days)).as_py():
        return None

    discovery_window = orbit_results.apply_mask(
        orbit_results.observation_end.equals(
            orbit_discovery_date.discovery_date, precision="ms"
        )
    )

    return discovery_window.impact_probability[0].as_py()


def summarize_impact_study_object_results(
    impactor_orbits: ImpactorOrbits,
    observations: Observations,
    results_timing: ResultsTiming,
    window_results: WindowResult,
    orbit_id: str,
) -> ImpactorResultSummary:
    """
    Summarize the impact study results for a single object.
    """
    impactor_orbit = impactor_orbits.select("orbit_id", orbit_id)
    assert (
        len(impactor_orbit) == 1
    ), f"{orbit_id} had {len(impactor_orbit)} impactor orbits, expected 1"

    orbit_observations = observations.select("orbit_id", orbit_id)
    orbit_results_timing = results_timing.select("orbit_id", orbit_id)
    orbit_window_results = window_results.select("orbit_id", orbit_id)
    orbit_discovery_dates = compute_discovery_dates(orbit_observations)
    completed_window_results = orbit_window_results.apply_mask(
        orbit_window_results.complete()
    )
    orbit_observation_cadence = compute_observation_cadence(orbit_observations)

    all_orbit_windows_completed = pc.all(
        pc.equal(orbit_window_results.status, "complete")
    ).as_py()

    first_observation = orbit_observations.coordinates.time.min()
    last_observation = orbit_observations.coordinates.time.max()

    ip_threshold_0_dot_01_percent = compute_ip_threshold_date(
        impactor_orbit, completed_window_results, 1e-4
    )
    ip_threshold_1_percent = compute_ip_threshold_date(
        impactor_orbit, completed_window_results, 1e-2
    )
    ip_threshold_10_percent = compute_ip_threshold_date(
        impactor_orbit, completed_window_results, 0.1
    )
    ip_threshold_50_percent = compute_ip_threshold_date(
        impactor_orbit, completed_window_results, 0.5
    )
    ip_threshold_90_percent = compute_ip_threshold_date(
        impactor_orbit, completed_window_results, 0.9
    )
    ip_threshold_100_percent = compute_ip_threshold_date(
        impactor_orbit, completed_window_results, 1.0
    )

    ip_at_discovery_time = _select_ip_at_discovery_time(
        orbit_id, orbit_window_results, orbit_discovery_dates
    )

    if len(orbit_observations) == 0:
        return ImpactorResultSummary.from_kwargs(
            orbit=impactor_orbit,
            mean_impact_time=Timestamp.nulls(1, scale="tdb"),
            windows=[0],
            nights=[0],
            observations=[0],
            singletons=[0],
            tracklets=[0],
            first_observation=Timestamp.nulls(1, scale="utc"),
            last_observation=Timestamp.nulls(1, scale="utc"),
            discovery_time=Timestamp.nulls(1, scale="utc"),
            ip_at_discovery_time=[ip_at_discovery_time],
            ip_threshold_0_dot_01_percent=Timestamp.nulls(1, scale="utc"),
            ip_threshold_1_percent=Timestamp.nulls(1, scale="utc"),
            ip_threshold_10_percent=Timestamp.nulls(1, scale="utc"),
            ip_threshold_50_percent=Timestamp.nulls(1, scale="utc"),
            ip_threshold_90_percent=Timestamp.nulls(1, scale="utc"),
            ip_threshold_100_percent=Timestamp.nulls(1, scale="utc"),
            maximum_impact_probability=[0],
            results_timing=orbit_results_timing,
            error=[None],
            status=["complete"],
        )

    # If the observations are not linked, we can return early
    if not pc.all(pc.equal(orbit_observations.linked, True)).as_py():
        return ImpactorResultSummary.from_kwargs(
            orbit=impactor_orbit,
            mean_impact_time=Timestamp.nulls(1, scale="tdb"),
            windows=[len(orbit_window_results)],
            nights=[len(orbit_observations.observing_night.unique())],
            observations=[len(orbit_observations)],
            singletons=[pc.sum(orbit_observation_cadence.singletons)],
            tracklets=[pc.sum(orbit_observation_cadence.tracklets)],
            first_observation=first_observation,
            last_observation=last_observation,
            discovery_time=orbit_discovery_dates.discovery_date,
            ip_at_discovery_time=[ip_at_discovery_time],
            ip_threshold_0_dot_01_percent=ip_threshold_0_dot_01_percent.date,
            ip_threshold_1_percent=ip_threshold_1_percent.date,
            ip_threshold_10_percent=ip_threshold_10_percent.date,
            ip_threshold_50_percent=ip_threshold_50_percent.date,
            ip_threshold_90_percent=ip_threshold_90_percent.date,
            ip_threshold_100_percent=ip_threshold_100_percent.date,
            maximum_impact_probability=[
                pc.max(orbit_window_results.impact_probability)
            ],
            results_timing=orbit_results_timing,
            status=["complete" if all_orbit_windows_completed else "incomplete"],
        )

    if len(orbit_window_results) == 0:
        return ImpactorResultSummary.from_kwargs(
            orbit=impactor_orbit,
            mean_impact_time=Timestamp.nulls(1, scale="tdb"),
            windows=[0],
            nights=[len(orbit_observations.observing_night.unique())],
            observations=[len(orbit_observations)],
            singletons=[pc.sum(orbit_observation_cadence.singletons)],
            tracklets=[pc.sum(orbit_observation_cadence.tracklets)],
            first_observation=first_observation,
            last_observation=last_observation,
            discovery_time=orbit_discovery_dates.discovery_date,
            ip_at_discovery_time=[ip_at_discovery_time],
            ip_threshold_0_dot_01_percent=ip_threshold_0_dot_01_percent.date,
            ip_threshold_1_percent=ip_threshold_1_percent.date,
            ip_threshold_10_percent=ip_threshold_10_percent.date,
            ip_threshold_50_percent=ip_threshold_50_percent.date,
            ip_threshold_90_percent=ip_threshold_90_percent.date,
            ip_threshold_100_percent=ip_threshold_100_percent.date,
            maximum_impact_probability=[0],
            results_timing=orbit_results_timing,
            error=["Orbit has no windows"],
            status=["incomplete"],
        )

    if not all_orbit_windows_completed:
        return ImpactorResultSummary.from_kwargs(
            orbit=impactor_orbit,
            mean_impact_time=Timestamp.nulls(1, scale="tdb"),
            windows=[len(orbit_window_results)],
            nights=[len(orbit_observations.observing_night.unique())],
            observations=[len(orbit_observations)],
            singletons=[pc.sum(orbit_observation_cadence.singletons)],
            tracklets=[pc.sum(orbit_observation_cadence.tracklets)],
            first_observation=first_observation,
            last_observation=last_observation,
            discovery_time=orbit_discovery_dates.discovery_date,
            ip_at_discovery_time=[ip_at_discovery_time],
            ip_threshold_0_dot_01_percent=ip_threshold_0_dot_01_percent.date,
            ip_threshold_1_percent=ip_threshold_1_percent.date,
            ip_threshold_10_percent=ip_threshold_10_percent.date,
            ip_threshold_50_percent=ip_threshold_50_percent.date,
            ip_threshold_90_percent=ip_threshold_90_percent.date,
            ip_threshold_100_percent=ip_threshold_100_percent.date,
            maximum_impact_probability=[
                pc.max(orbit_window_results.impact_probability)
            ],
            results_timing=orbit_results_timing,
            error=["Orbit has incomplete windows"],
            status=["incomplete"],
        )

    mean_impact_mjd = pc.mean(orbit_window_results.mean_impact_time.mjd()).as_py()
    if mean_impact_mjd is None:
        mean_impact_time = Timestamp.nulls(1, scale="tdb")
    else:
        mean_impact_time = Timestamp.from_mjd([mean_impact_mjd], "tdb")

    return ImpactorResultSummary.from_kwargs(
        orbit=impactor_orbit,
        mean_impact_time=mean_impact_time,
        windows=[len(orbit_window_results)],
        nights=[len(orbit_observations.observing_night.unique())],
        observations=[len(orbit_observations)],
        singletons=[pc.sum(orbit_observation_cadence.singletons)],
        tracklets=[pc.sum(orbit_observation_cadence.tracklets)],
        first_observation=first_observation,
        last_observation=last_observation,
        discovery_time=orbit_discovery_dates.discovery_date,
        ip_at_discovery_time=[ip_at_discovery_time],
        ip_threshold_0_dot_01_percent=ip_threshold_0_dot_01_percent.date,
        ip_threshold_1_percent=ip_threshold_1_percent.date,
        ip_threshold_10_percent=ip_threshold_10_percent.date,
        ip_threshold_50_percent=ip_threshold_50_percent.date,
        ip_threshold_90_percent=ip_threshold_90_percent.date,
        ip_threshold_100_percent=ip_threshold_100_percent.date,
        maximum_impact_probability=[pc.max(orbit_window_results.impact_probability)],
        results_timing=orbit_results_timing,
        status=["complete"],
    )


# Create remote version
summarize_impact_study_object_results_remote = ray.remote(
    summarize_impact_study_object_results
)


def summarize_impact_study_results(
    impactor_orbits: ImpactorOrbits,
    observations: Observations,
    results_timing: ResultsTiming,
    window_results: WindowResult,
    out_dir: Union[str, pathlib.Path],
    max_processes: Optional[int] = 1,
) -> ImpactorResultSummary:
    """
    Summarize the impact study results
    """

    # Initialize ray cluster
    use_ray = initialize_use_ray(num_cpus=max_processes)

    unique_orbit_ids = pc.unique(impactor_orbits.orbit_id).to_pylist()

    futures = []
    results = ImpactorResultSummary.empty()

    # Place our objects in the object store if we are using ray
    if use_ray:
        impactor_orbits_ref = ray.put(impactor_orbits)
        observations_ref = ray.put(observations)
        results_timing_ref = ray.put(results_timing)
        window_results_ref = ray.put(window_results)

    for orbit_id in unique_orbit_ids:
        if use_ray:
            futures.append(
                summarize_impact_study_object_results_remote.remote(
                    impactor_orbits_ref,
                    observations_ref,
                    results_timing_ref,
                    window_results_ref,
                    orbit_id,
                )
            )

            if len(futures) >= max_processes * 1.5:
                finished, futures = ray.wait(futures, num_returns=1)
                result = ray.get(finished[0])
                results = qv.concatenate([results, result])

        else:
            results = qv.concatenate(
                [
                    results,
                    summarize_impact_study_object_results(
                        impactor_orbits,
                        observations,
                        results_timing,
                        window_results,
                        orbit_id,
                    ),
                ]
            )

    while len(futures) > 0:
        finished, futures = ray.wait(futures, num_returns=1)
        result = ray.get(finished[0])
        results = qv.concatenate([results, result])

    out_dir_path = pathlib.Path(out_dir).absolute()
    out_dir_path.mkdir(parents=True, exist_ok=True)
    results.to_parquet(out_dir_path / "impactor_results_summary.parquet")
    logger.info(f"Saved impact study results to {out_dir_path}")

    return results


def run_all_analysis(
    run_dir: str,
    out_dir: str,
    summary_plots: bool = True,
    individual_plots: bool = True,
) -> None:
    """
    Perform all analysis on the impact study results.
    """
    # Collect all the results
    impactor_orbits, observations, results_timing, window_results = collect_all_results(
        run_dir
    )

    # Persist collected results to output directory
    impactor_orbits.to_parquet(out_dir / "impactor_orbits.parquet")
    observations.to_parquet(out_dir / "observations.parquet")
    results_timing.to_parquet(out_dir / "results_timing.parquet")
    window_results.to_parquet(out_dir / "window_results.parquet")

    # Summarize the results
    summary_results = summarize_impact_study_results(
        impactor_orbits, observations, results_timing, window_results, out_dir
    )

    # Make the summary plots
    if summary_plots:
        make_analysis_plots(summary_results, window_results, out_dir)

    # Make the individual plots
    if individual_plots:
        plot_individual_orbit_ip_over_time(
            impactor_orbits, window_results, out_dir, summary_results=summary_results
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--summary-plots", type=bool, default=True)
    parser.add_argument("--individual-plots", type=bool, default=True)
    args = parser.parse_args()
    run_all_analysis(
        args.run_dir, args.out_dir, args.summary_plots, args.individual_plots
    )
