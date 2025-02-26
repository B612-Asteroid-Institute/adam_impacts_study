import glob
import logging
import multiprocessing as mp
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
import ray
from adam_core.ray_cluster import initialize_use_ray
from adam_core.time import Timestamp

from adam_impact_study.analysis.plots import make_analysis_plots
from adam_impact_study.analysis.utils import collect_orbit_window_results
from adam_impact_study.types import (
    ImpactorOrbits,
    ImpactorResultSummary,
    Observations,
    ResultsTiming,
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
    filtered_results = filtered_results.drop_duplicates(
        subset=["orbit_id"], keep="first"
    )

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


def summarize_impact_study_object_results(
    run_dir: str, orbit_id: str
) -> ImpactorResultSummary:
    """
    Summarize the impact study results for a single object.
    """
    paths = get_study_paths(run_dir, orbit_id)
    orbit_dir = paths["orbit_base_dir"]
    impact_results = collect_orbit_window_results(run_dir, orbit_id)
    impactor_orbits = ImpactorOrbits.from_parquet(f"{orbit_dir}/impactor_orbit.parquet")

    if len(impact_results) == 0:
        return ImpactorResultSummary.from_kwargs(
            orbit=impactor_orbits,
            mean_impact_time=Timestamp.nulls(1, scale="tdb"),
            windows=[0],
            nights=[0],
            observations=[0],
            singletons=[0],
            tracklets=[0],
            discovery_time=Timestamp.nulls(1, scale="utc"),
            maximum_impact_probability=[0],
        )

    results_timings = ResultsTiming.from_parquet(f"{orbit_dir}/timings.parquet")

    discovery_dates = compute_discovery_dates(impact_results)
    warning_times = compute_warning_time(impactor_orbits, impact_results)
    realization_times = compute_realization_time(
        impactor_orbits, impact_results, discovery_dates
    )

    mean_impact_mjd = pc.mean(impact_results.mean_impact_time.mjd()).as_py()
    if mean_impact_mjd is None:
        mean_impact_time = Timestamp.nulls(1, scale="tdb")
    else:
        mean_impact_time = Timestamp.from_mjd(
            [mean_impact_mjd],
            impact_results.mean_impact_time.scale,
        )

    # Load sorcha observations
    observations = Observations.from_parquet(
        f"{paths['sorcha_dir']}/observations_{orbit_id}.parquet"
    )

    # Compute the number of singletons and tracklets in each window
    observation_cadence = compute_observation_cadence(observations)

    return ImpactorResultSummary.from_kwargs(
        orbit=impactor_orbits,
        mean_impact_time=mean_impact_time,
        windows=[len(impact_results)],
        nights=[pc.max(impact_results.observation_nights)],
        observations=[pc.max(impact_results.observation_count)],
        singletons=[pc.sum(observation_cadence.singletons)],
        tracklets=[pc.sum(observation_cadence.tracklets)],
        discovery_time=discovery_dates.discovery_date,
        warning_time=warning_times.warning_time,
        realization_time=realization_times.realization_time,
        maximum_impact_probability=[pc.max(impact_results.impact_probability)],
        results_timing=results_timings,
    )


summarize_impact_study_object_results_remote = ray.remote(
    summarize_impact_study_object_results
)


def summarize_impact_study_results(
    run_dir: str, out_dir: str, plot: bool = True, max_processes: Optional[int] = 1
) -> ImpactorResultSummary:
    """
    Summarize the impact study results.
    """
    assert run_dir != out_dir, "run_dir and out_dir must be different"

    if max_processes is None:
        max_processes = mp.cpu_count()

    if max_processes > 1:
        initialize_use_ray()

    orbit_ids = [os.path.basename(dir) for dir in glob.glob(f"{run_dir}/*")]
    results = ImpactorResultSummary.empty()
    futures = []
    for orbit_id in orbit_ids:

        if max_processes > 1:
            futures.append(
                summarize_impact_study_object_results_remote.remote(run_dir, orbit_id)
            )
        else:
            try:
                result = summarize_impact_study_object_results(run_dir, orbit_id)

                results = qv.concatenate([results, result])
            except Exception as e:
                logger.error(
                    f"Error summarizing impact study results for {orbit_id}: {e}"
                )

        if len(futures) > max_processes * 1.5:
            finished, futures = ray.wait(futures, num_returns=1)
            try:
                result = ray.get(finished[0])
                results = qv.concatenate([results, result])
            except Exception as e:
                logger.error(
                    f"Error summarizing impact study results for {orbit_id}: {e}"
                )

    while len(futures) > 0:
        finished, futures = ray.wait(futures, num_returns=1)
        try:
            result = ray.get(finished[0])
            results = qv.concatenate([results, result])
        except Exception as e:
            logger.error(f"Error summarizing impact study results for {orbit_id}: {e}")

    os.makedirs(out_dir, exist_ok=True)
    results.to_parquet(os.path.join(out_dir, "impactor_results_summary.parquet"))
    logger.info(f"Saved impact study results to {out_dir}")

    if plot:
        make_analysis_plots(results, run_dir, out_dir)

    return results
