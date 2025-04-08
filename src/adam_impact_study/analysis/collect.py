import logging
import multiprocessing as mp
import os
import pathlib
from typing import Iterator, List, Tuple, Union

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import quivr as qv
import ray
from adam_core.ray_cluster import initialize_use_ray
from adam_core.time import Timestamp
from adam_core.utils.iter import _iterate_chunks

from adam_impact_study.types import (
    ImpactorOrbits,
    Observations,
    ResultsTiming,
    WindowResult,
)
from adam_impact_study.utils import get_study_paths

logger = logging.getLogger(__name__)


def collect_orbit_window_results(
    run_dir: Union[str, pathlib.Path], orbit_id: str
) -> WindowResult:
    """Collect window results for a single orbit.

    Parameters
    ----------
    run_dir : str
        Base directory for the run
    orbit_id : str
        ID of the orbit to collect results for

    Returns
    -------
    WindowResult
        Combined window results for the orbit
    """
    run_dir_path = pathlib.Path(run_dir).absolute()
    paths = get_study_paths(run_dir_path, orbit_id)
    orbit_dir = paths["orbit_base_dir"]
    window_results = WindowResult.empty()
    window_dirs = sorted(orbit_dir.glob("windows/*"))
    if len(window_dirs) == 0:
        return WindowResult.empty()
    for window_dir in window_dirs:
        window_result_file = f"{window_dir}/window_result.parquet"
        if not os.path.exists(window_result_file):
            window_result = WindowResult.from_kwargs(
                orbit_id=[orbit_id],
                window=[os.path.basename(window_dir)],
                status=["incomplete"],
                observation_start=Timestamp.nulls(1, scale="utc"),
                observation_end=Timestamp.nulls(1, scale="utc"),
                mean_impact_time=Timestamp.nulls(1, scale="tdb"),
                minimum_impact_time=Timestamp.nulls(1, scale="tdb"),
                maximum_impact_time=Timestamp.nulls(1, scale="tdb"),
            )
        else:
            # Backwards compatibility for older results
            try:
                window_result = WindowResult.from_parquet(window_result_file)

            except ValueError:
                window_result_table = pq.read_table(window_result_file)
                if "condition_id" not in window_result_table.columns:
                    window_result_table = window_result_table.add_column(
                        3, "condition_id", pa.array(["default"], pa.large_string())
                    )

                if "status" not in window_result_table.columns:
                    window_result_table = window_result_table.add_column(
                        4, "status", pa.array(["complete"], pa.large_string())
                    )
                window_result = WindowResult.from_pyarrow(window_result_table)

        window_results = qv.concatenate([window_results, window_result])

    return window_results


def collect_all_window_results(run_dir: Union[str, pathlib.Path]) -> WindowResult:
    """Collect all window results from a run directory.

    Parameters
    ----------
    run_dir : str
        Base directory for the run

    Returns
    -------
    WindowResult
        Combined window results for all orbits
    """
    run_dir_path = pathlib.Path(run_dir).absolute()

    # Initialize empty results
    window_results = []

    # Get all window result files directly
    orbit_dirs = [d for d in run_dir_path.glob("*") if d.is_dir()]

    if not orbit_dirs:
        logger.warning(f"No orbit directories found in {run_dir_path}")
        return WindowResult.empty()

    window_results = WindowResult.empty()
    for orbit_dir in orbit_dirs:
        window_results_orbit = collect_orbit_window_results(
            run_dir_path, orbit_dir.name
        )
        window_results = qv.concatenate([window_results, window_results_orbit])

    return window_results


@ray.remote
def load_quivr_table_ray_worker(
    klass: qv.Table, paths: List[Union[str, pathlib.Path]]
) -> qv.Table:
    """Load a quivr table from a path.

    Parameters
    ----------
    klass : qv.Table
        The class of the table to load
    paths : List[Union[str, pathlib.Path]]
        The paths to the files to load

    Returns
    -------
    qv.Table
        The loaded table
    """
    results = klass.empty()
    for path in paths:
        results = qv.concatenate([results, klass.from_parquet(path)])
    return results


def load_quivr_table_ray(
    klass: qv.Table,
    paths: Iterator[Union[str, pathlib.Path]],
    chunk_size: int = 100,
    max_processes: int = 1,
) -> qv.Table:
    """Load a quivr table from a path.

    Parameters
    ----------
    klass : qv.Table
        The class of the table to load
    paths : List[Union[str, pathlib.Path]]
        The paths to the files to load

    Returns
    -------
    qv.Table
        The loaded table
    """

    # Split the list of paths into max_processes chunks
    results = klass.empty()
    paths = list(paths)

    # If single process, skip ray
    if max_processes == 1:
        for path in paths:
            results = qv.concatenate([results, klass.from_parquet(path)])
        return results

    initialize_use_ray(num_cpus=max_processes)

    futures = []
    for path_chunk in _iterate_chunks(paths, chunk_size):
        futures.append(load_quivr_table_ray_worker.remote(klass, path_chunk))

        if len(futures) > max_processes * 1.5:
            finished, futures = ray.wait(futures, num_returns=1)
            results = qv.concatenate([results, ray.get(finished[0])], validate=False)

    while len(futures) > 0:
        logger.info(f"Waiting for {len(futures)} futures to finish")
        finished, futures = ray.wait(futures, num_returns=1)
        results = qv.concatenate([results, ray.get(finished[0])], validate=False)

    # Validate at the end instead of during the loop to speed things up.
    results.validate()
    return results


def collect_all_window_results_new(
    run_dir: Union[str, pathlib.Path], max_processes: int = 1
) -> WindowResult:
    """Collect all window results from a run directory.

    Parameters
    ----------
    run_dir : str
        Base directory for the run

    Returns
    -------
    WindowResult
        Combined window results for all orbits
    """
    logger.info(f"Collecting window results for {run_dir}")
    run_dir_path = pathlib.Path(run_dir).absolute()
    window_results = WindowResult.empty()
    window_files = list(run_dir_path.glob("**/*window_result.parquet"))
    logger.info(f"Loading {len(window_files)} window results")
    window_results = load_quivr_table_ray(
        WindowResult, window_files, max_processes=max_processes
    )
    return window_results


def create_missing_window_results_worker(
    observations: Observations, window_results: WindowResult, orbit_ids: List[str]
) -> WindowResult:
    """Create missing window results for observations that do not have correspondingwindow results.

    Parameters
    ----------
    observations : Observations
        Observations to create missing window results for
    window_results : WindowResult
        Window results to create missing window results for
    orbit_ids : List[str]
        Orbit IDs to create missing window results for

    Returns
    -------
    WindowResult
        Window results with missing window results created
    """
    missing_window_results = WindowResult.empty()
    for orbit_id in orbit_ids:
        orbit_observations = observations.select("orbit_id", orbit_id)
        unique_nights = pc.unique(orbit_observations.observing_night).sort()
        for night in unique_nights[2:]:
            mask = pc.less_equal(orbit_observations.observing_night, night)
            observations_window = orbit_observations.apply_mask(mask)
            if len(observations_window) < 6:
                logger.debug(
                    f"Not enough observations for a least-squares fit for night {night}"
                )
                continue

            start_night = pc.min(observations_window.observing_night)
            end_night = pc.max(observations_window.observing_night)
            window = f"{start_night.as_py()}_{end_night.as_py()}"

            # Check if the window already exists
            existing = window_results.select("window", window).select(
                "orbit_id", orbit_id
            )
            if len(existing) > 0:
                continue

            # Populate the missing window result
            missing_window_results = qv.concatenate(
                [
                    missing_window_results,
                    WindowResult.from_kwargs(
                        orbit_id=[orbit_id],
                        object_id=[orbit_id],
                        condition_id=[None],
                        status=["incomplete"],
                        window=[window],
                        observation_start=observations_window.coordinates.time.min(),
                        observation_end=observations_window.coordinates.time.max(),
                        observation_count=[len(observations_window)],
                        observations_rejected=[None],
                        observation_nights=[
                            len(pc.unique(observations_window.observing_night))
                        ],
                        impact_probability=[None],
                        mean_impact_time=Timestamp.nulls(1, scale="tdb"),
                        minimum_impact_time=Timestamp.nulls(1, scale="tdb"),
                        maximum_impact_time=Timestamp.nulls(1, scale="tdb"),
                        stddev_impact_time=[None],
                        error=[None],
                        od_runtime=[None],
                        ip_runtime=[None],
                        window_runtime=[None],
                    ),
                ]
            )

    return missing_window_results


create_missing_window_results_worker_remote = ray.remote(
    create_missing_window_results_worker
)


def create_missing_window_results(
    observations: Observations, window_results: WindowResult, max_processes: int = 1, chunk_size: int = 100
) -> WindowResult:
    """Create missing window results for observations that do not have correspondingwindow results.

    Parameters
    ----------
    observations : Observations
        Observations to create missing window results for
    window_results : WindowResult
        Window results to create missing window results for

    Returns
    -------
    WindowResult
        Window results with missing window results created
    """
    logger.info("Creating missing window results")
    missing_window_results = WindowResult.empty()
    unique_orbit_ids = pc.unique(observations.orbit_id).sort()

    if max_processes == 1:
        missing_window_results = create_missing_window_results_worker(
            observations, window_results, unique_orbit_ids
        )
        logger.info(f"Created {len(missing_window_results)} missing window results")
        return qv.concatenate([window_results, missing_window_results])

    initialize_use_ray(num_cpus=max_processes)
    futures = []
    observations_ref = ray.put(observations)
    window_results_ref = ray.put(window_results)
    orbit_id_chunks = list(_iterate_chunks(unique_orbit_ids, chunk_size))
    completed_chunks = 0
    total_chunks = len(orbit_id_chunks)
    logger.info(f"Creating missing window results for {total_chunks} chunks")
    for orbit_id_chunk in orbit_id_chunks:
        futures.append(
            create_missing_window_results_worker_remote.remote(
                observations_ref, window_results_ref, orbit_id_chunk
            )
        )

        if len(futures) > max_processes * 1.5:
            finished, futures = ray.wait(futures, num_returns=1)
            missing_window_results = qv.concatenate(
                [missing_window_results, ray.get(finished[0])], validate=False
            )
            completed_chunks += 1
            logger.info(f"Completed {completed_chunks}/{total_chunks} chunks")

    while len(futures) > 0:
        finished, futures = ray.wait(futures, num_returns=1)
        missing_window_results = qv.concatenate(
            [missing_window_results, ray.get(finished[0])], validate=False
        )
        completed_chunks += 1
        logger.info(f"Completed {completed_chunks}/{total_chunks} chunks")

    logger.info(f"Created {len(missing_window_results)} missing window results")

    return qv.concatenate([window_results, missing_window_results])


def collect_all_observations(
    run_dir: Union[str, pathlib.Path], max_processes: int = 1
) -> Observations:
    """Collect all observations from a run directory.

    Parameters
    ----------
    run_dir : str
        Base directory for the run

    Returns
    -------
    Observations
        Combined observations for all orbits
    """
    logger.info(f"Collecting observations for {run_dir}")
    run_dir_path = pathlib.Path(run_dir).absolute()
    # Find all files that match the pattern observations_{orbit_id}.parquet
    # that may be in layers of subdirectories
    observations_files = list(run_dir_path.glob("**/*observations_*.parquet"))
    logger.info(f"Loading {len(observations_files)} observations files")
    observations = load_quivr_table_ray(
        Observations, observations_files, max_processes=max_processes
    )
    return observations


def collect_all_timings(
    run_dir: Union[str, pathlib.Path], max_processes: int = 1
) -> ResultsTiming:
    """Collect all timings from a run directory.

    Parameters
    ----------
    run_dir : str
        Base directory for the run

    Returns
    -------
    Timings
        Combined timings for all orbits
    """
    logger.info(f"Collecting timings for {run_dir}")
    run_dir_path = pathlib.Path(run_dir).absolute()
    # Find all files that match the pattern timings_{orbit_id}.parquet
    # that may be in layers of subdirectories
    timings_files = list(run_dir_path.glob("**/*timings.parquet"))
    logger.info(f"Loading {len(timings_files)} timings files")
    timings = load_quivr_table_ray(
        ResultsTiming, timings_files, max_processes=max_processes
    )

    return timings


def collect_all_impactor_orbits(
    run_dir: Union[str, pathlib.Path], max_processes: int = 1
) -> ImpactorOrbits:
    """Collect all impactor orbits from a run directory.

    Parameters
    ----------
    run_dir : str
        Base directory for the run

    Returns
    -------
    ImpactorOrbits
        Combined impactor orbits for all orbits
    """
    logger.info(f"Collecting impactor orbits for {run_dir}")
    run_dir_path = pathlib.Path(run_dir).absolute()
    # Find all files that match the pattern impactor_orbit.parquet
    # that may be in layers of subdirectories
    impactor_orbits_files = list(run_dir_path.glob("**/*impactor_orbit.parquet"))
    logger.info(f"Loading {len(impactor_orbits_files)} impactor orbits files")
    impactor_orbits = load_quivr_table_ray(
        ImpactorOrbits, impactor_orbits_files, max_processes=max_processes
    )

    return impactor_orbits


def collect_all_results(
    run_dir: Union[str, pathlib.Path],
    max_processes: int = 1,
) -> Tuple[ImpactorOrbits, Observations, ResultsTiming, WindowResult]:
    """Collect all results from a run directory.

    Parameters
    ----------
    run_dir : str
        Base directory for the run

    Returns
    -------
    Tuple[ImpactorOrbits, Observations, ResultsTiming, WindowResult]
        Combined results for all orbits
    """
    impactor_orbits = collect_all_impactor_orbits(run_dir, max_processes=max_processes)
    observations = collect_all_observations(run_dir, max_processes=max_processes)
    timings = collect_all_timings(run_dir, max_processes=max_processes)
    window_results = collect_all_window_results_new(
        run_dir, max_processes=max_processes
    )
    window_results = create_missing_window_results(observations, window_results)
    return impactor_orbits, observations, timings, window_results
