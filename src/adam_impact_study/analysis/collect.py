import logging
import os
import pathlib
from typing import Tuple, Union

import pyarrow as pa
import pyarrow.parquet as pq
import quivr as qv
from adam_core.time import Timestamp

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


def collect_all_observations(run_dir: Union[str, pathlib.Path]) -> Observations:
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
    run_dir_path = pathlib.Path(run_dir).absolute()
    # Find all files that match the pattern observations_{orbit_id}.parquet
    # that may be in layers of subdirectories
    observations_files = run_dir_path.glob("**/*observations_*.parquet")
    observations = Observations.empty()
    for observations_file in observations_files:
        observations = qv.concatenate([observations, Observations.from_parquet(observations_file)])

    return observations


def collect_all_timings(run_dir: Union[str, pathlib.Path]) -> ResultsTiming:
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
    run_dir_path = pathlib.Path(run_dir).absolute()
    # Find all files that match the pattern timings_{orbit_id}.parquet
    # that may be in layers of subdirectories
    timings_files = run_dir_path.glob("**/*timings.parquet")
    timings = ResultsTiming.empty()
    for timings_file in timings_files:
        timings = qv.concatenate([timings, ResultsTiming.from_parquet(timings_file)])

    return timings


def collect_all_impactor_orbits(run_dir: Union[str, pathlib.Path]) -> ImpactorOrbits:
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
    run_dir_path = pathlib.Path(run_dir).absolute()
    # Find all files that match the pattern impactor_orbit.parquet
    # that may be in layers of subdirectories
    impactor_orbits_files = run_dir_path.glob("**/*impactor_orbit.parquet")
    impactor_orbits = ImpactorOrbits.empty()
    for impactor_orbits_file in impactor_orbits_files:
        impactor_orbits = qv.concatenate([impactor_orbits, ImpactorOrbits.from_parquet(impactor_orbits_file)])

    return impactor_orbits


def collect_all_results(run_dir: Union[str, pathlib.Path]) -> Tuple[ImpactorOrbits, Observations, ResultsTiming, WindowResult]:
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
    impactor_orbits = collect_all_impactor_orbits(run_dir)
    observations = collect_all_observations(run_dir)
    timings = collect_all_timings(run_dir)
    window_results = collect_all_window_results(run_dir)

    return impactor_orbits, observations, timings, window_results
