import logging
import os
import pathlib
from typing import Union

import pyarrow as pa
import pyarrow.parquet as pq
import quivr as qv
from adam_core.time import Timestamp

from adam_impact_study.types import WindowResult
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
    orbit_dirs = run_dir_path.glob("*")

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
