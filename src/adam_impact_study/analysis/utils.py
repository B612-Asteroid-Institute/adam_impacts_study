import glob
import logging
import os

import pyarrow as pa
import pyarrow.parquet as pq
import quivr as qv
from adam_core.time import Timestamp

from adam_impact_study.types import WindowResult
from adam_impact_study.utils import get_study_paths

logger = logging.getLogger(__name__)


def collect_orbit_window_results(run_dir: str, orbit_id: str) -> WindowResult:
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
    paths = get_study_paths(run_dir, orbit_id)
    orbit_dir = paths["orbit_base_dir"]
    window_results = WindowResult.empty()
    window_dirs = sorted(glob.glob(f"{orbit_dir}/windows/*"))
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
            # TODO: Backwards compatibility with old window results (DELETE THIS LATER)
            window_result_table = pq.read_table(window_result_file)
            if "status" not in window_result_table.columns:
                window_result = WindowResult.from_pyarrow(
                    window_result_table.add_column(
                        3, "status", pa.array(["complete"], pa.large_string())
                    )
                )
            else:
                window_result = WindowResult.from_pyarrow(window_result_table)

        window_results = qv.concatenate([window_results, window_result])

    return window_results


def collect_all_window_results(run_dir: str) -> WindowResult:
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
    # Initialize empty results
    window_results = []

    # Get all window result files directly
    window_files = glob.glob(f"{run_dir}/*/windows/*/window_result.parquet")

    if not window_files:
        logger.warning(f"No window results found in {run_dir}")
        return WindowResult.empty()

    # Single pass concatenation
    for f in window_files:
        try:
            window_results.append(WindowResult.from_parquet(f))
        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")

    window_results = qv.concatenate(window_results)

    return window_results
