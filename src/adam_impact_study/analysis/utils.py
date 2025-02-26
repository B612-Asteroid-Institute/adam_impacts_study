import glob
import logging
import os
from typing import Optional

import quivr as qv

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
    window_result_files = sorted(
        glob.glob(f"{orbit_dir}/windows/*/window_result.parquet")
    )
    window_results = WindowResult.empty()
    for f in window_result_files:
        window_results = qv.concatenate([window_results, WindowResult.from_parquet(f)])
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
