import os
import pathlib
from dataclasses import asdict

os.environ["RAY_DEDUP_LOGS"] = "0"
import argparse
import logging
import sqlite3
from typing import Optional

import pyarrow as pa
import pyarrow.compute as pc
from adam_core.constants import KM_P_AU
from adam_core.constants import Constants as c
from adam_core.coordinates import Origin
from adam_core.dynamics.impacts import CollisionConditions
from adam_core.time import Timestamp
from adam_fo.config import check_build_exists

from adam_impact_study.analysis.plots import plot_ip_over_time
from adam_impact_study.impacts_study import run_impact_study_all
from adam_impact_study.types import ImpactorOrbits, RunConfiguration

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = c.R_EARTH_EQUATORIAL * KM_P_AU


def run_impact_study(
    orbit_file: str,
    run_dir: str,
    run_config: RunConfiguration,
    orbit_id_filter: Optional[str] = None,
    overwrite: bool = False,
) -> None:
    """Run impact study on provided orbits."""
    # Test that the build fo script has been run
    check_build_exists()

    # Load orbits directly from parquet
    logger.info(f"Loading orbits from {orbit_file}")
    impactor_orbits = ImpactorOrbits.from_parquet(orbit_file)

    # User passed a comma-delimited list of substrings to filter orbit id
    # by. We need to filter by the parent object id instead.
    filtered_orbits = impactor_orbits
    if orbit_id_filter:
        orbit_id_filter = [
            orbit_id_filter.strip() for orbit_id_filter in orbit_id_filter.split(",")
        ]
        mask = pa.array([False] * len(impactor_orbits))
        for f in orbit_id_filter:
            mask = pc.or_(mask, pc.match_substring(impactor_orbits.orbit_id, f))

        filtered_orbits = impactor_orbits.apply_mask(mask)
        logger.info(f"Selected {len(filtered_orbits)}/{len(impactor_orbits)} orbits")

    # Extract the date of the first pointing from the pointing file
    conn = sqlite3.connect(run_config.pointing_database_file)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT observationStartMJD as observationStartMJD_TAI FROM observations ORDER BY observationStartMJD_TAI LIMIT 1"
    )
    survey_start = cursor.fetchone()[0]
    survey_start = Timestamp.from_mjd([survey_start], scale="tai")
    conn.close()

    # Note, we want to remove this hard-coded value and replace with a superclass that includes impact date
    # If any orbits impact date is before the survey start, throw a ValueError
    impact_date = filtered_orbits.impact_time
    if impact_date.min().mjd()[0].as_py() < survey_start.mjd()[0].as_py():
        raise ValueError(
            f"Orbit impact date is before survey start: {impact_date.min().mjd()[0].as_py()} < {survey_start.mjd()[0].as_py()}"
        )

    logger.info(f"Processing {len(filtered_orbits)} orbits")

    # Create output directory
    run_dir_path = pathlib.Path(run_dir).absolute()
    run_dir_path.mkdir(parents=True, exist_ok=True)

    # Load run configuration
    logger.info(f"Run configuration: {asdict(run_config)}")
    run_config.to_json(run_dir_path / "run_config.json")

    # Set default collision conditions
    conditions = CollisionConditions.from_kwargs(
        condition_id=["Default - Earth"],
        collision_object=Origin.from_kwargs(code=["EARTH"]),
        collision_distance=[EARTH_RADIUS_KM],
        stopping_condition=[True],
    )

    # Run impact study
    logger.info("Starting impact study...")
    impact_study_results, results_timings = run_impact_study_all(
        filtered_orbits,
        run_config.pointing_database_file,
        run_dir_path,
        assist_initial_dt=run_config.assist_initial_dt,
        assist_min_dt=run_config.assist_min_dt,
        assist_adaptive_mode=run_config.assist_adaptive_mode,
        assist_epsilon=run_config.assist_epsilon,
        monte_carlo_samples=run_config.monte_carlo_samples,
        conditions=conditions,
        max_processes=run_config.max_processes,
        seed=run_config.seed,
        overwrite=overwrite,
    )


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(description="Run impact study")
    parser.add_argument("orbit_file", help="Path to orbit file (parquet format)")
    parser.add_argument("run_dir", help="Directory for this study run")
    parser.add_argument("--pointing-file", help="Path to pointing database file")
    parser.add_argument("--run-config", help="Path to run configuration file")
    parser.add_argument(
        "--max-processes",
        type=int,
        default=None,
        help="Maximum number of processes to use (override run config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for pipeline (override value in run config)",
        default=None,
    )

    parser.add_argument(
        "--orbit-id-filter",
        help="Comma-delimited list of substrings to filter orbit id by",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results in run directory",
        default=False,
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        os.environ["ADAM_LOG_LEVEL"] = "DEBUG"
    else:
        logging.basicConfig(level=logging.INFO)
        os.environ["ADAM_LOG_LEVEL"] = "INFO"

    # Use a default run config, unless one is specified
    run_config = RunConfiguration(
        monte_carlo_samples=1000,
        assist_epsilon=1e-6,
        assist_min_dt=1e-9,
        assist_initial_dt=1e-6,
        assist_adaptive_mode=1,
        seed=612,
        max_processes=1,
        pointing_database_file=args.pointing_file,
    )

    if args.run_config is not None:
        run_config = RunConfiguration.from_json(args.run_config)

    if args.seed is not None:
        run_config.seed = args.seed
        logger.info(f"Overriding run config seed {args.seed}.")

    if args.max_processes is not None:
        run_config.max_processes = args.max_processes
        logger.info(f"Overriding run config max processes {args.max_processes}.")

    run_impact_study(
        args.orbit_file,
        args.run_dir,
        run_config=run_config,
        orbit_id_filter=args.orbit_id_filter,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
