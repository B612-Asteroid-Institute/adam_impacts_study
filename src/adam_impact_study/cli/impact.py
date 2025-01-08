import argparse
import logging
import os
import sqlite3
from typing import Optional

import pyarrow as pa
import pyarrow.compute as pc
from adam_core.orbits import Orbits
from adam_core.time import Timestamp

from adam_impact_study.analysis import plot_ip_over_time
from adam_impact_study.impacts_study import run_impact_study_all

logger = logging.getLogger(__name__)


def run_impact_study(
    orbit_file: str,
    run_dir: str,
    max_processes: int = 1,
    pointing_file: Optional[str] = None,
    population_config: Optional[str] = None,
    object_id: Optional[str] = None,
    seed: Optional[int] = None,
) -> None:
    """Run impact study on provided orbits."""
    # Load orbits directly from parquet
    logger.info(f"Loading orbits from {orbit_file}")
    impactor_orbits = Orbits.from_parquet(orbit_file)

    if object_id:
        object_ids = object_id.split(",")
        impactor_orbits = impactor_orbits.apply_mask(
            pc.is_in(impactor_orbits.object_id, pa.array(object_ids))
        )
        logger.info(f"Filtered objects: {object_ids}")

    # Extract the date of the first pointing from the pointing file
    conn = sqlite3.connect(pointing_file)
    cursor = conn.cursor()
    cursor.execute(
        f"SELECT observationStartMJD as observationStartMJD_TAI FROM observations ORDER BY observationStartMJD_TAI LIMIT 1"
    )
    survey_start = cursor.fetchone()[0]
    survey_start = Timestamp.from_mjd([survey_start], scale="tai")
    conn.close()

    # Note, we want to remove this hard-coded value and replace with a superclass that includes impact date
    # If any orbits impact date is before the survey start, throw a ValueError
    impact_date = impactor_orbits.coordinates.time.add_days(30)
    if impact_date.min().mjd()[0].as_py() < survey_start.mjd()[0].as_py():
        raise ValueError(
            f"Orbit impact date is before survey start: {impact_date.min().mjd()[0].as_py()} < {survey_start.mjd()[0].as_py()}"
        )

    logger.info(f"Processing {len(impactor_orbits)} orbits")

    # Create output directory
    os.makedirs(run_dir, exist_ok=True)

    # Run impact study
    logger.info("Starting impact study...")
    impact_study_results = run_impact_study_all(
        impactor_orbits,
        population_config,
        pointing_file,
        run_dir,
        max_processes=max_processes,
        seed=seed,
    )

    logger.info("Generating plots...")
    plot_ip_over_time(impact_study_results, run_dir, impactor_orbits, survey_start)
    logger.info(f"Results saved to {run_dir}")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(description="Run impact study")
    parser.add_argument("orbit_file", help="Path to orbit file (parquet format)")
    parser.add_argument("run_dir", help="Directory for this study run")
    parser.add_argument(
        "--max-processes",
        type=int,
        default=1,
        help="Maximum number of processes to use",
    )
    parser.add_argument("--pointing-file", help="Path to pointing database file")
    parser.add_argument(
        "--population-config", help="Path to population configuration file"
    )
    parser.add_argument("--object-id", help="Specific object ID to process")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--seed", type=int, help="Seed for Sorcha", default=None)

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    run_impact_study(
        args.orbit_file,
        args.run_dir,
        max_processes=args.max_processes,
        pointing_file=args.pointing_file,
        population_config=args.population_config,
        object_id=args.object_id,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
